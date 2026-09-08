"""Opt-in backend for RF-DETR's one-pass TorchMetrics adapter.

Validated against RF-DETR commit 39c2d3a26be81d863abbe84e153445c7813241e5 and
TorchMetrics 1.8.2. This module requires RF-DETR's training dependencies. It
changes only the supplied metric instance, preserving its update, reduction,
reset and distributed-state lifecycle. It does not replace global imports.
"""
import numpy as np
from torchmetrics.detection.helpers import CocoBackend

from ultrafast_pycocotools import COCO, COCOeval, mask


class _RFDETRCOCOeval(COCOeval):
    """Use RF-DETR's largest maxDets limit for aggregate AP, including 500."""

    def summarize(self) -> None:
        super().summarize()
        if self.params.iouType in ('bbox', 'segm') and self.params.maxDets[-1] != 100:
            self.stats[0] = self._summarize(1, maxDets=self.params.maxDets[-1])


class _MaskTools:
    """TorchMetrics passes boolean masks; the COCO mask API consumes uint8."""

    @staticmethod
    def encode(value):
        if value.dtype == np.bool_:
            value = np.asfortranarray(value, dtype=np.uint8)
        return mask.encode(value)

    def __getattr__(self, name):
        return getattr(mask, name)


_MASK_TOOLS = _MaskTools()


class _UltrafastBackend(CocoBackend):
    """Retain TorchMetrics' format conversion and substitute local COCO tools."""

    @property
    def coco(self):
        return COCO

    @property
    def cocoeval(self):
        return _RFDETRCOCOeval

    @property
    def mask_utils(self):
        return _MASK_TOOLS


def use_ultrafast(metric):
    """Select ultrafast for a newly constructed RF-DETR one-pass metric.

    Call before the first update, for each train/validation/EMA metric. The
    existing faster-coco-eval dependency remains required by RF-DETR's constructor.
    Unsupported RF-DETR/TorchMetrics layouts fail through the upstream contract
    validator. Keypoint and legacy CocoEvaluator integrations are outside scope.
    """
    from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

    if not isinstance(metric, OnePassCocoMeanAveragePrecision):
        raise TypeError('Expected RF-DETR OnePassCocoMeanAveragePrecision')
    if metric.has_updates:
        raise ValueError('Select the backend before updating the metric')
    previous = metric._coco_backend
    metric._coco_backend = _UltrafastBackend('faster_coco_eval')
    try:
        metric._validate_private_contract()
    except Exception:
        metric._coco_backend = previous
        raise
    return metric
