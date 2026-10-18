"""Precision, recall and F1 for batches of detections, and over many batches.

:class:`DetectionMetrics` evaluates each batch with the same engine and
matching as ``COCOeval`` and keeps, per detection, only its category, its
score and one outcome byte per IoU threshold. No boxes or masks are retained
between batches, yet the accumulated result equals one evaluation over every
image: matching is per image, and a score threshold keeps a prefix of each
image's score order (see ``_detection_metrics``), so the stored outcomes are
exactly the ones a single evaluation would count.

Example::

    from ultrafast_pycocotools.metrics import DetectionMetrics

    metric = DetectionMetrics(iou_thrs=0.5, score_thr="best_f1")
    for images, targets in loader:
        preds = model(images)          # [{"boxes", "scores", "labels"}, ...]
        batch = metric.update(preds, targets)
    total = metric.compute()
    total["macro"]["f1"], total["score_thr"]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import _detection_metrics
from . import mask as maskUtils
from .coco import COCO
from .cocoeval import COCOeval

__all__ = ["DetectionMetrics", "detection_metrics"]

# `max_dets=None`: a per-(image, category) limit no batch reaches. The engine
# only compares against it; it sizes nothing. Fits a 32-bit usize.
_UNLIMITED = 2**31 - 1


def _quiet(*_):
    pass


def _to_numpy(value: Any) -> np.ndarray:
    """NumPy view of an array, list or tensor; tensors move to the CPU first.

    Duck-typed so that PyTorch is never imported.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class DetectionMetrics:
    """Per-batch and accumulated precision, recall and F1.

    Args:
        iou_type: ``"bbox"`` or ``"segm"`` for :meth:`update`; ``update_coco``
            also takes ``"keypoints"``.
        iou_thrs: one IoU threshold or a sequence, used as given. A scalar
            gives results without a threshold axis.
        score_thr: ``None`` (every detection), a number (``score >=
            score_thr``) or ``"best_f1"`` (the cut that maximizes F1 per IoU
            threshold).
        optimize: the F1 ``"best_f1"`` maximizes, ``"macro"`` or ``"micro"``.
        class_agnostic: match across labels, as ``params.useCats = 0``.
        box_format: ``"xyxy"`` or ``"xywh"``, for :meth:`update`'s boxes.
        area_range: ``(lo, hi)`` applies COCO's area rule (ground truth outside
            it is ignored, and so is an unmatched detection outside it);
            ``None`` filters nothing.
        max_dets: detections kept per (image, category), highest scores first,
            as COCO's ``maxDets``; ``None`` keeps all.

    Results have the layout of ``COCOeval.detection_metrics``. Each image may
    be added once; the same image twice would be counted twice.
    """

    def __init__(
        self,
        iou_type: str = "bbox",
        iou_thrs: float | Sequence[float] = 0.5,
        score_thr: float | str | None = None,
        optimize: str = "macro",
        class_agnostic: bool = False,
        box_format: str = "xyxy",
        area_range: Sequence[float] | None = None,
        max_dets: int | None = None,
    ):
        if iou_type not in ("bbox", "segm", "keypoints"):
            raise ValueError(f"iou_type must be 'bbox', 'segm' or 'keypoints', not {iou_type!r}")
        if box_format not in ("xyxy", "xywh"):
            raise ValueError(f"box_format must be 'xyxy' or 'xywh', not {box_format!r}")
        _detection_metrics.check_options(score_thr, optimize)
        self._scalar = np.ndim(iou_thrs) == 0
        self.iou_thrs = np.atleast_1d(np.asarray(iou_thrs, dtype=np.float64))
        if self.iou_thrs.ndim != 1 or self.iou_thrs.size == 0:
            raise ValueError("iou_thrs must be a number or a non-empty sequence of numbers")
        if area_range is not None:
            area_range = [float(area_range[0]), float(area_range[1])]
        if max_dets is not None and int(max_dets) < 0:
            raise ValueError(f"max_dets must be None or at least 0, not {max_dets}")
        self.iou_type = iou_type
        self.score_thr = score_thr
        self.optimize = optimize
        self.class_agnostic = bool(class_agnostic)
        self.box_format = box_format
        self.area_range = area_range
        self.max_dets = None if max_dets is None else int(max_dets)
        self.reset()

    def reset(self) -> None:
        """Forget every batch added so far."""
        self._cats: list[np.ndarray] = []
        self._scores: list[np.ndarray] = []
        self._outcomes: list[np.ndarray] = []
        self._npig: dict[int, int] = {}
        self._seen: set = set()
        self._next_id = 1

    # ------------------------------------------------------------------
    # adding batches
    # ------------------------------------------------------------------

    def update(self, preds: Sequence[dict], targets: Sequence[dict]) -> dict:
        """Add one batch of per-image arrays and return that batch's metrics.

        ``preds[i]`` has ``boxes`` (``(N, 4)``, for bbox), ``masks``
        (``(N, H, W)``, for segm), ``scores`` and ``labels``. ``targets[i]``
        has ``boxes`` or ``masks`` and ``labels``, and optionally ``iscrowd``
        and ``area`` (default: box area, or mask pixel count). Either may give
        ``image_id``; without one, ids are numbered across updates.
        """
        if self.iou_type == "keypoints":
            raise ValueError("keypoints take COCO input; use update_coco()")
        if len(preds) != len(targets):
            raise ValueError(f"preds has {len(preds)} images but targets has {len(targets)}")
        image_ids = self._image_ids(preds, targets)

        geometry = "masks" if self.iou_type == "segm" else "boxes"
        images, gt_anns, dt_anns, labels = [], [], [], set()
        for i, (pred, target, img_id) in enumerate(zip(preds, targets, image_ids)):
            for name, item, keys in (("preds", pred, (geometry, "scores", "labels")),
                                     ("targets", target, (geometry, "labels"))):
                for key in keys:
                    if key not in item:
                        raise ValueError(f"{name}[{i}] has no {key!r}")
            gt_geom, gt_cats, gt_extra = self._read_side(target, i, "targets", geometry)
            dt_geom, dt_cats, _ = self._read_side(pred, i, "preds", geometry)
            scores = _to_numpy(pred["scores"]).astype(np.float64).ravel()
            if scores.shape[0] != len(dt_cats):
                raise ValueError(f"preds[{i}] has {len(dt_cats)} labels but {scores.shape[0]} scores")
            image = {"id": img_id}
            if geometry == "masks":
                sizes = {m.shape[1:] for m in (gt_geom, dt_geom) if m.shape[0]}
                if len(sizes) > 1:
                    raise ValueError(f"image {i}: preds and targets masks differ in size {sorted(sizes)}")
                shape = sizes.pop() if sizes else gt_geom.shape[1:]
                image["height"], image["width"] = int(shape[0]), int(shape[1])
            images.append(image)
            labels.update(gt_cats.tolist())
            labels.update(dt_cats.tolist())

            gt_rles = self._encode(gt_geom) if geometry == "masks" else None
            iscrowd, areas = gt_extra
            for n in range(len(gt_cats)):
                ann = {"id": len(gt_anns) + 1, "image_id": img_id,
                       "category_id": int(gt_cats[n]), "iscrowd": int(iscrowd[n])}
                if gt_rles is None:
                    ann["bbox"] = gt_geom[n].tolist()
                    ann["area"] = float(gt_geom[n, 2] * gt_geom[n, 3]) if areas is None else float(areas[n])
                else:
                    ann["segmentation"] = gt_rles[n]
                    ann["bbox"] = maskUtils.toBbox(gt_rles[n]).tolist()
                    ann["area"] = float(maskUtils.area(gt_rles[n])) if areas is None else float(areas[n])
                gt_anns.append(ann)
            dt_rles = self._encode(dt_geom) if geometry == "masks" else None
            for n in range(len(dt_cats)):
                ann = {"image_id": img_id, "category_id": int(dt_cats[n]), "score": float(scores[n])}
                if dt_rles is None:
                    ann["bbox"] = dt_geom[n].tolist()
                else:
                    ann["segmentation"] = dt_rles[n]
                dt_anns.append(ann)

        gt = COCO({"images": images, "annotations": gt_anns,
                   "categories": [{"id": c} for c in sorted(labels)]}, verbose=False)
        dt = gt.loadRes(dt_anns)
        return self._add(gt, dt, image_ids, sorted(labels))

    def update_coco(self, gt: COCO | dict | str, dt: Any) -> dict:
        """Add COCO data and return its metrics.

        ``gt`` is a ``COCO`` object, a dataset dictionary or a path; ``dt`` is
        a result handle from ``gt.loadRes`` or anything ``loadRes`` accepts.
        Categories are the dataset's, as in ``COCOeval``.
        """
        gt = gt if isinstance(gt, COCO) else COCO(gt, verbose=False)
        dt = dt if isinstance(dt, COCO) else gt.loadRes(dt)
        image_ids = sorted(gt.getImgIds())
        repeated = self._seen.intersection(image_ids)
        if repeated:
            raise ValueError(f"image ids already added: {sorted(repeated)[:10]}")
        return self._add(gt, dt, image_ids, sorted(gt.getCatIds()))

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------

    def compute(self) -> dict:
        """Metrics over every image added since construction or ``reset()``."""
        if self._cats:
            cats = np.concatenate(self._cats)
            scores = np.concatenate(self._scores)
            outcomes = np.concatenate(self._outcomes)
        else:
            cats = np.zeros(0, dtype=np.int64)
            scores = np.zeros(0, dtype=np.float64)
            outcomes = np.zeros((0, self.iou_thrs.size), dtype=np.uint8)
        return self._summarize(cats, scores, outcomes, self._npig)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _image_ids(self, preds, targets) -> list:
        given = []
        for i, (pred, target) in enumerate(zip(preds, targets)):
            ids = {item["image_id"] for item in (pred, target) if "image_id" in item}
            ids = {int(_to_numpy(v)) for v in ids}
            if len(ids) > 1:
                raise ValueError(f"image {i}: preds and targets give different image_id {sorted(ids)}")
            given.append(ids.pop() if ids else None)
        if all(v is None for v in given):
            start = self._next_id
            while any(i in self._seen for i in range(start, start + len(given))):
                start += 1
            image_ids = list(range(start, start + len(given)))
            self._next_id = start + len(given)
        elif any(v is None for v in given):
            raise ValueError("give image_id for every image in a batch or for none")
        else:
            image_ids = given
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("image_id repeats within the batch")
        repeated = self._seen.intersection(image_ids)
        if repeated:
            raise ValueError(f"image ids already added: {sorted(repeated)[:10]}")
        return image_ids

    def _read_side(self, item, i, name, geometry):
        labels = _to_numpy(item["labels"]).astype(np.int64).ravel()
        n = labels.shape[0]
        geom = _to_numpy(item[geometry])
        if geometry == "boxes":
            geom = geom.astype(np.float64)
            if geom.size == 0:
                geom = geom.reshape(0, 4)
            if geom.shape != (n, 4):
                raise ValueError(f"{name}[{i}]['boxes'] must have shape ({n}, 4), not {geom.shape}")
            if self.box_format == "xyxy":
                geom = np.column_stack([geom[:, 0], geom[:, 1],
                                        geom[:, 2] - geom[:, 0], geom[:, 3] - geom[:, 1]])
        else:
            if geom.ndim != 3 or geom.shape[0] != n:
                raise ValueError(f"{name}[{i}]['masks'] must have shape ({n}, H, W), not {geom.shape}")
        extra = (None, None)
        if name == "targets":
            iscrowd = (_to_numpy(item["iscrowd"]).astype(np.int64).ravel()
                       if "iscrowd" in item else np.zeros(n, dtype=np.int64))
            areas = _to_numpy(item["area"]).astype(np.float64).ravel() if "area" in item else None
            for key, value in (("iscrowd", iscrowd), ("area", areas)):
                if value is not None and value.shape[0] != n:
                    raise ValueError(f"targets[{i}][{key!r}] has {value.shape[0]} entries, not {n}")
            extra = (iscrowd, areas)
        return geom, labels, extra

    @staticmethod
    def _encode(masks: np.ndarray) -> list:
        if masks.shape[0] == 0:
            return []
        return maskUtils.encode(np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8)))

    def _add(self, gt: COCO, dt: COCO, image_ids: list, cat_ids: list) -> dict:
        if not image_ids:
            # COCOeval reads an empty imgIds as "every image"; nothing to add.
            return self._summarize(np.zeros(0, dtype=np.int64), np.zeros(0),
                                   np.zeros((0, self.iou_thrs.size), dtype=np.uint8), {})
        ev = COCOeval(gt, dt, self.iou_type, print_function=_quiet)
        p = ev.params
        p.imgIds = list(image_ids)
        p.catIds = list(cat_ids)
        p.useCats = 0 if self.class_agnostic else 1
        p.iouThrs = self.iou_thrs.copy()
        p.areaRng = [self.area_range if self.area_range is not None else [0.0, float("inf")]]
        p.areaRngLbl = ["all"]
        p.maxDets = [_UNLIMITED if self.max_dets is None else self.max_dets]
        # The curves are not read; one recall point keeps their tensors small.
        p.recThrs = np.array([0.0])
        ev.evaluate()
        out = ev._engine.outcomes(0, p.maxDets[0])

        slots = np.asarray(ev.params.catIds if p.useCats else [-1], dtype=np.int64)
        outcome = out["outcome"]
        # A detection ignored at every threshold is in no count and no cut.
        counted = (outcome != _detection_metrics.IGNORED).any(axis=1)
        cats = slots[out["cat_slot"].astype(np.int64)[counted]]
        scores = out["score"][counted]
        outcome = outcome[counted]
        npig = {int(c): int(v) for c, v in zip(slots, out["npig"])}

        self._cats.append(cats)
        self._scores.append(scores)
        self._outcomes.append(outcome)
        for c, v in npig.items():
            self._npig[c] = self._npig.get(c, 0) + v
        self._seen.update(image_ids)
        return self._summarize(cats, scores, outcome, npig)

    def _summarize(self, cats, scores, outcomes, npig: dict) -> dict:
        # Sorted ids, as COCOeval sorts catIds: the macro mean sums in this order.
        ids = np.array(sorted(set(npig) | set(np.unique(cats).tolist())), dtype=np.int64)
        result = _detection_metrics.summarize_outcomes(
            np.searchsorted(ids, cats), scores, outcomes,
            np.array([npig.get(int(c), 0) for c in ids], dtype=np.int64), ids, self.iou_thrs,
            score_thr=self.score_thr, optimize=self.optimize,
        )
        return _detection_metrics.drop_threshold_axis(result) if self._scalar else result


def detection_metrics(preds: Sequence[dict], targets: Sequence[dict], **config) -> dict:
    """Metrics for one batch: ``DetectionMetrics(**config).update(preds, targets)``."""
    return DetectionMetrics(**config).update(preds, targets)
