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

from . import _detection_metrics, _ufcoco
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
    if type(value) is np.ndarray:
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _convert(value: Any, kind: str):
    """``box_batch``'s fallback for values that are not NumPy arrays of a
    common dtype: the conversion the dictionary path applies."""
    if kind == "int":
        return int(_to_numpy(value))
    return np.asarray(_to_numpy(value), dtype=np.float64 if kind == "f64" else np.int64)


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
        self._gt_cats: list[np.ndarray] = []
        self._npig: list[np.ndarray] = []
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
        if self.iou_type == "bbox":
            if not len(preds):
                return self._keep(None, [])
            raw = _ufcoco.box_batch(
                preds if type(preds) is list else list(preds),
                targets if type(targets) is list else list(targets),
                self.box_format == "xyxy", self.iou_thrs.tolist(),
                self.area_range if self.area_range is not None else [0.0, float("inf")],
                _UNLIMITED if self.max_dets is None else self.max_dets,
                not self.class_agnostic, _convert,
            )
            return self._keep(raw, self._image_ids(raw["image_ids"]))
        image_ids = self._image_ids(self._given_ids(preds, targets))

        geometry = "masks"
        sides = []
        for i, (pred, target) in enumerate(zip(preds, targets)):
            for name, item, keys in (("preds", pred, (geometry, "scores", "labels")),
                                     ("targets", target, (geometry, "labels"))):
                for key in keys:
                    if key not in item:
                        raise ValueError(f"{name}[{i}] has no {key!r}")
            gt_geom, gt_cats, (iscrowd, areas) = self._read_side(target, i, "targets", geometry)
            dt_geom, dt_cats, _ = self._read_side(pred, i, "preds", geometry)
            scores = np.asarray(_to_numpy(pred["scores"]), dtype=np.float64).ravel()
            if scores.shape[0] != len(dt_cats):
                raise ValueError(f"preds[{i}] has {len(dt_cats)} labels but {scores.shape[0]} scores")
            sides.append((gt_geom, gt_cats, iscrowd, areas, dt_geom, dt_cats, scores))
        if not image_ids:
            return self._keep(None, image_ids)

        images, gt_anns, dt_anns, labels = [], [], [], set()
        for i, (side, img_id) in enumerate(zip(sides, image_ids)):
            gt_geom, gt_cats, iscrowd, areas, dt_geom, dt_cats, scores = side
            sizes = {m.shape[1:] for m in (gt_geom, dt_geom) if m.shape[0]}
            if len(sizes) > 1:
                raise ValueError(f"image {i}: preds and targets masks differ in size {sorted(sizes)}")
            shape = sizes.pop() if sizes else gt_geom.shape[1:]
            images.append({"id": img_id, "height": int(shape[0]), "width": int(shape[1])})
            labels.update(gt_cats.tolist())
            labels.update(dt_cats.tolist())

            gt_rles = self._encode(gt_geom)
            for n in range(len(gt_cats)):
                gt_anns.append({
                    "id": len(gt_anns) + 1, "image_id": img_id, "category_id": int(gt_cats[n]),
                    "iscrowd": int(iscrowd[n]), "segmentation": gt_rles[n],
                    "bbox": maskUtils.toBbox(gt_rles[n]).tolist(),
                    "area": float(maskUtils.area(gt_rles[n])) if areas is None else float(areas[n]),
                })
            dt_rles = self._encode(dt_geom)
            for n in range(len(dt_cats)):
                dt_anns.append({"image_id": img_id, "category_id": int(dt_cats[n]),
                                "score": float(scores[n]), "segmentation": dt_rles[n]})

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
        if not self._cats:
            return self._summarize(None)
        return self._summarize({
            "det_cat": np.concatenate(self._cats), "score": np.concatenate(self._scores),
            "outcome": np.concatenate(self._outcomes), "gt_cat": np.concatenate(self._gt_cats),
            "npig": np.concatenate(self._npig),
        })

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _given_ids(preds, targets) -> list:
        given = []
        for i, (pred, target) in enumerate(zip(preds, targets)):
            ids = {item["image_id"] for item in (pred, target) if "image_id" in item}
            ids = {int(_to_numpy(v)) for v in ids}
            if len(ids) > 1:
                raise ValueError(f"image {i}: preds and targets give different image_id {sorted(ids)}")
            given.append(ids.pop() if ids else None)
        return given

    def _image_ids(self, given: list) -> list:
        """Ids for a batch's images: as given, or numbered; never one already added."""
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
        labels = np.asarray(_to_numpy(item["labels"]), dtype=np.int64).ravel()
        n = labels.shape[0]
        geom = _to_numpy(item[geometry])
        if geom.ndim != 3 or geom.shape[0] != n:
            raise ValueError(f"{name}[{i}]['masks'] must have shape ({n}, H, W), not {geom.shape}")
        extra = (None, None)
        if name == "targets":
            iscrowd = (np.asarray(_to_numpy(item["iscrowd"]), dtype=np.int64).ravel()
                       if "iscrowd" in item else np.zeros(n, dtype=np.int64))
            areas = (np.asarray(_to_numpy(item["area"]), dtype=np.float64).ravel()
                     if "area" in item else None)
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
            return self._keep(None, image_ids)
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
        # A detection ignored at every threshold is in no count and no cut.
        counted = (out["outcome"] != _detection_metrics.IGNORED).any(axis=1)
        return self._keep({
            "det_cat": slots[out["cat_slot"].astype(np.int64)[counted]],
            "score": out["score"][counted], "outcome": out["outcome"][counted],
            "gt_cat": slots, "npig": out["npig"].astype(np.int64),
        }, image_ids)

    def _keep(self, batch: dict | None, image_ids: list) -> dict:
        """Store a batch's counted outcomes and return its metrics."""
        if batch is not None:
            self._cats.append(batch["det_cat"])
            self._scores.append(batch["score"])
            self._outcomes.append(batch["outcome"])
            self._gt_cats.append(batch["gt_cat"])
            self._npig.append(batch["npig"])
        self._seen.update(image_ids)
        return self._summarize(batch)

    def _summarize(self, batch: dict | None) -> dict:
        if batch is None:
            batch = {"det_cat": np.zeros(0, dtype=np.int64), "score": np.zeros(0),
                     "outcome": np.zeros((0, self.iou_thrs.size), dtype=np.uint8),
                     "gt_cat": np.zeros(0, dtype=np.int64), "npig": np.zeros(0, dtype=np.int64)}
        result = _detection_metrics.summarize_ids(
            batch["det_cat"], batch["score"], batch["outcome"], batch["gt_cat"], batch["npig"],
            self.iou_thrs, score_thr=self.score_thr, optimize=self.optimize,
        )
        return _detection_metrics.drop_threshold_axis(result) if self._scalar else result


def detection_metrics(preds: Sequence[dict], targets: Sequence[dict], **config) -> dict:
    """Metrics for one batch: ``DetectionMetrics(**config).update(preds, targets)``."""
    return DetectionMetrics(**config).update(preds, targets)
