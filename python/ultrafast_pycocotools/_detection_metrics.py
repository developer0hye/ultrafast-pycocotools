"""Precision, recall and F1 counted from the engine's per-detection outcomes.

The engine reports, for every evaluated detection and IoU threshold, the
outcome accumulation reads: ignored, true positive or false positive. Counting
those bytes, instead of matching again, is what keeps these numbers consistent
with the AP printed beside them. ``COCOeval.detection_metrics`` and
``metrics.DetectionMetrics`` both summarize through this module.

A score threshold is applied *after* matching, which equals matching only the
detections that pass it. Matching visits an (image, category)'s detections in
stable descending score order, and each detection's match depends only on the
detections before it. The detections with ``score >= s`` are a prefix of that
order (ties at ``s`` are all kept and are adjacent; NaN sorts last and fails
``>= s``), and a prefix matches the same whether or not the rest of the list
exists. ``maxDets`` truncation is a prefix too, so the order of truncating and
filtering does not matter.
"""

from __future__ import annotations

import numpy as np

from . import _ufcoco

IGNORED = _ufcoco.OUTCOME_IGNORED
TRUE_POSITIVE = _ufcoco.OUTCOME_TRUE_POSITIVE
FALSE_POSITIVE = _ufcoco.OUTCOME_FALSE_POSITIVE


def check_options(score_thr, optimize: str) -> None:
    """Reject option values before any evaluation work is spent on them."""
    if optimize not in ("macro", "micro"):
        raise ValueError(f"optimize must be 'macro' or 'micro', not {optimize!r}")
    if isinstance(score_thr, str):
        if score_thr != "best_f1":
            raise ValueError(f"score_thr must be None, a number or 'best_f1', not {score_thr!r}")
    elif score_thr is not None and np.isnan(float(score_thr)):
        raise ValueError("score_thr must not be NaN")


def summarize_ids(det_cat, score, outcome, gt_cat, npig, iou_thrs, *, score_thr=None,
                  optimize: str = "macro", exhaustive: bool = False) -> dict:
    """Per-category, micro and macro precision / recall / F1.

    Args:
        det_cat: ``(D,)`` category id of each detection.
        score: ``(D,)`` detection scores.
        outcome: ``(D, T)`` engine outcomes.
        gt_cat / npig: category ids and their non-ignored ground truth.
        iou_thrs: ``(T,)`` the thresholds behind ``outcome``'s columns.
        score_thr: None (every detection), a number (``score >= score_thr``)
            or ``"best_f1"``.
        optimize: which F1 ``"best_f1"`` maximizes, ``"macro"`` or ``"micro"``.
        exhaustive: evaluate every cut of a macro sweep exactly (tests only).

    Categories are reported, in ascending id, when they have non-ignored
    ground truth or a detection that is not ignored at some threshold. Macro
    values average the categories with ground truth; zero denominators give 0.
    The arithmetic is ``ufcoco_core::metrics``'s.
    """
    check_options(score_thr, optimize)
    score = np.ascontiguousarray(score, dtype=np.float64)
    if score_thr is None:
        cut, thr = "all", 0.0
    elif isinstance(score_thr, str):
        cut, thr = "best_f1", 0.0
    else:
        cut, thr = "at_least", float(score_thr)
    raw = _ufcoco.summarize_outcomes(
        np.ascontiguousarray(det_cat, dtype=np.int64), score,
        np.ascontiguousarray(outcome, dtype=np.uint8),
        np.ascontiguousarray(gt_cat, dtype=np.int64), np.ascontiguousarray(npig, dtype=np.int64),
        cut, thr, optimize == "micro", exhaustive,
    )
    names = ("tp", "fp", "fn", "precision", "recall", "f1")
    return {
        "iou_thrs": np.asarray(iou_thrs, dtype=np.float64),
        "category_ids": raw["category_ids"],
        "score_thr": raw["score_thr"],
        "per_category": {name: raw[name] for name in names},
        "micro": {name: raw[f"micro_{name}"] for name in names},
        "macro": {name: raw[f"macro_{name}"] for name in names[3:]},
    }


def summarize_outcomes(cat_slot, score, outcome, npig, cat_ids, iou_thrs, *,
                       score_thr=None, optimize: str = "macro", exhaustive: bool = False) -> dict:
    """:func:`summarize_ids` for detections given as slots into ``cat_ids``,
    with ``npig`` per slot."""
    cat_ids = np.asarray(cat_ids, dtype=np.int64)
    return summarize_ids(cat_ids[np.asarray(cat_slot, dtype=np.int64)], score, outcome, cat_ids,
                         npig, iou_thrs, score_thr=score_thr, optimize=optimize,
                         exhaustive=exhaustive)


def drop_threshold_axis(result: dict) -> dict:
    """The result for a single IoU threshold, without the leading ``T`` axis."""
    out = {
        "iou_thr": float(result["iou_thrs"][0]),
        "category_ids": result["category_ids"],
        "score_thr": float(result["score_thr"][0]),
    }
    for group in ("per_category", "micro", "macro"):
        out[group] = {k: v[0] for k, v in result[group].items()}
    return out
