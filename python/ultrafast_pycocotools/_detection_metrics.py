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

# Cells of the (detections x categories) cumulative-count blocks the macro
# sweep builds at once; bounds its memory for datasets with many categories.
_SWEEP_BLOCK_CELLS = 1 << 20


def check_options(score_thr, optimize: str) -> None:
    """Reject option values before any evaluation work is spent on them."""
    if optimize not in ("macro", "micro"):
        raise ValueError(f"optimize must be 'macro' or 'micro', not {optimize!r}")
    if isinstance(score_thr, str):
        if score_thr != "best_f1":
            raise ValueError(f"score_thr must be None, a number or 'best_f1', not {score_thr!r}")
    elif score_thr is not None and np.isnan(float(score_thr)):
        raise ValueError("score_thr must not be NaN")


def _divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den``, with 0 where ``den`` is 0."""
    out = np.zeros(np.broadcast(num, den).shape, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _ratios(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray):
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    f1 = _divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def _macro(values: np.ndarray, has_gt: np.ndarray) -> np.ndarray:
    """Mean over the last axis, over categories with ground truth; 0 if none.

    Summed one category at a time, left to right. ``np.mean`` picks its
    summation order from the array's shape, so the same row could round
    differently alone than inside a sweep or a multi-threshold result; this
    order is the same for every shape.
    """
    selected = values[..., has_gt]
    n = selected.shape[-1]
    total = np.zeros(values.shape[:-1], dtype=np.float64)
    if n == 0:
        return total
    for j in range(n):
        total = total + selected[..., j]
    return total / n


def _counts(cat_slot, score, outcome, n_cats, thr, keep_all):
    """True and false positives per (threshold, category) at score cut ``thr``."""
    t_n = outcome.shape[1]
    tp = np.empty((t_n, n_cats), dtype=np.int64)
    fp = np.empty((t_n, n_cats), dtype=np.int64)
    for t in range(t_n):
        keep = np.ones(score.shape, dtype=bool) if keep_all else score >= thr[t]
        col = outcome[:, t]
        tp[t] = np.bincount(cat_slot[keep & (col == TRUE_POSITIVE)], minlength=n_cats)
        fp[t] = np.bincount(cat_slot[keep & (col == FALSE_POSITIVE)], minlength=n_cats)
    return tp, fp


def _macro_f1_at_cuts(cats, hits, cuts, npig):
    """Macro F1 after each cut of the score-sorted detections.

    ``cats`` / ``hits`` are the sorted detections' category slots and
    true-positive flags; ``cuts`` are the last positions of each score. Only
    categories with ground truth enter the mean, so only they are tracked.
    """
    gt_cats = np.flatnonzero(npig > 0)
    if gt_cats.size == 0:
        return np.zeros(cuts.size, dtype=np.float64)
    column = np.full(npig.size, -1, dtype=np.int64)
    column[gt_cats] = np.arange(gt_cats.size)
    cols = column[cats]
    npig_gt = npig[gt_cats]
    all_gt = np.ones(gt_cats.size, dtype=bool)
    rows_per_block = max(1, _SWEEP_BLOCK_CELLS // gt_cats.size)
    carry_tp = np.zeros(gt_cats.size, dtype=np.int64)
    carry_fp = np.zeros(gt_cats.size, dtype=np.int64)
    out = np.empty(cuts.size, dtype=np.float64)
    for lo in range(0, cats.size, rows_per_block):
        hi = min(cats.size, lo + rows_per_block)
        rows = np.flatnonzero(cols[lo:hi] >= 0)
        step_tp = np.zeros((hi - lo, gt_cats.size), dtype=np.int64)
        step_fp = np.zeros((hi - lo, gt_cats.size), dtype=np.int64)
        step_tp[rows, cols[lo:hi][rows]] = hits[lo:hi][rows]
        step_fp[rows, cols[lo:hi][rows]] = ~hits[lo:hi][rows]
        cum_tp = np.cumsum(step_tp, axis=0) + carry_tp
        cum_fp = np.cumsum(step_fp, axis=0) + carry_fp
        carry_tp, carry_fp = cum_tp[-1], cum_fp[-1]
        here = (cuts >= lo) & (cuts < hi)
        at = cuts[here] - lo
        _, _, f1 = _ratios(cum_tp[at], cum_fp[at], npig_gt - cum_tp[at])
        out[here] = _macro(f1, all_gt)
    return out


def _best_thresholds(cat_slot, score, outcome, npig, optimize):
    """The score cut that maximizes F1 at each IoU threshold.

    Candidates are the distinct scores of detections that count (not ignored,
    not NaN), so a cut never splits tied scores. Ties in F1 go to the highest
    threshold; with nothing to keep the threshold is ``+inf``.
    """
    best = np.full(outcome.shape[1], np.inf)
    for t in range(outcome.shape[1]):
        col = outcome[:, t]
        live = (col != IGNORED) & ~np.isnan(score)
        if not live.any():
            continue
        order = np.argsort(-score[live], kind="stable")
        scores = score[live][order]
        cats = cat_slot[live][order]
        hits = col[live][order] == TRUE_POSITIVE
        cuts = np.flatnonzero(np.append(scores[1:] != scores[:-1], True))
        if optimize == "micro":
            tp = np.cumsum(hits)[cuts]
            fp = cuts + 1 - tp
            _, _, f1 = _ratios(tp, fp, int(npig.sum()) - tp)
        else:
            f1 = _macro_f1_at_cuts(cats, hits, cuts, npig)
        best[t] = scores[cuts[int(np.argmax(f1))]]
    return best


def summarize_outcomes(cat_slot, score, outcome, npig, cat_ids, iou_thrs, *,
                       score_thr=None, optimize: str = "macro") -> dict:
    """Per-category, micro and macro precision / recall / F1.

    Args:
        cat_slot: ``(D,)`` category slot of each detection.
        score: ``(D,)`` detection scores.
        outcome: ``(D, T)`` engine outcomes.
        npig: ``(K,)`` non-ignored ground truth per category slot.
        cat_ids: ``(K,)`` category id of each slot.
        iou_thrs: ``(T,)`` the thresholds behind ``outcome``'s columns.
        score_thr: None (every detection), a number (``score >= score_thr``)
            or ``"best_f1"``.
        optimize: which F1 ``"best_f1"`` maximizes, ``"macro"`` or ``"micro"``.

    Categories are reported when they have non-ignored ground truth or a
    detection that is not ignored at some threshold. Macro values average the
    categories with ground truth; zero denominators give 0.
    """
    check_options(score_thr, optimize)
    cat_slot = np.asarray(cat_slot, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.uint8)
    npig = np.asarray(npig, dtype=np.int64)
    n_cats = npig.size
    t_n = outcome.shape[1]

    keep_all = score_thr is None
    if keep_all:
        thr = np.full(t_n, -np.inf)
    elif isinstance(score_thr, str):
        thr = _best_thresholds(cat_slot, score, outcome, npig, optimize)
    else:
        thr = np.full(t_n, float(score_thr))

    tp, fp = _counts(cat_slot, score, outcome, n_cats, thr, keep_all)
    counted = np.bincount(cat_slot[(outcome != IGNORED).any(axis=1)], minlength=n_cats) > 0
    reported = (npig > 0) | counted
    tp, fp = tp[:, reported], fp[:, reported]
    fn = npig[reported] - tp
    has_gt = npig[reported] > 0

    precision, recall, f1 = _ratios(tp, fp, fn)
    micro_tp, micro_fp, micro_fn = tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1)
    micro_p, micro_r, micro_f1 = _ratios(micro_tp, micro_fp, micro_fn)
    return {
        "iou_thrs": np.asarray(iou_thrs, dtype=np.float64),
        "category_ids": np.asarray(cat_ids, dtype=np.int64)[reported],
        "score_thr": thr,
        "per_category": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        },
        "micro": {
            "tp": micro_tp, "fp": micro_fp, "fn": micro_fn,
            "precision": micro_p, "recall": micro_r, "f1": micro_f1,
        },
        "macro": {
            "precision": _macro(precision, has_gt),
            "recall": _macro(recall, has_gt),
            "f1": _macro(f1, has_gt),
        },
    }


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
