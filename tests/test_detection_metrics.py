"""Precision, recall and F1 (``COCOeval.detection_metrics``).

There is no pycocotools oracle for these numbers, so the tests pin them to
the ones there is: recall must reproduce ``eval['recall']`` bit for bit, and a
score threshold applied after matching must give the same counts as an
evaluation that never saw the detections below it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ultrafast_pycocotools as ufc
from ultrafast_pycocotools import _detection_metrics


def _quiet(*_):
    pass


def _evaluate(gt, dt, iou_type, configure=None):
    ev = ufc.COCOeval(gt, dt, iou_type, print_function=_quiet)
    if configure is not None:
        configure(ev.params)
    ev.evaluate()
    ev.accumulate()
    return ev


def _load(gt_path, dt_path, iou_type="bbox"):
    gt = ufc.COCO(str(gt_path), verbose=False)
    dt = gt.loadRes(str(dt_path))
    return gt, dt


def _by_category(result):
    """``{category_id: (tp, fp, fn)}`` for categories with anything counted."""
    pc = result["per_category"]
    out = {}
    for i, cid in enumerate(result["category_ids"].tolist()):
        counts = (int(pc["tp"][i]), int(pc["fp"][i]), int(pc["fn"][i]))
        if any(counts):
            out[cid] = counts
    return out


def _assert_same_result(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            for name in a[key]:
                assert np.array_equal(a[key][name], b[key][name]), (key, name)
        else:
            assert np.array_equal(a[key], b[key]), key


@pytest.fixture(scope="module")
def bbox_eval(synthetic):
    return _evaluate(*_load(*synthetic), "bbox")


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_recall_reproduces_eval_recall_bit_for_bit(synthetic, real_coco, iou_type):
    """With no score threshold, ``tp / (tp + fn)`` is COCO's recall.

    Checked for every IoU threshold, area range and maxDets, on synthetic data
    (crowds, area boundaries, ties) and the committed real COCO slice.
    """
    for paths in (synthetic, real_coco):
        ev = _evaluate(*_load(*paths), iou_type)
        p = ev.params
        for a, area in enumerate(p.areaRngLbl):
            for m, max_det in enumerate(p.maxDets):
                r = ev.detection_metrics(iou_thr=list(p.iouThrs), area=area, max_dets=max_det)
                slots = [list(p.catIds).index(c) for c in r["category_ids"]]
                expected = ev.eval["recall"][:, slots, a, m]
                pc = r["per_category"]
                has_gt = (pc["tp"] + pc["fn"])[0] > 0
                assert has_gt.any()
                assert np.array_equal(pc["recall"][:, has_gt], expected[:, has_gt]), (area, max_det)
                # Categories without ground truth are -1 in eval['recall'].
                assert np.all(expected[:, ~has_gt] == -1)


def test_keypoint_recall_reproduces_eval_recall(synthetic_kp):
    ev = _evaluate(*_load(*synthetic_kp), "keypoints")
    p = ev.params
    for a, area in enumerate(p.areaRngLbl):
        r = ev.detection_metrics(iou_thr=list(p.iouThrs), area=area)
        assert np.array_equal(r["per_category"]["recall"], ev.eval["recall"][:, :, a, -1])


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_score_threshold_after_matching_equals_matching_fewer_detections(synthetic, iou_type):
    """The prefix argument in ``_detection_metrics``'s module docstring.

    Includes a threshold equal to a tied score, every area range and a
    maxDets cut.
    """
    gt_path, dt_path = synthetic
    gt, dt = _load(gt_path, dt_path)
    ev = _evaluate(gt, dt, iou_type)
    dets = json.loads(dt_path.read_text())
    scores = [d["score"] for d in dets]
    values, counts = np.unique(scores, return_counts=True)
    tied = float(values[counts > 1][len(values[counts > 1]) // 2])
    for thr in (tied, float(np.quantile(scores, 0.25)), float(np.quantile(scores, 0.75))):
        kept = [d for d in dets if d["score"] >= thr]
        ev_kept = _evaluate(gt, gt.loadRes(kept), iou_type)
        for area in ev.params.areaRngLbl:
            for max_det in (10, 100):
                after = ev.detection_metrics(score_thr=thr, area=area, max_dets=max_det)
                before = ev_kept.detection_metrics(area=area, max_dets=max_det)
                assert _by_category(after) == _by_category(before), (thr, area, max_det)


def test_keypoint_score_threshold_after_matching(synthetic_kp):
    gt_path, dt_path = synthetic_kp
    gt, dt = _load(gt_path, dt_path)
    ev = _evaluate(gt, dt, "keypoints")
    dets = json.loads(dt_path.read_text())
    thr = float(np.median([d["score"] for d in dets]))
    ev_kept = _evaluate(gt, gt.loadRes([d for d in dets if d["score"] >= thr]), "keypoints")
    assert _by_category(ev.detection_metrics(score_thr=thr)) == _by_category(
        ev_kept.detection_metrics()
    )


@pytest.mark.parametrize("optimize", ["macro", "micro"])
def test_best_f1_is_the_best_cut_and_matches_a_direct_call(bbox_eval, synthetic, optimize):
    ev = bbox_eval
    thrs = [0.5, 0.75]
    best = ev.detection_metrics(iou_thr=thrs, score_thr="best_f1", optimize=optimize)
    candidates = np.unique([d["score"] for d in json.loads(synthetic[1].read_text())])
    for t, iou_thr in enumerate(thrs):
        chosen = float(best["score_thr"][t])
        direct = ev.detection_metrics(iou_thr=[iou_thr], score_thr=chosen)
        for group in ("per_category", "micro", "macro"):
            for name, value in best[group].items():
                assert np.array_equal(value[t], direct[group][name][0]), (group, name)
        # Every candidate cut through the public summary, from one matching.
        out = ev._engine.outcomes(0, ev.params.maxDets[-1])
        column = out["outcome"][:, [list(ev.params.iouThrs).index(iou_thr)]]
        f1 = [
            float(_detection_metrics.summarize_outcomes(
                out["cat_slot"], out["score"], column, out["npig"], ev.params.catIds,
                [iou_thr], score_thr=float(s),
            )[optimize]["f1"][0])
            for s in candidates
        ]
        assert best[optimize]["f1"][t] == max(f1)
        # Ties in F1 go to the highest threshold.
        assert chosen == max(s for s, v in zip(candidates, f1) if v == max(f1))


def _random_outcomes(seed, n=1500, n_cats=9, n_thrs=2):
    """Coarse scores (many ties), ignored detections, NaN scores and two
    categories without ground truth (false positives only)."""
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, n_cats, n)
    score = np.round(rng.random(n), 2)
    score[rng.random(n) < 0.01] = np.nan
    outcome = rng.choice(
        [_detection_metrics.IGNORED, _detection_metrics.TRUE_POSITIVE, _detection_metrics.FALSE_POSITIVE],
        size=(n, n_thrs), p=[0.1, 0.4, 0.5],
    ).astype(np.uint8)
    npig = np.bincount(cats, minlength=n_cats) // 2 + 1
    npig[:2] = 0
    outcome[np.isin(cats, [0, 1])] = _detection_metrics.FALSE_POSITIVE
    return cats, score, outcome, npig


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("optimize", ["macro", "micro"])
def test_best_f1_sweep_equals_brute_force(seed, optimize):
    """Every candidate cut through the fixed-threshold path, against the sweep."""
    cats, score, outcome, npig = _random_outcomes(seed)
    ids, thrs = np.arange(npig.size), [0.5, 0.75]
    best = _detection_metrics.summarize_outcomes(
        cats, score, outcome, npig, ids, thrs, score_thr="best_f1", optimize=optimize)
    for t in range(len(thrs)):
        live = (outcome[:, t] != _detection_metrics.IGNORED) & ~np.isnan(score)
        candidates = np.unique(score[live])[::-1]
        f1 = np.array([
            _detection_metrics.summarize_outcomes(
                cats, score, outcome[:, [t]], npig, ids, [thrs[t]], score_thr=float(s)
            )[optimize]["f1"][0]
            for s in candidates
        ])
        assert best["score_thr"][t] == candidates[int(np.argmax(f1))]
        assert best[optimize]["f1"][t] == f1.max()


def test_best_macro_f1_without_detections_of_ground_truth_categories():
    """Only false positives of a category without ground truth at the first
    threshold; every cut has macro F1 0, so the highest threshold wins."""
    tp, fp, ign = (_detection_metrics.TRUE_POSITIVE, _detection_metrics.FALSE_POSITIVE,
                   _detection_metrics.IGNORED)
    cats = np.array([1, 1, 0])
    score = np.array([0.9, 0.4, 0.6])
    outcome = np.array([[fp, fp], [fp, fp], [ign, tp]], dtype=np.uint8)
    r = _detection_metrics.summarize_outcomes(
        cats, score, outcome, np.array([1, 0]), [7, 8], [0.5, 0.75], score_thr="best_f1")
    assert r["score_thr"].tolist() == [0.9, 0.6]
    assert r["macro"]["f1"].tolist() == [0.0, 1.0]


def test_macro_candidates_equal_evaluating_every_cut():
    """The candidate search against an exact evaluation of every cut."""
    for seed in range(4):
        args = _random_outcomes(seed, n=4000, n_cats=30)
        ids = np.arange(args[3].size)
        for optimize in ("macro", "micro"):
            bounded = _detection_metrics.summarize_outcomes(
                *args, ids, [0.5, 0.75], score_thr="best_f1", optimize=optimize)
            every = _detection_metrics.summarize_outcomes(
                *args, ids, [0.5, 0.75], score_thr="best_f1", optimize=optimize, exhaustive=True)
            _assert_same_result(bounded, every)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("score_thr", [None, 0.5, 0.0, "best_f1"])
@pytest.mark.parametrize("optimize", ["macro", "micro"])
def test_rust_summary_equals_the_numpy_implementation(seed, score_thr, optimize):
    """``ufcoco_core::metrics`` against the NumPy code it replaced, bit for bit."""
    import reference_detection_metrics as reference

    args = _random_outcomes(seed, n=2500, n_cats=11, n_thrs=3)
    ids = np.arange(100, 100 + args[3].size)
    kwargs = {"score_thr": score_thr, "optimize": optimize}
    _assert_same_result(
        _detection_metrics.summarize_outcomes(*args, ids, [0.5, 0.6, 0.7], **kwargs),
        reference.summarize_outcomes(*args, ids, [0.5, 0.6, 0.7], **kwargs),
    )


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_rust_summary_equals_the_numpy_implementation_on_coco(real_coco, iou_type):
    import reference_detection_metrics as reference

    ev = _evaluate(*_load(*real_coco), iou_type)
    out = ev._engine.outcomes(0, 100)
    for score_thr in (None, 0.3, "best_f1"):
        for optimize in ("macro", "micro"):
            kwargs = {"score_thr": score_thr, "optimize": optimize}
            args = (out["cat_slot"], out["score"], out["outcome"], out["npig"], ev.params.catIds,
                    ev.params.iouThrs)
            _assert_same_result(_detection_metrics.summarize_outcomes(*args, **kwargs),
                                reference.summarize_outcomes(*args, **kwargs))


def test_counts_agree_with_confusion_matrix(bbox_eval):
    p = bbox_eval.params
    for thr in (0.0, 0.3, 0.7):
        r = bbox_eval.detection_metrics(score_thr=thr)
        cm, _ = bbox_eval.confusion_matrix(score_thr=thr)
        n = cm.shape[0] - 1
        slots = [list(p.catIds).index(c) for c in r["category_ids"]]
        assert np.array_equal(r["per_category"]["tp"], np.diag(cm)[slots])
        assert np.array_equal(r["per_category"]["fp"], cm[slots, n])
        assert np.array_equal(r["per_category"]["fn"], cm[n, slots])
        assert r["micro"]["tp"] == np.trace(cm)
        assert r["micro"]["fp"] == cm[:n, n].sum()
        assert r["micro"]["fn"] == cm[n, :n].sum()


def test_scalar_and_sequence_iou_thresholds_agree(bbox_eval):
    seq = bbox_eval.detection_metrics(iou_thr=[0.5, 0.75], score_thr=0.4)
    for t, thr in enumerate([0.5, 0.75]):
        one = bbox_eval.detection_metrics(iou_thr=thr, score_thr=0.4)
        assert one["iou_thr"] == seq["iou_thrs"][t]
        assert one["score_thr"] == 0.4
        for group in ("per_category", "micro", "macro"):
            for name, value in one[group].items():
                assert np.array_equal(value, seq[group][name][t]), (group, name)


def _tiny():
    """Category 1 found, category 2 missed, category 3 only false positives."""
    gt = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": c, "name": str(c)} for c in (1, 2, 3)],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [20, 20, 10, 10], "area": 100, "iscrowd": 0},
        ],
    }
    dets = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [80, 80, 5, 5], "score": 0.8},
        {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.7},
    ]
    coco = ufc.COCO(gt, verbose=False)
    return _evaluate(coco, coco.loadRes(dets), "bbox")


def test_hand_computed_values_and_zero_denominators():
    r = _tiny().detection_metrics()
    assert r["category_ids"].tolist() == [1, 2, 3]
    pc = r["per_category"]
    assert pc["tp"].tolist() == [1, 0, 0]
    assert pc["fp"].tolist() == [1, 0, 1]
    assert pc["fn"].tolist() == [1, 1, 0]
    assert pc["precision"].tolist() == [0.5, 0.0, 0.0]
    assert pc["recall"].tolist() == [0.5, 0.0, 0.0]
    assert pc["f1"].tolist() == [0.5, 0.0, 0.0]
    # Micro counts every false positive, including category 3's.
    assert (r["micro"]["tp"], r["micro"]["fp"], r["micro"]["fn"]) == (1, 2, 2)
    assert r["micro"]["precision"] == 1 / 3
    assert r["micro"]["recall"] == 1 / 3
    # Macro averages the categories with ground truth (1 and 2) only.
    assert r["macro"]["precision"] == 0.25
    assert r["macro"]["recall"] == 0.25
    assert r["macro"]["f1"] == 0.25


def test_best_f1_with_nothing_to_keep_is_infinite_and_zero(tmp_path):
    gt = {
        "images": [{"id": 1, "width": 10, "height": 10}],
        "categories": [{"id": 1, "name": "a"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}
        ],
    }
    coco = ufc.COCO(gt, verbose=False)
    ev = ufc.COCOeval(coco, coco.loadRes([]), "bbox", print_function=_quiet)
    ev.evaluate()
    r = ev.detection_metrics(score_thr="best_f1")
    assert r["score_thr"] == np.inf
    assert (r["micro"]["tp"], r["micro"]["fn"], r["macro"]["f1"]) == (0, 1, 0.0)


def test_class_agnostic(synthetic):
    gt, dt = _load(*synthetic)
    ev = _evaluate(gt, dt, "bbox", lambda p: setattr(p, "useCats", 0))
    r = ev.detection_metrics()
    assert r["category_ids"].tolist() == [-1]
    assert r["micro"]["recall"] == ev.eval["recall"][0, 0, 0, -1]
    assert r["macro"]["f1"] == r["micro"]["f1"]


def test_custom_threshold_grid(synthetic):
    gt, dt = _load(*synthetic)
    ev = _evaluate(gt, dt, "bbox", lambda p: setattr(p, "iouThrs", np.array([0.3])))
    assert ev.detection_metrics(iou_thr=0.3)["iou_thr"] == 0.3


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"iou_thr": 0.3}, "not in params.iouThrs"),
        ({"iou_thr": []}, "must not be empty"),
        ({"area": "tiny"}, "areaRngLbl"),
        ({"max_dets": 101}, "max_dets"),
        ({"max_dets": -1}, "max_dets"),
        ({"score_thr": "best"}, "score_thr"),
        ({"score_thr": float("nan")}, "NaN"),
        ({"optimize": "weighted"}, "optimize"),
    ],
)
def test_errors(bbox_eval, kwargs, match):
    with pytest.raises(ValueError, match=match):
        bbox_eval.detection_metrics(**kwargs)


def test_requires_evaluate(synthetic):
    gt, dt = _load(*synthetic)
    with pytest.raises(Exception, match="evaluate"):
        ufc.COCOeval(gt, dt, "bbox", print_function=_quiet).detection_metrics()
