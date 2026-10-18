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


def test_best_f1_sweep_does_not_depend_on_block_size(bbox_eval, monkeypatch):
    whole = bbox_eval.detection_metrics(score_thr="best_f1")
    monkeypatch.setattr(_detection_metrics, "_SWEEP_BLOCK_CELLS", 7)
    assert whole["score_thr"] == bbox_eval.detection_metrics(score_thr="best_f1")["score_thr"]


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
