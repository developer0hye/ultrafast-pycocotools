"""The extension APIs, and the property that makes them worth having.

Every extension is derived from the *same* matching the AP numbers came from.
A confusion matrix that disagrees with the mAP printed next to it is worse
than no confusion matrix, so the tests here mostly check consistency with the
core result rather than re-deriving expected values.
"""

from __future__ import annotations

import numpy as np
import pytest

import ultrafast_pycocotools as ufc


@pytest.fixture(scope="module")
def evaluated(synthetic):
    gt_path, dt_path = synthetic
    gt = ufc.COCO(str(gt_path), verbose=False)
    dt = gt.loadRes(str(dt_path))
    ev = ufc.COCOeval(gt, dt, "bbox", print_function=lambda *_: None)
    ev.run()
    return ev


def test_stats_as_dict_matches_positional(evaluated):
    d = evaluated.stats_as_dict
    assert d["AP"] == float(evaluated.stats[0])
    assert d["AP_50"] == float(evaluated.stats[1])
    assert d["AP_large"] == float(evaluated.stats[5])
    assert len(d) == len(evaluated.stats)


def test_per_category_stats_average_to_overall_ap(evaluated):
    """Per-class AP must average back to the reported mAP.

    COCO's mAP is the mean over categories with any valid entry, so this is a
    real invariant, not an approximation — if per-class numbers came from a
    different slice of the precision array they would not reconcile.
    """
    per = evaluated.per_category_stats(area="all", max_dets=100)
    aps = [v["AP"] for v in per.values() if not np.isnan(v["AP"])]
    assert aps, "no category produced a finite AP"
    assert np.isclose(float(np.mean(aps)), float(evaluated.stats[0]), rtol=0, atol=1e-12)


def test_per_category_stats_has_names(evaluated):
    per = evaluated.per_category_stats()
    some = next(iter(per.values()))
    assert set(some) == {"name", "AP", "AP_50", "AP_75", "AR"}
    assert isinstance(some["name"], str)


def test_pr_curve_shape_and_monotonicity(evaluated):
    curve = evaluated.pr_curve(iou_thr=0.5)
    assert curve["recall"].shape == curve["precision"].shape == curve["score"].shape
    assert curve["recall"].shape == np.asarray(evaluated.params.recThrs).shape
    valid = curve["precision"] >= 0
    p = curve["precision"][valid]
    # Interpolated precision is non-increasing in recall by construction.
    assert np.all(np.diff(p) <= 1e-12), "precision must not increase with recall"


def test_pr_curve_single_category(evaluated):
    cid = evaluated.params.catIds[0]
    curve = evaluated.pr_curve(cat_id=cid, iou_thr=0.5)
    assert curve["precision"].shape == np.asarray(evaluated.params.recThrs).shape


def test_matches_are_consistent_with_recall(evaluated):
    """Matched pairs are exactly the true positives behind AR.

    ``AR@[0.5]`` over all categories is (matched non-ignored GT) / (non-ignored
    GT), so the match list has to be able to reproduce it.
    """
    m = evaluated.matches(iou_thr=0.5, area="all", max_dets=100)
    for key in ("image_id", "category_id", "dt_id", "gt_id", "score", "iou"):
        assert key in m
    n = len(m["dt_id"])
    assert n > 0
    assert len(set(zip(m["image_id"].tolist(), m["dt_id"].tolist()))) == n, (
        "a detection may be matched at most once per setting"
    )
    # Every reported IoU must clear the threshold it was matched at.
    assert np.all(m["iou"] >= 0.5 - 1e-12)


def test_matches_ious_agree_with_mask_api(evaluated):
    """Spot-check reported IoUs against a direct computation."""
    from ultrafast_pycocotools import mask as mask_util

    m = evaluated.matches(iou_thr=0.5)
    gt, dt = evaluated.cocoGt, evaluated.cocoDt
    for i in range(0, min(50, len(m["dt_id"]))):
        d = dt.anns[int(m["dt_id"][i])]
        g = gt.anns[int(m["gt_id"][i])]
        want = np.asarray(
            mask_util.iou([d["bbox"]], [g["bbox"]], [int(g.get("iscrowd", 0))])
        )[0, 0]
        assert want == m["iou"][i], f"row {i}: {want} vs {m['iou'][i]}"


def test_mean_iou_in_range(evaluated):
    v = evaluated.mean_iou(iou_thr=0.5)
    assert 0.5 <= v <= 1.0


def test_per_instance_covers_every_evaluated_instance(evaluated):
    dets, gts = evaluated.per_instance(iou_thr=0.5)
    # Ground truth is reported once per (image, category) evaluation, which for
    # area='all' with useCats=1 is once per annotation.
    assert len(gts["gt_id"]) == len(evaluated.cocoGt.anns)
    # Crowd annotations are the ones COCO ignores.
    n_crowd = sum(1 for a in evaluated.cocoGt.anns.values() if a.get("iscrowd", 0))
    assert int(gts["ignore"].sum()) == n_crowd
    # Every detection that survived the maxDets cut appears exactly once.
    assert len(set(dets["dt_id"].tolist())) == len(dets["dt_id"])


def test_confusion_matrix_reconciles_with_ap_accounting(evaluated):
    cm, labels = evaluated.confusion_matrix(iou_thr=0.5, score_thr=0.0)
    n = len(evaluated.params.catIds)
    assert cm.shape == (n + 1, n + 1)
    assert labels[-1] == "background"

    dets, gts = evaluated.per_instance(iou_thr=0.5)
    live = ~dets["ignore"]
    tp = int(((dets["gt_id"] >= 0) & live).sum())
    fp = int(((dets["gt_id"] < 0) & live).sum())
    assert int(np.trace(cm[:n, :n])) == tp
    assert int(cm[:n, n].sum()) == fp
    # Every non-ignored ground truth is either found or counted as missed.
    real_gt = int((~gts["ignore"]).sum())
    found = len(set(dets["gt_id"][live & (dets["gt_id"] >= 0)].tolist()))
    assert int(cm[n, :n].sum()) + found == real_gt

    # matches() is the true-positive view of the same data.
    assert len(evaluated.matches(iou_thr=0.5)["dt_id"]) == tp


def test_confusion_matrix_score_threshold_monotone(evaluated):
    low, _ = evaluated.confusion_matrix(score_thr=0.0)
    high, _ = evaluated.confusion_matrix(score_thr=0.9)
    n = len(evaluated.params.catIds)
    # Raising the score threshold cannot create true positives.
    assert np.trace(high[:n, :n]) <= np.trace(low[:n, :n])


def test_boundary_iou_runs_and_is_not_looser_than_mask(synthetic):
    """Boundary IoU is min(mask IoU, boundary IoU), so it cannot exceed segm AP."""
    gt_path, dt_path = synthetic
    out = {}
    for iou_type in ("segm", "boundary"):
        gt = ufc.COCO(str(gt_path), verbose=False)
        dt = gt.loadRes(str(dt_path))
        ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None)
        ev.run()
        out[iou_type] = float(ev.stats[0])
    assert out["boundary"] <= out["segm"] + 1e-12


def test_init_as_pycocotools_registers_modules():
    import sys

    saved = {k: sys.modules.get(k) for k in list(sys.modules) if k.startswith("pycocotools")}
    try:
        for k in list(sys.modules):
            if k.startswith("pycocotools"):
                del sys.modules[k]
        ufc.init_as_pycocotools()
        from pycocotools.coco import COCO as Patched
        from pycocotools.cocoeval import COCOeval as PatchedEval

        assert Patched is ufc.COCO
        assert PatchedEval is ufc.COCOeval
    finally:
        for k in list(sys.modules):
            if k.startswith("pycocotools"):
                del sys.modules[k]
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
