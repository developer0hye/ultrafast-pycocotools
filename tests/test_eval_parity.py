"""Evaluation must agree with ``pycocotools`` bit for bit.

The comparison is on the whole ``precision`` / ``recall`` / ``scores`` arrays
(``T x R x K x A x M``, so roughly 800k doubles for COCO), not on the twelve
summary numbers. Summaries average away disagreements: a curve can be wrong in
a hundred places and still round to the same 0.065. Comparing raw bytes is the
only version of "no difference in AP" that means anything.

Each implementation gets its own ``COCO`` objects because pycocotools rewrites
``ann['segmentation']`` and ``ann['ignore']`` in place during ``_prepare``;
sharing handles would let the first run change the second one's input.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import subset_dt, subset_gt

import ultrafast_pycocotools as ufc

REAL_SUBSET_IMAGES = 400


def run_reference(gt_path, dt_path, iou_type, tweak=None):
    from pycocotools.coco import COCO as RefCOCO
    from pycocotools.cocoeval import COCOeval as RefEval

    gt = RefCOCO(str(gt_path))
    dt = gt.loadRes(str(dt_path))
    ev = RefEval(gt, dt, iou_type)
    if tweak:
        tweak(ev.params)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def run_ours(gt_path, dt_path, iou_type, tweak=None, **kwargs):
    gt = ufc.COCO(str(gt_path), verbose=False)
    dt = gt.loadRes(str(dt_path))
    ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None, **kwargs)
    if tweak:
        tweak(ev.params)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def assert_bit_identical(ref_ev, our_ev, label: str) -> None:
    for key in ("precision", "recall", "scores"):
        a = np.ascontiguousarray(ref_ev.eval[key], dtype=np.float64)
        b = np.ascontiguousarray(our_ev.eval[key], dtype=np.float64)
        assert a.shape == b.shape, f"{label}: {key} shape {a.shape} vs {b.shape}"
        if a.tobytes() != b.tobytes():
            diff = np.flatnonzero(a.ravel() != b.ravel())
            first = diff[0]
            raise AssertionError(
                f"{label}: {key} differs at {len(diff)}/{a.size} positions; "
                f"first at flat index {first}: {a.ravel()[first]!r} vs "
                f"{b.ravel()[first]!r} (delta {a.ravel()[first] - b.ravel()[first]:.3e})"
            )
    a = np.asarray(ref_ev.stats, dtype=np.float64)
    b = np.asarray(our_ev.stats, dtype=np.float64)
    assert a.tobytes() == b.tobytes(), f"{label}: stats differ\n{a}\n{b}"


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_synthetic(synthetic, iou_type):
    gt_path, dt_path = synthetic
    assert_bit_identical(
        run_reference(gt_path, dt_path, iou_type),
        run_ours(gt_path, dt_path, iou_type),
        f"synthetic/{iou_type}",
    )


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_real_coco_subset(real_pair, tmp_path, iou_type):
    gt_full, dt_full = real_pair
    gt_path = subset_gt(gt_full, REAL_SUBSET_IMAGES, tmp_path)
    dt_path = subset_dt(dt_full, gt_path, tmp_path)
    assert_bit_identical(
        run_reference(gt_path, dt_path, iou_type),
        run_ours(gt_path, dt_path, iou_type),
        f"coco/{iou_type}",
    )


def test_keypoints(synthetic_kp):
    gt_path, dt_path = synthetic_kp
    assert_bit_identical(
        run_reference(gt_path, dt_path, "keypoints"),
        run_ours(gt_path, dt_path, "keypoints"),
        "synthetic/keypoints",
    )


def test_use_cats_off(synthetic):
    """Class-agnostic (proposal) scoring merges every category per image.

    The merge is category-major, not annotation order, and getting that wrong
    reorders ties in detection score.
    """
    gt_path, dt_path = synthetic

    def tweak(p):
        p.useCats = 0

    assert_bit_identical(
        run_reference(gt_path, dt_path, "bbox", tweak),
        run_ours(gt_path, dt_path, "bbox", tweak),
        "useCats=0",
    )


def test_custom_area_ranges(synthetic):
    def tweak(p):
        p.areaRng = [[0, 1e10], [0, 500], [500, 5000], [5000, 1e10], [100, 1000]]
        p.areaRngLbl = ["all", "tiny", "mid", "big", "overlapping"]

    assert_bit_identical(
        run_reference(synthetic[0], synthetic[1], "bbox", tweak),
        run_ours(synthetic[0], synthetic[1], "bbox", tweak),
        "custom areaRng",
    )


def test_custom_max_dets(synthetic):
    def tweak(p):
        p.maxDets = [3, 25, 300]

    assert_bit_identical(
        run_reference(synthetic[0], synthetic[1], "bbox", tweak),
        run_ours(synthetic[0], synthetic[1], "bbox", tweak),
        "custom maxDets",
    )


def test_custom_iou_thresholds(synthetic):
    def tweak(p):
        p.iouThrs = np.linspace(0.3, 0.9, 7, endpoint=True)

    assert_bit_identical(
        run_reference(synthetic[0], synthetic[1], "bbox", tweak),
        run_ours(synthetic[0], synthetic[1], "bbox", tweak),
        "custom iouThrs",
    )


def test_image_subset(synthetic):
    def tweak(p):
        p.imgIds = sorted(p.imgIds)[:40]

    assert_bit_identical(
        run_reference(synthetic[0], synthetic[1], "bbox", tweak),
        run_ours(synthetic[0], synthetic[1], "bbox", tweak),
        "image subset",
    )


def test_category_subset(synthetic):
    def tweak(p):
        p.catIds = sorted(p.catIds)[:3]

    assert_bit_identical(
        run_reference(synthetic[0], synthetic[1], "bbox", tweak),
        run_ours(synthetic[0], synthetic[1], "bbox", tweak),
        "category subset",
    )


def test_no_detections(synthetic, tmp_path):
    """Every category present in ground truth but nothing predicted.

    Recall must be 0 and precision 0 — not the -1 "category absent" sentinel,
    which would silently drop those categories out of the mAP average.
    """
    gt_path, _ = synthetic
    empty = tmp_path / "empty_dt.json"
    empty.write_text(json.dumps([]))
    gt = ufc.COCO(str(gt_path), verbose=False)
    dt = gt.loadRes([])
    ev = ufc.COCOeval(gt, dt, "bbox", print_function=lambda *_: None)
    ev.run()
    assert float(ev.stats[0]) == 0.0
    assert float(ev.stats[8]) == 0.0


def test_all_scores_tied(synthetic, tmp_path):
    """Every detection at the same score.

    This is the worst case for tie-breaking: the entire ordering is decided by
    the stability of the sort, so any instability shows up immediately.
    """
    gt_path, dt_path = synthetic
    dets = json.loads(dt_path.read_text())
    for d in dets:
        d["score"] = 0.5
    tied = tmp_path / "tied_dt.json"
    tied.write_text(json.dumps(dets))
    assert_bit_identical(
        run_reference(gt_path, tied, "bbox"),
        run_ours(gt_path, tied, "bbox"),
        "all scores tied",
    )


def test_eval_imgs_shape_matches_reference(synthetic):
    """``evalImgs`` is opt-in, but when asked for it must match upstream."""
    gt_path, dt_path = synthetic
    ref_ev = run_reference(gt_path, dt_path, "bbox")
    our_ev = run_ours(gt_path, dt_path, "bbox", store_eval_imgs=True)
    assert len(ref_ev.evalImgs) == len(our_ev.evalImgs)
    for i, (a, b) in enumerate(zip(ref_ev.evalImgs, our_ev.evalImgs)):
        if a is None or b is None:
            assert a is None and b is None, f"evalImgs[{i}] presence differs"
            continue
        assert list(a["dtIds"]) == list(b["dtIds"]), i
        assert list(a["gtIds"]) == list(b["gtIds"]), i
        np.testing.assert_array_equal(a["dtMatches"], b["dtMatches"], err_msg=str(i))
        np.testing.assert_array_equal(a["gtMatches"], b["gtMatches"], err_msg=str(i))
        np.testing.assert_array_equal(a["dtIgnore"], b["dtIgnore"], err_msg=str(i))
        np.testing.assert_array_equal(a["gtIgnore"], b["gtIgnore"], err_msg=str(i))
