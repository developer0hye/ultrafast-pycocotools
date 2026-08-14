"""The fixture must actually contain the hard cases everything else assumes.

`bench/make_dataset.py` generates crowd regions, boundary-exact areas, tied
scores and degenerate polygons *probabilistically*. Nothing downstream checks
that any of them came out. If a probability drifted to zero — or a refactor
dropped a branch — every crowd test would keep passing, because pycocotools and
this library agree perfectly on crowd-free data. The suite would go green while
covering less, which is the worst possible failure mode for a parity suite: it
gets quieter, not louder.

So this file asserts the fixture's *composition*. It is the guard on the other
guards.
"""

from __future__ import annotations

import json

import numpy as np


def test_fixture_has_crowd_regions(synthetic):
    """Crowd handling is a whole branch of the matcher (a crowd may absorb
    several detections and is excluded from the recall denominator)."""
    gt = json.loads(synthetic[0].read_text())
    crowd = [a for a in gt["annotations"] if a.get("iscrowd", 0)]
    assert len(crowd) >= 20, f"only {len(crowd)} crowd annotations"
    # Crowd ground truth is stored as uncompressed RLE, which is its own
    # decode path.
    assert any(isinstance(a["segmentation"], dict) for a in crowd)


def test_fixture_has_areas_exactly_on_the_scale_boundaries(synthetic):
    """COCO splits small/medium/large at 32^2 and 96^2 and the comparison is
    inclusive, so annotations sitting exactly on a bound are the ones that
    catch an off-by-one."""
    gt = json.loads(synthetic[0].read_text())
    areas = {a["area"] for a in gt["annotations"]}
    assert 32.0**2 in areas, "no annotation with area exactly 1024"
    assert 96.0**2 in areas, "no annotation with area exactly 9216"


def test_fixture_has_degenerate_polygons(synthetic):
    """A repeated consecutive vertex makes `rleFrPoly` divide by zero and cast
    the NaN — the path `c_i32` exists for."""
    gt = json.loads(synthetic[0].read_text())
    found = 0
    for ann in gt["annotations"]:
        seg = ann["segmentation"]
        if not isinstance(seg, list):
            continue
        for ring in seg:
            pts = list(zip(ring[0::2], ring[1::2]))
            if any(a == b for a, b in zip(pts, pts[1:])):
                found += 1
                break
    assert found >= 5, f"only {found} polygons with a repeated vertex"


def test_fixture_has_heavily_tied_scores(synthetic):
    """Ties are where an unstable sort changes which ground truth a detection
    claims, and they only exist because scores are quantised."""
    dets = json.loads(synthetic[1].read_text())
    scores = [d["score"] for d in dets]
    _, counts = np.unique(scores, return_counts=True)
    # Both a deep tie and broad tying: one very deep tie in an otherwise
    # distinct set would exercise far less of the ordering than this does.
    assert counts.max() >= 5, f"deepest score tie is only {counts.max()}"
    tied = int(counts[counts > 1].sum())
    assert tied / len(scores) > 0.8, f"only {tied}/{len(scores)} detections are tied"


def test_fixture_has_lopsided_images(synthetic):
    """Images with ground truth but no detections exercise the recall
    denominator; the reverse exercises the false-positive path. Both are
    skipped entirely if every image has some of each."""
    gt = json.loads(synthetic[0].read_text())
    dets = json.loads(synthetic[1].read_text())
    with_gt = {a["image_id"] for a in gt["annotations"]}
    with_dt = {d["image_id"] for d in dets}
    all_imgs = {i["id"] for i in gt["images"]}
    assert with_gt - with_dt, "no image has ground truth and no detections"
    assert with_dt - with_gt, "no image has detections and no ground truth"
    assert all_imgs - with_gt - with_dt, "no empty image"


def test_fixture_has_wrong_class_detections(synthetic):
    """Detections landing on an object of a different category are what make
    per-category AP mean anything."""
    gt = json.loads(synthetic[0].read_text())
    dets = json.loads(synthetic[1].read_text())
    per_img_gt_cats: dict = {}
    for a in gt["annotations"]:
        per_img_gt_cats.setdefault(a["image_id"], set()).add(a["category_id"])
    wrong = sum(
        1
        for d in dets
        if d["category_id"] not in per_img_gt_cats.get(d["image_id"], set())
    )
    assert wrong >= 50, f"only {wrong} detections in a category absent from their image"


def test_keypoint_fixture_has_invisible_and_empty_annotations(synthetic_kp):
    """`num_keypoints == 0` forces a ground truth to be ignored, and invisible
    keypoints take the `k1 == 0` branch of OKS, which measures distance to a
    doubled bounding box instead of to the keypoints."""
    gt = json.loads(synthetic_kp[0].read_text())
    anns = [a for a in gt["annotations"] if "keypoints" in a]
    assert anns
    assert any(a["num_keypoints"] == 0 for a in anns), "no fully invisible instance"
    assert any(
        any(v == 0 for v in a["keypoints"][2::3]) for a in anns
    ), "no partially invisible instance"
