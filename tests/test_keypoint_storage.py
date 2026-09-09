"""Compact pose storage preserves general joints and visibility semantics."""

import copy
import json

import numpy as np
import pytest
from pycocotools.coco import COCO as ReferenceCOCO
from pycocotools.cocoeval import COCOeval as ReferenceEval
from ultrafast_pycocotools import COCO, COCOeval


@pytest.mark.parametrize("use_area", [True, False])
@pytest.mark.parametrize("joints", [3, 17, 25])
@pytest.mark.parametrize("visibility", ["mixed", "zero", "nan"])
@pytest.mark.parametrize("mode", ["file", "list", "tuple", "numpy"])
def test_general_keypoints_complete_arrays(
    tmp_path, joints, visibility, mode, use_area
):
    points = [[float(i), float(i + 1), float(i % 3)] for i in range(joints)]
    for point in points:
        if visibility == "zero":
            point[2] = 0.0
        elif visibility == "nan" and point[2] == 0:
            point[2] = float("nan")
    triplets = np.asarray(points).reshape(-1).tolist()
    gt_data = dict(
        images=[dict(id=1, height=100, width=100)],
        categories=[dict(id=7)],
        annotations=[
            dict(
                id=1,
                image_id=1,
                category_id=7,
                bbox=[0.0, 0.0, 32.0, 32.0],
                area=1024.0,
                iscrowd=0,
                num_keypoints=joints,
                keypoints=triplets,
            )
        ],
    )
    detections = [
        dict(
            image_id=1,
            category_id=7,
            bbox=[0.0, 0.0, 32.0, 32.0],
            score=0.7,
            keypoints=[float(i) for i in range(joints * 3)],
        )
        for _ in range(3)
    ]
    # DT visibility is validated but must not affect OKS.
    detections[1]["keypoints"][2::3] = [float("nan")] * joints
    reference_gt = ReferenceCOCO()
    reference_gt.dataset = copy.deepcopy(gt_data)
    if not use_area:
        # Match the alternate OKS denominator; an all-area grid keeps area
        # filtering independent of this deliberately changed reference field.
        reference_gt.dataset["annotations"][0]["area"] = 32.0 * 32.0 * 0.53
    reference_gt.createIndex()
    reference = ReferenceEval(
        reference_gt, reference_gt.loadRes(copy.deepcopy(detections)), "keypoints"
    )
    if mode == "file":
        # Keep one standard JSON case on the compact geometry parser.
        for annotation in detections:
            annotation["keypoints"][2::3] = [0.0] * joints
        gp, dp = tmp_path / "gt.json", tmp_path / "dt.json"
        gp.write_text(json.dumps(gt_data))
        dp.write_text(json.dumps(detections))
        gt = COCO(gp, verbose=False)
        dt = gt.loadRes(dp)
        if visibility != "nan":
            assert gt._compact is not None and dt._compact is not None
    else:
        convert = {"list": list, "tuple": tuple, "numpy": np.asarray}[mode]
        for annotation in [*gt_data["annotations"], *detections]:
            annotation["keypoints"] = convert(annotation["keypoints"])
        gt = COCO(gt_data, verbose=False)
        dt = gt.loadRes(detections)
    actual = COCOeval(
        gt, dt, "keypoints", use_area=use_area, print_function=lambda *_: None
    )
    for cap in (1, 20):
        for evaluator in (reference, actual):
            evaluator.params.kpt_oks_sigmas = np.linspace(0.025, 0.095, joints)
            if not use_area:
                evaluator.params.areaRng = [[0.0, 1e10]]
                evaluator.params.areaRngLbl = ["all"]
            evaluator.params.maxDets = [cap]
            evaluator.params.iouThrs = np.array([0.0, 0.5, 0.75, 1.0])
            evaluator.evaluate()
            evaluator.accumulate()
        for key in ("precision", "recall", "scores"):
            assert reference.eval[key].shape == actual.eval[key].shape
            assert reference.eval[key].tobytes() == actual.eval[key].tobytes()
        assert (
            np.asarray(reference.computeOks(1, 7)).tobytes()
            == np.asarray(actual.computeOks(1, 7)).tobytes()
        )


@pytest.mark.parametrize("use_cats", [0, 1])
def test_pose_cap_snapshot_reloads_and_preserves_diagnostics(tmp_path, use_cats):
    """A cap drops payload only; ties, zero caps and later larger caps still work."""
    gt_data = dict(
        images=[dict(id=i, width=200, height=200) for i in (1, 2, 3)],
        categories=[dict(id=i) for i in (1, 7)],
        annotations=[dict(id=i, image_id=1, category_id=i, bbox=[0, 0, 40, 40],
                          area=1600, iscrowd=0, num_keypoints=17,
                          keypoints=[float(i), 2.0, 2.0] * 17) for i in (1, 7)],
    )
    detections = [dict(image_id=image, category_id=cat, bbox=[0, 0, 40, 40],
                       score=score, keypoints=[float(x), 2.0, 0.0] * 17)
                  for image in (1, 2)
                  for cat, x, score in [(7, 7, .8), (1, 1, .8), (1, 50, .9),
                                         (7, 70, .8), (1, 1, -0.0), (1, 10, 0.0)]]
    gp, dp = tmp_path / "gt.json", tmp_path / "dt.json"
    gp.write_text(json.dumps(gt_data)); dp.write_text(json.dumps(detections))
    gt = COCO(gp, verbose=False); dt = gt.loadRes(dp)
    assert gt._compact is not None and dt._compact is not None
    # No later parsing may depend on a mutable or still-existing source file.
    gp.write_text("{}"); dp.unlink()
    actual = COCOeval(gt, dt, "keypoints", store_eval_imgs=True,
                      print_function=lambda *_: None)
    for cap in (2, 0, 10, 1):
        dense_gt = COCO(copy.deepcopy(gt_data), verbose=False)
        dense = COCOeval(dense_gt, dense_gt.loadRes(copy.deepcopy(detections)),
                         "keypoints", store_eval_imgs=True, print_function=lambda *_: None)
        evaluators = [actual, dense]
        if use_cats:
            ref_gt = ReferenceCOCO(); ref_gt.dataset = copy.deepcopy(gt_data); ref_gt.createIndex()
            evaluators.append(ReferenceEval(ref_gt, ref_gt.loadRes(copy.deepcopy(detections)), "keypoints"))
        for ev in evaluators:
            ev.params.useCats = use_cats
            ev.params.maxDets = [cap]
            ev.evaluate(); ev.accumulate()
        for expected in evaluators[1:]:
            for key in ("precision", "recall", "scores"):
                assert actual.eval[key].tobytes() == expected.eval[key].tobytes()
            assert len(actual.evalImgs) == len(expected.evalImgs)
            for a, b in zip(actual.evalImgs, expected.evalImgs):
                assert (a is None) == (b is None)
                if a is not None:
                    for key in ("dtIds", "gtIds", "dtMatches", "gtMatches", "dtIgnore", "gtIgnore"):
                        np.testing.assert_array_equal(a[key], b[key])
        for a, b in zip(actual.per_instance(max_dets=10), dense.per_instance(max_dets=10)):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        assert gt._compact is not None and dt._compact is not None
    np.testing.assert_array_equal(actual.computeOks(1, 1), dense.computeOks(1, 1))
    assert dt.anns[6]["keypoints"] == detections[5]["keypoints"]


@pytest.mark.parametrize("bad_points", [[1.0, 2.0], ["bad", 2.0, 0.0] * 17, [1.0, 2.0, 0.0] * 3])
def test_pose_cap_still_validates_discarded_coordinates(tmp_path, bad_points):
    gt_data = dict(images=[dict(id=1, width=100, height=100)], categories=[dict(id=1)],
                   annotations=[dict(id=1, image_id=1, category_id=1, bbox=[0, 0, 40, 40],
                                     area=1600, iscrowd=0, num_keypoints=17,
                                     keypoints=[1.0, 2.0, 2.0] * 17)])
    detections = [dict(image_id=1, category_id=1, bbox=[0, 0, 40, 40], score=score,
                       keypoints=points)
                  for score, points in [(0.9, [1.0, 2.0, 0.0] * 17), (0.1, bad_points)]]
    gp, dp = tmp_path / "gt.json", tmp_path / "dt.json"
    gp.write_text(json.dumps(gt_data)); dp.write_text(json.dumps(detections))
    gt = COCO(gp, verbose=False); dt = gt.loadRes(dp)
    assert dt._compact is not None
    ev = COCOeval(gt, dt, "keypoints", print_function=lambda *_: None)
    ev.params.maxDets = [1]
    with pytest.raises(ValueError, match="keypoint|triplet|coordinate"):
        ev.evaluate()
