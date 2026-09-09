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
