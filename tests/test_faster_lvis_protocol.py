"""Preserve the COCO-style federated protocol used by Ultralytics."""
import copy

import numpy as np
import pytest

from ultrafast_pycocotools import COCO, COCOeval
from test_lvis import dataset


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
@pytest.mark.parametrize('limited', [False, True])
def test_coco_style_lvis_matches_existing_backend(iou_type, limited):
    faster = pytest.importorskip('faster_coco_eval')
    data, predictions = dataset()
    # Force the distinction between a global image cap and per-category caps.
    predictions = [dict(predictions[0], score=.99) for _ in range(301)] + predictions
    # COCO-style evaluation derives ignore from iscrowd; official LVIS does not.
    data['annotations'][0]['ignore'] = 1
    data['annotations'][0]['iscrowd'] = 0
    results = []
    for coco, evaluator in [(faster.COCO, faster.COCOeval_faster), (COCO, COCOeval)]:
        gt = coco(copy.deepcopy(data))
        extra = {'lvis_protocol': 'coco'} if evaluator is COCOeval else {}
        ev = evaluator(gt, gt.loadRes(copy.deepcopy(predictions)), iou_type,
                       lvis_style=True, print_function=lambda *_: None, **extra)
        if limited:
            ev.params.catIds = [1, 2]
            ev.params.maxDets = [1, 2, 10]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        results.append(ev)
    reference, actual = results
    assert reference.params.maxDets == actual.params.maxDets
    for key in ['precision', 'recall', 'scores']:
        np.testing.assert_allclose(actual.eval[key], reference.eval[key], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.stats, reference.stats, rtol=0, atol=1e-12)
    for key in ['AP_all', 'AP_50', 'AP_small', 'AP_medium', 'AP_large', 'APr', 'APc', 'APf']:
        assert actual.stats_as_dict[key] == pytest.approx(reference.stats_as_dict[key], rel=0, abs=1e-12)
