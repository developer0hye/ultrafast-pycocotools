"""Validate the federated protocol against the official LVIS API, not COCO AP."""
import copy
import json
import warnings

import numpy as np
import pytest

import ultrafast_pycocotools as ufc


def dataset():
    categories = [{'id': i, 'name': f'category{i}', 'frequency': freq}
                  for i, freq in enumerate(['r', 'c', 'f', 'r'], 1)]
    images = [{'id': i, 'height': 256, 'width': 256,
               'neg_category_ids': [2, 3], 'not_exhaustive_category_ids': [1] if i == 2 else []}
              for i in range(1, 5)]
    annotations = []
    for i, cat, side, ignore in [(1, 1, 16, 0), (2, 1, 40, 0), (3, 2, 100, 0), (3, 3, 16, 1)]:
        box = [10, 10, side, side]
        annotations.append({'id': len(annotations) + 1, 'image_id': i, 'category_id': cat,
                            'bbox': box, 'area': side * side, 'ignore': ignore,
                            'iscrowd': ignore, 'segmentation': [[10, 10, 10, 10+side, 10+side, 10+side, 10+side, 10]]})
    predictions = [
        {'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 16, 16], 'score': .8},
        {'image_id': 2, 'category_id': 1, 'bbox': [10, 10, 40, 40], 'score': .8},
        {'image_id': 2, 'category_id': 1, 'bbox': [150, 150, 20, 20], 'score': .9},  # non-exhaustive FP ignored
        {'image_id': 1, 'category_id': 2, 'bbox': [150, 150, 20, 20], 'score': .9},  # verified-negative FP
        {'image_id': 1, 'category_id': 4, 'bbox': [10, 10, 20, 20], 'score': .99},  # unverified category
        {'image_id': 3, 'category_id': 2, 'bbox': [10, 10, 100, 100], 'score': .7},
        {'image_id': 3, 'category_id': 3, 'bbox': [10, 10, 16, 16], 'score': .95},  # ignored GT
        {'image_id': 4, 'category_id': 2, 'bbox': [10, 10, 16, 16], 'score': .6},  # empty image FP
    ]
    return {'images': images, 'categories': categories, 'annotations': annotations}, predictions


def official(tmp_path, data, predictions, iou_type, cap, categories, monkeypatch):
    pytest.importorskip("lvis", reason="Install .[lvis-test] for the official reference")
    from lvis import LVIS, LVISResults, LVISEval
    # LVIS 0.5.3 uses NumPy's removed np.float spelling for float64 only.
    monkeypatch.setattr(np, 'float', float, raising=False)
    path = tmp_path / 'lvis.json'
    path.write_text(json.dumps(data))
    gt = LVIS(str(path))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="The 'warn' method is deprecated", category=DeprecationWarning)
        dt = LVISResults(gt, copy.deepcopy(predictions), max_dets=cap)
    ev = LVISEval(gt, dt, iou_type)
    ev.params.max_dets = cap
    if categories is not None:
        ev.params.cat_ids = categories
    ev.run()
    return ev


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
@pytest.mark.parametrize('cap,categories', [(300, None), (2, None), (2, [1, 2])])
@pytest.mark.parametrize('file_results', [False, True])
def test_lvis_matches_official_federated_protocol(tmp_path, monkeypatch, iou_type, cap, categories, file_results):
    data, predictions = dataset()
    before = copy.deepcopy(data)
    reference = official(tmp_path, data, predictions, iou_type, cap, categories, monkeypatch)
    gt = ufc.COCO(data, verbose=False)
    prediction_path = tmp_path / 'predictions.json'
    prediction_path.write_text(json.dumps(predictions))
    dt = gt.loadRes(prediction_path if file_results else copy.deepcopy(predictions))
    ev = ufc.COCOeval(gt, dt, iou_type, lvis_style=True, print_function=lambda *_: None, store_eval_imgs=True)
    ev.params.maxDets = [cap]
    if categories is not None:
        ev.params.catIds = categories
    ev.run()
    for key in ('precision', 'recall'):
        actual = ev.eval[key][..., 0]
        assert actual.shape == reference.eval[key].shape
        assert actual.tobytes() == np.ascontiguousarray(reference.eval[key]).tobytes(), key
    expected_stats = np.asarray(list(reference.results.values()), dtype=np.float64)
    assert ev.stats.tobytes() == expected_stats.tobytes()
    metrics = ev.stats_as_dict
    for key, value in reference.results.items():
        assert metrics[key] == value
    assert metrics['AP_all'] == metrics['AP']
    assert metrics['AP_50'] == metrics['AP50']
    assert metrics['AP_small'] == metrics['APs']
    assert data == before
    # Public per-image evaluation must apply the same non-exhaustive ignore rule.
    img = ev.evaluateImg(2, 1, ev.params.areaRng[0], cap)
    if img is not None:
        stored = next(x for x in ev.evalImgs if x and x['image_id'] == 2 and x['category_id'] == 1 and list(x['aRng']) == ev.params.areaRng[0])
        np.testing.assert_array_equal(img['dtIgnore'], stored['dtIgnore'])


def test_global_limit_precedes_unverified_category_filter(tmp_path, monkeypatch):
    data, predictions = dataset()
    predictions = [{'image_id': 1, 'category_id': 4, 'bbox': [0, 0, 10, 10], 'score': .99} for _ in range(301)] + predictions
    reference = official(tmp_path, data, predictions, 'bbox', 300, [1], monkeypatch)
    gt = ufc.COCO(data, verbose=False)
    ev = ufc.COCOeval(gt, gt.loadRes(copy.deepcopy(predictions)), 'bbox', lvis_style=True, print_function=lambda *_: None)
    ev.params.catIds = [1]
    ev.run()
    assert ev.eval['precision'][..., 0].tobytes() == np.ascontiguousarray(reference.eval['precision']).tobytes()


def test_lvis_requires_federated_metadata():
    data, predictions = dataset()
    del data['images'][0]['neg_category_ids']
    gt = ufc.COCO(data, verbose=False)
    ev = ufc.COCOeval(gt, gt.loadRes(predictions), 'bbox', lvis_style=True)
    with pytest.raises(ValueError, match='federated annotation metadata'):
        ev.evaluate()


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_public_lvis_example_matches_official(tmp_path, monkeypatch, iou_type):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / 'bench/data'
    if not (root / 'lvis_gt_100.json').is_file():
        pytest.skip('Run python bench/fetch_lvis_fixture.py for the official public LVIS example')
    data = json.loads((root / 'lvis_gt_100.json').read_text())
    predictions = json.loads((root / 'lvis_dt_100.json').read_text())
    reference = official(tmp_path, data, predictions, iou_type, 300, None, monkeypatch)
    gt = ufc.COCO(data, verbose=False)
    ev = ufc.COCOeval(gt, gt.loadRes(copy.deepcopy(predictions)),
                     iou_type, lvis_style=True, print_function=lambda *_: None)
    ev.run()
    for key in ('precision', 'recall'):
        assert ev.eval[key][..., 0].tobytes() == np.ascontiguousarray(reference.eval[key]).tobytes(), key
    assert ev.stats.tobytes() == np.asarray(list(reference.results.values()), dtype=np.float64).tobytes()


def test_coco_ignores_foreign_lvis_mark_metadata():
    data, predictions = dataset()
    gt = ufc.COCO(data, verbose=False)
    outputs = []
    for marked in (False, True):
        dets = copy.deepcopy(predictions)
        if marked:
            for det in dets:
                det['lvis_mark'] = True
        ev = ufc.COCOeval(gt, gt.loadRes(dets), 'bbox', print_function=lambda *_: None)
        ev.run()
        outputs.append(ev.eval['precision'].tobytes())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('protocol', ['official', 'coco'])
@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_file_results_filter_before_materializing_and_keep_mutable_views(tmp_path, protocol, iou_type):
    from test_eval_parity import assert_bit_identical
    data, predictions = dataset()
    # Early unverified high scores consume the official cap. Tied verified
    # detections keep their original IDs/order across filtering and reuse.
    predictions = [dict(predictions[4], score=.99) for _ in range(5)] + predictions
    predictions += [dict(predictions[0], score=.8) for _ in range(4)]
    path = tmp_path / 'predictions.json'
    path.write_text(json.dumps(predictions))
    gt = ufc.COCO(copy.deepcopy(data), verbose=False)
    dt = gt.loadRes(path)
    for cap, cats in [(2, [1, 2]), (300, [1, 2, 3, 4])]:
        results = []
        for detections in [dt, gt.loadRes(copy.deepcopy(predictions))]:
            ev = ufc.COCOeval(gt, detections, iou_type, lvis_style=True,
                              lvis_protocol=protocol, print_function=lambda *_: None,
                              store_eval_imgs=True)
            ev.params.maxDets = [cap] if protocol == 'official' else [1, 2, cap]
            ev.params.catIds = cats
            ev.run()
            results.append(ev)
        assert_bit_identical(results[0], results[1], f'{protocol}/{iou_type}/{cap}')
        for a, b in zip(results[0].evalImgs, results[1].evalImgs):
            assert (a is None) == (b is None)
            if a is not None:
                for key in ('dtIds', 'dtScores', 'dtMatches', 'dtIgnore'):
                    np.testing.assert_array_equal(a[key], b[key])
        assert dt._compact is not None
    dt.anns[6]['score'] = .01
    assert dt._compact is None
    predictions[5]['score'] = .01
    reference_dt = gt.loadRes(copy.deepcopy(predictions))
    evaluations = [ufc.COCOeval(gt, d, iou_type, lvis_style=True,
                    lvis_protocol=protocol, print_function=lambda *_: None)
                   for d in (dt, reference_dt)]
    for ev in evaluations:
        ev.run()
    assert_bit_identical(*evaluations, 'mutable detection score')
