"""File snapshots must preserve mutable COCO API behavior and exact evaluation."""
import copy
import json
import pickle

import pytest

import ultrafast_pycocotools as ufc
from test_eval_parity import assert_bit_identical, run_reference


def write_inputs(tmp_path):
    data = {'images':[{'id':2,'height':128,'width':128},{'id':1,'height':128,'width':128}],
            'annotations':[{'id':3,'image_id':2,'category_id':1,'bbox':[2,3,12,14],'area':168,'iscrowd':0},
                           {'id':7,'image_id':1,'category_id':2,'bbox':[0,0,40,50],'area':2000,'iscrowd':1}],
            'categories':[{'id':1},{'id':2}], 'info':{'custom':'retained'}}
    dets = [{'image_id':2,'category_id':1,'bbox':[2,3,12,14],'score':.8},
            {'image_id':1,'category_id':2,'bbox':[1,1,38,48],'score':.8},
            {'image_id':2,'category_id':2,'bbox':[10,10,4,5],'score':.8}]
    gt, dt = tmp_path/'gt.json', tmp_path/'dt.json'
    gt.write_text(json.dumps(data));dt.write_text(json.dumps(dets))
    return gt, dt, data, dets


def evaluate(gt, dt):
    ev = ufc.COCOeval(gt, dt, 'bbox', print_function=lambda *_: None)
    ev.run()
    return ev


@pytest.mark.parametrize('view', ['dataset','anns','imgToAnns','catToImgs'])
def test_public_views_materialize_real_objects_and_disable_snapshot(tmp_path, view):
    gp, dp, data, dets = write_inputs(tmp_path)
    gt = ufc.COCO(gp, verbose=False)
    assert gt._compact is not None
    assert gt.getImgIds() == [2,1]
    assert gt.getCatIds() == [1,2]
    assert gt._compact is not None
    getattr(gt, view)
    assert gt._compact is None
    assert type(gt.dataset) is dict
    assert type(gt.dataset['annotations']) is list
    assert gt.dataset == data
    assert list(gt.dataset) == list(data)
    assert gt.anns[3] is gt.dataset['annotations'][0]
    assert gt.imgToAnns[2][0] is gt.anns[3]
    gt.anns[3]['bbox'] = [80,80,10,10]
    changed = evaluate(gt, gt.loadRes(dp))
    reference_gt = ufc.COCO(copy.deepcopy(gt.dataset), verbose=False)
    expected = evaluate(reference_gt, reference_gt.loadRes(copy.deepcopy(dets)))
    assert_bit_identical(expected, changed, view)


def test_input_snapshot_survives_file_changes_and_deletion(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    ref = run_reference(gp, dp, 'bbox')
    gt = ufc.COCO(gp, verbose=False);dt = gt.loadRes(dp)
    assert gt._compact is not None and dt._compact is not None
    gp.write_text('{}');dp.unlink()
    assert_bit_identical(ref, evaluate(gt, dt), 'immutable snapshot')
    assert gt.dataset == data
    assert dt.anns[1]['bbox'] == dets[0]['bbox']
    assert dt.anns[1]['id'] == 1 and dt.anns[1]['area'] == 168


@pytest.mark.parametrize('input_types', [('file','file'),('file','dict'),('dict','file')])
def test_mixed_file_and_in_memory_inputs(tmp_path, input_types):
    gp, dp, data, dets = write_inputs(tmp_path)
    ref = run_reference(gp, dp, 'bbox')
    gt = ufc.COCO(gp if input_types[0]=='file' else data, verbose=False)
    dt = gt.loadRes(dp if input_types[1]=='file' else dets)
    actual = evaluate(gt, dt)
    assert_bit_identical(ref, actual, str(input_types))
    assert (gt._compact is not None) == (input_types[0]=='file')
    assert (dt._compact is not None) == (input_types[1]=='file')
    # Diagnostics may materialize annotations, but must describe the same matches.
    assert actual.computeIoU(2, 1).shape == (1, 1)


def test_pickle_and_deepcopy_preserve_index_identity(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    gt = ufc.COCO(gp, verbose=False)
    for restored in [pickle.loads(pickle.dumps(gt)), copy.deepcopy(gt)]:
        assert restored.dataset == data
        assert restored.anns[3] is restored.dataset['annotations'][0]
        assert restored.imgToAnns[2][0] is restored.anns[3]


def test_duplicate_gt_ids_fall_back_to_dictionary_semantics(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    data['annotations'][1]['id'] = 3
    gp.write_text(json.dumps(data))
    gt = ufc.COCO(gp, verbose=False)
    assert gt._compact is None
    assert gt.anns[3]['image_id'] == 1


def test_invalid_detection_image_rejected_at_load_time(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    dets[0]['image_id'] = 999
    dp.write_text(json.dumps(dets))
    with pytest.raises(AssertionError, match='current coco set'):
        ufc.COCO(gp, verbose=False).loadRes(dp)


def test_replacing_dataset_preserves_old_indexes_until_rebuild(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    gt = ufc.COCO(gp, verbose=False)
    gt.dataset = dict(data, annotations=[])
    assert 3 in gt.anns
    gt.createIndex()
    assert gt.anns == {}


@pytest.mark.parametrize('params', [
    {'imgIds': [2]}, {'catIds': [1]}, {'imgIds': [1], 'catIds': [1]},
    {'useCats': 0}, {'useCats': 0, 'catIds': [2]}, {'maxDets': [1, 2, 100]},
    {'imgIds': []}, {'catIds': []},
])
def test_filtered_and_repeated_evaluation(tmp_path, params):
    gp, dp, data, dets = write_inputs(tmp_path)
    def tweak(p):
        for name, value in params.items():
            setattr(p, name, value)
    ref = run_reference(gp, dp, 'bbox', tweak)
    gt = ufc.COCO(gp, verbose=False)
    dt = gt.loadRes(dp)
    ev = ufc.COCOeval(gt, dt, 'bbox', print_function=lambda *_: None)
    tweak(ev.params)
    ev.run()
    assert_bit_identical(ref, ev, str(params))
    ev.run()
    assert_bit_identical(ref, ev, 'repeated ' + str(params))
    assert gt._compact is not None and dt._compact is not None


def test_materialization_preserves_independently_edited_metadata_indexes(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    gt = ufc.COCO(gp, verbose=False)
    images, categories = gt.imgs, gt.cats
    gt.imgs[99] = {'id': 99}
    gt.cats[99] = {'id': 99}
    assert len(gt.anns) == 2
    assert gt.imgs is images and gt.cats is categories
    assert 99 in gt.imgs and 99 in gt.cats
    gt.createIndex()
    assert 99 not in gt.imgs and 99 not in gt.cats


def test_caption_precedence_and_explicit_polygons_fall_back(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    gt = ufc.COCO(gp, verbose=False)
    explicit = gt.loadRes(dp, derive_segmentation=True)
    assert explicit._compact is None
    assert 'segmentation' in explicit.anns[1]
    dets[0]['caption'] = 'An example caption.'
    dp.write_text(json.dumps(dets))
    captions = gt.loadRes(dp)
    assert captions._compact is None
    assert 'area' not in captions.anns[1]


def test_legacy_pickle_state(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    original = ufc.COCO(data, verbose=False)
    state = original.__getstate__().copy()
    for public, private in [('dataset', '_dataset'), ('anns', '_anns'),
                            ('imgToAnns', '_img_to_anns'), ('catToImgs', '_cat_to_imgs')]:
        state[public] = state.pop(private)
    state.pop('_compact')
    restored = ufc.COCO.__new__(ufc.COCO)
    restored.__setstate__(state)
    assert restored.dataset == data
    assert restored.anns[3] is restored.dataset['annotations'][0]


def test_subclass_dataset_access_is_respected(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    class CustomCOCO(ufc.COCO):
        reads = 0
        @property
        def dataset(self):
            self.reads += 1
            return super().dataset
        @dataset.setter
        def dataset(self, value):
            ufc.COCO.dataset.fset(self, value)
    gt = CustomCOCO(gp, verbose=False)
    assert gt._compact is None
    before = gt.reads
    gt.getCatIds()
    assert gt.reads > before
    before = gt.reads
    gt.loadRes(dp)
    assert gt.reads > before


def test_compact_detection_defaults_and_ids_survive_filtering(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    # loadRes overwrites these fields; compact storage must derive the same values.
    for d in dets:
        d.update(id=999, area=-1, iscrowd=1)
    dp.write_text(json.dumps(dets))
    gt = ufc.COCO(gp, verbose=False)
    dt = gt.loadRes(dp)
    ev = ufc.COCOeval(gt, dt, 'bbox', print_function=lambda *_: None)
    ev.params.imgIds = [2]
    ev.run()
    predictions, _ = ev.per_instance(iou_thr=.5)
    assert sorted(predictions['dt_id'].tolist()) == [1, 3]
    assert gt._compact is not None and dt._compact is not None
    ref = run_reference(gp, dp, 'bbox', lambda p: setattr(p, 'imgIds', [2]))
    assert_bit_identical(ref, ev, 'derived detection fields')
    assert dt.anns[3]['id'] == 3
    assert dt.anns[3]['area'] == 20
    assert dt.anns[3]['iscrowd'] == 0


@pytest.mark.parametrize('thresholds', [[0., 0., .25, .5, 1.], [.01, .2, .3, .99]])
def test_compact_recall_samples_with_custom_grids(tmp_path, thresholds):
    import numpy as np
    gp, dp, data, dets = write_inputs(tmp_path)
    def tweak(p):
        p.recThrs = np.array(thresholds)
        p.maxDets = [1, 3, 7]
    ref = run_reference(gp, dp, 'bbox', tweak)
    gt = ufc.COCO(gp, verbose=False)
    ev = ufc.COCOeval(gt, gt.loadRes(dp), 'bbox', print_function=lambda *_: None)
    tweak(ev.params)
    ev.run()
    assert_bit_identical(ref, ev, 'custom recall samples')


@pytest.mark.parametrize('coordinate', [2**53 + 1, -(2**53 + 1), 2**63])
def test_large_integer_coordinates_keep_ordinary_json_semantics(tmp_path, coordinate):
    gp, dp, data, dets = write_inputs(tmp_path)
    data['annotations'][0]['bbox'][0] = coordinate
    gp.write_text(json.dumps(data))
    gt = ufc.COCO(gp, verbose=False)
    assert gt._compact is None
    assert gt.anns[3]['bbox'][0] == coordinate
    assert isinstance(gt.anns[3]['bbox'][0], int)


@pytest.mark.parametrize('iou_type', ['segm', 'keypoints'])
@pytest.mark.parametrize('input_types', [('file', 'file'), ('file', 'dict'), ('dict', 'file')])
def test_nonbbox_keeps_file_snapshots_and_exact_curves(synthetic, synthetic_kp, iou_type, input_types):
    gp, dp = synthetic_kp if iou_type == 'keypoints' else synthetic
    def tweak(p):
        p.imgIds = p.imgIds[::3]
        p.catIds = p.catIds[::2]
    reference = run_reference(gp, dp, iou_type, tweak)
    gt = ufc.COCO(gp if input_types[0] == 'file' else json.loads(gp.read_text()), verbose=False)
    dt = gt.loadRes(dp if input_types[1] == 'file' else json.loads(dp.read_text()))
    ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None)
    tweak(ev.params)
    for _ in range(2):
        ev.run()
        assert_bit_identical(reference, ev, f'{iou_type}/{input_types}')
        assert (gt._compact is not None) == (input_types[0] == 'file')
        assert (dt._compact is not None) == (input_types[1] == 'file')


def test_keypoint_json_boolean_coordinates_and_mutable_views(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    for ann in data['annotations']:
        ann.update(keypoints=[True, False, 2] * 17, num_keypoints=17)
    for ann in dets:
        ann['keypoints'] = [True, False, 2] * 17
    gp.write_text(json.dumps(data))
    dp.write_text(json.dumps(dets))
    reference = run_reference(gp, dp, 'keypoints')
    gt = ufc.COCO(gp, verbose=False)
    ev = ufc.COCOeval(gt, gt.loadRes(dp), 'keypoints', print_function=lambda *_: None)
    ev.run()
    assert_bit_identical(reference, ev, 'boolean coordinates')
    gt.anns[3]['keypoints'] = [90, 90, 2] * 17
    gp.write_text(json.dumps(gt.dataset))
    ev.run()
    assert_bit_identical(run_reference(gp, dp, 'keypoints'), ev, 'mutated public keypoints')


def test_duplicate_annotation_arrays_keep_last_json_value(tmp_path):
    gp, dp, data, dets = write_inputs(tmp_path)
    text = json.dumps(data)
    gp.write_text(text[:-1] + ', "annotations": []}')
    gt = ufc.COCO(gp, verbose=False)
    assert gt._compact is None
    assert gt.dataset['annotations'] == []
    assert gt.anns == {}


@pytest.mark.parametrize('use_cats', [0, 1])
@pytest.mark.parametrize('iou_type', ['segm', 'boundary', 'keypoints'])
def test_nonbbox_compact_matches_materialized_diagnostics(synthetic, synthetic_kp, iou_type, use_cats):
    import numpy as np
    gp, dp = synthetic_kp if iou_type == 'keypoints' else synthetic
    evaluations = []
    for materialized in (False, True):
        gt = ufc.COCO(gp, verbose=False)
        dt = gt.loadRes(dp)
        if materialized:
            gt.anns
            dt.anns
        ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None)
        ev.params.useCats = use_cats
        ev.params.catIds = ev.params.catIds[::2]
        ev.params.maxDets = [1, 3, 20] if iou_type != 'keypoints' else [20]
        ev.run()
        assert (gt._compact is None) == materialized
        evaluations.append(ev)
    first, second = evaluations
    assert_bit_identical(first, second, f'{iou_type}/useCats={use_cats}')
    for actual, expected in zip(first.per_instance(iou_thr=.5), second.per_instance(iou_thr=.5)):
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize('field', ['segmentation', 'keypoints', 'num_keypoints'])
def test_duplicate_geometry_fields_keep_last_json_value(tmp_path, field):
    gp, dp, data, dets = write_inputs(tmp_path)
    iou_type = 'segm' if field == 'segmentation' else 'keypoints'
    for ann in data['annotations'] + dets:
        if iou_type == 'segm':
            ann['segmentation'] = [[2, 3, 14, 3, 14, 17, 2, 17]]
        else:
            ann.update(keypoints=[5, 5, 2] * 17, num_keypoints=17)
    for path, value in [(gp, data), (dp, dets)]:
        path.write_text(json.dumps(value).replace(f'"{field}":', f'"{field}": null, "{field}":'))
    reference = run_reference(gp, dp, iou_type)
    gt = ufc.COCO(gp, verbose=False)
    dt = gt.loadRes(dp)
    actual = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None)
    actual.run()
    assert_bit_identical(reference, actual, f'duplicate {field}')
    assert gt.anns[3][field] == data['annotations'][0][field]


@pytest.mark.parametrize('iou_type', ['segm', 'boundary'])
@pytest.mark.parametrize('use_cats', [0, 1])
def test_mask_cap_preserves_ties_and_reloads_geometry_when_limit_grows(tmp_path, iou_type, use_cats):
    import numpy as np
    gp, dp, data, dets = write_inputs(tmp_path)
    # Interleave categories; category-major order, not file order, breaks
    # cross-category ties when useCats=0. Each mask has a distinct location.
    dets = [dict(image_id=2, category_id=cat, bbox=[x, 3, 12, 14], score=score)
            for cat, x, score in [(2, 90, .8), (1, 2, .8), (1, 50, .9),
                                  (2, 20, .8), (1, 70, -0.0), (1, 2, 0.0)]]
    for ann in data['annotations'] + dets:
        x, y, w, h = ann['bbox']
        ann['segmentation'] = [[x, y, x+w, y, x+w, y+h, x, y+h]]
    gp.write_text(json.dumps(data)); dp.write_text(json.dumps(dets))
    gt = ufc.COCO(gp, verbose=False)
    dt = gt.loadRes(dp)
    ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None, store_eval_imgs=True)
    ev.params.useCats = use_cats
    for cap in (2, 0, 10):
        ev.params.maxDets = [cap]
        ev.evaluate(); ev.accumulate()
        if iou_type == 'segm':
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            rg = COCO(str(gp)); rd = rg.loadRes(str(dp))
            ref = COCOeval(rg, rd, iou_type)
        else:
            rg = ufc.COCO(copy.deepcopy(data), verbose=False)
            ref = ufc.COCOeval(rg, rg.loadRes(copy.deepcopy(dets)), iou_type,
                              print_function=lambda *_: None, store_eval_imgs=True)
        ref.params.useCats = use_cats
        ref.params.maxDets = [cap]
        ref.evaluate(); ref.accumulate()
        for key in ('precision', 'recall', 'scores'):
            np.testing.assert_array_equal(ev.eval[key], ref.eval[key])
        for actual, expected in zip(ev.evalImgs, ref.evalImgs):
            assert (actual is None) == (expected is None)
            if actual is not None:
                for key in ('dtIds', 'gtIds', 'dtMatches', 'gtMatches', 'dtIgnore', 'gtIgnore'):
                    np.testing.assert_array_equal(actual[key], expected[key])
        assert gt._compact is not None and dt._compact is not None


@pytest.mark.parametrize('use_cats', [0, 1])
def test_segmentation_spans_keep_escaped_counts_exact(tmp_path, use_cats):
    # The RLE alphabet contains a backslash, which JSON escapes. Compact files
    # keep such strings in the snapshot and decode them lazily; strings with any
    # other escape take the ordinary decoder. Both must match pycocotools.
    import numpy as np
    from pycocotools import mask as ref_mask
    rng = np.random.default_rng(7)
    h, w = 96, 128
    images = [{'id': i, 'height': h, 'width': w} for i in (1, 2, 3)]
    annotations, dets = [], []
    for i in range(90):
        image, category = 1 + i % 3, 1 + i % 2
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        bw, bh = int(rng.integers(6, 40)), int(rng.integers(6, 40))
        annotations.append({'id': i + 1, 'image_id': image, 'category_id': category,
                            'bbox': [x, y, bw, bh], 'area': bw * bh, 'iscrowd': 0,
                            'segmentation': [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]})
        for _ in range(2):
            m = np.zeros((h, w), np.uint8)
            dx, dy = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))
            m[max(y + dy, 0):y + dy + bh, max(x + dx, 0):x + dx + bw] = 1
            m[rng.integers(0, h, 25), rng.integers(0, w, 25)] = 1
            rle = ref_mask.encode(np.asfortranarray(m))
            # A file bbox, as detectors write it; its area differs from the mask area.
            dets.append({'image_id': image, 'category_id': category, 'score': float(rng.random()),
                         'bbox': [x + dx, y + dy, bw, bh],
                         'segmentation': {'size': [h, w], 'counts': rle['counts'].decode()}})
    data = {'images': images, 'annotations': annotations, 'categories': [{'id': 1}, {'id': 2}]}
    escaped = [i for i, d in enumerate(dets) if '\\' in d['segmentation']['counts']]
    assert len(escaped) > 10
    gp, dp = tmp_path / 'gt.json', tmp_path / 'dt.json'
    gp.write_text(json.dumps(data))
    text = json.dumps(dets)
    # One string uses a unicode escape for its backslash instead of a pair.
    first = json.dumps(dets[escaped[0]]['segmentation']['counts'])
    text = text.replace(first, first.replace('\\\\', '\\u005c'), 1)
    assert '\\u005c' in text
    dp.write_text(text)

    def tweak(params):
        params.useCats = use_cats

    reference = run_reference(gp, dp, 'segm', tweak)
    gt = ufc.COCO(gp, verbose=False)
    dt = gt.loadRes(dp)
    actual = ufc.COCOeval(gt, dt, 'segm', print_function=lambda *_: None)
    tweak(actual.params)
    actual.run()
    assert gt._compact is not None and dt._compact is not None
    assert_bit_identical(reference, actual, f'escaped counts useCats={use_cats}')


def _large_results(tmp_path, count=45000):
    # Above the parallel-parsing threshold. String values imitate record
    # boundaries (`}, {`) and contain escapes, so a chunk may start inside one.
    import numpy as np
    rng = np.random.default_rng(11)
    images = [{'id': i, 'height': 480, 'width': 640} for i in range(1, 301)]
    annotations = []
    for i in range(1500):
        x, y, w, h = [float(v) for v in rng.uniform(0, 400, 2)] + [float(v) for v in rng.uniform(4, 200, 2)]
        annotations.append({'id': i + 1, 'image_id': 1 + i % 300, 'category_id': 1 + i % 3,
                            'bbox': [x, y, w, h], 'area': w * h, 'iscrowd': 0})
    dets = []
    for i in range(count):
        a = annotations[int(rng.integers(0, len(annotations)))]
        box = [v + float(rng.normal(0, 3)) for v in a['bbox']]
        name = ['plain.jpg', 'a}, {"image_id": 1, "bbox": [0,0,1,1]}, {"x', 'q\\"}, {\\\\', '},{'][i % 4]
        dets.append({'image_id': a['image_id'], 'file_name': name, 'category_id': a['category_id'],
                     'bbox': box, 'score': round(float(rng.random()), 3)})
    data = {'images': images, 'annotations': annotations, 'categories': [{'id': c} for c in (1, 2, 3)]}
    gp, dp = tmp_path / 'gt.json', tmp_path / 'dt.json'
    gp.write_text(json.dumps(data))
    text = json.dumps(dets)
    assert len(text) > 4 << 20
    dp.write_text(text)
    return gp, dp, dets


def test_parallel_result_parsing_matches_sequential_semantics(tmp_path):
    gp, dp, dets = _large_results(tmp_path)
    reference = run_reference(gp, dp, 'bbox')
    gt = ufc.COCO(gp, verbose=False)
    dt = gt.loadRes(dp)
    assert dt._compact is not None and dt._compact.annotation_count == len(dets)
    actual = ufc.COCOeval(gt, dt, 'bbox', print_function=lambda *_: None)
    actual.run()
    assert_bit_identical(reference, actual, 'parallel result parsing')
    expected = gt.loadRes(copy.deepcopy(dets))
    assert dt.anns == expected.anns


@pytest.mark.parametrize('damage', ['trailing comma', 'trailing data', 'truncated', 'bad separator',
                                    'non-object element', 'bad record'])
def test_parallel_result_parsing_rejects_like_the_sequential_parser(tmp_path, damage):
    gp, dp, dets = _large_results(tmp_path)
    text = dp.read_text()
    middle = text.index('}, {', len(text) // 2) + 1
    text = {
        'trailing comma': text[:-1] + ', ]',
        'trailing data': text + ' x',
        'truncated': text[:-7],
        'bad separator': text[:middle] + ';' + text[middle + 1:],
        'non-object element': text[:middle] + ', 5' + text[middle:],
        'bad record': text[:middle] + ', {"image_id": 1}' + text[middle:],
    }[damage]
    dp.write_text(text)
    gt = ufc.COCO(gp, verbose=False)

    def load(**options):
        try:
            return gt.loadRes(dp, **options), None
        except Exception as error:  # noqa: BLE001 - the outcome is compared
            return None, type(error)

    # `derive_segmentation=True` bypasses compact loading: the ordinary loader
    # is what a rejected compact file must fall back to.
    actual, actual_error = load()
    expected, expected_error = load(derive_segmentation=True)
    assert actual_error is expected_error
    if expected_error is None:
        assert actual._compact is None and len(actual.anns) == len(expected.anns)


def _crowded_bbox_inputs(tmp_path):
    import numpy as np
    rng = np.random.default_rng(3)
    images = [{'id': i, 'height': 200, 'width': 200} for i in range(1, 41)]
    annotations, dets = [], []
    for i in range(400):
        image, category = int(rng.integers(1, 41)), int(rng.integers(1, 4))
        x, y = [float(v) for v in rng.uniform(0, 150, 2)]
        w, h = [float(v) for v in rng.uniform(3, 60, 2)]
        annotations.append({'id': i + 1, 'image_id': image, 'category_id': category, 'bbox': [x, y, w, h],
                            'area': w * h, 'iscrowd': int(rng.random() < .05)})
        for _ in range(int(rng.integers(0, 4))):
            box = [x + rng.normal(0, 4), y + rng.normal(0, 4), w * rng.uniform(.7, 1.3), h * rng.uniform(.7, 1.3)]
            dets.append({'image_id': image, 'category_id': category if rng.random() < .8 else int(rng.integers(1, 4)),
                         'bbox': [float(v) for v in box], 'score': float(np.round(rng.random(), 2))})
    for _ in range(600):
        box = [float(v) for v in rng.uniform(0, 150, 2)] + [float(v) for v in rng.uniform(2, 80, 2)]
        dets.append({'image_id': int(rng.integers(1, 41)), 'category_id': int(rng.integers(1, 4)), 'bbox': box,
                     'score': float(np.round(rng.random(), 2))})
    gp, dp = tmp_path / 'gt.json', tmp_path / 'dt.json'
    gp.write_text(json.dumps({'images': images, 'annotations': annotations,
                              'categories': [{'id': c} for c in (1, 2, 3)]}))
    dp.write_text(json.dumps(dets))
    return gp, dp


@pytest.mark.parametrize('max_dets', [[1, 10, 100], [3, 2], [0, 5]])
@pytest.mark.parametrize('thresholds', [[0., 0., .25, .5, 1.], [-1., 0., 1., 1.5], [.3, .3, .3], [2., .1]])
def test_precision_envelope_with_tied_scores_and_unusual_grids(tmp_path, thresholds, max_dets):
    # Tied scores, crowds, several detections per ground truth and
    # out-of-range or repeated recall thresholds all reach the sampled curve.
    import numpy as np
    gp, dp = _crowded_bbox_inputs(tmp_path)

    def tweak(p):
        p.recThrs = np.array(thresholds)
        p.maxDets = max_dets

    import contextlib
    import io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        reference_gt = COCO(str(gp))
        reference = COCOeval(reference_gt, reference_gt.loadRes(str(dp)), 'bbox')
        tweak(reference.params)
        reference.evaluate()
        reference.accumulate()
    gt = ufc.COCO(gp, verbose=False)
    actual = ufc.COCOeval(gt, gt.loadRes(dp), 'bbox', print_function=lambda *_: None)
    tweak(actual.params)
    actual.evaluate()
    actual.accumulate()
    # summarize() indexes maxDets[2]; compare the complete arrays instead.
    for key in ('precision', 'recall', 'scores'):
        expected = np.ascontiguousarray(reference.eval[key], dtype=np.float64)
        assert expected.tobytes() == np.ascontiguousarray(actual.eval[key]).tobytes(), key
