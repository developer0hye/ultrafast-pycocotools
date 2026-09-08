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
