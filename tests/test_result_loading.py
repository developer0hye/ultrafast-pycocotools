"""Compact results preserve masks, metrics and public query semantics."""
import copy

import numpy as np
import pytest

import ultrafast_pycocotools as ufc


def test_compact_bbox_results_support_public_mask_helpers():
    data = {'images':[{'id':7,'height':100,'width':100}], 'categories':[{'id':1}], 'annotations':[]}
    predictions = [{'image_id':7,'category_id':1,'bbox':[10.25,20.75,12.5,18.25],'score':.9}]
    gt = ufc.COCO(data, verbose=False)
    regular = gt.loadRes(copy.deepcopy(predictions), derive_segmentation=True)
    compact = gt.loadRes(copy.deepcopy(predictions))
    assert 'segmentation' in regular.anns[1]
    assert 'segmentation' not in compact.anns[1]
    np.testing.assert_array_equal(compact.annToMask(compact.anns[1]), regular.annToMask(regular.anns[1]))
    assert 'segmentation' not in compact.anns[1], 'on-demand masks must not retain derived polygons'


@pytest.mark.parametrize('image_ids,category_ids', [([2,1],[1]), ([1,2],[]), ([],[1]), ([],[])])
def test_eval_collection_preserves_duplicate_id_resolution(image_ids, category_ids):
    data = {'images':[{'id':1},{'id':2}], 'categories':[{'id':1},{'id':2}],
            'annotations':[{'id':9,'image_id':1,'category_id':1},
                           {'id':10,'image_id':2,'category_id':1},
                           {'id':9,'image_id':2,'category_id':2}]}
    gt = ufc.COCO(data, verbose=False)
    expected = gt.loadAnns(gt.getAnnIds(imgIds=image_ids,catIds=category_ids))
    assert gt._eval_annotations(image_ids,category_ids) == expected
    assert gt.catToImgs[1] == [1,2]
    assert gt.catToImgs[2] == [2]


def test_in_memory_annotations_and_ultralytics_metric_access(real_coco):
    import json
    gt_path, dt_path = real_coco
    data = json.loads(gt_path.read_text())
    gt = ufc.COCO(data, verbose=False)
    assert gt.dataset is data
    ev = ufc.COCOeval(gt, gt.loadRes(str(dt_path)),
                     'bbox', lvis_style=False, print_function=lambda *_: None)
    ev.run()
    for alias, index in [('AP_all',0),('AP_50',1),('AP_small',3),('AP_medium',4),('AP_large',5)]:
        assert ev.stats_as_dict[alias] == ev.stats[index]


def test_default_results_draw_without_retaining_polygons():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt = ufc.COCO({'images':[{'id':1,'height':32,'width':32}],
                   'categories':[{'id':1}], 'annotations':[]}, verbose=False)
    dt = gt.loadRes([{'image_id':1,'category_id':1,'bbox':[2,3,10,12],'score':.9}])
    fig, ax = plt.subplots()
    try:
        dt.showAnns(dt.loadAnns([1]))
        assert len(ax.collections[0].get_paths()) == 1
        assert 'segmentation' not in dt.anns[1]
    finally:
        plt.close(fig)


def test_materialized_results_match_reference_annotation_fields():
    from pycocotools.coco import COCO as ReferenceCOCO
    data = {'images':[{'id':1,'height':32,'width':32}],
            'categories':[{'id':1}], 'annotations':[]}
    preds = [{'image_id':1,'category_id':1,'bbox':[2,3,10,12],'score':.9}]
    ref = ReferenceCOCO()
    ref.dataset = copy.deepcopy(data)
    ref.createIndex()
    actual = ufc.COCO(data, verbose=False).loadRes(copy.deepcopy(preds), derive_segmentation=True)
    assert actual.anns == ref.loadRes(copy.deepcopy(preds)).anns


def test_keypoint_results_do_not_gain_implicit_segmentation():
    gt = ufc.COCO({'images':[{'id':1,'height':32,'width':32}],
                   'categories':[{'id':1}], 'annotations':[]}, verbose=False)
    dt = gt.loadRes([{'image_id':1,'category_id':1,'keypoints':[2,3,2,10,12,2],'score':.9}])
    assert 'bbox' in dt.anns[1]  # Derived keypoint bounds are not a box prediction.
    with pytest.raises(KeyError, match='segmentation'):
        dt.annToMask(dt.anns[1])
