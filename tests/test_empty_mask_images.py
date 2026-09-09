"""Empty COCO images do not need rasterization dimensions."""
import copy

import numpy as np
import pytest
from pycocotools.coco import COCO as ReferenceCOCO
from pycocotools.cocoeval import COCOeval as ReferenceEval

from ultrafast_pycocotools import COCO, COCOeval, mask


@pytest.mark.parametrize('kind', ['segm', 'boundary'])
def test_empty_images_without_dimensions(kind):
    segmentation = mask.encode(np.asfortranarray(np.ones((4, 4), dtype=np.uint8)))
    annotation = dict(id=1, image_id=1, category_id=1, bbox=[0., 0., 4., 4.],
                      segmentation=segmentation, area=16., iscrowd=0)
    gt_data = dict(images=[dict(id=1, height=4, width=4), dict(id=2), dict(id=3)],
                   categories=[dict(id=1)], annotations=[annotation])
    dt_data = copy.deepcopy(gt_data)
    dt_data['images'] = [dict(id=1), dict(id=2, height=4, width=4), dict(id=3)]
    dt_data['annotations'][0].update(image_id=2, score=.9)
    actual = COCOeval(COCO(copy.deepcopy(gt_data)), COCO(copy.deepcopy(dt_data)), kind)
    actual.evaluate()
    actual.accumulate()
    # A fully annotated counterpart must produce the same boundary arrays too.
    if kind == 'boundary':
        for dataset in (gt_data, dt_data):
            for image in dataset['images']:
                image.update(height=4, width=4)
        reference = COCOeval(COCO(gt_data), COCO(dt_data), kind)
    else:
        gt, dt = ReferenceCOCO(), ReferenceCOCO()
        gt.dataset, dt.dataset = gt_data, dt_data
        gt.createIndex()
        dt.createIndex()
        reference = ReferenceEval(gt, dt, kind)
    reference.evaluate()
    reference.accumulate()
    for key in ('precision', 'recall', 'scores'):
        assert actual.eval[key].tobytes() == reference.eval[key].tobytes()


def test_nonempty_image_still_requires_dimensions():
    segmentation = mask.encode(np.asfortranarray(np.ones((4, 4), dtype=np.uint8)))
    data = dict(images=[dict(id=1)], categories=[dict(id=1)], annotations=[
        dict(id=1, image_id=1, category_id=1, bbox=[0., 0., 4., 4.],
             segmentation=segmentation, area=16., iscrowd=0, score=.9)])
    with pytest.raises(KeyError, match='height'):
        COCOeval(COCO(data), COCO(copy.deepcopy(data)), 'segm').evaluate()
