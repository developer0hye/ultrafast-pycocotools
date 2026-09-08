"""The scaling workload must preserve original order and keep every category."""
import json
from pathlib import Path

from scale import prepare


def test_nested_subsets_preserve_order_categories_and_full_inputs(tmp_path):
    gt = {'images': [{'id': i} for i in [9, 3, 5, 1]],
          'categories': [{'id': 1}, {'id': 2}],
          'annotations': [{'id': j, 'image_id': i, 'category_id': 1}
                          for j, i in enumerate([5, 9, 1, 3, 5])]}
    pred = [{'image_id': i, 'category_id': 1, 'score': .5} for i in [1, 3, 9, 5, 1]]
    gt_path, pred_path = tmp_path / 'gt.json', tmp_path / 'pred.json'
    gt_path.write_text(json.dumps(gt))
    pred_path.write_text(json.dumps(pred))
    before = gt_path.read_bytes(), pred_path.read_bytes()
    prepare(gt_path, pred_path, tmp_path, [1, 2, 4], 123)
    points = json.loads((tmp_path / 'inputs.json').read_text())
    previous = set()
    for point in points:
        subset = json.loads(Path(point['gt']).read_text())
        detections = json.loads(Path(point['pred']).read_text())
        selected = {im['id'] for im in subset['images']}
        assert previous <= selected
        assert len(selected) == point['images']
        assert subset['categories'] == gt['categories']
        assert subset['annotations'] == [a for a in gt['annotations'] if a['image_id'] in selected]
        assert detections == [d for d in pred if d['image_id'] in selected]
        assert point['annotations'] == len(subset['annotations'])
        assert point['detections'] == len(detections)
        previous = selected
    assert points[-1]['gt'] == str(gt_path)
    assert points[-1]['pred'] == str(pred_path)
    assert before == (gt_path.read_bytes(), pred_path.read_bytes())
