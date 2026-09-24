"""Ultralytics validation with ultrafast as its COCO evaluator.

Runs the real YOLO validators on the small coco8 datasets, captures every
evaluator Ultralytics builds, and re-scores the same predictions and ground
truth with pycocotools. The complete arrays, the summary statistics and the
metrics Ultralytics reports must match exactly.

Two Ultralytics routes are covered:

* in-memory annotations it converts from detection labels (``gdict``);
* ``coco_evaluate`` on files, as for official COCO runs: the saved
  ``predictions.json`` and a COCO annotation file built from the same dataset
  labels. The coco8 images come from train2017, so the official val2017
  annotation files would not contain them.

Needs an Ultralytics revision that evaluates with ultrafast-pycocotools
(ultralytics/ultralytics#26101) and network access for the coco8 datasets and
model weights. Point ``YOLO_CONFIG_DIR`` at a scratch directory to keep the
downloads out of the user's Ultralytics dataset directory.
"""
import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('ultralytics', reason='optional Ultralytics integration dependencies')
pytest.importorskip('torch')
from pycocotools.coco import COCO as ReferenceCOCO
from pycocotools.cocoeval import COCOeval as ReferenceCOCOeval

import ultrafast_pycocotools

TASKS = {
    'detect': ('yolo26n.pt', 'coco8.yaml', ['bbox'], ['Box']),
    'segment': ('yolo26n-seg.pt', 'coco8-seg.yaml', ['bbox', 'segm'], ['Box', 'Mask']),
    'pose': ('yolo26n-pose.pt', 'coco8-pose.yaml', ['bbox', 'keypoints'], ['Box', 'Pose']),
}


@pytest.fixture
def captured(monkeypatch):
    """Every ultrafast COCOeval Ultralytics creates."""
    evaluators = []
    original = ultrafast_pycocotools.COCOeval

    def recording(*args, **options):
        # Return the real class: a subclass would take the dictionary route,
        # because only exact COCOeval instances use compact file inputs.
        evaluator = original(*args, **options)
        evaluators.append(evaluator)
        return evaluator

    # Ultralytics imports the names inside coco_evaluate, so patching the module suffices.
    monkeypatch.setattr(ultrafast_pycocotools, 'COCOeval', recording)
    return evaluators


def validator_for(task, tmp_path):
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator
    model, data, _, _ = TASKS[task]
    kind = {'detect': DetectionValidator, 'segment': SegmentationValidator, 'pose': PoseValidator}[task]
    return kind(args={'model': model, 'data': data, 'save_json': True, 'imgsz': 320, 'batch': 8,
                      'project': str(tmp_path), 'name': task, 'plots': False})


def reference_scores(gt_source, dt_source, iou_type, image_ids):
    """pycocotools on the same ground truth, detections and image list."""
    dt_source = copy.deepcopy(dt_source)
    for annotation in dt_source:  # loadRes derives these again.
        for key in ('id', 'area', 'iscrowd'):
            annotation.pop(key, None)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = ReferenceCOCO()
        gt.dataset = copy.deepcopy(gt_source)
        gt.createIndex()
        reference = ReferenceCOCOeval(gt, gt.loadRes(dt_source), iou_type)
        reference.params.imgIds = list(image_ids)
        reference.evaluate()
        reference.accumulate()
        reference.summarize()
    return reference


def assert_identical(evaluator, reference):
    for key in ('precision', 'recall', 'scores'):
        expected = np.ascontiguousarray(reference.eval[key], dtype=np.float64)
        actual = np.ascontiguousarray(evaluator.eval[key], dtype=np.float64)
        assert expected.shape == actual.shape, key
        assert expected.tobytes() == actual.tobytes(), f'{evaluator.params.iouType} {key}'
    assert np.asarray(reference.stats).tobytes() == np.asarray(evaluator.stats).tobytes()


def coco_ground_truth(validator):
    """COCO annotations from the validator's own dataset labels, keyed by file stem."""
    images, annotations = [], []
    for label in validator.dataloader.dataset.labels:
        h, w = (int(v) for v in label['shape'])
        image_id = int(Path(label['im_file']).stem)
        images.append({'id': image_id, 'height': h, 'width': w})
        segments = label.get('segments') or []
        keypoints = label.get('keypoints')
        for i, (cls, (cx, cy, bw, bh)) in enumerate(zip(label['cls'][:, 0], label['bboxes'])):
            box = [float((cx - bw / 2) * w), float((cy - bh / 2) * h), float(bw * w), float(bh * h)]
            annotation = {'id': len(annotations) + 1, 'image_id': image_id,
                          'category_id': int(validator.class_map[int(cls)]), 'bbox': box,
                          'area': box[2] * box[3], 'iscrowd': 0}
            if len(segments) > i:
                annotation['segmentation'] = [(np.asarray(segments[i]) * [w, h]).ravel().tolist()]
            if keypoints is not None and len(keypoints) > i:
                points = np.asarray(keypoints[i], dtype=np.float64).copy()
                points[:, 0] *= w
                points[:, 1] *= h
                annotation['keypoints'] = points.ravel().tolist()
                annotation['num_keypoints'] = int((points[:, 2] > 0).sum())
            annotations.append(annotation)
    return {'images': images, 'annotations': annotations,
            'categories': [{'id': int(c)} for c in validator.class_map]}


def test_detection_labels_route_matches_pycocotools(captured, tmp_path):
    validator = validator_for('detect', tmp_path)
    stats = validator()
    assert [e.params.iouType for e in captured] == ['bbox'], 'Ultralytics did not evaluate with ultrafast'
    evaluator = captured[0]
    assert len(evaluator.cocoDt.dataset['annotations']) > 0
    reference = reference_scores(evaluator.cocoGt.dataset, evaluator.cocoDt.dataset['annotations'],
                                 'bbox', evaluator.params.imgIds)
    assert_identical(evaluator, reference)
    for area in ('small', 'medium', 'large'):
        assert stats[f'metrics/mAP_{area}(B)'] == evaluator.stats_as_dict[f'AP_{area}']


@pytest.mark.parametrize('task', TASKS)
def test_coco_file_route_matches_pycocotools(task, captured, tmp_path):
    validator = validator_for(task, tmp_path)
    validator()
    captured.clear()
    gt_path = tmp_path / f'{task}-annotations.json'
    gt_source = coco_ground_truth(validator)
    gt_path.write_text(json.dumps(gt_source))
    predictions = validator.save_dir / 'predictions.json'
    # Score as an official COCO run: files on both sides, COCO metrics reported.
    validator.is_coco, validator.gdict, validator._coco_api = True, None, None
    _, _, iou_types, suffixes = TASKS[task]
    stats = validator.coco_evaluate({}, predictions, gt_path, iou_types, suffix=suffixes)
    assert [e.params.iouType for e in captured] == iou_types, 'Ultralytics did not evaluate with ultrafast'
    detections = json.loads(predictions.read_text())
    assert detections
    for evaluator, suffix in zip(captured, suffixes):
        assert evaluator.cocoGt._compact is not None and evaluator.cocoDt._compact is not None
        reference = reference_scores(gt_source, detections, evaluator.params.iouType, evaluator.params.imgIds)
        assert_identical(evaluator, reference)
        assert stats[f'metrics/mAP50-95({suffix[0]})'] == reference.stats[0]
        assert stats[f'metrics/mAP50({suffix[0]})'] == reference.stats[1]
