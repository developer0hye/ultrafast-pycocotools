"""Side effects of compact file inputs that decode or parse lazily.

Segmentation masks from files are decoded from the owned snapshot during
evaluation, and large result files are parsed in parallel chunks. These tests
check that this changes nothing observable: file changes after loading,
diagnostic APIs, materialized views and thread counts.
"""
import hashlib
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import ultrafast_pycocotools as ufc
from test_eval_parity import assert_bit_identical, run_reference

KEYPOINTS = 17


def _mask_inputs(tmp_path, images=12, per_image=8, seed=5):
    """Polygon ground truth and RLE detections, many with escaped counts."""
    from pycocotools import mask as ref_mask
    rng = np.random.default_rng(seed)
    h, w = 96, 128
    data = {'images': [{'id': i, 'height': h, 'width': w} for i in range(1, images + 1)],
            'annotations': [], 'categories': [{'id': 1}, {'id': 2}]}
    dets = []
    for i in range(images * per_image):
        image, category = 1 + i % images, 1 + i % 2
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        bw, bh = int(rng.integers(6, 40)), int(rng.integers(6, 40))
        data['annotations'].append({
            'id': i + 1, 'image_id': image, 'category_id': category, 'bbox': [x, y, bw, bh],
            'area': bw * bh, 'iscrowd': int(i % 29 == 0),
            'segmentation': [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]})
        for _ in range(2):
            m = np.zeros((h, w), np.uint8)
            dx, dy = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))
            m[max(y + dy, 0):y + dy + bh, max(x + dx, 0):x + dx + bw] = 1
            m[rng.integers(0, h, 25), rng.integers(0, w, 25)] = 1
            rle = ref_mask.encode(np.asfortranarray(m))
            dets.append({'image_id': image, 'category_id': category, 'score': float(rng.random()),
                         'bbox': [x + dx, y + dy, bw, bh],
                         'segmentation': {'size': [h, w], 'counts': rle['counts'].decode()}})
    assert sum('\\' in d['segmentation']['counts'] for d in dets) > 5
    gp, dp = tmp_path / 'gt.json', tmp_path / 'dt.json'
    gp.write_text(json.dumps(data))
    dp.write_text(json.dumps(dets))
    return gp, dp, data, dets


def _evaluate(gt, dt, iou_type='segm', **options):
    ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None, **options)
    ev.run()
    return ev


def _diagnostics(ev):
    matches = ev.matches(iou_thr=0.5)
    confusion = ev.confusion_matrix(iou_thr=0.5)
    return {name: np.asarray(values) for name, values in matches.items()}, confusion


def test_segmentation_snapshot_survives_file_changes_before_evaluation(tmp_path):
    # Masks are decoded from the snapshot only when evaluated.
    gp, dp, data, dets = _mask_inputs(tmp_path)
    reference = run_reference(gp, dp, 'segm')
    gt = ufc.COCO(gp, verbose=False)
    dt = gt.loadRes(dp)
    assert gt._compact is not None and dt._compact is not None
    gp.write_text('{}')
    dp.write_text('[' + ' ' * 1000 + ']')
    os.unlink(dp)
    first = _evaluate(gt, dt)
    assert_bit_identical(reference, first, 'segm after file change')
    second = _evaluate(gt, dt)  # A second engine decodes the same snapshot again.
    assert_bit_identical(reference, second, 'segm re-evaluated')
    assert dt.anns[1]['segmentation'] == dets[0]['segmentation']


def test_segmentation_diagnostics_match_materialized_inputs(tmp_path):
    gp, dp, data, dets = _mask_inputs(tmp_path)
    compact = _evaluate(ufc.COCO(gp, verbose=False), ufc.COCO(gp, verbose=False).loadRes(dp),
                        store_eval_imgs=True)
    gt = ufc.COCO(gp, verbose=False)
    dt = gt.loadRes(dp)
    gt.anns, dt.anns  # Materialize both before evaluating.
    plain = _evaluate(gt, dt, store_eval_imgs=True)
    assert gt._compact is None and dt._compact is None
    assert_bit_identical(plain, compact, 'segm compact vs materialized')
    (matches, confusion), (expected, expected_confusion) = _diagnostics(compact), _diagnostics(plain)
    assert matches.keys() == expected.keys()
    for name in matches:
        np.testing.assert_array_equal(matches[name], expected[name])
    assert len(confusion) == len(expected_confusion)
    for part, expected_part in zip(confusion, expected_confusion):
        np.testing.assert_array_equal(np.asarray(part), np.asarray(expected_part))
    for actual, reference in zip(compact.evalImgs, plain.evalImgs):
        assert (actual is None) == (reference is None)
        if actual is not None:
            for key in ('dtIds', 'gtIds', 'dtMatches', 'gtMatches', 'dtIgnore', 'gtIgnore'):
                np.testing.assert_array_equal(actual[key], reference[key])
    # Diagnostics re-run IoU from the snapshot on every call; results are stable.
    again, _ = _diagnostics(compact)
    for name in matches:
        np.testing.assert_array_equal(again[name], matches[name])


def test_views_materialized_after_evaluation_leave_the_engine_valid(tmp_path):
    gp, dp, data, dets = _mask_inputs(tmp_path)
    reference = run_reference(gp, dp, 'segm')
    gt = ufc.COCO(gp, verbose=False)
    dt = gt.loadRes(dp)
    ev = _evaluate(gt, dt)
    before, _ = _diagnostics(ev)
    dt.anns, gt.anns  # Releases the compact caches; the engine keeps its snapshot.
    assert gt._compact is None and dt._compact is None
    after, _ = _diagnostics(ev)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
    assert_bit_identical(reference, _evaluate(gt, dt), 'segm from materialized views')


RUNNER = textwrap.dedent('''
    import hashlib, json, sys
    import numpy as np
    import ultrafast_pycocotools as ufc
    gt_path, dt_path = sys.argv[1], sys.argv[2]
    out = {}
    for iou_type in ('bbox', 'segm', 'keypoints'):
        gt = ufc.COCO(gt_path, verbose=False)
        dt = gt.loadRes(dt_path)
        assert dt._compact is not None
        ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *a, **k: None)
        ev.run()
        digest = hashlib.sha256()
        for key in ('precision', 'recall', 'scores'):
            digest.update(np.ascontiguousarray(ev.eval[key], dtype=np.float64).tobytes())
        out[iou_type] = digest.hexdigest()
    out['anns'] = hashlib.sha256(json.dumps(dt.anns, sort_keys=True).encode()).hexdigest()
    print(json.dumps(out))
''')


def _large_mixed_results(tmp_path):
    """Above the parallel threshold. Only some chunks contain segmentation or
    keypoints, so chunk columns must be padded when joined."""
    rng = np.random.default_rng(21)
    h, w = 120, 160
    images = [{'id': i, 'height': h, 'width': w} for i in range(1, 201)]
    annotations = []
    for i in range(800):
        x, y = [float(v) for v in rng.uniform(0, 100, 2)]
        bw, bh = [float(v) for v in rng.uniform(5, 50, 2)]
        points = [[float(x + rng.uniform(0, bw)), float(y + rng.uniform(0, bh)), 2] for _ in range(KEYPOINTS)]
        annotations.append({'id': i + 1, 'image_id': 1 + i % 200, 'category_id': 1, 'iscrowd': 0,
                            'bbox': [x, y, bw, bh], 'area': bw * bh, 'num_keypoints': KEYPOINTS,
                            'keypoints': [v for p in points for v in p],
                            'segmentation': [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]})
    dets = []
    count = 40000
    for i in range(count):
        a = annotations[int(rng.integers(0, len(annotations)))]
        x, y, bw, bh = [v + float(rng.normal(0, 2)) for v in a['bbox']]
        det = {'image_id': a['image_id'], 'category_id': 1, 'bbox': [x, y, bw, bh],
               'score': round(float(rng.random()), 3), 'file_name': 'a}, {"x": 1}, {' if i % 3 else 'b.jpg'}
        if i >= count // 4:  # No segmentation in the first quarter (-> box masks).
            det['segmentation'] = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
        if i >= count * 3 // 4:  # Keypoints only in the last quarter.
            det['keypoints'] = [round(float(v + rng.normal(0, 2)), 3) if j % 3 < 2 else 1
                                for j, v in enumerate(a['keypoints'])]
        dets.append(det)
    gp, dp = tmp_path / 'gt.json', tmp_path / 'dt.json'
    gp.write_text(json.dumps({'images': images, 'annotations': annotations, 'categories': [{'id': 1}]}))
    text = json.dumps(dets)
    assert len(text) > 4 << 20
    dp.write_text(text)
    return gp, dp


def test_parallel_parsing_with_partial_geometry_does_not_depend_on_threads(tmp_path):
    gp, dp = _large_mixed_results(tmp_path)

    def run(threads):
        env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
        result = subprocess.run([sys.executable, '-c', RUNNER, str(gp), str(dp)],
                                capture_output=True, text=True, env=env, check=True)
        return json.loads(result.stdout.strip().splitlines()[-1])

    sequential = run(1)  # One thread keeps the sequential parser.
    for threads in (2, 5):
        assert run(threads) == sequential, threads
