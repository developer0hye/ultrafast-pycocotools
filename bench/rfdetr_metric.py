"""Replay identical CPU tensor batches through RF-DETR's real one-pass metric.

Preparation runs separately. Scoring includes metric construction, update,
single-process merge and compute, but excludes JSON/trace loading and inference.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
try:
    import resource
except ImportError:
    resource = None
import time

import numpy as np

from predict_coco_rfdetr import digest


def prepare(gt_path, pred_path, out):
    from collections import defaultdict
    gt = json.loads(gt_path.read_text())
    predictions = json.loads(pred_path.read_text())
    images = sorted(im['id'] for im in gt['images'])
    allowed = set(images)
    arrays = {'image_ids': np.asarray(images, dtype=np.int64)}
    for prefix, records in [('gt', gt['annotations']), ('dt', predictions)]:
        grouped = defaultdict(list)
        for r in records:
            if r['image_id'] not in allowed:
                raise ValueError('Unexpected prediction image ID')
            grouped[r['image_id']].append(r)
        offsets, boxes, labels, scores, crowds, areas = [0], [], [], [], [], []
        for image_id in images:
            for r in grouped[image_id]:
                x, y, w, h = r['bbox']
                boxes.append([x, y, x+w, y+h])
                labels.append(r['category_id'])
                if prefix == 'dt':
                    scores.append(r['score'])
                else:
                    crowds.append(r.get('iscrowd', 0))
                    areas.append(r.get('area', w*h))
            offsets.append(len(boxes))
        arrays[prefix+'_offsets'] = np.asarray(offsets, dtype=np.int64)
        arrays[prefix+'_boxes'] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        arrays[prefix+'_labels'] = np.asarray(labels, dtype=np.int64)
        if prefix == 'dt':
            arrays['dt_scores'] = np.asarray(scores, dtype=np.float32)
        else:
            arrays['gt_iscrowd'] = np.asarray(crowds, dtype=np.int64)
            arrays['gt_area'] = np.asarray(areas, dtype=np.float32)
    np.savez_compressed(out, **arrays)
    out.with_suffix('.metadata.json').write_text(json.dumps({
        'gt_sha256': digest(gt_path), 'pred_sha256': digest(pred_path),
        'trace_sha256': digest(out), 'images': len(images), 'detections': len(predictions),
        'box_format': 'xyxy', 'box_score_area_dtype': 'float32', 'labels_crowds_dtype': 'int64',
    }, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--gt', type=Path)
    p.add_argument('--pred', type=Path)
    p.add_argument('--backend', choices=['faster-coco-eval', 'ultrafast', 'pycocotools'])
    p.add_argument('--out', type=Path)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--max-dets', type=int, default=500)
    a = p.parse_args()
    if a.prepare:
        if a.trace.exists() or not a.gt or not a.pred:
            p.error('Preparation requires GT/prediction paths and a new trace path')
        prepare(a.gt, a.pred, a.trace)
        return
    if not a.backend or not a.out or a.batch < 1:
        p.error('Scoring requires backend, out and a positive batch size')
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    torch.set_num_interop_threads(1)
    with np.load(a.trace) as archive:
        trace = {k: torch.from_numpy(archive[k]) for k in archive.files}
    n = len(trace['image_ids'])
    predictions, targets = [], []
    for prefix, destination in [('dt', predictions), ('gt', targets)]:
        for i in range(n):
            start, end = trace[prefix+'_offsets'][i:i+2].tolist()
            fields = ['boxes', 'labels', 'scores'] if prefix == 'dt' else ['boxes', 'labels', 'iscrowd', 'area']
            destination.append({k: trace[prefix+'_'+k][start:end] for k in fields})
    times = {}
    with contextlib.redirect_stdout(io.StringIO()):
        started = time.perf_counter()
        metric = OnePassCocoMeanAveragePrecision(class_metrics=True, max_detection_thresholds=[1, 10, a.max_dets])
        if a.backend == 'ultrafast':
            from ultrafast_pycocotools.integrations.rfdetr import use_ultrafast
            use_ultrafast(metric)
        elif a.backend == 'pycocotools':
            from torchmetrics.detection.helpers import CocoBackend
            from pycocotools.cocoeval import COCOeval
            from rfdetr.evaluation.coco_eval import patched_pycocotools_summarize
            class ReferenceEval(COCOeval):
                def summarize(self):
                    patched_pycocotools_summarize(self, log_summary=False)
            class ReferenceBackend(CocoBackend):
                @property
                def cocoeval(self):
                    return ReferenceEval
            metric._coco_backend = ReferenceBackend('pycocotools')
            metric._validate_private_contract()
        times['construct_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        for i in range(0, n, a.batch):
            metric.update(predictions[i:i+a.batch], targets[i:i+a.batch])
        times['update_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        metric.merge_distributed_state()
        times['merge_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        result = metric.compute()
        times['compute_seconds'] = time.perf_counter() - started
    arrays = {k: v.detach().cpu().numpy() for k, v in result.items()}
    np.savez_compressed(a.out/'metrics.npz', **arrays)
    metadata = json.loads(a.trace.with_suffix('.metadata.json').read_text())
    report = {'backend': a.backend, 'images': n, 'trace': metadata,
              'max_dets': a.max_dets, 'batch': a.batch, 'class_metrics': True,
              'times': times, 'total_seconds': sum(times.values()),
              'peak_rss_MiB': (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                               (1024 * 1024 if sys.platform == 'darwin' else 1024)) if resource else None,
              'metrics': {k: v.tolist() for k, v in arrays.items()},
              'metric_hashes': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
              'metric_shapes': {k: list(v.shape) for k, v in arrays.items()},
              'versions': {k: importlib.metadata.version(k) for k in
                           ['torch', 'torchmetrics', 'rfdetr', 'faster-coco-eval', 'ultrafast-pycocotools', 'numpy', 'pycocotools']}}
    (a.out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(a.backend, round(report['total_seconds'], 3), round(report['peak_rss_MiB'], 1), report['metrics']['map'])


if __name__ == '__main__':
    main()
