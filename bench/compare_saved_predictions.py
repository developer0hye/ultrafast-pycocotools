"""Compare COCO scorers on identical saved predictions, without inference."""
import argparse
import contextlib
import hashlib
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


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', type=Path, required=True)
    p.add_argument('--pred', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', choices=['pycocotools', 'faster-coco-eval', 'ultrafast'], required=True)
    p.add_argument('--repeats', type=int, default=1)
    a = p.parse_args()
    if a.repeats < 1:
        p.error('repeats must be positive')
    if a.backend == 'pycocotools':
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    elif a.backend == 'faster-coco-eval':
        from faster_coco_eval import COCO, COCOeval_faster as COCOeval
    else:
        from ultrafast_pycocotools import COCO, COCOeval
    a.out.mkdir(parents=True, exist_ok=False)
    t = time.perf_counter()
    gt_dict = json.loads(a.gt.read_text())
    predictions = json.loads(a.pred.read_text())
    parse_seconds = time.perf_counter() - t
    image_ids = sorted(i['id'] for i in gt_dict['images'])
    assert {d['image_id'] for d in predictions} <= set(image_ids)
    runs = []
    previous = None
    for repeat in range(a.repeats):
        times = {}
        with contextlib.redirect_stdout(io.StringIO()):
            t = time.perf_counter()
            gt = COCO()
            gt.dataset = gt_dict
            gt.createIndex()
            times['gt_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            # Match an in-memory evaluator loadRes call, including defensive copies.
            dt = gt.loadRes([dict(d) for d in predictions])
            times['load_res_seconds'] = time.perf_counter() - t
            ev = COCOeval(gt, dt, 'bbox')
            ev.params.imgIds = image_ids
            for method in ('evaluate', 'accumulate', 'summarize'):
                t = time.perf_counter()
                getattr(ev, method)()
                times[method + '_seconds'] = time.perf_counter() - t
        arrays = {k: np.ascontiguousarray(ev.eval[k], dtype=np.float64)
                  for k in ('precision', 'recall', 'scores')}
        arrays['stats'] = np.ascontiguousarray(ev.stats, dtype=np.float64)
        if previous is not None:
            assert all(arrays[k].tobytes() == previous[k].tobytes() for k in arrays)
        else:
            np.savez_compressed(a.out / 'arrays.npz', **arrays)
            previous = arrays
        times['total_scoring_seconds'] = sum(times.values())
        runs.append(times)
        del ev, gt, dt
    result = {'backend': a.backend, 'images': len(image_ids), 'detections': len(predictions),
              'gt_sha256': digest(a.gt), 'pred_sha256': digest(a.pred),
              'input_json_parse_seconds': parse_seconds, 'runs': runs,
              'stats': previous['stats'].tolist(),
              'array_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in previous.items()},
              'peak_rss_MiB': (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                               (1024 * 1024 if sys.platform == 'darwin' else 1024))
                               if resource else None,
              'rayon_threads': os.environ.get('RAYON_NUM_THREADS'),
              'omp_threads': os.environ.get('OMP_NUM_THREADS')}
    from importlib.metadata import version
    distribution = {'ultrafast': 'ultrafast-pycocotools'}.get(a.backend, a.backend)
    result['package_version'] = version(distribution)
    result['result_loading'] = 'package default'
    (a.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'backend': a.backend, 'images': len(image_ids),
                      'detections': len(predictions), 'AP': result['stats'][0], 'runs': runs}))


if __name__ == '__main__':
    main()
