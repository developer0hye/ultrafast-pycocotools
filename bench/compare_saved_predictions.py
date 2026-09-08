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
    p.add_argument('--input-mode', choices=['in-memory', 'files'], default='in-memory',
                   help='files includes JSON parsing in scoring time and avoids preloading Python inputs')
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
    if a.input_mode == 'in-memory':
        t = time.perf_counter()
        gt_dict = json.loads(a.gt.read_text())
        predictions = json.loads(a.pred.read_text())
        parse_seconds = time.perf_counter() - t
        image_ids = sorted(i['id'] for i in gt_dict['images'])
        prediction_count = len(predictions)
        assert {d['image_id'] for d in predictions} <= set(image_ids)
    else:
        parse_seconds = None  # File parsing is included in the load timings below.
    storage = None
    native_timings = []
    runs = []
    previous = None
    for repeat in range(a.repeats):
        times = {}
        cpu_times = {}
        with contextlib.redirect_stdout(io.StringIO()):
            t = time.perf_counter()
            c = time.process_time()
            if a.input_mode == 'files':
                gt = COCO(str(a.gt))
            else:
                gt = COCO()
                gt.dataset = gt_dict
                gt.createIndex()
            times['gt_seconds'] = time.perf_counter() - t
            cpu_times['gt_seconds'] = time.process_time() - c
            t = time.perf_counter()
            c = time.process_time()
            # Match an in-memory evaluator loadRes call, including defensive copies.
            dt = gt.loadRes(str(a.pred) if a.input_mode == 'files' else [dict(d) for d in predictions])
            times['load_res_seconds'] = time.perf_counter() - t
            cpu_times['load_res_seconds'] = time.process_time() - c
            if a.input_mode == 'files':
                image_ids = sorted(gt.getImgIds())
                compact = getattr(dt, '_compact', None)
                prediction_count = compact.annotation_count if compact is not None else len(dt.anns)
                storage = {name: ({'snapshot_bytes': handle._compact.snapshot_bytes,
                                  'column_bytes': handle._compact.column_bytes}
                                 if getattr(handle, '_compact', None) is not None else None)
                           for name, handle in [('gt', gt), ('dt', dt)]}
                del compact
            ev = COCOeval(gt, dt, 'bbox')
            ev.params.imgIds = image_ids
            for method in ('evaluate', 'accumulate', 'summarize'):
                t = time.perf_counter()
                c = time.process_time()
                getattr(ev, method)()
                times[method + '_seconds'] = time.perf_counter() - t
                cpu_times[method + '_seconds'] = time.process_time() - c
        engine = getattr(ev, '_engine', None)
        if engine is not None:
            native_timings.append(engine.timings())
        del engine
        arrays = {k: np.ascontiguousarray(ev.eval[k], dtype=np.float64)
                  for k in ('precision', 'recall', 'scores')}
        arrays['stats'] = np.ascontiguousarray(ev.stats, dtype=np.float64)
        if previous is not None:
            assert all(arrays[k].tobytes() == previous[k].tobytes() for k in arrays)
        else:
            np.savez_compressed(a.out / 'arrays.npz', **arrays)
            previous = arrays
        times['total_scoring_seconds'] = sum(times.values())
        cpu_times['total_scoring_seconds'] = sum(cpu_times.values())
        times['cpu_seconds'] = cpu_times
        runs.append(times)
        del ev, gt, dt
    result = {'backend': a.backend, 'images': len(image_ids), 'detections': prediction_count,
              'gt_sha256': digest(a.gt), 'pred_sha256': digest(a.pred),
              'input_json_parse_seconds': parse_seconds, 'runs': runs,
              'input_mode': a.input_mode, 'compact_storage': storage,
              'native_timings': native_timings,
              'output_array_bytes': sum(v.nbytes for v in previous.values()),
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
                      'detections': prediction_count, 'AP': result['stats'][0], 'runs': runs}))


if __name__ == '__main__':
    main()
