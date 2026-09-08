"""Score official LVIS inputs with a selected backend, retaining complete arrays."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt', type=Path, default=Path('bench/data/lvis_gt_100.json'))
    p.add_argument('--pred', type=Path, default=Path('bench/data/lvis_dt_100.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', choices=['lvis', 'faster-coco-eval', 'ultrafast'], required=True)
    p.add_argument('--iou-type', choices=['bbox', 'segm'], default='bbox')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.backend == 'lvis':
        from lvis import LVIS, LVISResults, LVISEval
        np.float = float  # Official LVIS 0.5.3's removed NumPy spelling; same float64 type.
    elif a.backend == 'faster-coco-eval':
        from faster_coco_eval import COCO, COCOeval_faster as COCOeval
    else:
        from ultrafast_pycocotools import COCO, COCOeval
    names = ['AP','AP50','AP75','APs','APm','APl','APr','APc','APf','AR@300','ARs@300','ARm@300','ARl@300']
    start = time.perf_counter()
    predictions = json.loads(a.pred.read_text())
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="The 'warn' method is deprecated", category=DeprecationWarning)
        if a.backend == 'lvis':
            gt = LVIS(str(a.gt))
            dt = LVISResults(gt, predictions, max_dets=300)
            ev = LVISEval(gt, dt, a.iou_type)
        else:
            gt = COCO(str(a.gt))
            if a.backend == 'faster-coco-eval':
                # COCO-style result loaders do not necessarily apply LVIS's global image cap.
                groups = {}
                for det in predictions:
                    groups.setdefault(det['image_id'], []).append(det)
                predictions = [d for group in groups.values() for d in
                               (sorted(group, key=lambda x: x['score'], reverse=True)[:300] if len(group) > 300 else group)]
            dt = gt.loadRes(predictions)
            ev = COCOeval(gt, dt, a.iou_type, lvis_style=True, print_function=lambda *_: None)
            ev.params.maxDets = [300]
        load_seconds = time.perf_counter() - start
        score_start = time.perf_counter()
        ev.run() if a.backend != 'faster-coco-eval' else (ev.evaluate(), ev.accumulate(), ev.summarize())
        scoring_seconds = time.perf_counter() - score_start
    arrays = {key: np.ascontiguousarray(ev.eval[key] if a.backend == 'lvis' else ev.eval[key][..., 0])
              for key in ('precision','recall')}
    if a.backend == 'lvis':
        metrics = ev.results
    else:
        metrics = ev.stats_as_dict
        if a.backend == 'faster-coco-eval':
            other = ['AP_all','AP_50','AP_75','AP_small','AP_medium','AP_large','APr','APc','APf','AR_all','AR_small','AR_medium','AR_large']
            metrics = {k:metrics[v] for k,v in zip(names,other)}
    arrays['stats'] = np.array([metrics[k] for k in names], dtype=np.float64)
    np.savez_compressed(a.out/'arrays.npz', **arrays)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024 if sys.platform=='darwin' else 1024)
    except ImportError:
        rss = None
    result = {'backend':a.backend,'iou_type':a.iou_type,'load_seconds':load_seconds,
              'scoring_seconds':scoring_seconds,'total_seconds':load_seconds+scoring_seconds,
              'peak_rss_MiB':rss,'stats':dict(zip(names,arrays['stats'].tolist())),
              'gt_sha256':hashlib.sha256(a.gt.read_bytes()).hexdigest(),
              'pred_sha256':hashlib.sha256(a.pred.read_bytes()).hexdigest(),
              'array_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('backend','iou_type','total_seconds','peak_rss_MiB')}))


if __name__ == '__main__':
    main()
