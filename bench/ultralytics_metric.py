"""Measure actual Ultralytics validation, or cold/cached saved-prediction replay.

Select source revisions through PYTHONPATH. --arrays is an untimed diagnostic:
profiling observes evaluator objects without replacing the evaluation algorithm.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('full', 'replay'), default='replay')
    p.add_argument('--task', choices=('detect', 'segment', 'pose'), default='detect')
    p.add_argument('--gt', required=True)
    p.add_argument('--pred')
    p.add_argument('--data')
    p.add_argument('--model')
    p.add_argument('--output', required=True)
    p.add_argument('--lvis', action='store_true')
    p.add_argument('--arrays', action='store_true')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    import numpy as np
    import psutil
    import torch
    import ultralytics
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.models.yolo.segment import SegmentationValidator

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(a.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_dir = output.parent / output.stem
    save_dir.mkdir(exist_ok=True)
    validator_type = {'detect': DetectionValidator, 'segment': SegmentationValidator, 'pose': PoseValidator}[a.task]
    args = {'task': a.task, 'save_json': True, 'plots': False, 'verbose': False}
    if a.mode == 'full':
        assert a.model and a.data and not a.arrays
        args.update(model=a.model, data=a.data, device='0', imgsz=640, batch=16, workers=2,
                    conf=0.001, iou=0.7, max_det=300, rect=True)
    v = validator_type(args=args, save_dir=save_dir)
    record = {
        'mode': a.mode, 'task': a.task, 'diagnostic_only': a.arrays,
        'python': sys.version, 'platform': platform.platform(),
        'cpu': platform.processor(), 'logical_cpus': psutil.cpu_count(),
        'physical_cpus': psutil.cpu_count(logical=False), 'ram_bytes': psutil.virtual_memory().total,
        'threads': a.threads, 'load_before': os.getloadavg(),
        'versions': {name: importlib.metadata.version(name) for name in
                     ('ultralytics', 'faster-coco-eval', 'ultrafast-pycocotools', 'numpy', 'torch', 'torchvision')},
        'source': str(Path(ultralytics.__file__).resolve()),
        'source_hashes': {str(path.relative_to(Path(ultralytics.__file__).parent)): sha256(path)
                          for path in (Path(ultralytics.__file__).parent / 'models/yolo').glob('*/val.py')},
        'gt_sha256': sha256(a.gt), 'calls': [],
    }
    record['native_hashes'] = {}
    for name in ('faster-coco-eval', 'ultrafast-pycocotools'):
        dist = importlib.metadata.distribution(name)
        record['native_hashes'][name] = {str(f): sha256(dist.locate_file(f)) for f in dist.files
                                         if str(f).endswith(('.so', '.pyd', '.dylib'))}
    if Path('/proc/cpuinfo').exists():
        record['cpu_models'] = sorted({line.split(':', 1)[1].strip() for line in
                                       Path('/proc/cpuinfo').read_text().splitlines()
                                       if line.startswith('model name')})
    record['evaluator_statistics'] = {}
    snapshots = {}
    def observe(frame, event, arg):
        if event == 'return' and frame.f_code.co_name == 'summarize':
            obj = frame.f_locals.get('self')
            if obj is not None and getattr(obj, 'eval', None) and hasattr(obj, 'params'):
                for name in ('precision', 'recall', 'scores'):
                    snapshots[f"call{len(record['calls'])}_{obj.params.iouType}_{name}"] = np.asarray(obj.eval[name]).copy()
                if hasattr(obj, 'stats_as_dict'):
                    record['evaluator_statistics'][f"call{len(record['calls'])}_{obj.params.iouType}"] = obj.stats_as_dict

    def measure(call):
        start_cpu, start = time.process_time(), time.perf_counter()
        stats = call()
        duration = time.perf_counter() - start
        cpu = time.process_time() - start_cpu
        assert 'fitness' in stats, stats
        return {'seconds': duration, 'cpu_seconds': cpu, 'metrics': dict(stats)}

    if a.mode == 'full':
        evaluate = v.eval_json
        def timed_evaluate(stats):
            record['pre_evaluator_metrics'] = dict(stats)
            entry = measure(lambda: evaluate(stats))
            record['evaluator'] = entry
            return entry['metrics']
        v.eval_json = timed_evaluate
        record['calls'].append(measure(lambda: v()))
        assert 'evaluator' in record
        a.pred = str(save_dir / 'predictions.json')
        record.update(model_sha256=sha256(a.model), data_sha256=sha256(a.data),
                      gpu=torch.cuda.get_device_name(), validator_args=vars(v.args),
                      speed_ms_per_image=v.speed)
    else:
        assert a.pred
        v.training, v.is_coco, v.is_lvis, v.gdict = False, not a.lvis, a.lvis, None
        v.jdict = json.loads(Path(a.pred).read_text())
        image_ids = [im['id'] for im in json.loads(Path(a.gt).read_text())['images']]
        v.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=[f'{i}.jpg' for i in image_ids]))
        types, suffixes = {'detect': (['bbox'], ['Box']), 'segment': (['bbox', 'segm'], ['Box', 'Mask']),
                           'pose': (['bbox', 'keypoints'], ['Box', 'Pose'])}[a.task]
        if a.arrays:
            sys.setprofile(observe)
        cached = None
        try:
            for i in range(a.repeats):
                entry = measure(lambda: v.coco_evaluate({}, a.pred, a.gt, types, suffixes))
                for suffix in suffixes:
                    assert f'metrics/mAP50-95({suffix[0]})' in entry['metrics'], entry
                entry['ground_truth'] = 'cold' if i == 0 else 'cached'
                if i:
                    assert v._coco_api is cached
                    assert entry['metrics'] == record['calls'][0]['metrics']
                cached = v._coco_api
                record['calls'].append(entry)
        finally:
            sys.setprofile(None)
        record.update(images=len(image_ids), predictions=len(v.jdict))
        if a.arrays:
            assert len(snapshots) == a.repeats * len(types) * 3, list(snapshots)
            np.savez_compressed(output.with_suffix('.npz'), **snapshots)
    record.update(pred_sha256=sha256(a.pred), prediction_file=a.pred,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                  (1024 * 1024 if sys.platform == 'darwin' else 1024),
                  load_after=os.getloadavg())
    output.write_text(json.dumps(record, indent=2, default=str) + '\n')
    print(json.dumps({'output': str(output), 'calls': record['calls']}))


if __name__ == '__main__':
    main()
