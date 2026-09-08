"""Same-compiler native evaluator comparison, without the Ultralytics caller."""
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

root = Path(__file__).resolve().parents[3]
inputs = Path('/Users/yhkwon/Documents/ultrafast-pycocotools/bench/out/ultralytics-pr26101-evidence-20260909/inputs')
out = root / 'bench/out/mask-cap-lvis/m2-final-v019'
out.mkdir(exist_ok=True)
interpreters = {
    'baseline': str(root / '.venv/bin/python'),
    'candidate': str(root / '.venv/bin/python'),
}
sources = {'baseline': '/Users/yhkwon/Documents/ultrafast-pycocotools-nonbbox-memory/python', 'candidate': str(root / 'python')}
records = []
for task, annotation, prediction, iou, protocol in (
        ('segm', 'instances_val2017.json', 'segment-predictions.json', 'segm', None),
        ('lvis_official', 'lvis_gt_100.json', 'lvis_dt_100.json', 'bbox', 'official'),
        ('lvis_mask', 'lvis_gt_93_annotated.json', 'lvis_yolo26n_seg_93_annotated.json', 'segm', 'coco')):
    reference = None
    for iteration in range(7):
        for variant in (('baseline', 'candidate') if iteration % 2 == 0 else ('candidate', 'baseline')):
            name = f'{task}-{iteration}-{variant}'
            result = out / f'{name}.json'
            cmd = [interpreters[variant], str(root/'bench/run_impl.py'), '--impl', 'ufcoco',
                   '--file-inputs', '--threads', '2', '--iou-type', iou,
                   '--gt', str(inputs/annotation), '--dt', str(inputs/prediction), '--json-out', str(result)]
            if protocol:
                cmd += ['--lvis-protocol', protocol]
            if iteration == 6:
                cmd += ['--arrays-out', str(result.with_suffix('.npz'))]
            print('START', name, flush=True)
            samples = []
            start = time.perf_counter()
            psutil.cpu_percent()
            with result.with_suffix('.log').open('w') as log:
                proc = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, OPENBLAS_NUM_THREADS='2', PYTHONPATH=sources[variant]))
                while proc.poll() is None:
                    samples.append(dict(seconds=time.perf_counter()-start,
                                        host_cpu_percent=psutil.cpu_percent(),
                                        available_ram=psutil.virtual_memory().available))
                    try:
                        proc.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        pass
            assert proc.returncode == 0, name
            data = json.loads(result.read_text())
            if reference is None:
                reference = data
            assert data['digests'] == reference['digests'], name
            assert data['stats'] == reference['stats'], name
            records.append(dict(name=name, command=cmd, samples=samples))
            (out/'execution.json').write_text(json.dumps(records, indent=2))
            print('PASS', name, flush=True)
