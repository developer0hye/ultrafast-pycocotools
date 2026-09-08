"""Balanced native-build comparison on frozen actual Ultralytics predictions."""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil

root = Path('/tmp/ultralytics-pr26101')
out = root / 'results/nonbbox-direct-g'
out.mkdir(exist_ok=True)
interpreters = {
    'baseline': '/tmp/ultrafast-nonbbox-opt-20260909/.venv/bin/python',
    'candidate': '/tmp/ultrafast-nonbbox-direct-20260909/.venv/bin/python',
}
env = dict(os.environ, PYTHONPATH=str(root / 'replacement'), RAYON_NUM_THREADS='2',
           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', YOLO_AUTOINSTALL='false')
manifest = []
for task in ('segment',):
    gt = root / 'data/coco/annotations' / (
        'instances_val2017.json' if task == 'segment' else 'person_keypoints_val2017.json')
    pred = root / f'results/formal/{task}-full-0-reference/predictions.json'
    for iteration in range(7):
        for variant in (('baseline', 'candidate') if iteration % 2 == 0 else ('candidate', 'baseline')):
            diagnostic = iteration == 6
            name = f'{task}-{iteration}-{variant}'
            output = out / f'{name}.json'
            command = [interpreters[variant], str(root / 'ultralytics_metric.py'), '--task', task,
                       '--gt', str(gt), '--pred', str(pred), '--output', str(output)]
            if diagnostic:
                command.append('--arrays')
            print('START', name, flush=True)
            samples = []
            psutil.cpu_percent()
            start = time.perf_counter()
            with output.with_suffix('.log').open('w') as log:
                worker = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                process = psutil.Process(worker.pid)
                done = threading.Event()
                def wait():
                    worker.wait()
                    done.set()
                waiter = threading.Thread(target=wait)
                waiter.start()
                while not done.is_set():
                    try:
                        samples.append({'elapsed': time.perf_counter() - start,
                                        'host_cpu_percent': psutil.cpu_percent(),
                                        'available_ram': psutil.virtual_memory().available,
                                        'rss': process.memory_info().rss})
                    except psutil.NoSuchProcess:
                        pass
                    done.wait(.5)
                waiter.join()
            manifest.append(dict(name=name, command=command, exit_code=worker.returncode,
                                 wall_seconds=time.perf_counter()-start, samples=samples))
            (out / 'execution.json').write_text(json.dumps(manifest, indent=2))
            assert worker.returncode == 0, name
            record = json.loads(output.read_text())
            formal = json.loads((root / f'results/formal/{task}-replay-0-replacement.json').read_text())
            assert record['pred_sha256'] == formal['pred_sha256']
            assert record['gt_sha256'] == formal['gt_sha256']
            assert all(call['metrics'] == formal['calls'][0]['metrics'] for call in record['calls'])
            if diagnostic:
                actual = np.load(output.with_suffix('.npz'))
                expected = np.load(root / f'results/formal/{task}-diagnostic-0-replacement.npz')
                assert actual.files == expected.files
                for key in actual.files:
                    np.testing.assert_array_equal(actual[key], expected[key], err_msg=name+'/'+key)
            print('PASS', name, flush=True)
