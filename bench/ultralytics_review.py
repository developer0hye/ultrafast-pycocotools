"""Run alternating fresh-process validation and evaluator replay on fixed COCO data."""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--full-rounds', type=int, default=4)
    p.add_argument('--replay-rounds', type=int, default=6)
    p.add_argument('--tasks', nargs='+', default=['detect', 'segment', 'pose', 'lvis', 'lvis_segment'])
    a = p.parse_args()
    root = a.root.resolve()
    results = root / 'results/formal'
    results.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, YOLO_AUTOINSTALL='false', RAYON_NUM_THREADS='2', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    manifest = {'execution': [], 'full_rounds': a.full_rounds, 'replay_rounds': a.replay_rounds}
    for task in a.tasks:
        pred = str(root / f'results/warmup-{task}-reference/predictions.json') if a.full_rounds == 0 else None
        model = {'detect': 'yolo26n.pt', 'segment': 'yolo26n-seg.pt', 'pose': 'yolo26n-pose.pt'}.get(task)
        gt = root / ('data/lvis_gt_93_annotated.json' if task == 'lvis_segment'
                     else 'data/lvis_gt_100.json' if task == 'lvis' else 'data/coco/annotations/' +
                     ('person_keypoints_val2017.json' if task == 'pose' else 'instances_val2017.json'))
        for mode in ('full', 'replay', 'diagnostic'):
            if task.startswith('lvis') and mode == 'full':
                continue
            rounds = a.full_rounds if mode == 'full' else a.replay_rounds if mode == 'replay' else 1
            for iteration in range(rounds):
                for revision in (('reference', 'replacement') if iteration % 2 == 0 else ('replacement', 'reference')):
                    name = f'{task}-{mode}-{iteration}-{revision}'
                    output = results / f'{name}.json'
                    command = [sys.executable, str(root / 'ultralytics_metric.py'), '--task',
                               'segment' if task == 'lvis_segment' else 'detect' if task == 'lvis' else task,
                               '--gt', str(gt), '--output', str(output)]
                    if mode == 'full':
                        command += ['--mode', 'full', '--model', str(root / model), '--data',
                                    str(root / ('data/coco-pose.yaml' if task == 'pose' else 'data/coco.yaml'))]
                    else:
                        command += ['--pred', str(root / 'data/lvis_yolo26n_seg_93_annotated.json') if task == 'lvis_segment'
                                    else str(root / 'data/lvis_dt_100.json') if task == 'lvis' else pred]
                        if task.startswith('lvis'): command += ['--lvis']
                        if mode == 'diagnostic': command += ['--arrays']
                    env['PYTHONPATH'] = str(root / revision)
                    print('START', name, flush=True)
                    samples = []
                    psutil.cpu_percent()
                    start = time.perf_counter()
                    with output.with_suffix('.log').open('w') as log:
                        worker = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                        process = psutil.Process(worker.pid)
                        completed, ended = threading.Event(), []
                        def wait_for_exit(worker=worker, ended=ended, completed=completed):
                            worker.wait()
                            ended.append(time.perf_counter())
                            completed.set()
                        waiter = threading.Thread(target=wait_for_exit)
                        waiter.start()
                        while not completed.is_set():
                            try:
                                tree = [process] + process.children(recursive=True)
                                samples.append({'elapsed': time.perf_counter() - start,
                                                'host_cpu_percent': psutil.cpu_percent(),
                                                'available_ram': psutil.virtual_memory().available,
                                                'tree_rss': sum(x.memory_info().rss for x in tree if x.is_running())})
                            except psutil.NoSuchProcess:
                                pass
                            completed.wait(0.5)
                        waiter.join()
                    entry = {'name': name, 'command': command, 'revision': revision, 'exit_code': worker.returncode,
                             'fresh_process_seconds': ended[0] - start, 'samples': samples}
                    manifest['execution'].append(entry)
                    (results / 'execution.json').write_text(json.dumps(manifest, indent=2) + '\n')
                    if worker.returncode: raise RuntimeError(f'{name} failed: see {output.with_suffix(".log")}')
                    data = json.loads(output.read_text())
                    if mode == 'full':
                        if pred is None:
                            pred = str(root / data['prediction_file'])
                            expected_hash = data['pred_sha256']
                        if data['pred_sha256'] != expected_hash:
                            raise AssertionError(f'Predictions differ in {name}')
                    print('DONE', name, round(entry['fresh_process_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
