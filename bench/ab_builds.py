"""Alternate two builds of ultrafast-pycocotools in fresh processes.

Each round runs every task and thread count once per build, swapping the
build order (and the thread order) every round, so host drift affects both
builds alike. Usage::

    python bench/ab_builds.py --inputs INPUTS --base VENV_A/bin/python --new VENV_B/bin/python --out OUT
"""
import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
TASKS = {'bbox': ('instances_val2017', 'detect'), 'segm': ('instances_val2017', 'segment'),
         'keypoints': ('person_keypoints_val2017', 'pose')}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs', type=Path, required=True)
parser.add_argument('--base', required=True, help='Python of the baseline build')
parser.add_argument('--new', required=True, help='Python of the build under test')
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--rounds', type=int, default=6)
args = parser.parse_args()
I, OUT = args.inputs, args.out
OUT.mkdir(parents=True, exist_ok=False)
PYTHON = {'base': args.base, 'new': args.new}
rows = []
psutil.cpu_percent()
for r in range(args.rounds):
    builds = ('base', 'new') if r % 2 == 0 else ('new', 'base')
    for threads in ((2, 1) if r % 2 == 0 else (1, 2)):
        for task, (gt, dt) in TASKS.items():
            for build in builds:
                env = dict(os.environ, RAYON_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                target = OUT / f'{task}-t{threads}-{build}-{r}.json'
                cpu_before = psutil.cpu_percent()
                subprocess.run([PYTHON[build], str(REPO / 'bench/run_impl.py'),
                                '--impl', 'ufcoco', '--threads', str(threads), '--iou-type', task,
                                '--gt', str(I / f'{gt}.json'), '--dt', str(I / f'{dt}-predictions.json'),
                                '--file-inputs', '--json-out', str(target)], env=env, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                res = json.loads(target.read_text())
                rows.append(dict(task=task, threads=threads, build=build, round=r, wall=res['wall_total'],
                                 cpu=res['cpu_total'], rss=res['peak_rss_mb'], digests=res['digests'],
                                 host_cpu=psutil.cpu_percent()))
(OUT / 'rows.json').write_text(json.dumps(rows, indent=1))
print('| Task | Threads | Wall s (main → branch) | CPU s | Peak RSS MB |\n| --- | ---: | ---: | ---: | ---: |')
for task in TASKS:
    for threads in (2, 1):
        m = {b: [x for x in rows if x['task'] == task and x['threads'] == threads and x['build'] == b] for b in ('base', 'new')}
        med = lambda b, k: statistics.median(x[k] for x in m[b])
        same = len({json.dumps(x['digests'], sort_keys=True) for x in m['base'] + m['new']}) == 1
        print(f"| {task} | {threads} | {med('base','wall'):.3f} → {med('new','wall'):.3f} ({100*(med('new','wall')/med('base','wall')-1):+.1f}%) "
              f"| {med('base','cpu'):.3f} → {med('new','cpu'):.3f} | {med('base','rss'):.1f} → {med('new','rss'):.1f} "
              f"({100*(med('new','rss')/med('base','rss')-1):+.1f}%) | digests equal: {same}")
print('median host cpu', statistics.median(x['host_cpu'] for x in rows))
