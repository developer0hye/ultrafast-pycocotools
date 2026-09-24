"""Per-phase wall time, CPU time and memory for the public file-input route.

``run_impl.py`` times the public calls as a user makes them; ``profile_engine.py``
reads the engine's internal timers but bypasses the public ``evaluate()``.
This script does both in one fresh process per run: it times each public call
and splits ``evaluate()`` into its Python prelude, native extraction
(``Evaluator(...)``), the engine run and the Python epilogue. After every phase
it records current RSS, the process high-water mark (``VmHWM``), and, when the
extension is built with ``--features alloc-stats``, the Rust allocator's live
bytes, per-phase peak and allocation count.

Usage::

    python bench/profile_tasks.py --inputs DIR --task bbox --runs 5 --threads 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

TASKS = {
    'bbox': ('instances_val2017.json', 'detect-predictions.json'),
    'segm': ('instances_val2017.json', 'segment-predictions.json'),
    'keypoints': ('person_keypoints_val2017.json', 'pose-predictions.json'),
}


def status_kb(field: str) -> int:
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(field + ':'):
            return int(line.split()[1])
    raise KeyError(field)


def child(gt_path: str, dt_path: str, iou_type: str) -> dict:
    import ultrafast_pycocotools as ufc
    from ultrafast_pycocotools import _ufcoco, cocoeval

    phases: list[dict] = []
    last = {'wall': time.perf_counter(), 'cpu': time.process_time()}
    _ufcoco.reset_alloc_peak()

    def mark(name: str) -> None:
        wall, cpu = time.perf_counter(), time.process_time()
        stats = _ufcoco.alloc_stats()
        phases.append(dict(phase=name, wall=wall - last['wall'], cpu=cpu - last['cpu'],
                           rss_mb=status_kb('VmRSS') / 1000, hwm_mb=status_kb('VmHWM') / 1000,
                           rust_live_mb=stats['live_bytes'] / 1e6 if stats['enabled'] else None,
                           rust_peak_mb=stats['peak_bytes'] / 1e6 if stats['enabled'] else None,
                           rust_allocs=stats['allocations'] if stats['enabled'] else None))
        _ufcoco.reset_alloc_peak()
        last['wall'], last['cpu'] = time.perf_counter(), time.process_time()

    native = _ufcoco.Evaluator
    engines = []

    class Timed:
        def __init__(self, *args):
            mark('evaluate:prelude')
            self._engine = native(*args)
            engines.append(self._engine)
            mark('evaluate:extract')

        def __getattr__(self, name):
            return getattr(self._engine, name)

        def run(self, *args):
            result = self._engine.run(*args)
            mark('evaluate:engine')
            return result

    class Shim:
        def __getattr__(self, name):
            return Timed if name == 'Evaluator' else getattr(_ufcoco, name)

    cocoeval._ufcoco = Shim()
    mark('import')
    gt = ufc.COCO(gt_path, verbose=False)
    mark('load_gt')
    dt = gt.loadRes(dt_path)
    mark('load_dt')
    ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None)
    mark('construct')
    ev.evaluate()
    mark('evaluate:epilogue')
    ev.accumulate()
    mark('accumulate')
    ev.summarize()
    mark('summarize')
    return dict(phases=phases, engine_timings=dict(engines[0].timings()), stats=[float(x) for x in ev.stats])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--child', nargs=2, metavar=('GT', 'DT'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child(*args.child, args.task)))
        return
    if args.inputs is None:
        parser.error('--inputs is required')
    gt, dt = (str(args.inputs / name) for name in TASKS[args.task])
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    runs = [json.loads(subprocess.check_output(
        [sys.executable, __file__, '--task', args.task, '--child', gt, dt], env=env))
        for _ in range(args.runs)]
    names = [p['phase'] for p in runs[0]['phases']]
    print(f'{args.task}: median of {args.runs} fresh processes, {args.threads} threads')
    print(f'{"phase":20s} {"wall s":>8s} {"cpu s":>8s} {"RSS MB":>8s} {"HWM MB":>8s} '
          f'{"dHWM":>7s} {"rust live":>9s} {"rust peak":>9s} {"allocs":>9s}')
    previous_hwm = None
    for i, name in enumerate(names):
        med = {k: statistics.median(r['phases'][i][k] for r in runs)
               for k in ('wall', 'cpu', 'rss_mb', 'hwm_mb')}
        rust = runs[0]['phases'][i]
        delta = '' if previous_hwm is None else f'{med["hwm_mb"] - previous_hwm:+7.1f}'
        previous_hwm = med['hwm_mb']
        extra = ('' if rust['rust_live_mb'] is None else
                 f' {rust["rust_live_mb"]:9.1f} {rust["rust_peak_mb"]:9.1f} {rust["rust_allocs"]:9d}')
        print(f'{name:20s} {med["wall"]:8.3f} {med["cpu"]:8.3f} {med["rss_mb"]:8.1f} '
              f'{med["hwm_mb"]:8.1f} {delta:>7s}{extra}')
    total = statistics.median(sum(p['wall'] for p in r['phases'] if p['phase'] != 'import') for r in runs)
    print(f'{"total (no import)":20s} {total:8.3f}')
    print('engine timings (run 1):', {k: round(v, 4) for k, v in runs[0]['engine_timings'].items()})
    if args.json_out:
        args.json_out.write_text(json.dumps(runs, indent=1))


if __name__ == '__main__':
    main()
