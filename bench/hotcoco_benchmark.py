"""Compare released COCO evaluators in balanced fresh processes.

Inputs use the filenames in the published Ultralytics evidence archive.
Oracle/array-export runs are separate from timed repetitions. No run is
discarded based on its speed, memory use, background load, or agreement.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
TASKS = {
    'bbox': ('instances_val2017.json', 'detect-predictions.json'),
    'segm': ('instances_val2017.json', 'segment-predictions.json'),
    'keypoints': ('person_keypoints_val2017.json', 'pose-predictions.json'),
}
PACKAGES = ('hotcoco', 'ultrafast-pycocotools', 'pycocotools', 'numpy', 'psutil')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def snapshot():
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    return dict(utc=datetime.now(timezone.utc).isoformat(),
                cpu_percent=psutil.cpu_percent(), available_ram=memory.available,
                swap_used=swap.used, swap_in=swap.sin, swap_out=swap.sout)


def spread(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values))


def compare_arrays(reference, candidate):
    with np.load(reference) as a, np.load(candidate) as b:
        results = {}
        for key in ('precision', 'recall', 'scores'):
            x, y = a[key], b[key]
            same_shape = x.shape == y.shape
            results[key] = dict(reference_shape=list(x.shape), candidate_shape=list(y.shape),
                                byte_identical=same_shape and x.tobytes() == y.tobytes(),
                                unequal_values=int(np.count_nonzero(x != y)) if same_shape else None,
                                within_1e_12=same_shape and bool(np.allclose(x, y, rtol=0, atol=1e-12, equal_nan=True)),
                                max_abs_diff=float(np.max(np.abs(x-y))) if same_shape else None)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-description', default='Released wheels',
                        help='Provenance label when comparing an unreleased candidate build')
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--tasks', choices=TASKS, nargs='+', default=list(TASKS))
    parser.add_argument('--modes', choices=['files', 'list'], nargs='+', default=['files', 'list'])
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or any(t < 1 for t in args.threads):
        parser.error('use a positive even number of rounds and positive thread counts')
    inputs, out = args.inputs.resolve(), args.out.resolve()
    files = {name: inputs / name for task in args.tasks for name in TASKS[task]}
    if not all(path.is_file() for path in files.values()):
        parser.error('missing input file: ' + ', '.join(str(p) for p in files.values() if not p.is_file()))
    out.mkdir(parents=True, exist_ok=False)
    cpu = platform.processor()
    if sys.platform == 'darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    elif Path('/proc/cpuinfo').is_file():
        cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith('model name')), cpu)
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  cpu=cpu, physical_cores=psutil.cpu_count(logical=False), logical_cpus=psutil.cpu_count(),
                  ram_bytes=psutil.virtual_memory().total, platform=platform.platform(), python=sys.version,
                  packages={p: importlib.metadata.version(p) for p in PACKAGES},
                  input_sha256={n: digest(p) for n, p in files.items()},
                  script_sha256={p.name: digest(p) for p in (Path(__file__), HERE/'run_impl.py')},
                  threads=args.threads, rounds=args.rounds, modes=args.modes, tasks=args.tasks,
                  method=args.build_description + '; 1 excluded array-export warmup per backend/task/mode/thread; '
                         'alternating backend order and reversed thread order each round; fresh process each run. '
                         'Wall/CPU include GT and DT loading, construction, evaluation and summary; imports and '
                         'post-timing hashes excluded. RSS captured before output conversion/hash/export. '
                         'List route includes Python JSON parsing of DT; GT is a file in both routes. '
                         'No timing-based exclusions; host telemetry includes child startup and verification.',
                  runs=[], parity={}, summary={})
    psutil.cpu_percent()
    time.sleep(1)
    result['baseline_load'] = snapshot()

    def save():
        (out/'results.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(task, mode, impl, threads, label, arrays=False):
        name = f'{task}-{mode}-t{threads}-{impl}-{label}'
        target = out/(name+'.json')
        command = [sys.executable, str(HERE/'run_impl.py'), '--impl', impl, '--threads', str(threads),
                   '--iou-type', task, '--gt', str(inputs/TASKS[task][0]),
                   '--dt', str(inputs/TASKS[task][1]), '--json-out', str(target)]
        if mode == 'files':
            command.append('--file-inputs')
        if arrays:
            command.extend(['--arrays-out', str(out/(name+'.npz'))])
        environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                           RAYON_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
        samples = [snapshot()]
        print('START', name, flush=True)
        with (out/(name+'.log')).open('w') as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    samples.append(snapshot())
        samples.append(snapshot())
        entry = dict(name=name, task=task, mode=mode, impl=impl, threads=threads,
                     label=label, diagnostic=arrays, command=command,
                     returncode=process.returncode, telemetry=samples)
        if process.returncode == 0:
            entry['result'] = json.loads(target.read_text())
            print('PASS', name, entry['result']['wall_total'], entry['result']['peak_rss_mb'], flush=True)
        else:
            print('FAILED', name, 'see', out/(name+'.log'), flush=True)
        result['runs'].append(entry)
        save()
        return entry

    for task in args.tasks:
        for mode in args.modes:
            oracle = run(task, mode, 'pycocotools', 1, 'oracle', arrays=True)
            if oracle['returncode']:
                raise RuntimeError('Oracle failed; inspect saved log before benchmarking')
            checks = result['parity'].setdefault(task+'-'+mode, {})
            warmups = {}
            for threads in args.threads:
                for impl in ('hotcoco', 'ufcoco'):
                    warmup = run(task, mode, impl, threads, 'warmup', arrays=True)
                    warmups[impl, threads] = warmup
                    if warmup['returncode'] == 0:
                        checks[f'{impl}-t{threads}'] = {
                            'arrays': compare_arrays(out/(oracle['name']+'.npz'), out/(warmup['name']+'.npz')),
                            'stats_max_abs_diff': float(np.max(np.abs(np.array(oracle['result']['stats'])-
                                                                      warmup['result']['stats']))),
                            'same_params': oracle['result']['params'] == warmup['result']['params'],
                        }
                    else:
                        checks[f'{impl}-t{threads}'] = {'failed': True}
                    save()
            for iteration in range(args.rounds):
                thread_order = args.threads if iteration % 2 == 0 else list(reversed(args.threads))
                order = ('hotcoco', 'ufcoco') if iteration % 2 == 0 else ('ufcoco', 'hotcoco')
                for threads in thread_order:
                    for impl in order:
                        if warmups[impl, threads]['returncode']:
                            continue
                        entry = run(task, mode, impl, threads, f'round{iteration}')
                        if entry['returncode']:
                            raise RuntimeError('Timed process failed; preserve all results and inspect log')
                        entry['stable_digests'] = entry['result']['digests'] == warmups[impl, threads]['result']['digests']
                        if not entry['stable_digests']:
                            save()
                            raise RuntimeError('Within-backend nondeterminism; inspect output hashes')
            for threads in args.threads:
                for impl in ('hotcoco', 'ufcoco'):
                    values = [r['result'] for r in result['runs'] if not r['diagnostic'] and
                              (r['task'],r['mode'],r['threads'],r['impl']) == (task,mode,threads,impl)]
                    if values:
                        result['summary'][f'{task}-{mode}-t{threads}-{impl}'] = {
                            key: spread([r[key] for r in values])
                            for key in ('wall_total', 'cpu_total', 'eval_total', 'peak_rss_mb')}
            save()
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    main()
