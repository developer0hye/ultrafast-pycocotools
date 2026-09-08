"""Record fresh-process, balanced comparisons and system load on a busy host.

Requires psutil in addition to the three evaluator packages. Each six-round
block uses all backend order permutations; thread configurations alternate
order between rounds. No samples are discarded based on their timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import psutil

from reproduce import compare, compare_numerically, digest


HERE = Path(__file__).resolve().parent
BACKENDS = ('pycocotools', 'faster-coco-eval', 'ultrafast')


def snapshot():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        'utc': datetime.now(timezone.utc).isoformat(),
        'cpu_percent': psutil.cpu_percent(),
        'load_average': list(os.getloadavg()),
        'available_memory_bytes': memory.available,
        'memory_percent': memory.percent,
        'swap_used_bytes': swap.used,
        'swap_in_bytes': swap.sin,
        'swap_out_bytes': swap.sout,
    }


def spread(values):
    return {'min': min(values), 'median': statistics.median(values),
            'max': max(values), 'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--pred', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--input-mode', choices=['files', 'in-memory'], default='files')
    args = parser.parse_args()
    if args.rounds < 1 or any(t < 1 for t in args.threads):
        parser.error('rounds and threads must be positive')
    args.gt, args.pred = args.gt.resolve(), args.pred.resolve()
    if not args.gt.is_file() or not args.pred.is_file():
        parser.error('gt and pred must be existing files')
    args.out.mkdir(parents=True, exist_ok=False)
    result = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, *sys.argv],
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
        'benchmark_script_sha256': {p.name: digest(p) for p in
                                   [Path(__file__), HERE / 'compare_saved_predictions.py']},
        'platform': platform.platform(), 'machine': platform.machine(),
        'python': platform.python_version(), 'logical_cpus': psutil.cpu_count(),
        'memory_bytes': psutil.virtual_memory().total,
        'versions': {p: importlib.metadata.version(p) for p in
                     ['numpy', 'pycocotools', 'faster-coco-eval',
                      'ultrafast-pycocotools', 'psutil']},
        'gt_sha256': digest(args.gt), 'pred_sha256': digest(args.pred),
        'input_mode': args.input_mode, 'threads': args.threads,
        'rounds': args.rounds, 'cpu_affinity': None,
        'method': 'One excluded warmup per backend/thread configuration; fresh process '
                  'per run; all six backend permutations; thread order reversed on odd '
                  'rounds; whole-host telemetry includes the benchmark process, imports '
                  'and serialization; no timing-based exclusions. Thread limits do not '
                  'pin cores. CPU time remains sensitive to frequency/cache effects.',
        'runs': [],
    }
    psutil.cpu_percent()
    time.sleep(1)
    result['baseline'] = [snapshot()]
    for _ in range(4):
        time.sleep(1)
        result['baseline'].append(snapshot())

    def save():
        (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(backend, threads, round_index, warmup):
        label = f'{"warmup" if warmup else f"round-{round_index + 1}"}-t{threads}-{backend}'
        folder = args.out / label
        cmd = [sys.executable, str(HERE / 'compare_saved_predictions.py'),
               '--backend', backend, '--gt', str(args.gt), '--pred', str(args.pred),
               '--out', str(folder), '--input-mode', args.input_mode]
        env = {**os.environ, 'RAYON_NUM_THREADS': str(threads),
               'OMP_NUM_THREADS': str(threads), 'OPENBLAS_NUM_THREADS': '1',
               'VECLIB_MAXIMUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        samples = [snapshot()]
        with (args.out / f'{label}.log').open('w') as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(1)
                samples.append(snapshot())
        if process.returncode:
            raise RuntimeError(f'{label} failed: see {args.out / (label + ".log")}')
        measurement = json.loads((folder / 'result.json').read_text())
        entry = {'label': label, 'backend': backend, 'threads': threads,
                 'round': round_index, 'warmup': warmup, 'telemetry': samples,
                 'measurement': measurement}
        result['runs'].append(entry)
        save()
        seconds = measurement['runs'][0]['total_scoring_seconds']
        print(f'{label}: {seconds:.4f} s', flush=True)
        return folder

    orders = list(itertools.permutations(BACKENDS))
    for round_index in range(-1, args.rounds):
        order = BACKENDS if round_index == -1 else orders[round_index % len(orders)]
        thread_order = args.threads if round_index % 2 == 0 else list(reversed(args.threads))
        for threads in thread_order:
            folders = {backend: run(backend, threads, round_index, round_index == -1)
                       for backend in order}
            exact = compare(folders['pycocotools'], folders['ultrafast'])
            numerical = compare_numerically(folders['pycocotools'], folders['faster-coco-eval'])
            for entry in result['runs'][-len(BACKENDS):]:
                entry['ultrafast_byte_identical'] = exact['byte_identical']
                entry['faster_agreement'] = numerical
            save()
            if not all(numerical['within_absolute_tolerance'].values()):
                raise ValueError('faster-coco-eval failed tolerance; see results.json')

    summary = {}
    for threads in args.threads:
        summary[str(threads)] = {}
        for backend in BACKENDS:
            values = [r['measurement'] for r in result['runs']
                      if not r['warmup'] and r['threads'] == threads and r['backend'] == backend]
            # Determinism across ALL fresh processes, not only within each round.
            assert all(v['array_sha256'] == values[0]['array_sha256'] for v in values)
            summary[str(threads)][backend] = {
                'scoring_wall_seconds': spread([v['runs'][0]['total_scoring_seconds'] for v in values]),
                'scoring_cpu_seconds': spread([v['runs'][0]['cpu_seconds']['total_scoring_seconds'] for v in values]),
                'peak_rss_MiB': spread([v['peak_rss_MiB'] for v in values]),
            }
    result['summary'] = summary
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    print('PASS: all rounds satisfy parity; raw measurements and telemetry in', args.out / 'results.json')


if __name__ == '__main__':
    main()
