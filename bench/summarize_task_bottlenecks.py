"""Condense the task-bottleneck measurements into one JSON file and tables.

Inputs are the `hotcoco_benchmark.py --backends ufcoco` output directories
for the baseline and optimized builds (one per thread count), and the
`profile_tasks.py --json-out` files of both `alloc-stats` builds.

Usage::

    python bench/summarize_task_bottlenecks.py --runs DIR --profiles DIR --out FILE
"""

import argparse
import json
import statistics
from pathlib import Path

TASKS = ('bbox', 'segm', 'keypoints')
METRICS = ('wall_total', 'cpu_total', 'peak_rss_mb')
PHASES = ('load_gt', 'load_dt', 'evaluate:prelude', 'evaluate:extract', 'evaluate:engine',
          'evaluate:epilogue', 'accumulate', 'summarize')
ENGINE = ('gt_read', 'dt_read', 'group_index', 'iou_cpu', 'match_cpu', 'accumulate_cpu')


def headline(runs: Path) -> dict:
    out = {}
    for build in ('base', 'new'):
        for threads in (2, 1):
            result = json.loads((runs / f'{build}-t{threads}' / 'results.json').read_text())
            parity = result['parity']
            for task in TASKS:
                summary = result['summary'][f'{task}-files-t{threads}-ufcoco']
                arrays = parity[f'{task}-files'][f'ufcoco-t{threads}']['arrays']
                out.setdefault(task, {}).setdefault(f't{threads}', {})[build] = dict(
                    {metric: summary[metric] for metric in METRICS},
                    byte_identical=all(arrays[key]['byte_identical'] for key in arrays),
                    packages=result['packages'], started_utc=result['started_utc'],
                    input_sha256=result['input_sha256'])
    return out


def phases(profiles: Path) -> dict:
    out = {}
    for build in ('base', 'new'):
        for task in TASKS:
            runs = json.loads((profiles / f'{build}-{task}.json').read_text())
            table = {}
            for name in PHASES:
                rows = [next(p for p in run['phases'] if p['phase'] == name) for run in runs]
                table[name] = dict(
                    wall=statistics.median(r['wall'] for r in rows),
                    cpu=statistics.median(r['cpu'] for r in rows),
                    rust_peak_mb=statistics.median(r['rust_peak_mb'] for r in rows),
                    rust_live_mb=statistics.median(r['rust_live_mb'] for r in rows),
                    allocations=statistics.median(r['rust_allocs'] for r in rows))
            engine = {key: statistics.median(run['engine_timings'][key] for run in runs)
                      for key in ENGINE}
            out.setdefault(task, {})[build] = dict(phases=table, engine=engine, runs=len(runs))
    return out


def change(before: float, after: float) -> str:
    return f'{100 * (after / before - 1):+.1f}%' if before else 'n/a'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = dict(headline=headline(args.runs), phases=phases(args.profiles))
    args.out.write_text(json.dumps(data, indent=1) + '\n')

    print('| Task | Threads | Wall s (main → branch) | CPU s | Peak RSS MB | Arrays |')
    print('| --- | ---: | ---: | ---: | ---: | --- |')
    for task in TASKS:
        for threads in ('t2', 't1'):
            b, n = (data['headline'][task][threads][k] for k in ('base', 'new'))
            same = 'bit-identical' if b['byte_identical'] and n['byte_identical'] else 'DIFFER'
            print(f"| {task} | {threads[1:]} | {b['wall_total']['median']:.3f} → {n['wall_total']['median']:.3f} "
                  f"({change(b['wall_total']['median'], n['wall_total']['median'])}) "
                  f"| {b['cpu_total']['median']:.3f} → {n['cpu_total']['median']:.3f} "
                  f"| {b['peak_rss_mb']['median']:.1f} → {n['peak_rss_mb']['median']:.1f} "
                  f"({change(b['peak_rss_mb']['median'], n['peak_rss_mb']['median'])}) | {same} |")
    for task in TASKS:
        b, n = data['phases'][task]['base'], data['phases'][task]['new']
        print(f'\n{task}\n\n| Phase | Wall s | CPU s | Rust peak in phase MB | Rust allocations so far |')
        print('| --- | ---: | ---: | ---: | ---: |')
        for name in PHASES:
            x, y = b['phases'][name], n['phases'][name]
            print(f"| {name} | {x['wall']:.3f} → {y['wall']:.3f} | {x['cpu']:.3f} → {y['cpu']:.3f} "
                  f"| {x['rust_peak_mb']:.1f} → {y['rust_peak_mb']:.1f} "
                  f"| {x['allocations']:,.0f} → {y['allocations']:,.0f} |")
        print('\n| Engine timer | main → branch |\n| --- | ---: |')
        for key in ENGINE:
            print(f"| {key} | {b['engine'][key]:.3f} → {n['engine'][key]:.3f} |")


if __name__ == '__main__':
    main()
