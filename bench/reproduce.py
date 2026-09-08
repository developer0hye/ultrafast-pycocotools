"""Generate inputs and compare COCO scorers, keeping strict ultrafast parity."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def compare(reference, candidate):
    import numpy as np
    reference, candidate = Path(reference), Path(candidate)
    ref = json.loads((reference / 'result.json').read_text())
    fast = json.loads((candidate / 'result.json').read_text())
    for key in ('gt_sha256', 'pred_sha256', 'images', 'detections'):
        if ref[key] != fast[key]:
            raise ValueError(f'Inputs differ: {key}')
    equality = {}
    with np.load(reference / 'arrays.npz') as a, np.load(candidate / 'arrays.npz') as b:
        for key in ('precision', 'recall', 'scores', 'stats'):
            equality[key] = a[key].shape == b[key].shape and a[key].tobytes() == b[key].tobytes()
    if not all(equality.values()):
        raise ValueError(f'Array parity failed: {equality}')
    return {'byte_identical': equality, 'reference': ref, 'ultrafast': fast}


def compare_numerically(reference, candidate, atol=1e-12):
    """Report byte equality separately from numerical agreement; require identical inputs."""
    import numpy as np
    reference, candidate = Path(reference), Path(candidate)
    ref = json.loads((reference / 'result.json').read_text())
    other = json.loads((candidate / 'result.json').read_text())
    for key in ('gt_sha256', 'pred_sha256', 'images', 'detections'):
        if ref[key] != other[key]:
            raise ValueError(f'Inputs differ: {key}')
    result = {'absolute_tolerance': atol, 'relative_tolerance': 0,
              'byte_identical': {}, 'max_abs_difference': {},
              'different_elements': {}, 'within_absolute_tolerance': {}}
    with np.load(reference / 'arrays.npz') as a, np.load(candidate / 'arrays.npz') as b:
        for key in ('precision', 'recall', 'scores', 'stats'):
            if a[key].shape != b[key].shape:
                raise ValueError(f'Array shapes differ: {key}')
            result['byte_identical'][key] = a[key].tobytes() == b[key].tobytes()
            delta = np.abs(a[key] - b[key])
            result['max_abs_difference'][key] = float(np.max(delta)) if np.isfinite(delta).all() else None
            result['different_elements'][key] = int(np.count_nonzero(a[key] != b[key]))
            result['within_absolute_tolerance'][key] = bool(np.allclose(a[key], b[key], rtol=0, atol=atol))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('quick', 'objects365', 'predictions'))
    p.add_argument('--out', type=Path, required=True, help='New output directory; never overwritten')
    p.add_argument('--gt', type=Path, help='Original annotation JSON; required except in quick mode')
    p.add_argument('--pred', type=Path, help='Existing predictions, for predictions mode')
    p.add_argument('--include-faster', action='store_true', help='Also measure faster-coco-eval and report numerical agreement')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, help='Defaults to 0 for quick, 1234 for Objects365')
    p.add_argument('--cpus', help='Optional Linux CPU affinity, e.g. 0,1')
    p.add_argument('--verify-published', choices=('quick', 'objects365', 'yolo11m'),
                   help='Require the input hashes recorded in the published benchmark')
    a = p.parse_args()
    if a.threads < 1 or a.repeats < 1:
        p.error('threads and repeats must be positive')
    if a.mode != 'quick' and (a.gt is None or not a.gt.is_file()):
        p.error('--gt must name an existing annotation JSON')
    if a.mode == 'predictions' and (a.pred is None or not a.pred.is_file()):
        p.error('--pred must name an existing prediction JSON')
    if a.cpus:
        if not hasattr(os, 'sched_setaffinity'):
            p.error('--cpus requires Linux CPU affinity support')
        os.sched_setaffinity(0, {int(x) for x in a.cpus.split(',')})
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    seed = a.seed if a.seed is not None else (0 if a.mode == 'quick' else 1234)
    env = {**os.environ, 'RAYON_NUM_THREADS': str(a.threads),
           'OMP_NUM_THREADS': str(a.threads), 'OPENBLAS_NUM_THREADS': '1'}
    commands = []

    def run(script, arguments, log):
        cmd = [sys.executable, str(HERE / script), *map(str, arguments)]
        commands.append(cmd)
        print(f'Running {script}; log: {log.name}', flush=True)
        with log.open('w') as stream:
            subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    if a.mode == 'quick':
        run('make_dataset.py', ['--images', 64, '--cats', 8, '--gt-per-image', 12,
                               '--dt-per-image', 40, '--seed', seed, '--out', a.out / 'inputs'],
            a.out / 'generation.log')
        gt, pred = a.out / 'inputs/gt_64.json', a.out / 'inputs/dt_64.json'
    elif a.mode == 'objects365':
        gt, pred = a.gt.resolve(), a.out / 'predictions.json'
        run('make_dets.py', ['--gt', gt, '--out', pred, '--seed', seed, '--recall', .75,
                            '--wrong-class', .12, '--fp-ratio', 2.0, '--score-decimals', 3],
            a.out / 'generation.log')
    else:
        gt, pred = a.gt.resolve(), a.pred.resolve()
    if a.verify_published:
        published = json.loads((HERE / 'results/public_benchmarks.json').read_text())
        expected = published['cases'][a.verify_published]['reference']
        for key, path in [('gt_sha256', gt), ('pred_sha256', pred)]:
            if digest(path) != expected[key]:
                raise ValueError(f'Published input mismatch: {key}; refusing an expensive nonmatching run')
    for backend in ('pycocotools', 'ultrafast'):
        run('compare_saved_predictions.py', ['--backend', backend, '--gt', gt, '--pred', pred,
                                             '--out', a.out / backend, '--repeats', a.repeats],
            a.out / f'{backend}.log')
    result = compare(a.out / 'pycocotools', a.out / 'ultrafast')
    if a.include_faster:
        run('compare_saved_predictions.py', ['--backend', 'faster-coco-eval', '--gt', gt, '--pred', pred,
                                             '--out', a.out / 'faster-coco-eval', '--repeats', a.repeats],
            a.out / 'faster-coco-eval.log')
        result['faster_coco_eval'] = json.loads((a.out / 'faster-coco-eval/result.json').read_text())
        result['faster_agreement'] = compare_numerically(a.out / 'pycocotools', a.out / 'faster-coco-eval')
    if a.verify_published and result['reference']['array_sha256'] != expected['array_sha256']:
        raise ValueError('Inputs match but evaluation arrays differ from the published reference; check versions')
    result.update(mode=a.mode, seed=seed, python=platform.python_version(),
                  versions={name: importlib.metadata.version(name) for name in
                            ('numpy', 'pycocotools', 'ultrafast-pycocotools')},
                  threads=a.threads, cpu_affinity=sorted(os.sched_getaffinity(0))
                  if hasattr(os, 'sched_getaffinity') else None, commands=commands)
    if a.include_faster:
        result['versions']['faster-coco-eval'] = importlib.metadata.version('faster-coco-eval')
    (a.out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: precision, recall, scores and stats are byte-identical (pycocotools vs ultrafast).', flush=True)
    print(f'Evidence: {a.out / "comparison.json"}', flush=True)
    if a.include_faster:
        agreement = result['faster_agreement']
        print('faster-coco-eval versus reference:', json.dumps(agreement), flush=True)
        if not all(agreement['within_absolute_tolerance'].values()):
            raise ValueError('faster-coco-eval differs beyond tolerance; see comparison.json')


if __name__ == '__main__':
    main()
