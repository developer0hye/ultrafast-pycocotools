"""Measure nested Objects365 subsets in fresh scorer processes; never fit curves."""
from __future__ import annotations

import argparse
import json
import importlib.metadata
import platform
import os
from pathlib import Path
import random
import subprocess
import sys

from reproduce import compare, compare_numerically, digest

HERE = Path(__file__).resolve().parent


def prepare(gt_path, pred_path, out, sizes, seed):
    """Run in a separate process so input preparation cannot inflate scorer RSS."""
    gt = json.loads(gt_path.read_text())
    predictions = json.loads(pred_path.read_text())
    ids = sorted(im['id'] for im in gt['images'])
    random.Random(seed).shuffle(ids)
    if sizes[-1] > len(ids):
        raise ValueError('Requested subset exceeds available images')
    points = []
    for count in sizes:
        selected = set(ids[:count])
        if count == len(ids):
            g, p = gt_path, pred_path
            anns, dets = gt['annotations'], predictions
        else:
            subset = {**gt, 'images': [im for im in gt['images'] if im['id'] in selected],
                      'annotations': [ann for ann in gt['annotations'] if ann['image_id'] in selected]}
            anns = subset['annotations']
            dets = [d for d in predictions if d['image_id'] in selected]
            g, p = out / f'gt_{count}.json', out / f'pred_{count}.json'
            g.write_text(json.dumps(subset))
            p.write_text(json.dumps(dets))
        points.append(dict(images=count, annotations=len(anns), detections=len(dets),
                           categories=len(gt['categories']), gt=str(g), pred=str(p)))
    (out / 'inputs.json').write_text(json.dumps(points, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--pred', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 5000, 10000, 20000, 40000, 80000])
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--include-faster', action='store_true', help='Also measure faster-coco-eval')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cpus', help='Optional Linux CPU affinity, e.g. 0,1')
    parser.add_argument('--reuse-full', type=Path, help='Reuse Objects365 endpoint from a public_benchmarks.json file after verifying input hashes')
    parser.add_argument('--prepare-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or min(args.sizes) < 1:
        parser.error('threads and sizes must be positive')
    if args.include_faster and args.reuse_full:
        parser.error('--include-faster requires fresh reference arrays; omit --reuse-full')
    args.sizes = sorted(set(args.sizes))
    args.gt, args.pred, args.out = args.gt.resolve(), args.pred.resolve(), args.out.resolve()
    if not args.gt.is_file() or not args.pred.is_file():
        parser.error('annotation and prediction JSON files must exist')
    if args.prepare_only:
        prepare(args.gt, args.pred, args.out, args.sizes, args.seed)
        return
    if args.cpus:
        if not hasattr(os, 'sched_setaffinity'):
            parser.error('--cpus requires Linux')
        os.sched_setaffinity(0, {int(x) for x in args.cpus.split(',')})
    args.out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'RAYON_NUM_THREADS': str(args.threads),
           'OMP_NUM_THREADS': str(args.threads), 'OPENBLAS_NUM_THREADS': '1'}
    subprocess.run([sys.executable, str(HERE / 'scale.py'), '--prepare-only',
                    '--gt', str(args.gt), '--pred', str(args.pred), '--out', str(args.out),
                    '--seed', str(args.seed), '--sizes', *map(str, args.sizes)], check=True, env=env)
    points = json.loads((args.out / 'inputs.json').read_text())
    result = {'schema_version': 1, 'workload': 'Objects365 v2 validation; seeded synthetic predictions',
              'subset_seed': args.seed, 'selection': 'Seeded shuffle of sorted image IDs; nested prefixes; preserve source annotation and prediction order; keep all categories',
              'threads': args.threads, 'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
              'runs_per_backend_per_size': 1,
              'python': platform.python_version(),
              'versions': {name: importlib.metadata.version(name) for name in
                           ('numpy', 'pycocotools', 'ultrafast-pycocotools')},
              'points': []}
    for index, point in enumerate(points):
        count = point['images']
        cached = None
        if args.reuse_full:
            published = json.loads(args.reuse_full.read_text())
            candidate = published['cases']['objects365']
            if count == candidate['reference']['images']:
                for key, path in [('gt_sha256', point['gt']), ('pred_sha256', point['pred'])]:
                    if digest(path) != candidate['reference'][key]:
                        raise ValueError(f'Reused endpoint input mismatch: {key}')
                if not all(candidate['byte_identical'].values()):
                    raise ValueError('Reused endpoint must have verified array parity')
                cached = candidate
                result['endpoint_provenance'] = published['provenance']
        if cached:
            measured = cached
            source = 'previously published full-dataset measurement; same input hashes'
        else:
            # Alternate order across sizes; separate processes isolate memory high-water marks.
            order = ('pycocotools', 'ultrafast') if index % 2 == 0 else ('ultrafast', 'pycocotools')
            for backend in order:
                destination = args.out / f'{count}_{backend}'
                print(f'{count:,} images: {backend}', flush=True)
                with (args.out / f'{count}_{backend}.log').open('w') as log:
                    subprocess.run([sys.executable, str(HERE / 'compare_saved_predictions.py'),
                                    '--gt', point['gt'], '--pred', point['pred'], '--out', str(destination),
                                    '--backend', backend], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            measured = compare(args.out / f'{count}_pycocotools', args.out / f'{count}_ultrafast')
            source = 'new measurement'
        if args.include_faster:
            if cached:
                raise ValueError('--include-faster requires a fresh reference run; omit --reuse-full')
            destination = args.out / f'{count}_faster-coco-eval'
            print(f'{count:,} images: faster-coco-eval', flush=True)
            with (args.out / f'{count}_faster-coco-eval.log').open('w') as log:
                subprocess.run([sys.executable, str(HERE / 'compare_saved_predictions.py'),
                                '--gt', point['gt'], '--pred', point['pred'], '--out', str(destination),
                                '--backend', 'faster-coco-eval'], check=True, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
            measured['faster_coco_eval'] = json.loads((destination / 'result.json').read_text())
            measured['faster_agreement'] = compare_numerically(args.out / f'{count}_pycocotools', destination)
            result['versions']['faster-coco-eval'] = importlib.metadata.version('faster-coco-eval')
        result['points'].append({**{k: v for k, v in point.items() if k not in ('gt', 'pred')},
                                 'measurement_source': source, **measured})
        (args.out / 'scaling.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f'{count:,} images: PASS, pycocotools/ultrafast arrays byte-identical', flush=True)
        if args.include_faster and not all(measured['faster_agreement']['within_absolute_tolerance'].values()):
            raise ValueError('faster-coco-eval differs beyond tolerance; see scaling.json')


if __name__ == '__main__':
    main()
