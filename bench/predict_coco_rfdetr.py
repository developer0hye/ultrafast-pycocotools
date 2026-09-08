"""Generate reproducible COCO bbox predictions with a public RF-DETR model.

Inference runs once; the saved predictions can be evaluated by every backend.
Official COCO checkpoints already use COCO category IDs, so no remapping is
applied. This script records the exact model and input hashes for reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--ann', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', default='nano', choices=['nano', 'small', 'medium', 'large'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.001)
    p.add_argument('--limit', type=int, default=0, help='First N sorted images; 0 means all')
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--source-revision', help='Pinned upstream Git commit used for this run')
    a = p.parse_args()
    if a.batch < 1 or a.threads < 1 or a.limit < 0:
        p.error('batch/threads must be positive and limit must be nonnegative')
    if a.shards < 1 or not 0 <= a.shard_index < a.shards:
        p.error('shard-index must be in [0, shards)')
    if a.out.exists() or a.out.with_suffix('.metadata.json').exists():
        p.error('Output already exists; use a new path')
    if a.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    from PIL import Image
    import torch
    import rfdetr

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    ann = json.loads(a.ann.read_text())
    images = sorted(ann['images'], key=lambda x: x['file_name'])
    if a.limit:
        images = images[:a.limit]
    total_images = len(images)
    images = images[total_images * a.shard_index // a.shards:total_images * (a.shard_index + 1) // a.shards]
    for image in images:
        if not (a.images / image['file_name']).is_file():
            raise FileNotFoundError(image['file_name'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    factory = getattr(rfdetr, 'RFDETR' + a.variant.capitalize())
    model = factory(device=a.device)
    # Fuse the inference model without a device-specific compilation warmup.
    model.optimize_for_inference(compile=False)
    config = model.model_config
    weights = Path(config.pretrain_weights)
    metadata = {'model': 'RF-DETR-' + a.variant, 'rfdetr': importlib.metadata.version('rfdetr'),
                'source_revision': a.source_revision, 'torch': torch.__version__,
                'transformers': importlib.metadata.version('transformers'),
                'supervision': importlib.metadata.version('supervision'), 'numpy': np.__version__,
                'device': a.device, 'threads': a.threads, 'batch': a.batch,
                'resolution': config.resolution, 'num_select': config.num_select,
                'threshold': a.threshold, 'optimized': True, 'compiled': False,
                'bbox_decimal_places': 3, 'weights_sha256': digest(weights),
                'gt_sha256': digest(a.ann), 'images': len(images),
                'shards': a.shards, 'shard_index': a.shard_index, 'total_images': total_images,
                'image_ids_sha256': hashlib.sha256(json.dumps([im['id'] for im in images]).encode()).hexdigest()}
    del ann
    temporary = a.out.with_suffix('.partial.json')
    count = 0
    started = time.perf_counter()
    with temporary.open('w') as stream, torch.inference_mode():
        stream.write('[')
        for start in range(0, len(images), a.batch):
            chunk = images[start:start + a.batch]
            inputs = []
            for im in chunk:
                with Image.open(a.images / im['file_name']) as image:
                    inputs.append(image.convert('RGB'))
            results = model.predict(inputs, threshold=a.threshold)
            if not isinstance(results, list):
                results = [results]
            if len(results) != len(chunk):
                raise ValueError('Model output count does not match the input batch')
            for im, prediction in zip(chunk, results):
                boxes = np.asarray(prediction.xyxy, dtype=np.float64)
                for box, score, category in zip(boxes, prediction.confidence, prediction.class_id):
                    x0, y0, x1, y1 = box
                    record = {'image_id': int(im['id']), 'category_id': int(category),
                              'bbox': [round(float(v), 3) for v in (x0, y0, x1 - x0, y1 - y0)],
                              'score': float(score)}
                    if count:
                        stream.write(',')
                    stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False))
                    count += 1
            done = start + len(chunk)
            if start == 0 or done % 128 < a.batch or done == len(images):
                elapsed = time.perf_counter() - started
                print(f'{done}/{len(images)} images; {count} predictions; {elapsed:.1f}s; '
                      f'ETA {elapsed / done * (len(images) - done):.1f}s', flush=True)
        stream.write(']')
    temporary.replace(a.out)
    metadata.update(detections=count, prediction_seconds=time.perf_counter() - started,
                    pred_sha256=digest(a.out))
    a.out.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Completed: {count} predictions', flush=True)


if __name__ == '__main__':
    main()
