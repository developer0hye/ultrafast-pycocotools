"""Verify and merge contiguous RF-DETR prediction shards in their original order."""
import argparse
import hashlib
import json
from pathlib import Path

from predict_coco_rfdetr import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann', type=Path, required=True)
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output already exists')
    annotations = json.loads(a.ann.read_text())
    image_ids = [im['id'] for im in sorted(annotations['images'], key=lambda x: x['file_name'])]
    shards = sorted([(json.loads(path.with_suffix('.metadata.json').read_text()), path)
                     for path in a.inputs], key=lambda item: item[0]['shard_index'])
    common = ['model', 'rfdetr', 'source_revision', 'torch', 'transformers', 'supervision',
              'device', 'threads', 'batch', 'resolution', 'num_select', 'threshold',
              'optimized', 'compiled', 'bbox_decimal_places', 'weights_sha256',
              'gt_sha256', 'shards', 'total_images']
    first = shards[0][0]
    assert len(shards) == first['shards']
    assert first['gt_sha256'] == digest(a.ann)
    assert first['total_images'] == len(image_ids)
    for i, (meta, path) in enumerate(shards):
        assert meta['shard_index'] == i
        assert all(meta[k] == first[k] for k in common)
        ids = image_ids[len(image_ids)*i//len(shards):len(image_ids)*(i+1)//len(shards)]
        assert meta['images'] == len(ids)
        assert meta['image_ids_sha256'] == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        assert meta['pred_sha256'] == digest(path)
    temporary = a.out.with_suffix('.partial.json')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with temporary.open('wb') as out:
        out.write(b'[')
        for meta, path in shards:
            raw = path.read_bytes().strip()
            assert raw.startswith(b'[') and raw.endswith(b']')
            if meta['detections']:
                if count:
                    out.write(b',')
                out.write(memoryview(raw)[1:-1])
                count += meta['detections']
        out.write(b']')
    temporary.replace(a.out)
    metadata = {k: first[k] for k in common}
    metadata.update(images=len(image_ids), detections=count, pred_sha256=digest(a.out),
                    image_ids_sha256=hashlib.sha256(json.dumps(image_ids).encode()).hexdigest(),
                    shard_prediction_seconds=[meta['prediction_seconds'] for meta, _ in shards],
                    shard_pred_sha256=[meta['pred_sha256'] for meta, _ in shards])
    a.out.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Merged {len(shards)} verified shards: {len(image_ids)} images, {count} predictions')


if __name__ == '__main__':
    main()
