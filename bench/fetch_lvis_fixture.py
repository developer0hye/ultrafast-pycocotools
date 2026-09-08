"""Fetch the official 100-image LVIS example, pinned by source commit and SHA-256."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--coco-seg-pred', type=Path,
                        help='Also map real COCO segmentation predictions onto the official LVIS synsets')
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parent / 'results/lvis_fixture_sources.json').read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, item in manifest['files'].items():
        path = args.out / name
        if path.exists():
            payload = path.read_bytes()
        else:
            with urllib.request.urlopen(item['url'], timeout=60) as response:
                payload = response.read()
        if hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError(f'LVIS fixture hash mismatch: {name}')
        if not path.exists():
            path.write_bytes(payload)
        print(f'Verified {name}')

    if args.coco_seg_pred:
        gt = json.loads((args.out / 'lvis_gt_100.json').read_text())
        mapping = json.loads((args.out / 'lvis_coco_to_synset.json').read_text())
        synsets = {category['synset']: category['id'] for category in gt['categories']}
        categories = {item['coco_cat_id']: synsets[item['synset']] for item in mapping.values()
                      if item['synset'] in synsets}
        images = {image['id'] for image in gt['images']}
        predictions = json.loads(args.coco_seg_pred.read_text())
        selected = [dict(p, category_id=categories[p['category_id']]) for p in predictions
                    if p['image_id'] in images and p['category_id'] in categories]
        assert selected and all('segmentation' in p for p in selected)
        payload = (json.dumps(selected, separators=(',', ':')) + '\n').encode()
        (args.out / 'lvis_yolo26n_seg_100.json').write_bytes(payload)
        provenance = {
            'source_prediction_sha256': hashlib.sha256(args.coco_seg_pred.read_bytes()).hexdigest(),
            'mapped_prediction_sha256': hashlib.sha256(payload).hexdigest(),
            'mapping': categories, 'images': len(images), 'predictions': len(selected),
            'unmapped_coco_categories': [item['coco_cat_id'] for item in mapping.values()
                                         if item['synset'] not in synsets],
            'scope': 'Real COCO-trained YOLO masks, official synset mapping, 100 LVIS example images; '
                     'compatibility check, not a full-taxonomy LVIS model accuracy claim',
        }
        (args.out / 'lvis_yolo26n_seg_100_sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
        # The previous backend raises during mask preparation on this input,
        # which contains seven images without ground-truth annotations.
        # Keep the original 100-image case, and save an explicitly labelled subset
        # for an additional mask-array comparison that both backends can execute.
        annotated = {ann['image_id'] for ann in gt['annotations']}
        subset = dict(gt, images=[image for image in gt['images'] if image['id'] in annotated])
        (args.out / 'lvis_gt_93_annotated.json').write_text(json.dumps(subset, separators=(',', ':')) + '\n')
        subset_predictions = [p for p in selected if p['image_id'] in annotated]
        (args.out / 'lvis_yolo26n_seg_93_annotated.json').write_text(
            json.dumps(subset_predictions, separators=(',', ':')) + '\n')
        assert len(annotated) == 93
        print(f'Mapped {len(selected)} real masks across {len(categories)} COCO categories')


if __name__ == '__main__':
    main()
