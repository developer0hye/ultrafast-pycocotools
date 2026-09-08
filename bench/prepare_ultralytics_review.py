"""Fetch hash-pinned real COCO review inputs and prepare all 5000 validation images."""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import yaml

ASSETS = {
    'data/val2017.zip': ('https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip',
                         '4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05'),
    'data/annotations_trainval2017.zip': ('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                         '113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268'),
    'data/coco2017labels-segments.zip': ('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip',
                                       '039d39b6548fa9dbcc1bb84e722953d25ebc04935772805e5d5c7badefcb349b'),
    'data/coco2017labels-pose.zip': ('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip',
                                   'a088116a01f10202d3697467a1d43b26c1cfe3f76dc997ac3f57f6a0c0bece86'),
    'yolo26n.pt': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt',
                  '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'),
    'yolo26n-seg.pt': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt',
                      '361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f'),
    'yolo26n-pose.pt': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt',
                       'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Contains reference/ and replacement/ Ultralytics checkouts')
    a = p.parse_args()
    root = a.root.resolve()
    data = root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in ASSETS.items():
        target = root / name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=60) as response, target.open('wb') as out:
                for chunk in iter(lambda: response.read(1048576), b''):
                    out.write(chunk)
        digest = hashlib.sha256()
        with target.open('rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                digest.update(chunk)
        assert digest.hexdigest() == expected, f'Hash mismatch: {target}'
        print('Verified', name, flush=True)
    with zipfile.ZipFile(data / 'val2017.zip') as archive:
        archive.extractall(data / 'coco/images')
    for kind in ('segments', 'pose'):
        with zipfile.ZipFile(data / f'coco2017labels-{kind}.zip') as archive:
            for name in archive.namelist():
                if '/labels/val2017/' in name:
                    archive.extract(name, data)
    with zipfile.ZipFile(data / 'annotations_trainval2017.zip') as archive:
        for name in ('instances_val2017.json', 'person_keypoints_val2017.json'):
            archive.extract('annotations/' + name, data / 'coco')
    for name in ('images', 'annotations'):
        destination = data / 'coco-pose' / name
        if not destination.exists():
            destination.symlink_to(data / 'coco' / name, target_is_directory=True)
    for name in ('coco', 'coco-pose'):
        dataset = data / name
        images = sorted((dataset / 'images/val2017').glob('*.jpg'))
        assert len(images) == 5000
        (dataset / 'val2017.txt').write_text(''.join(str(image) + '\n' for image in images))
        config = yaml.safe_load((root / 'reference/ultralytics/cfg/datasets' / f'{name}.yaml').read_text())
        config.pop('download', None)
        config.update(path=str(dataset), train='val2017.txt', val='val2017.txt')
        (data / f'{name}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == '__main__':
    main()
