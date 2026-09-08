"""Fetch the official 100-image LVIS example, pinned by source commit and SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'data')
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


if __name__ == '__main__':
    main()
