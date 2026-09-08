"""Validate the complete tested release set before allowing a PyPI upload."""
import hashlib
import itertools
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import parse_wheel_filename


def main():
    directory = Path(sys.argv[1])
    version = tomllib.loads(Path('Cargo.toml').read_text())['workspace']['package']['version']
    expected = set(itertools.product(
        ['cp39', 'cp310', 'cp311', 'cp312', 'cp313', 'cp314'],
        ['linux-x86_64', 'windows-x86_64', 'macos-x86_64', 'macos-arm64'],
    ))
    found = set()
    files = sorted(directory.iterdir())
    wheels = [p for p in files if p.suffix == '.whl']
    sources = [p for p in files if p.name.endswith('.tar.gz')]
    if len(wheels) != len(expected) or len(sources) != 1 or len(files) != len(expected) + 1:
        raise SystemExit('Expected exactly 24 wheels and one source archive')
    for wheel in wheels:
        name, wheel_version, _, tags = parse_wheel_filename(wheel.name)
        if name != 'ultrafast-pycocotools' or str(wheel_version) != version:
            raise SystemExit(f'Unexpected package/version: {wheel.name}')
        coverage = set()
        for tag in tags:
            platform = tag.platform
            if platform.startswith('manylinux_2_17_') or platform.startswith('manylinux2014_'):
                system = 'linux-x86_64' if platform.endswith('_x86_64') else None
            elif platform == 'win_amd64':
                system = 'windows-x86_64'
            elif platform.startswith('macosx_') and platform.endswith('_x86_64'):
                system = 'macos-x86_64'
            elif platform.startswith('macosx_') and platform.endswith('_arm64'):
                system = 'macos-arm64'
            else:
                system = None
            if system is None or tag.abi != tag.interpreter:
                raise SystemExit(f'Unexpected wheel tag: {tag}')
            coverage.add((tag.interpreter, system))
        if len(coverage) != 1 or found & coverage:
            raise SystemExit(f'Duplicate or ambiguous wheel: {wheel.name}')
        found.update(coverage)
        with zipfile.ZipFile(wheel) as archive:
            paths = archive.namelist()
            required = ['__init__.py', 'coco.py', 'cocoeval.py', 'mask.py', 'integrations/rfdetr.py']
            for path in required:
                if 'ultrafast_pycocotools/' + path not in paths:
                    raise SystemExit(f'Missing {path} in {wheel.name}')
            native = [p for p in paths if '/_ufcoco.' in p and p.endswith(('.so', '.pyd'))]
            if len(native) != 1:
                raise SystemExit(f'Expected exactly one native extension in {wheel.name}')
            metadata_path = next(p for p in paths if p.endswith('.dist-info/METADATA'))
            metadata = BytesParser().parsebytes(archive.read(metadata_path))
            if metadata['Version'] != version or metadata['Name'] != 'ultrafast-pycocotools':
                raise SystemExit(f'Inconsistent wheel metadata: {wheel.name}')
    if found != expected:
        raise SystemExit(f'Incorrect platform coverage: missing={expected-found}, extra={found-expected}')
    with tarfile.open(sources[0]) as archive:
        paths = archive.getnames()
        metadata_path = next(p for p in paths if p.endswith('/PKG-INFO'))
        metadata = BytesParser().parsebytes(archive.extractfile(metadata_path).read())
        if metadata['Version'] != version or metadata['Name'] != 'ultrafast-pycocotools':
            raise SystemExit('Inconsistent source archive metadata')
        for suffix in ['/Cargo.lock', '/pyproject.toml', '/rust/ufcoco-core/src/lib.rs',
                       '/rust/ufcoco-py/src/lib.rs', '/python/ultrafast_pycocotools/integrations/rfdetr.py']:
            if not any(p.endswith(suffix) for p in paths):
                raise SystemExit(f'Source archive is missing {suffix}')
    Path('SHA256SUMS').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n' for p in files
    ))
    print(f'Validated {len(wheels)} wheels and one source archive for {version}')


if __name__ == '__main__':
    main()
