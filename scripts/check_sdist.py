"""Check the license paths PyPI enforces for source distribution metadata."""
import sys
import tarfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


def check_sdist(path):
    with tarfile.open(path) as archive:
        names = archive.getnames()
        metadata_paths = [name for name in names if name.count('/') == 1 and name.endswith('/PKG-INFO')]
        if len(metadata_paths) != 1:
            raise ValueError('Expected one top-level PKG-INFO')
        metadata_path = metadata_paths[0]
        metadata = BytesParser().parsebytes(archive.extractfile(metadata_path).read())
        root = PurePosixPath(metadata_path).parent
        licenses = metadata.get_all('License-File', [])
        if not licenses:
            raise ValueError('Source metadata must declare its license file')
        for filename in licenses:
            location = str(root / filename)
            if location not in names or not archive.getmember(location).isfile():
                raise ValueError(f'Missing declared license file: {location}')
            if not archive.extractfile(location).read().strip():
                raise ValueError(f'Empty declared license file: {location}')
    print(f'Validated source license files: {Path(path).name}')


if __name__ == '__main__':
    for argument in sys.argv[1:]:
        check_sdist(Path(argument))
