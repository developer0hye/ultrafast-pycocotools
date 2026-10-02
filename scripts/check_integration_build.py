"""Check an upstream integration environment before its tests run.

The upstream projects depend on the released ultrafast-pycocotools wheel, which
the workflow replaces with a build of this checkout. Fail early if the
replacement did not happen, if the framework is not the CPU build, or if the
upstream revision no longer evaluates with ultrafast-pycocotools, so a green
run always means this revision was exercised. SAHI's evaluation path does not
use torch, so its job does not install one.
"""

import importlib.metadata
import inspect
import json
import sys
from pathlib import Path


def installed_from_checkout() -> str:
    distribution = importlib.metadata.distribution('ultrafast-pycocotools')
    direct_url = distribution.read_text('direct_url.json')
    assert direct_url, 'ultrafast-pycocotools was installed from an index, not from this checkout'
    url = json.loads(direct_url)['url']
    assert url == Path.cwd().resolve().as_uri(), f'ultrafast-pycocotools was installed from {url}'
    return distribution.version


def main() -> None:
    project = sys.argv[1]
    version = installed_from_checkout()
    if project in ('rfdetr', 'ultralytics'):
        import torch
        assert torch.version.cuda is None, 'expected the CPU build of torch'
        framework = f'; torch {torch.__version__}'
    else:
        framework = ''
    if project == 'rfdetr':
        from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision
        # The constructor rejects unknown backends; its private registry has
        # been renamed upstream, so do not read it directly.
        try:
            OnePassCocoMeanAveragePrecision(backend='ufcoco')
        except ValueError as error:
            raise AssertionError('RF-DETR revision has no ufcoco backend') from error
        upstream = importlib.metadata.version('rfdetr')
    elif project == 'ultralytics':
        from ultralytics.models.yolo.detect import DetectionValidator
        source = inspect.getsource(DetectionValidator.coco_evaluate)
        assert 'ultrafast_pycocotools' in source, 'Ultralytics revision does not evaluate with ultrafast-pycocotools'
        upstream = importlib.metadata.version('ultralytics')
    elif project == 'sahi':
        from sahi.scripts.coco_evaluation import evaluate
        source = inspect.getsource(evaluate)
        assert 'ultrafast_pycocotools' in source, 'SAHI revision has no ultrafast evaluation backend'
        upstream = importlib.metadata.version('sahi')
    else:
        raise SystemExit(f'unknown project {project}')
    print(f'ultrafast-pycocotools {version} from this checkout; {project} {upstream}{framework}')


if __name__ == '__main__':
    main()
