"""SAHI's ``coco evaluate`` with ultrafast as its COCO evaluator.

SAHI (obss/sahi#1452, released in 0.12.7) selects the evaluator with
``backend="ultrafast"`` and otherwise runs the same code as its pycocotools
backend: it loads the dataset from a temporary file, passes the result list to
``loadRes``, and sets custom ``catIds``, ``maxDets``, ``iouThrs`` and
``areaRng`` before reading ``stats`` and ``eval['precision']``.

Each test runs SAHI once per backend on the committed real COCO fixture,
captures the evaluator SAHI builds each time, and requires the complete
``precision``, ``recall`` and ``scores`` arrays, the summary statistics and the
metrics SAHI reports to match exactly.
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sahi', reason='optional SAHI integration dependency')
import pycocotools.cocoeval
from sahi.scripts.coco_evaluation import evaluate

import ultrafast_pycocotools

DATA = Path(__file__).parent / 'data'
GT = DATA / 'coco_subset_gt.json'
DT = DATA / 'coco_subset_dt.json'


@pytest.fixture
def captured(monkeypatch):
    """Every COCOeval SAHI creates, per backend."""
    evaluators = {'pycocotools': [], 'ultrafast': []}

    def recording(backend, original):
        def build(*args, **options):
            # Return the real class so the evaluator SAHI builds is the one tested.
            evaluator = original(*args, **options)
            evaluators[backend].append(evaluator)
            return evaluator
        return build

    # SAHI imports the names inside evaluate(), so patching the modules suffices.
    monkeypatch.setattr(pycocotools.cocoeval, 'COCOeval',
                        recording('pycocotools', pycocotools.cocoeval.COCOeval))
    monkeypatch.setattr(ultrafast_pycocotools, 'COCOeval',
                        recording('ultrafast', ultrafast_pycocotools.COCOeval))
    return evaluators


def run(tmp_path, backend, result=DT, **options):
    return evaluate(str(GT), str(result), out_dir=str(tmp_path / backend), backend=backend,
                    return_dict=True, **options)['eval_results']


def assert_identical(reference, candidate):
    for key in ('precision', 'recall', 'scores'):
        expected, actual = reference.eval[key], candidate.eval[key]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, key
        assert actual.tobytes() == expected.tobytes(), key
    assert candidate.stats.tobytes() == reference.stats.tobytes()


@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
@pytest.mark.parametrize('max_detections', [1, 100, 500])
@pytest.mark.parametrize('iou_thrs', [None, 0.5, [0.5, 0.75]])
def test_evaluate_matches_pycocotools(tmp_path, captured, iou_type, max_detections, iou_thrs):
    options = dict(type=iou_type, classwise=True, max_detections=max_detections, iou_thrs=iou_thrs)
    reference = run(tmp_path, 'pycocotools', **options)
    candidate = run(tmp_path, 'ultrafast', **options)
    assert candidate == reference
    assert len(captured['pycocotools']) == len(captured['ultrafast']) == 1
    assert_identical(captured['pycocotools'][0], captured['ultrafast'][0])


def test_custom_area_ranges(tmp_path, captured):
    options = dict(classwise=True, areas=[256, 4096, 10_000_000_000])
    assert run(tmp_path, 'ultrafast', **options) == run(tmp_path, 'pycocotools', **options)
    assert_identical(captured['pycocotools'][0], captured['ultrafast'][0])


def test_empty_results(tmp_path):
    result = tmp_path / 'result.json'
    result.write_text('[]')
    assert run(tmp_path, 'ultrafast', result=result) == run(tmp_path, 'pycocotools', result=result) == {}


def test_exported_metrics_match(tmp_path, captured):
    run(tmp_path, 'pycocotools', classwise=True)
    run(tmp_path, 'ultrafast', classwise=True)
    exported = {backend: json.loads((tmp_path / backend / 'eval.json').read_text())
                for backend in ('pycocotools', 'ultrafast')}
    assert exported['ultrafast'] == exported['pycocotools']
    assert np.array_equal(captured['ultrafast'][0].params.iouThrs,
                          captured['pycocotools'][0].params.iouThrs)
