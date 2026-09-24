"""``loadRes`` with an ``Nx7`` NumPy array builds compact columns directly.

The array route must keep pycocotools' semantics: ``int()`` truncation of IDs,
float32 arithmetic for the derived area and box polygon of float32 arrays,
public views equal to ``loadNumpyAnnotations`` + ``loadRes``, and the same
errors for unusable input.
"""
import contextlib
import copy
import io

import numpy as np
import pytest

import ultrafast_pycocotools as ufc
from test_eval_parity import assert_bit_identical


def _inputs(seed=4):
    rng = np.random.default_rng(seed)
    images = [{'id': i, 'height': 160, 'width': 200} for i in range(1, 31)]
    annotations = []
    for i in range(240):
        x, y = [float(v) for v in rng.uniform(0, 140, 2)]
        w, h = [float(v) for v in rng.uniform(3, 60, 2)]
        annotations.append({'id': i + 1, 'image_id': 1 + i % 30, 'category_id': 1 + i % 3,
                            'bbox': [x, y, w, h], 'area': w * h, 'iscrowd': int(i % 37 == 0),
                            'segmentation': [[x, y, x + w, y, x + w, y + h, x, y + h]]})
    rows = []
    for a in annotations:
        for _ in range(3):
            x, y, w, h = a['bbox']
            rows.append([a['image_id'] + rng.uniform(0, 0.99),  # int() truncates the fraction
                         x + rng.normal(0, 3), y + rng.normal(0, 3), w * rng.uniform(.8, 1.2),
                         h * rng.uniform(.8, 1.2), round(float(rng.random()), 2), a['category_id']])
    data = {'images': images, 'annotations': annotations, 'categories': [{'id': c} for c in (1, 2, 3)]}
    return data, np.array(rows)


def _reference(data, rows, iou_type, tweak=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(data)
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(rows), iou_type)
        if tweak:
            tweak(ev.params)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def _ours(data, rows, iou_type, tweak=None):
    gt = ufc.COCO(copy.deepcopy(data), verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(rows)
    ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None)
    if tweak:
        tweak(ev.params)
    ev.run()
    return dt, ev


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
@pytest.mark.parametrize('iou_type', ['bbox', 'segm'])
def test_array_results_match_pycocotools(dtype, iou_type):
    data, rows = _inputs()
    rows = rows.astype(dtype)
    dt, actual = _ours(data, rows, iou_type)
    assert dt._compact is not None and dt._compact.from_array
    assert_bit_identical(_reference(data, rows, iou_type), actual, f'{iou_type} {np.dtype(dtype)}')


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_array_views_equal_the_dictionary_route(dtype):
    data, rows = _inputs()
    rows = rows.astype(dtype)
    gt = ufc.COCO(copy.deepcopy(data), verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        expected = gt.loadRes(rows, derive_segmentation=True)  # the ordinary conversion
        actual = gt.loadRes(rows)
    assert actual._compact is not None
    for key in expected.anns:
        want, got = dict(expected.anns[key]), actual.anns[key]
        want.pop('segmentation')
        assert got.keys() == want.keys()
        for field in want:
            assert type(got[field]) is type(want[field]), field
            assert np.array_equal(np.asarray(got[field]), np.asarray(want[field])), field
        assert [type(v) for v in got['bbox']] == [type(v) for v in want['bbox']]
    assert actual._compact is None  # Views materialize, as for files.


def test_array_results_are_copied_at_load_time():
    data, rows = _inputs()
    reference = _reference(data, rows, 'bbox')
    gt = ufc.COCO(copy.deepcopy(data), verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(rows)
    rows[:, 1:6] = 0.0  # Later edits to the caller's array must not leak in.
    ev = ufc.COCOeval(gt, dt, 'bbox', print_function=lambda *_: None)
    ev.run()
    assert_bit_identical(reference, ev, 'copied rows')


@pytest.mark.parametrize('layout', ['fortran', 'strided'])
def test_non_contiguous_arrays(layout):
    data, rows = _inputs()
    view = np.asfortranarray(rows) if layout == 'fortran' else np.repeat(rows, 2, axis=1)[:, ::2]
    assert not view.flags.c_contiguous
    dt, actual = _ours(data, view, 'bbox')
    assert dt._compact is not None
    assert_bit_identical(_reference(data, np.ascontiguousarray(view), 'bbox'), actual, layout)


def test_ids_beyond_32_bits_and_filtered_evaluation():
    data, rows = _inputs()
    offset = 2.0 ** 40
    for image in data['images']:
        image['id'] += int(offset) if image['id'] % 2 else 0
    for ann in data['annotations']:
        ann['image_id'] += int(offset) if ann['image_id'] % 2 else 0
    odd = np.floor(rows[:, 0]) % 2 == 1
    rows[odd, 0] = np.floor(rows[odd, 0]) + offset

    def tweak(p):
        p.imgIds = sorted(p.imgIds)[::2]
        p.catIds = [1, 3]
        p.maxDets = [1, 5, 100]

    dt, actual = _ours(data, rows, 'bbox', tweak)
    assert dt._compact is not None
    assert_bit_identical(_reference(data, rows, 'bbox', tweak), actual, 'wide ids')


@pytest.mark.parametrize('options', [{'iouType': 'keypoints'},
                                     {'iouType': 'bbox', 'lvis_style': True, 'lvis_protocol': 'coco'},
                                     {'iouType': 'segm', 'lvis_style': True, 'lvis_protocol': 'coco'}],
                         ids=['keypoints', 'lvis-bbox', 'lvis-segm'])
def test_other_evaluations_match_the_dictionary_route(options):
    # Keypoint and LVIS evaluation read array results as dictionaries.
    data, rows = _inputs()
    for category, frequency in zip(data['categories'], 'rcf'):
        category['frequency'] = frequency  # LVIS metadata
    options = dict(options)
    iou_type = options.pop('iouType')
    evaluations = []
    for derive in (False, True):  # True takes the ordinary conversion.
        gt = ufc.COCO(copy.deepcopy(data), verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(rows, derive_segmentation=derive)
        ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None, **options)
        ev.run()
        evaluations.append(ev)
    assert_bit_identical(evaluations[1], evaluations[0], str(options))


@pytest.mark.parametrize('bad', ['nan-id', 'huge-id', 'foreign-image'])
def test_unusable_arrays_fail_like_the_dictionary_route(bad):
    data, rows = _inputs()
    rows = rows.copy()
    if bad == 'nan-id':
        rows[5, 0] = np.nan
    elif bad == 'huge-id':
        rows[5, 6] = 1e30
    else:
        rows[5, 0] = 999
    gt = ufc.COCO(copy.deepcopy(data), verbose=False)

    def outcome(**options):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gt.loadRes(rows, **options)
        except Exception as error:  # noqa: BLE001 - the outcome is compared
            return type(error)
        return None

    assert outcome() is outcome(derive_segmentation=True)


def test_empty_array_and_messages():
    data, rows = _inputs()
    gt = ufc.COCO(copy.deepcopy(data), verbose=False)
    for array in (np.zeros((0, 7)), rows):
        printed, expected = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(printed):
            actual = gt.loadRes(array)
        with contextlib.redirect_stdout(expected):
            reference = gt.loadRes(array, derive_segmentation=True)
        assert printed.getvalue() == expected.getvalue()
        assert len(actual.anns) == len(reference.anns) == len(array)
