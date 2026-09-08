"""The mask API must agree with ``pycocotools.mask`` bit for bit.

Not "to within a tolerance" — the RLE strings must be the same bytes, the
areas the same ``uint32``, the IoU matrices the same doubles. Mask IoU feeds
the greedy matcher, so a single-ULP disagreement can flip a match and move AP.

The awkward inputs are deliberate. ``rleFrPoly`` divides by an edge length
that is zero for a repeated vertex and casts the resulting NaN to ``int``,
which is undefined in C and lands on ``INT_MIN`` in practice; polygons that
leave the image; 1-pixel images; empty and full masks; crowd flags.
"""

from __future__ import annotations

import numpy as np
import pytest
from pycocotools import mask as ref

from ultrafast_pycocotools import mask as ours

RNG = np.random.default_rng(20260814)


def random_mask(h: int, w: int, fill: float = 0.3) -> np.ndarray:
    return np.asfortranarray((RNG.random((h, w)) < fill).astype(np.uint8))


def blob_mask(h: int, w: int) -> np.ndarray:
    """A connected-ish blob, which is what real annotations look like and
    what exercises long runs rather than the pathological alternating case."""
    m = np.zeros((h, w), dtype=np.uint8)
    cy, cx = RNG.integers(0, h), RNG.integers(0, w)
    ry, rx = max(1, h // 4), max(1, w // 4)
    ys, xs = np.ogrid[:h, :w]
    m[((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1.0] = 1
    return np.asfortranarray(m)


SHAPES = [(1, 1), (1, 17), (17, 1), (8, 8), (33, 27), (64, 96)]


@pytest.mark.parametrize("h,w", SHAPES)
def test_encode_decode_roundtrip_matches_reference(h, w):
    for maker in (random_mask, blob_mask):
        m = maker(h, w)
        a, b = ref.encode(m), ours.encode(m)
        assert a["size"] == b["size"]
        assert a["counts"] == b["counts"], (h, w, maker.__name__)
        assert isinstance(b["counts"], bytes)
        np.testing.assert_array_equal(ref.decode(a), ours.decode(b))


@pytest.mark.parametrize("h,w", SHAPES)
def test_degenerate_masks(h, w):
    for m in (
        np.asfortranarray(np.zeros((h, w), np.uint8)),
        np.asfortranarray(np.ones((h, w), np.uint8)),
    ):
        assert ref.encode(m)["counts"] == ours.encode(m)["counts"]
        assert ref.area(ref.encode(m)) == ours.area(ours.encode(m))
        np.testing.assert_array_equal(ref.toBbox(ref.encode(m)), ours.toBbox(ours.encode(m)))


def test_area_dtype_and_value():
    m = blob_mask(40, 50)
    a, b = ref.area(ref.encode(m)), ours.area(ours.encode(m))
    assert a == b
    assert np.asarray(a).dtype == np.asarray(b).dtype == np.uint32


def test_encode_3d_stack():
    stack = np.asfortranarray(np.stack([blob_mask(24, 31) for _ in range(5)], axis=2))
    a, b = ref.encode(stack), ours.encode(stack)
    assert [x["counts"] for x in a] == [x["counts"] for x in b]
    np.testing.assert_array_equal(ref.decode(a), ours.decode(b))


@pytest.mark.parametrize("shape", [(13, 17, 3), (1, 17, 3), (13, 1, 3),
                                   (13, 17, 1), (0, 17, 3), (13, 0, 3), (13, 17, 0)])
@pytest.mark.parametrize("layout", ["fortran", "c", "reverse_rows", "reverse_masks",
                                    "step_rows", "step_masks", "offset"])
def test_encode_layouts_match_reference(shape, layout):
    """Contiguous fast paths must retain the arbitrary-stride API extension."""
    rng = np.random.default_rng(90210)
    base = np.asfortranarray(rng.integers(0, 4, size=shape, dtype=np.uint8))
    views = {
        "fortran": base,
        "c": np.ascontiguousarray(base),
        "reverse_rows": base[::-1, :, :],
        "reverse_masks": base[:, :, ::-1],
        "step_rows": base[::2, :, :],
        "step_masks": base[:, :, ::2],
        "offset": base[:, :, 1:],
    }
    masks = views[layout]
    before = masks.copy()
    masks.flags.writeable = False
    # Cython requires Fortran layout; our API additionally accepts strided arrays.
    expected = ref.encode(np.asfortranarray(masks))
    assert ours.encode(masks) == expected
    np.testing.assert_array_equal(masks, before)
    assert not masks.flags.writeable


@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 63, 64, 65, 257])
def test_encode_run_boundaries_match_reference(length):
    for split in range(length + 1):
        for first, second in [(0, 1), (1, 0), (2, 255), (255, 2), (255, 255)]:
            mask = np.full((length, 1), first, dtype=np.uint8, order="F")
            mask[split:] = second
            assert ours.encode(mask) == ref.encode(mask)


@pytest.mark.parametrize("intersect", [0, 1])
def test_merge(intersect):
    rles = [ref.encode(blob_mask(40, 40)) for _ in range(4)]
    assert ref.merge(rles, intersect)["counts"] == ours.merge(rles, intersect)["counts"]


def test_merge_shape_mismatch_collapses():
    a = ref.encode(blob_mask(10, 10))
    b = ref.encode(blob_mask(12, 12))
    assert ref.merge([a, b])["counts"] == ours.merge([a, b])["counts"]


POLYS = [
    # A plain quadrilateral.
    [[10.0, 10.0, 10.0, 40.0, 45.0, 40.0, 45.0, 10.0]],
    # Sub-pixel coordinates, which is where the x5 upsample rounding shows.
    [[1.3, 2.7, 1.3, 9.1, 8.8, 9.1, 8.8, 2.7]],
    # A repeated vertex: the traced edge has zero length, rleFrPoly divides
    # by it, and the NaN gets cast to int.
    [[5.0, 5.0, 5.0, 5.0, 5.0, 20.0, 20.0, 20.0, 20.0, 5.0]],
    # Partly outside the image.
    [[-8.0, -8.0, -8.0, 30.0, 30.0, 30.0, 30.0, -8.0]],
    # Two rings, which annToRLE unions.
    [
        [2.0, 2.0, 2.0, 10.0, 10.0, 10.0, 10.0, 2.0],
        [20.0, 20.0, 20.0, 28.0, 28.0, 28.0, 28.0, 20.0],
    ],
    # A triangle with collinear points.
    [[0.0, 0.0, 15.0, 15.0, 30.0, 30.0, 0.0, 30.0]],
]


@pytest.mark.parametrize("poly", POLYS)
@pytest.mark.parametrize("hw", [(32, 32), (48, 60), (7, 9)])
def test_frpoly(poly, hw):
    h, w = hw
    a, b = ref.frPyObjects(poly, h, w), ours.frPyObjects(poly, h, w)
    assert [x["counts"] for x in a] == [x["counts"] for x in b]
    assert ref.merge(a)["counts"] == ours.merge(b)["counts"]


def test_frbbox_and_single_object_forms():
    boxes = [[1.0, 2.0, 10.0, 12.0], [0.0, 0.0, 3.5, 4.5]]
    # pycocotools routes a *list* of boxes to frBbox, which is typed to accept
    # only an ndarray, so upstream raises TypeError here. We accept it. That
    # is a deliberate widening: no result pycocotools could produce changes.
    with pytest.raises(TypeError):
        ref.frPyObjects(boxes, 40, 40)
    got = ours.frPyObjects(boxes, 40, 40)
    want = ref.frPyObjects(np.array(boxes, dtype=np.float64), 40, 40)
    assert [x["counts"] for x in got] == [x["counts"] for x in want]

    arr = np.array(boxes, dtype=np.float64)
    a3, b3 = ref.frPyObjects(arr, 40, 40), ours.frPyObjects(arr, 40, 40)
    assert [x["counts"] for x in a3] == [x["counts"] for x in b3]


def test_frpyobjects_single_object_forms_are_dead_code_upstream():
    """A bare ``[x, y, w, h]`` or flat polygon is documented but unreachable.

    ``frPyObjects`` tests ``len(pyobj[0]) == 4`` before it ever reaches the
    single-object branches, and ``pyobj[0]`` is a float there, so upstream
    raises ``TypeError``. We implement the documented behaviour instead, and
    check it against the list-wrapped form that upstream *can* evaluate.
    """
    box = [1.0, 2.0, 10.0, 12.0]
    poly = [3.0, 3.0, 3.0, 20.0, 25.0, 20.0, 25.0, 3.0]
    for obj in (box, poly):
        with pytest.raises(TypeError):
            ref.frPyObjects(obj, 40, 40)

    got = ours.frPyObjects(box, 40, 40)
    want = ref.frPyObjects(np.array([box], dtype=np.float64), 40, 40)[0]
    assert got["counts"] == want["counts"]

    got = ours.frPyObjects(poly, 40, 40)
    want = ref.frPyObjects([poly], 40, 40)[0]
    assert got["counts"] == want["counts"]


def test_fruncompressed():
    m = blob_mask(20, 25)
    rle = ref.encode(m)
    counts = []
    # Rebuild the uncompressed run list from the decoded mask.
    flat = m.flatten(order="F")
    run, prev = 0, 0
    for v in flat:
        if v != prev:
            counts.append(run)
            run, prev = 0, v
        run += 1
    counts.append(run)
    uc = {"size": [20, 25], "counts": counts}
    a, b = ref.frPyObjects(uc, 20, 25), ours.frPyObjects(uc, 20, 25)
    assert a["counts"] == b["counts"] == rle["counts"]


@pytest.mark.parametrize("crowd", [[0, 0, 0], [1, 0, 1], [1, 1, 1]])
def test_iou_boxes(crowd):
    dt = np.array(
        [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 10.0, 10.0], [50.0, 50.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]]
    )
    gt = np.array([[0.0, 0.0, 10.0, 10.0], [8.0, 8.0, 20.0, 20.0], [100.0, 100.0, 5.0, 5.0]])
    a = np.asarray(ref.iou(dt, gt, crowd))
    b = np.asarray(ours.iou(dt, gt, crowd))
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes(), "box IoU must be bit-identical"


@pytest.mark.parametrize("crowd", [[0, 0], [1, 0]])
def test_iou_masks(crowd):
    d = [ref.encode(blob_mask(50, 60)) for _ in range(3)]
    g = [ref.encode(blob_mask(50, 60)) for _ in range(2)]
    a = np.asarray(ref.iou(d, g, crowd))
    b = np.asarray(ours.iou(d, g, crowd))
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes(), "mask IoU must be bit-identical"


def test_iou_empty_returns_list():
    # cocoeval checks `len(ious) == 0`, so an empty ndarray would not do.
    assert ours.iou([], [], []) == []
    assert ours.iou(np.zeros((0, 4)), np.zeros((2, 4)), [0, 0]) == []


def test_iou_size_mismatch_marks_negative():
    d = [ref.encode(np.asfortranarray(np.ones((10, 10), np.uint8)))]
    g = [ref.encode(np.asfortranarray(np.ones((12, 12), np.uint8)))]
    a = np.asarray(ref.iou(d, g, [0]))
    b = np.asarray(ours.iou(d, g, [0]))
    assert a.tobytes() == b.tobytes()


def test_tobbox_matches():
    rles = [ref.encode(blob_mask(37, 41)) for _ in range(6)]
    np.testing.assert_array_equal(ref.toBbox(rles), ours.toBbox(rles))
    np.testing.assert_array_equal(ref.toBbox(rles[0]), ours.toBbox(rles[0]))


def test_counts_string_accepted_like_reference():
    """Result files carry ``counts`` as ``str`` after a JSON round-trip."""
    rle = ref.encode(blob_mask(30, 30))
    as_str = {"size": rle["size"], "counts": rle["counts"].decode("ascii")}
    np.testing.assert_array_equal(ref.decode(as_str), ours.decode(as_str))
    assert ref.area(as_str) == ours.area(as_str)
