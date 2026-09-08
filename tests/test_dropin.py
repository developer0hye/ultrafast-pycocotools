"""Can this actually replace pycocotools?

The parity suite proves the *numbers* match. This one asks a different
question: does code written against pycocotools keep working when the import
is swapped? Those fail in completely different ways — an evaluator can produce
identical AP and still break every caller by returning a `str` where a `bytes`
was expected, or by not having the method a subclass overrides.

So this suite is deliberately adversarial. It walks the reference package's
public surface rather than a list we wrote ourselves (a hand-written list only
ever contains the things we remembered to implement), checks types and not
just values, and reproduces the calling patterns of the libraries people
actually use: torchvision's `CocoEvaluator`, and the mmdetection habit of
subclassing `COCOeval` to override `evaluateImg`.

Deliberate, documented differences are asserted *as* differences, so that if
one silently disappears the test says so.
"""

from __future__ import annotations

import inspect
import io
import json
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest

import ultrafast_pycocotools as ufc
import ultrafast_pycocotools.coco as our_coco
import ultrafast_pycocotools.cocoeval as our_cocoeval
import ultrafast_pycocotools.mask as our_mask

ref_coco = pytest.importorskip("pycocotools.coco")
ref_cocoeval = pytest.importorskip("pycocotools.cocoeval")
ref_mask = pytest.importorskip("pycocotools.mask")


# ----------------------------------------------------------------------
# API surface
# ----------------------------------------------------------------------


def public_callables(cls) -> list[str]:
    return sorted(
        n
        for n in dir(cls)
        if not n.startswith("__") and callable(getattr(cls, n, None))
    )


@pytest.mark.parametrize(
    "ref_cls,our_cls,label",
    [
        (ref_coco.COCO, our_coco.COCO, "COCO"),
        (ref_cocoeval.COCOeval, our_cocoeval.COCOeval, "COCOeval"),
        (ref_cocoeval.Params, our_cocoeval.Params, "Params"),
    ],
)
def test_every_public_method_exists(ref_cls, our_cls, label):
    """Enumerated from the reference, not from a list we maintain.

    A hand-written list only contains what we remembered; this notices a
    method we never knew existed.
    """
    missing = [n for n in public_callables(ref_cls) if not hasattr(our_cls, n)]
    assert not missing, f"{label} is missing {missing}"


@pytest.mark.parametrize(
    "ref_cls,our_cls,label",
    [
        (ref_coco.COCO, our_coco.COCO, "COCO"),
        (ref_cocoeval.COCOeval, our_cocoeval.COCOeval, "COCOeval"),
    ],
)
def test_no_method_gained_a_required_parameter(ref_cls, our_cls, label):
    """Extra keyword arguments with defaults are fine; new required ones are not."""
    for name in public_callables(ref_cls):
        try:
            ref_sig = inspect.signature(getattr(ref_cls, name))
            our_sig = inspect.signature(getattr(our_cls, name))
        except (TypeError, ValueError):
            continue
        dropped = [p for p in ref_sig.parameters if p not in our_sig.parameters]
        assert not dropped, f"{label}.{name} no longer accepts {dropped}"
        added_required = [
            p
            for p, v in our_sig.parameters.items()
            if p not in ref_sig.parameters
            and v.default is inspect.Parameter.empty
            and v.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        assert not added_required, f"{label}.{name} now requires {added_required}"


@pytest.mark.parametrize("iou_type", ["bbox", "segm", "keypoints"])
def test_params_attributes(iou_type):
    ref = ref_cocoeval.Params(iouType=iou_type)
    ours = our_cocoeval.Params(iouType=iou_type)
    missing = [n for n in dir(ref) if not n.startswith("__") and not hasattr(ours, n)]
    assert not missing, f"Params({iou_type}) is missing {missing}"
    for name in ("iouThrs", "recThrs", "maxDets", "areaRng", "areaRngLbl", "useCats"):
        a, b = getattr(ref, name), getattr(ours, name)
        if isinstance(a, np.ndarray):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), name
        else:
            assert a == b, name


def test_mask_module_exports():
    missing = [
        n
        for n in dir(ref_mask)
        if not n.startswith("_")
        and callable(getattr(ref_mask, n))
        and not hasattr(our_mask, n)
    ]
    assert not missing, f"mask module is missing {missing}"


# ----------------------------------------------------------------------
# COCO behaviour, method by method
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair(synthetic):
    """Reference and our handle over the same files, loaded independently.

    Separate loads matter: pycocotools rewrites annotations during evaluation,
    so a shared handle would let one implementation change the other's input.
    """
    gt_path, dt_path = synthetic
    ref_gt = ref_coco.COCO(str(gt_path))
    ref_dt = ref_gt.loadRes(str(dt_path))
    our_gt = ufc.COCO(str(gt_path), verbose=False)
    our_dt = our_gt.loadRes(str(dt_path))
    return ref_gt, ref_dt, our_gt, our_dt


def test_index_structures_match(pair):
    ref_gt, _, our_gt, _ = pair
    assert sorted(ref_gt.anns) == sorted(our_gt.anns)
    assert sorted(ref_gt.imgs) == sorted(our_gt.imgs)
    assert sorted(ref_gt.cats) == sorted(our_gt.cats)
    assert sorted(ref_gt.imgToAnns) == sorted(our_gt.imgToAnns)
    assert sorted(ref_gt.catToImgs) == sorted(our_gt.catToImgs)
    for k in ref_gt.imgToAnns:
        assert [a["id"] for a in ref_gt.imgToAnns[k]] == [
            a["id"] for a in our_gt.imgToAnns[k]
        ], f"imgToAnns[{k}] order differs — tie-breaking depends on it"
    for k in ref_gt.catToImgs:
        assert ref_gt.catToImgs[k] == our_gt.catToImgs[k], k


def test_get_ann_ids_every_filter_combination(pair):
    """Order matters, not just membership: it decides score-tie resolution."""
    ref_gt, _, our_gt, _ = pair
    img_ids = sorted(ref_gt.getImgIds())[:25]
    cat_ids = sorted(ref_gt.getCatIds())[:4]
    cases = [
        {},
        {"imgIds": img_ids},
        {"catIds": cat_ids},
        {"imgIds": img_ids, "catIds": cat_ids},
        {"areaRng": [100, 10000]},
        {"imgIds": img_ids, "areaRng": [0, 1e9]},
        {"iscrowd": True},
        {"iscrowd": False},
        {"imgIds": img_ids, "catIds": cat_ids, "areaRng": [0, 5000], "iscrowd": False},
        {"imgIds": img_ids[0]},  # scalar, not a list
        {"catIds": cat_ids[0]},
        {"imgIds": [-1]},  # absent image
        {"imgIds": []},
    ]
    for kw in cases:
        assert ref_gt.getAnnIds(**kw) == our_gt.getAnnIds(**kw), kw


def test_get_cat_ids_and_img_ids(pair):
    ref_gt, _, our_gt, _ = pair
    names = [c["name"] for c in ref_gt.dataset["categories"][:3]]
    sups = list({c["supercategory"] for c in ref_gt.dataset["categories"]})
    ids = sorted(ref_gt.getCatIds())[:3]
    for kw in ({}, {"catNms": names}, {"supNms": sups}, {"catIds": ids},
               {"catNms": names, "catIds": ids}, {"catNms": names[0]}):
        assert ref_gt.getCatIds(**kw) == our_gt.getCatIds(**kw), kw
    for kw in ({}, {"imgIds": sorted(ref_gt.getImgIds())[:5]}, {"catIds": ids[:1]},
               {"catIds": ids}):
        assert sorted(ref_gt.getImgIds(**kw)) == sorted(our_gt.getImgIds(**kw)), kw


def test_load_helpers_return_the_same_objects(pair):
    ref_gt, _, our_gt, _ = pair
    ann_ids = sorted(ref_gt.getAnnIds())[:20]
    assert [a["id"] for a in ref_gt.loadAnns(ann_ids)] == [
        a["id"] for a in our_gt.loadAnns(ann_ids)
    ]
    # Scalar id is accepted and returns a one-element list.
    assert len(our_gt.loadAnns(ann_ids[0])) == 1
    assert len(our_gt.loadCats(sorted(ref_gt.getCatIds())[0])) == 1
    assert len(our_gt.loadImgs(sorted(ref_gt.getImgIds())[0])) == 1
    # loadAnns must hand back the *stored* dict, not a copy: callers mutate it.
    got = our_gt.loadAnns([ann_ids[0]])[0]
    assert got is our_gt.anns[ann_ids[0]]


def test_ann_to_rle_and_mask(pair):
    ref_gt, _, our_gt, _ = pair
    for ann_id in sorted(ref_gt.getAnnIds())[:40]:
        r = ref_gt.annToRLE(ref_gt.anns[ann_id])
        o = our_gt.annToRLE(our_gt.anns[ann_id])
        assert r["size"] == list(o["size"])
        assert r["counts"] == o["counts"], ann_id
        assert isinstance(o["counts"], bytes), "counts must stay bytes"
        np.testing.assert_array_equal(
            ref_gt.annToMask(ref_gt.anns[ann_id]),
            our_gt.annToMask(our_gt.anns[ann_id]),
        )


def test_load_res_derived_fields(pair):
    ref_gt, ref_dt, our_gt, our_dt = pair
    assert sorted(ref_dt.anns) == sorted(our_dt.anns)
    for ann_id in sorted(ref_dt.anns)[:50]:
        r, o = ref_dt.anns[ann_id], our_dt.anns[ann_id]
        assert r["id"] == o["id"]
        assert r["area"] == o["area"], "area decides the small/medium/large bucket"
        assert r["iscrowd"] == o["iscrowd"]
        assert r["bbox"] == o["bbox"]
        # Box polygons are derived on demand by default; public masks stay identical.
        if "segmentation" in o:
            assert r["segmentation"] == o["segmentation"]
        else:
            np.testing.assert_array_equal(ref_dt.annToMask(r), our_dt.annToMask(o))
    assert [i["id"] for i in ref_dt.dataset["images"]] == [
        i["id"] for i in our_dt.dataset["images"]
    ]
    assert ref_dt.dataset["categories"] == our_dt.dataset["categories"]


def test_load_res_from_numpy_array(synthetic):
    """torchvision hands results in as an Nx7 array."""
    gt_path, dt_path = synthetic
    dets = json.loads(dt_path.read_text())[:500]
    arr = np.array(
        [
            [d["image_id"], *d["bbox"], d["score"], d["category_id"]]
            for d in dets
        ],
        dtype=np.float64,
    )
    ref_gt = ref_coco.COCO(str(gt_path))
    our_gt = ufc.COCO(str(gt_path), verbose=False)
    with redirect_stdout(io.StringIO()):
        r = ref_gt.loadRes(arr.copy())
        o = our_gt.loadRes(arr.copy())
    assert sorted(r.anns) == sorted(o.anns)
    for k in sorted(r.anns)[:50]:
        assert r.anns[k]["bbox"] == o.anns[k]["bbox"]
        assert r.anns[k]["score"] == o.anns[k]["score"]
        assert r.anns[k]["category_id"] == o.anns[k]["category_id"]


def test_load_res_accepts_a_path_and_a_list(synthetic):
    gt_path, dt_path = synthetic
    our_gt = ufc.COCO(str(gt_path), verbose=False)
    from_path = our_gt.loadRes(str(dt_path))
    from_list = our_gt.loadRes(json.loads(dt_path.read_text()))
    assert sorted(from_path.anns) == sorted(from_list.anns)


def test_load_res_rejects_foreign_image_ids(synthetic):
    gt_path, _ = synthetic
    our_gt = ufc.COCO(str(gt_path), verbose=False)
    with pytest.raises(AssertionError):
        our_gt.loadRes([{"image_id": 10**9, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1.0}])


def test_info_prints(pair):
    _, _, our_gt, _ = pair
    buf = io.StringIO()
    with redirect_stdout(buf):
        our_gt.info()
    assert buf.getvalue().strip(), "info() should print the dataset info block"


# ----------------------------------------------------------------------
# COCOeval behaviour beyond the summary numbers
# ----------------------------------------------------------------------


def run_pair(gt_path, dt_path, iou_type):
    ref_gt = ref_coco.COCO(str(gt_path))
    ref_dt = ref_gt.loadRes(str(dt_path))
    ref = ref_cocoeval.COCOeval(ref_gt, ref_dt, iou_type)
    with redirect_stdout(io.StringIO()):
        ref.evaluate()
        ref.accumulate()
        ref.summarize()

    our_gt = ufc.COCO(str(gt_path), verbose=False)
    our_dt = our_gt.loadRes(str(dt_path))
    ours = ufc.COCOeval(our_gt, our_dt, iou_type, print_function=lambda *_: None)
    ours.run()
    return ref, ours


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_compute_iou_matches(synthetic, iou_type):
    """``computeIoU`` is public and people call it directly."""
    gt_path, dt_path = synthetic
    ref, ours = run_pair(gt_path, dt_path, iou_type)
    checked = 0
    for img_id in sorted(ref.params.imgIds)[:40]:
        for cat_id in sorted(ref.params.catIds)[:5]:
            a = np.asarray(ref.computeIoU(img_id, cat_id), dtype=np.float64)
            b = np.asarray(ours.computeIoU(img_id, cat_id), dtype=np.float64)
            assert a.shape == b.shape, (img_id, cat_id)
            assert a.tobytes() == b.tobytes(), (img_id, cat_id)
            checked += a.size
    assert checked > 0, "the fixture produced no overlapping (image, category)"


def test_compute_oks_matches(synthetic_kp):
    gt_path, dt_path = synthetic_kp
    ref, ours = run_pair(gt_path, dt_path, "keypoints")
    checked = 0
    for img_id in sorted(ref.params.imgIds)[:40]:
        for cat_id in ref.params.catIds:
            a = np.asarray(ref.computeOks(img_id, cat_id), dtype=np.float64)
            b = np.asarray(ours.computeOks(img_id, cat_id), dtype=np.float64)
            assert a.shape == b.shape, (img_id, cat_id)
            # Bit-exact, like the bbox sibling above. `atol=1e-12` used to sit
            # here and it let a real divergence through: OKS applies its three
            # divisions one at a time *because* fusing them is not the same in
            # floating point (see eval.rs), and that difference is ~1 ULP —
            # comfortably inside the old tolerance.
            assert a.tobytes() == b.tobytes(), (img_id, cat_id)
            checked += a.size
    assert checked > 0


def test_evaluate_img_matches(synthetic):
    """The method mmdetection-style subclasses override."""
    gt_path, dt_path = synthetic
    ref, ours = run_pair(gt_path, dt_path, "bbox")
    a_rng = ref.params.areaRng[0]
    max_det = ref.params.maxDets[-1]
    seen = 0
    for img_id in sorted(ref.params.imgIds)[:40]:
        for cat_id in sorted(ref.params.catIds)[:5]:
            r = ref.evaluateImg(img_id, cat_id, a_rng, max_det)
            o = ours.evaluateImg(img_id, cat_id, a_rng, max_det)
            if r is None or o is None:
                assert r is None and o is None, (img_id, cat_id)
                continue
            assert list(r["dtIds"]) == list(o["dtIds"])
            assert list(r["gtIds"]) == list(o["gtIds"])
            assert list(r["dtScores"]) == list(o["dtScores"])
            np.testing.assert_array_equal(r["dtMatches"], o["dtMatches"])
            np.testing.assert_array_equal(r["gtMatches"], o["gtMatches"])
            np.testing.assert_array_equal(r["gtIgnore"], o["gtIgnore"])
            np.testing.assert_array_equal(r["dtIgnore"], o["dtIgnore"])
            seen += 1
    assert seen > 0


def test_ious_dict_is_available_and_correct(synthetic):
    gt_path, dt_path = synthetic
    ref, ours = run_pair(gt_path, dt_path, "bbox")
    assert set(ref.ious) == set(ours.ious)
    for key, a in ref.ious.items():
        b = ours.ious[key]
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        assert a.shape == b.shape, key
        assert a.tobytes() == b.tobytes(), key


def test_gts_and_dts_views(synthetic):
    """Contents must match, and a missing key must still yield ``[]``.

    Key *sets* deliberately are not compared: both sides are ``defaultdict``s,
    so every lookup during evaluation inserts an empty entry, and upstream
    queries all image x category pairs while we only touch populated ones.
    That difference is an artefact of when each side happened to be asked, not
    of what either holds.
    """
    gt_path, dt_path = synthetic
    ref, ours = run_pair(gt_path, dt_path, "bbox")
    ours._prepare()

    for name in ("_gts", "_dts"):
        r, o = getattr(ref, name), getattr(ours, name)
        populated_ref = {k for k, v in r.items() if v}
        populated_our = {k for k, v in o.items() if v}
        assert populated_ref == populated_our, name
        for key in populated_ref:
            assert [a["id"] for a in r[key]] == [a["id"] for a in o[key]], (name, key)
        # defaultdict semantics: callers index straight in without checking.
        assert o[(10**9, 10**9)] == []


def test_summarize_output_is_character_identical(synthetic):
    """People grep this. The format string, spacing and rounding all count."""
    gt_path, dt_path = synthetic
    ref_gt = ref_coco.COCO(str(gt_path))
    ref_dt = ref_gt.loadRes(str(dt_path))
    ref = ref_cocoeval.COCOeval(ref_gt, ref_dt, "bbox")
    buf_ref = io.StringIO()
    with redirect_stdout(io.StringIO()):
        ref.evaluate()
        ref.accumulate()
    with redirect_stdout(buf_ref):
        ref.summarize()

    our_gt = ufc.COCO(str(gt_path), verbose=False)
    our_dt = our_gt.loadRes(str(dt_path))
    ours = ufc.COCOeval(our_gt, our_dt, "bbox")
    buf_our = io.StringIO()
    with redirect_stdout(io.StringIO()):
        ours.evaluate()
        ours.accumulate()
    with redirect_stdout(buf_our):
        ours.summarize()

    assert buf_our.getvalue() == buf_ref.getvalue()


def test_stats_is_a_numpy_array_of_the_right_length(synthetic):
    _, ours = run_pair(*synthetic, "bbox")
    assert isinstance(ours.stats, np.ndarray)
    assert ours.stats.shape == (12,)
    assert ours.stats.dtype == np.float64


def test_eval_dict_shape_and_keys(synthetic):
    ref, ours = run_pair(*synthetic, "bbox")
    assert set(ref.eval) <= set(ours.eval), set(ref.eval) - set(ours.eval)
    for k in ("precision", "recall", "scores"):
        assert ours.eval[k].shape == ref.eval[k].shape
        assert ours.eval[k].dtype == np.float64
    assert list(ours.eval["counts"]) == list(ref.eval["counts"])
    assert ours.eval["params"] is ours.params


def test_params_eval_is_a_snapshot(synthetic):
    _, ours = run_pair(*synthetic, "bbox")
    assert ours._paramsEval is not None
    assert ours._paramsEval is not ours.params
    assert list(ours._paramsEval.maxDets) == list(ours.params.maxDets)


# ----------------------------------------------------------------------
# calling patterns from real downstream libraries
# ----------------------------------------------------------------------


def test_subclass_overriding_evaluate_img(synthetic):
    """The mmdetection pattern: subclass and override a per-image hook.

    The override must actually be reachable — if the engine bypassed it the
    subclass would silently do nothing, which is worse than a crash.
    """
    gt_path, dt_path = synthetic
    calls = []

    class Counting(ufc.COCOeval):
        def evaluateImg(self, imgId, catId, aRng, maxDet):
            calls.append((imgId, catId))
            return super().evaluateImg(imgId, catId, aRng, maxDet)

    gt = ufc.COCO(str(gt_path), verbose=False)
    dt = gt.loadRes(str(dt_path))
    ev = Counting(gt, dt, "bbox", print_function=lambda *_: None)
    ev.evaluate()
    img_ids = sorted(ev.params.imgIds)[:5]
    for img_id in img_ids:
        ev.evaluateImg(img_id, ev.params.catIds[0], ev.params.areaRng[0], 100)
    assert len(calls) == len(img_ids)


def test_torchvision_style_accumulation_loop(synthetic):
    """torchvision's CocoEvaluator: one loadRes per batch, then evaluate once."""
    gt_path, dt_path = synthetic
    dets = json.loads(dt_path.read_text())
    gt = ufc.COCO(str(gt_path), verbose=False)

    merged: list = []
    for i in range(0, len(dets), 500):
        batch = dets[i : i + 500]
        merged.extend(batch)
    dt = gt.loadRes(merged)
    ev = ufc.COCOeval(gt, dt, "bbox", print_function=lambda *_: None)
    ev.params.imgIds = sorted({d["image_id"] for d in merged})
    ev.run()
    assert np.isfinite(ev.stats).all()


def test_init_as_pycocotools_serves_a_full_run(synthetic, monkeypatch):
    """The path a third-party library takes: it imports `pycocotools` itself."""
    gt_path, dt_path = synthetic
    saved = {k: v for k, v in sys.modules.items() if k.startswith("pycocotools")}
    try:
        for k in list(sys.modules):
            if k.startswith("pycocotools"):
                del sys.modules[k]
        ufc.init_as_pycocotools()

        from pycocotools import mask as patched_mask
        from pycocotools.coco import COCO as PatchedCOCO
        from pycocotools.cocoeval import COCOeval as PatchedEval

        assert PatchedCOCO is ufc.COCO
        assert patched_mask.encode is our_mask.encode

        gt = PatchedCOCO(str(gt_path))
        dt = gt.loadRes(str(dt_path))
        ev = PatchedEval(gt, dt, "bbox")
        with redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        assert len(ev.stats) == 12
    finally:
        for k in list(sys.modules):
            if k.startswith("pycocotools"):
                del sys.modules[k]
        sys.modules.update(saved)


# ----------------------------------------------------------------------
# documented differences, asserted as differences
# ----------------------------------------------------------------------


def test_annotations_are_not_rewritten_to_rle(synthetic):
    """Upstream edits the caller's data during `_prepare`; we do not.

    Asserted so the difference cannot disappear unnoticed in either direction.
    """
    gt_path, dt_path = synthetic
    ref_gt = ref_coco.COCO(str(gt_path))
    ref_dt = ref_gt.loadRes(str(dt_path))
    ref = ref_cocoeval.COCOeval(ref_gt, ref_dt, "segm")
    ann_id = sorted(ref_gt.anns)[0]
    assert isinstance(ref_gt.anns[ann_id]["segmentation"], list)
    with redirect_stdout(io.StringIO()):
        ref.evaluate()
    assert isinstance(ref_gt.anns[ann_id]["segmentation"], dict), (
        "reference behaviour changed; revisit this note"
    )

    our_gt = ufc.COCO(str(gt_path), verbose=False)
    our_dt = our_gt.loadRes(str(dt_path))
    ours = ufc.COCOeval(our_gt, our_dt, "segm", print_function=lambda *_: None)
    ours.evaluate()
    assert isinstance(our_gt.anns[ann_id]["segmentation"], list), (
        "we should have left the caller's annotation alone"
    )


def test_eval_imgs_is_opt_in(synthetic):
    gt_path, dt_path = synthetic
    gt = ufc.COCO(str(gt_path), verbose=False)
    dt = gt.loadRes(str(dt_path))
    lean = ufc.COCOeval(gt, dt, "bbox", print_function=lambda *_: None)
    lean.evaluate()
    assert lean.evalImgs == []

    full = ufc.COCOeval(
        gt, dt, "bbox", store_eval_imgs=True, print_function=lambda *_: None
    )
    full.evaluate()
    assert len(full.evalImgs) > 0


def test_accumulate_with_foreign_params_is_refused(synthetic):
    """Upstream silently accepts a different `p`; we refuse rather than
    quietly evaluate against parameters the run never used."""
    gt_path, dt_path = synthetic
    gt = ufc.COCO(str(gt_path), verbose=False)
    dt = gt.loadRes(str(dt_path))
    ev = ufc.COCOeval(gt, dt, "bbox", print_function=lambda *_: None)
    ev.evaluate()
    with pytest.raises(NotImplementedError):
        ev.accumulate(our_cocoeval.Params(iouType="bbox"))
