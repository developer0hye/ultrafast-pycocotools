"""``metrics.DetectionMetrics``: per-batch and accumulated precision/recall/F1.

The reference is ``COCOeval.detection_metrics`` (itself pinned to
``eval['recall']`` in ``test_detection_metrics.py``) on the same images with
the same parameters. Accumulating batches must reproduce one evaluation over
all of them bit for bit, and array input must reproduce COCO input.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import ultrafast_pycocotools as ufc
from ultrafast_pycocotools import mask as maskUtils
from ultrafast_pycocotools.metrics import _UNLIMITED, DetectionMetrics, detection_metrics


def _quiet(*_):
    pass


@pytest.fixture(scope="module")
def data(synthetic):
    gt_path, dt_path = synthetic
    return json.loads(gt_path.read_text()), json.loads(dt_path.read_text())


def _subset(gt, dets, image_ids):
    keep = set(image_ids)
    return (
        {
            "images": [im for im in gt["images"] if im["id"] in keep],
            "annotations": [a for a in gt["annotations"] if a["image_id"] in keep],
            "categories": gt["categories"],
        },
        [copy.deepcopy(d) for d in dets if d["image_id"] in keep],
    )


def _reference(gt, dets, iou_type, iou_thrs, score_thr=None, optimize="macro",
               class_agnostic=False, area_range=None, max_dets=None):
    coco = ufc.COCO(gt, verbose=False)
    ev = ufc.COCOeval(coco, coco.loadRes(copy.deepcopy(dets)), iou_type, print_function=_quiet)
    p = ev.params
    p.iouThrs = np.asarray(iou_thrs, dtype=np.float64)
    p.areaRng = [list(area_range) if area_range is not None else [0.0, float("inf")]]
    p.areaRngLbl = ["all"]
    p.maxDets = [_UNLIMITED if max_dets is None else max_dets]
    p.useCats = 0 if class_agnostic else 1
    ev.evaluate()
    return ev.detection_metrics(iou_thr=list(iou_thrs), score_thr=score_thr, optimize=optimize)


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            assert a[key].keys() == b[key].keys()
            for name in a[key]:
                assert np.array_equal(a[key][name], b[key][name]), (key, name)
        else:
            assert np.array_equal(a[key], b[key]), key


def _batches(gt, size):
    ids = sorted(im["id"] for im in gt["images"])
    return [ids[i:i + size] for i in range(0, len(ids), size)]


CONFIGS = [
    {},
    {"score_thr": 0.4},
    {"score_thr": "best_f1"},
    {"score_thr": "best_f1", "optimize": "micro"},
    {"class_agnostic": True, "score_thr": "best_f1"},
    {"area_range": (32.0**2, 96.0**2)},
    {"max_dets": 3, "score_thr": "best_f1"},
]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()) or "default")
def test_accumulated_batches_equal_one_evaluation(data, config):
    gt, dets = data
    thrs = [0.5, 0.75]
    metric = DetectionMetrics(iou_thrs=thrs, **config)
    for ids in _batches(gt, 32):
        batch_gt, batch_dt = _subset(gt, dets, ids)
        got = metric.update_coco(batch_gt, batch_dt)
        _assert_same(got, _reference(batch_gt, batch_dt, "bbox", thrs, **config))
    _assert_same(metric.compute(), _reference(gt, dets, "bbox", thrs, **config))


def test_accumulated_segm_batches_equal_one_evaluation(data):
    gt, dets = data
    metric = DetectionMetrics(iou_type="segm", iou_thrs=[0.5, 0.75], score_thr="best_f1")
    for ids in _batches(gt, 45):
        metric.update_coco(*_subset(gt, dets, ids))
    _assert_same(metric.compute(), _reference(gt, dets, "segm", [0.5, 0.75], score_thr="best_f1"))


def _arrays(gt, dets, image_ids, box_format="xywh"):
    preds, targets = [], []
    for img_id in image_ids:
        g = [a for a in gt["annotations"] if a["image_id"] == img_id]
        d = [x for x in dets if x["image_id"] == img_id]
        gt_boxes = np.array([a["bbox"] for a in g], dtype=np.float64).reshape(-1, 4)
        dt_boxes = np.array([x["bbox"] for x in d], dtype=np.float64).reshape(-1, 4)
        if box_format == "xyxy":
            gt_boxes = np.column_stack([gt_boxes[:, :2], gt_boxes[:, :2] + gt_boxes[:, 2:]])
            dt_boxes = np.column_stack([dt_boxes[:, :2], dt_boxes[:, :2] + dt_boxes[:, 2:]])
        targets.append({
            "image_id": img_id,
            "boxes": gt_boxes,
            "labels": np.array([a["category_id"] for a in g], dtype=np.int64),
            "iscrowd": np.array([a["iscrowd"] for a in g], dtype=np.int64),
            "area": np.array([a["area"] for a in g], dtype=np.float64),
        })
        preds.append({
            "boxes": dt_boxes,
            "scores": np.array([x["score"] for x in d], dtype=np.float64),
            "labels": np.array([x["category_id"] for x in d], dtype=np.int64),
        })
    return preds, targets


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()) or "default")
def test_array_input_equals_coco_input(data, config):
    gt, dets = data
    from_arrays = DetectionMetrics(iou_thrs=[0.5, 0.75], box_format="xywh", **config)
    from_coco = DetectionMetrics(iou_thrs=[0.5, 0.75], **config)
    for ids in _batches(gt, 40):
        _assert_same(from_arrays.update(*_arrays(gt, dets, ids)),
                     from_coco.update_coco(*_subset(gt, dets, ids)))
    _assert_same(from_arrays.compute(), from_coco.compute())


def test_missing_target_area_defaults_to_the_box_area(data):
    """Images without ``area`` take ``w * h``, as the engine does for COCO
    ground truth without an ``area`` field; other images keep theirs."""
    gt, dets = data
    ids = sorted(im["id"] for im in gt["images"])[:40]
    preds, targets = _arrays(gt, dets, ids)
    without = set(ids[::2])
    for target in targets:
        if target["image_id"] in without:
            del target["area"]
    batch_gt, batch_dt = copy.deepcopy(_subset(gt, dets, ids))
    for ann in batch_gt["annotations"]:
        if ann["image_id"] in without:
            del ann["area"]
    config = {"iou_thrs": [0.5, 0.75], "area_range": (32.0**2, 96.0**2)}
    _assert_same(DetectionMetrics(box_format="xywh", **config).update(preds, targets),
                 DetectionMetrics(**config).update_coco(batch_gt, batch_dt))


def _variant(items, kind):
    """The same values in another dtype or memory layout."""
    out = []
    for item in items:
        item = dict(item)
        if kind == "float32":
            for key in ("boxes", "scores", "area"):
                if key in item:
                    item[key] = item[key].astype(np.float32)
        elif kind == "int32":
            item["labels"] = item["labels"].astype(np.int32)
        elif kind == "small ints":
            item["labels"] = item["labels"].astype(np.int16)
            if "iscrowd" in item:
                item["iscrowd"] = item["iscrowd"].astype(bool)
        elif kind == "uint8 crowd":
            if "iscrowd" in item:
                item["iscrowd"] = item["iscrowd"].astype(np.uint8)
        elif kind == "strided":
            for key in ("boxes", "scores", "labels"):
                if key in item:
                    item[key] = np.repeat(item[key], 2, axis=0)[::2]
        elif kind == "fortran":
            item["boxes"] = np.asfortranarray(item["boxes"])
        elif kind == "lists":
            for key in ("boxes", "scores", "labels", "iscrowd"):
                if key in item:
                    item[key] = item[key].tolist()
        elif kind == "integer boxes":
            item["boxes"] = item["boxes"].astype(np.int64)
        out.append(item)
    return out


@pytest.mark.parametrize(
    "kind", ["float32", "int32", "small ints", "uint8 crowd", "strided", "fortran", "lists",
             "integer boxes"])
def test_dtypes_and_layouts_read_like_float64_and_int64(data, kind):
    """NumPy arrays of common dtypes are read in place; each must give what the
    same values as contiguous float64 / int64 give."""
    gt, dets = data
    ids = sorted(im["id"] for im in gt["images"])[:40]
    preds, targets = _arrays(gt, dets, ids)
    if kind == "float32":
        # Compare with the float32 values widened, which is exact.
        preds = [{**p, "boxes": p["boxes"].astype(np.float32).astype(np.float64),
                  "scores": p["scores"].astype(np.float32).astype(np.float64)} for p in preds]
        targets = [{**t, "boxes": t["boxes"].astype(np.float32).astype(np.float64),
                    "area": t["area"].astype(np.float32).astype(np.float64)} for t in targets]
    if kind == "integer boxes":
        preds = [{**p, "boxes": np.trunc(p["boxes"])} for p in preds]
        targets = [{**t, "boxes": np.trunc(t["boxes"])} for t in targets]
    config = {"iou_thrs": [0.5, 0.75], "box_format": "xywh", "score_thr": "best_f1"}
    _assert_same(DetectionMetrics(**config).update(preds, targets),
                 DetectionMetrics(**config).update(_variant(preds, kind), _variant(targets, kind)))


@pytest.mark.parametrize("kind", ["plain", "requires grad", "non-contiguous", "float32", "float16"])
def test_torch_tensors_read_like_numpy(data, kind):
    """CPU tensors are read through ``.numpy()``; others fall back to the
    Python conversion. Both must give what the same NumPy values give."""
    torch = pytest.importorskip("torch")
    gt, dets = data
    ids = sorted(im["id"] for im in gt["images"])[:30]
    preds, targets = _arrays(gt, dets, ids)
    if kind == "float32":
        preds = [{**p, "boxes": p["boxes"].astype(np.float32).astype(np.float64)} for p in preds]
    if kind == "float16":
        preds = [{**p, "scores": p["scores"].astype(np.float16).astype(np.float64)} for p in preds]

    def tensor(key, value):
        t = torch.from_numpy(np.ascontiguousarray(value))
        if kind == "requires grad" and t.is_floating_point():
            t = t.clone().requires_grad_(True)
        if kind == "non-contiguous" and t.ndim == 2:
            t = t.repeat_interleave(2, dim=0)[::2]
        if kind == "float32" and key == "boxes":
            t = t.float()
        if kind == "float16" and key == "scores":
            t = t.half()
        return t

    as_tensors = [[{k: tensor(k, v) if isinstance(v, np.ndarray) else v for k, v in item.items()}
                   for item in side] for side in (preds, targets)]
    config = {"iou_thrs": [0.5, 0.75], "box_format": "xywh", "score_thr": "best_f1"}
    _assert_same(DetectionMetrics(**config).update(preds, targets),
                 DetectionMetrics(**config).update(*as_tensors))


def test_zero_dimensional_labels(data):
    one = {"boxes": np.array([[0.0, 0.0, 10.0, 10.0]]), "scores": np.array([0.9]), "labels": np.array([3])}
    scalar = {**one, "labels": np.array(3)}
    target = {"boxes": np.array([[0.0, 0.0, 10.0, 10.0]]), "labels": np.array([3])}
    _assert_same(DetectionMetrics().update([one], [target]), DetectionMetrics().update([scalar], [target]))


def test_xyxy_boxes_are_converted_to_xywh(data):
    gt, dets = data
    ids = sorted(im["id"] for im in gt["images"])[:30]
    preds, targets = _arrays(gt, dets, ids, box_format="xyxy")
    converted = [
        {**item, "boxes": np.column_stack([item["boxes"][:, :2], item["boxes"][:, 2:] - item["boxes"][:, :2]])}
        for item in preds
    ]
    converted_t = [
        {**item, "boxes": np.column_stack([item["boxes"][:, :2], item["boxes"][:, 2:] - item["boxes"][:, :2]])}
        for item in targets
    ]
    _assert_same(
        DetectionMetrics(box_format="xyxy").update(preds, targets),
        DetectionMetrics(box_format="xywh").update(converted, converted_t),
    )


def test_mask_input_equals_coco_rle_input(data):
    """Masks become the same RLE a COCO file would hold.

    Detections are given as RLE on both sides, since COCO results that also
    carry a ``bbox`` take their area from the box rather than the mask.
    """
    gt, dets = data
    ids = sorted(im["id"] for im in gt["images"])[:12]
    batch_gt, batch_dt = _subset(gt, dets, ids)
    coco = ufc.COCO(batch_gt, verbose=False)
    boxes_as_masks = coco.loadRes(copy.deepcopy(batch_dt))
    rle_dt = [
        {"image_id": d["image_id"], "category_id": d["category_id"], "score": d["score"],
         "segmentation": boxes_as_masks.annToRLE(d)}
        for d in boxes_as_masks.dataset["annotations"]
    ]
    preds, targets = [], []
    for img_id in ids:
        g = [a for a in batch_gt["annotations"] if a["image_id"] == img_id]
        d = [x for x in rle_dt if x["image_id"] == img_id]
        im = coco.imgs[img_id]
        empty = np.zeros((0, im["height"], im["width"]), dtype=bool)
        targets.append({
            "image_id": img_id,
            "masks": np.stack([coco.annToMask(a) for a in g]) if g else empty,
            "labels": np.array([a["category_id"] for a in g], dtype=np.int64),
            "iscrowd": np.array([a["iscrowd"] for a in g], dtype=np.int64),
            "area": np.array([a["area"] for a in g], dtype=np.float64),
        })
        preds.append({
            "masks": np.stack([maskUtils.decode(x["segmentation"]) for x in d]) if d else empty,
            "scores": np.array([x["score"] for x in d], dtype=np.float64),
            "labels": np.array([x["category_id"] for x in d], dtype=np.int64),
        })
    config = {"iou_type": "segm", "iou_thrs": [0.5, 0.75], "score_thr": "best_f1"}
    _assert_same(DetectionMetrics(**config).update(preds, targets),
                 DetectionMetrics(**config).update_coco(batch_gt, rle_dt))


def test_scalar_iou_threshold_drops_the_axis(data):
    gt, dets = data
    ids = sorted(im["id"] for im in gt["images"])[:20]
    one = DetectionMetrics(iou_thrs=0.5).update(*_arrays(gt, dets, ids))
    seq = DetectionMetrics(iou_thrs=[0.5]).update(*_arrays(gt, dets, ids))
    assert one["iou_thr"] == 0.5
    assert np.isscalar(one["micro"]["f1"])
    for group in ("per_category", "micro", "macro"):
        for name, value in one[group].items():
            assert np.array_equal(value, seq[group][name][0])


def _box(x, y, label, score=None):
    item = {"boxes": np.array([[x, y, x + 10.0, y + 10.0]]), "labels": np.array([label])}
    if score is not None:
        item["scores"] = np.array([score])
    return item


def test_automatic_image_ids_continue_and_reset_clears():
    metric = DetectionMetrics()
    first = metric.update([_box(0, 0, 1, 0.9)], [_box(0, 0, 1)])
    assert first["micro"]["tp"] == 1
    second = metric.update([_box(0, 0, 1, 0.9)], [_box(50, 50, 1)])
    assert (second["micro"]["tp"], second["micro"]["fp"], second["micro"]["fn"]) == (0, 1, 1)
    total = metric.compute()
    assert (total["micro"]["tp"], total["micro"]["fp"], total["micro"]["fn"]) == (1, 1, 1)
    metric.reset()
    assert metric.compute()["category_ids"].size == 0
    assert metric.update([_box(0, 0, 1, 0.9)], [_box(0, 0, 1)])["micro"]["tp"] == 1


def test_empty_images_and_batches():
    metric = DetectionMetrics()
    empty = {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=int)}
    empty_pred = {**empty, "scores": np.zeros(0)}
    r = metric.update([empty_pred, _box(0, 0, 2, 0.5)], [_box(0, 0, 1), empty])
    assert r["category_ids"].tolist() == [1, 2]
    assert r["per_category"]["fn"].tolist() == [1, 0]
    assert r["per_category"]["fp"].tolist() == [0, 1]
    assert metric.update([], [])["category_ids"].size == 0
    # Real images with nothing on either side: no category at all.
    assert metric.update([empty_pred, empty_pred], [empty, empty])["category_ids"].size == 0
    assert metric.compute()["micro"]["fp"] == 1
    nothing = DetectionMetrics(score_thr="best_f1").compute()
    assert nothing["score_thr"] == np.inf
    assert nothing["macro"]["f1"] == 0.0


def test_best_f1_when_no_prediction_has_a_ground_truth_label():
    r = DetectionMetrics(score_thr="best_f1").update([_box(0, 0, 2, 0.5)], [_box(0, 0, 1)])
    assert r["score_thr"] == 0.5
    assert r["macro"]["f1"] == 0.0
    assert r["per_category"]["fp"].tolist() == [0, 1]


def test_batch_results_off_returns_none_and_accumulates_the_same(data):
    gt, dets = data
    with_batches = DetectionMetrics(iou_thrs=[0.5, 0.75], score_thr="best_f1", box_format="xywh")
    without = DetectionMetrics(iou_thrs=[0.5, 0.75], score_thr="best_f1", box_format="xywh",
                               batch_results=False)
    for ids in _batches(gt, 50):
        assert with_batches.update(*_arrays(gt, dets, ids)) is not None
        assert without.update(*_arrays(gt, dets, ids)) is None
    assert without.update_coco(*_subset(gt, dets, [])) is None
    _assert_same(with_batches.compute(), without.compute())


def test_detection_metrics_helper():
    r = detection_metrics([_box(0, 0, 1, 0.9)], [_box(0, 0, 1)], iou_thrs=0.5)
    assert r["micro"]["f1"] == 1.0


def test_torch_tensors():
    torch = pytest.importorskip("torch")
    pred = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1])}
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}
    assert detection_metrics([pred], [target])["micro"]["tp"] == 1


@pytest.mark.parametrize(
    "preds, targets, match",
    [
        ([_box(0, 0, 1, 0.9)], [], "preds has 1 images"),
        ([{"boxes": np.zeros((1, 4)), "labels": [1]}], [_box(0, 0, 1)], "no 'scores'"),
        ([_box(0, 0, 1, 0.9)], [{"labels": [1]}], "no 'boxes'"),
        ([{"boxes": np.zeros((1, 3)), "scores": [0.5], "labels": [1]}], [_box(0, 0, 1)], "shape"),
        ([{"boxes": np.zeros((1, 4)), "scores": [0.5, 0.4], "labels": [1]}], [_box(0, 0, 1)], "scores"),
        ([_box(0, 0, 1, 0.9), _box(0, 0, 1, 0.9)], [{**_box(0, 0, 1), "image_id": 3}, _box(0, 0, 1)],
         "every image"),
        ([{**_box(0, 0, 1, 0.9), "image_id": 4}], [{**_box(0, 0, 1), "image_id": 5}], "different image_id"),
        ([_box(0, 0, 1, 0.9)] * 2, [{**_box(0, 0, 1), "image_id": 6}] * 2, "repeats"),
        ([_box(0, 0, 1, 0.9)], [{**_box(0, 0, 1), "iscrowd": [0, 1]}], "iscrowd"),
    ],
)
def test_input_errors(preds, targets, match):
    with pytest.raises(ValueError, match=match):
        DetectionMetrics().update(preds, targets)


def test_an_image_cannot_be_added_twice(data):
    gt, dets = data
    metric = DetectionMetrics()
    batch = _subset(gt, dets, [1, 2])
    metric.update_coco(*batch)
    with pytest.raises(ValueError, match="already added"):
        metric.update_coco(*_subset(gt, dets, [2, 3]))
    with pytest.raises(ValueError, match="already added"):
        metric.update([_box(0, 0, 1, 0.9)], [{**_box(0, 0, 1), "image_id": 1}])
    # A failed update adds nothing, and automatic ids skip the ones in use.
    r = metric.update([_box(0, 0, 1, 0.9)] * 3, [_box(0, 0, 1)] * 3)
    assert r["micro"]["tp"] == 3


def test_option_errors():
    for kwargs, match in [
        ({"iou_type": "keypoint"}, "iou_type"),
        ({"box_format": "cxcywh"}, "box_format"),
        ({"iou_thrs": []}, "iou_thrs"),
        ({"score_thr": "best"}, "score_thr"),
        ({"optimize": "weighted"}, "optimize"),
        ({"max_dets": -1}, "max_dets"),
    ]:
        with pytest.raises(ValueError, match=match):
            DetectionMetrics(**kwargs)
    with pytest.raises(ValueError, match="update_coco"):
        DetectionMetrics(iou_type="keypoints").update([], [])
    masks = {"masks": np.zeros((1, 4, 4), bool), "labels": [1]}
    with pytest.raises(ValueError, match="differ in size"):
        DetectionMetrics(iou_type="segm").update(
            [{"masks": np.ones((1, 5, 5), bool), "scores": [0.5], "labels": [1]}], [masks])
