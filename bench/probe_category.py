"""Print the raw precision curve of one category from two implementations.

`diagnose_divergence.py` narrows a disagreement to a category; this shows the
numbers side by side so the *shape* of the difference is visible. Sentinel
handling, a shifted sampling index and a broken envelope all look completely
different here and identical in an aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run(impl: str, gt_path: str, dt_path: str, iou_type: str):
    if impl == "pycocotools":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    elif impl == "hotcoco":
        from hotcoco import COCO, COCOeval
    elif impl == "faster":
        from faster_coco_eval import COCO
        from faster_coco_eval import COCOeval_faster as COCOeval
    else:
        from ultrafast_pycocotools import COCO, COCOeval
    gt = COCO(gt_path)
    dt = gt.loadRes(dt_path)
    ev = COCOeval(gt, dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--iou-type", default="bbox")
    ap.add_argument("--other", default="hotcoco")
    ap.add_argument("--category", default="toaster")
    ap.add_argument("--t", type=int, default=0, help="IoU threshold index")
    ap.add_argument("--a", type=int, default=0, help="area range index")
    ap.add_argument("--m", type=int, default=2, help="maxDets index")
    args = ap.parse_args()

    meta = json.loads(Path(args.gt).read_text())
    cat_id = next(c["id"] for c in meta["categories"] if c["name"] == args.category)
    n_gt = sum(
        1
        for a in meta["annotations"]
        if a["category_id"] == cat_id and not a.get("iscrowd")
    )
    n_crowd = sum(
        1 for a in meta["annotations"] if a["category_id"] == cat_id and a.get("iscrowd")
    )
    dets = json.loads(Path(args.dt).read_text())
    n_dt = sum(1 for d in dets if d["category_id"] == cat_id)
    print(f"category {args.category!r} (id {cat_id}): {n_gt} ground truth "
          f"(+{n_crowd} crowd), {n_dt} detections")

    ref = run("pycocotools", args.gt, args.dt, args.iou_type)
    oth = run(args.other, args.gt, args.dt, args.iou_type)
    k = list(ref.params.catIds).index(cat_id)

    a = np.asarray(ref.eval["precision"], np.float64)[args.t, :, k, args.a, args.m]
    b = np.asarray(oth.eval["precision"], np.float64)[args.t, :, k, args.a, args.m]
    ra = np.asarray(ref.eval["recall"], np.float64)[args.t, k, args.a, args.m]
    rb = np.asarray(oth.eval["recall"], np.float64)[args.t, k, args.a, args.m]
    sa = np.asarray(ref.eval["scores"], np.float64)[args.t, :, k, args.a, args.m]
    sb = np.asarray(oth.eval["scores"], np.float64)[args.t, :, k, args.a, args.m]

    print(f"\nIoU thr index {args.t} ({ref.params.iouThrs[args.t]:.2f}), "
          f"area {ref.params.areaRngLbl[args.a]}, maxDets {ref.params.maxDets[args.m]}")
    print(f"recall: pycocotools {ra!r}   {args.other} {rb!r}")
    print(f"\n{'recThr':>7} {'pycocotools':>14} {args.other:>14} {'delta':>12}"
          f"   {'score(py)':>10} {'score(oth)':>10}")
    shown = 0
    for i, thr in enumerate(ref.params.recThrs):
        if a[i] == b[i] and shown > 6:
            continue
        mark = "" if a[i] == b[i] else "  <--"
        print(f"{thr:7.2f} {a[i]:14.9f} {b[i]:14.9f} {a[i] - b[i]:12.3e}"
              f"   {sa[i]:10.4f} {sb[i]:10.4f}{mark}")
        shown += 1
        if shown > 40:
            print("  ...")
            break

    print(f"\nsummary for this slice:")
    print(f"  pycocotools: {(a == -1).sum()} cells at -1, {(a == 0).sum()} at 0")
    print(f"  {args.other:<12} {(b == -1).sum()} cells at -1, {(b == 0).sum()} at 0")


if __name__ == "__main__":
    main()
