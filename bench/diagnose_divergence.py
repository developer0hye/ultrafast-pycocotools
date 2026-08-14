"""Localise where another implementation's AP stops matching pycocotools.

`max |diff|` says a difference exists; it does not say why. This walks the full
precision array, reports which IoU thresholds, categories and recall points
disagree, and then runs the decisive experiment: hand the other implementation
pycocotools' exact threshold grid and see whether the disagreement survives.

If it vanishes, the grid was the whole story. If it does not, something else is
wrong and the residue is what to look at next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(impl: str, gt_path: str, dt_path: str, iou_type: str, grid=None):
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
    applied = False
    if grid is not None:
        iou_thrs, rec_thrs = grid
        try:
            ev.params.iouThrs = np.asarray(iou_thrs, dtype=np.float64)
            ev.params.recThrs = np.asarray(rec_thrs, dtype=np.float64)
            applied = (
                np.asarray(ev.params.iouThrs, np.float64).tobytes() == iou_thrs.tobytes()
                and np.asarray(ev.params.recThrs, np.float64).tobytes() == rec_thrs.tobytes()
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ({impl} rejected the grid override: {exc})")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev, applied


def describe(ref, other, label: str, cat_names: dict) -> None:
    a = np.ascontiguousarray(ref.eval["precision"], dtype=np.float64)
    b = np.ascontiguousarray(other.eval["precision"], dtype=np.float64)
    print(f"\n=== {label} ===")
    if a.shape != b.shape:
        print(f"  shape differs: {a.shape} vs {b.shape}")
        return
    diff = a != b
    n = int(diff.sum())
    print(f"  precision cells differing: {n:,} of {a.size:,} ({n / a.size:.3%})")
    if n == 0:
        print("  bit-identical")
        return
    print(f"  max |delta|: {np.abs(a - b).max():.3e}")

    t_idx = np.asarray(ref.params.iouThrs, dtype=np.float64)
    by_t = diff.sum(axis=(1, 2, 3, 4))
    print("  by IoU threshold:")
    for i, thr in enumerate(t_idx):
        if by_t[i]:
            print(f"    {thr:.2f}: {by_t[i]:,} cells")

    by_k = diff.sum(axis=(0, 1, 3, 4))
    worst = np.argsort(by_k)[::-1][:6]
    print("  worst categories:")
    for k in worst:
        if by_k[k]:
            cid = int(ref.params.catIds[k])
            print(f"    {cat_names.get(cid, cid):<16} {by_k[k]:,} cells")

    # Where in the recall sweep? A grid problem clusters at the recall values
    # that land exactly on a threshold; a matching problem does not.
    by_r = diff.sum(axis=(0, 2, 3, 4))
    hit = np.flatnonzero(by_r)
    print(f"  recall points touched: {len(hit)} of {a.shape[1]}")
    print(f"    first few: {[f'{ref.params.recThrs[i]:.2f}' for i in hit[:12]]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--iou-type", default="bbox")
    ap.add_argument("--other", default="hotcoco")
    args = ap.parse_args()

    gt_meta = json.loads(Path(args.gt).read_text())
    cat_names = {c["id"]: c["name"] for c in gt_meta["categories"]}

    ref, _ = load("pycocotools", args.gt, args.dt, args.iou_type)
    other, _ = load(args.other, args.gt, args.dt, args.iou_type)
    describe(ref, other, f"{args.other}, default grid", cat_names)

    # The grid itself, bit for bit.
    for name in ("iouThrs", "recThrs"):
        a = np.asarray(getattr(ref.params, name), dtype=np.float64)
        b = np.asarray(getattr(other.params, name), dtype=np.float64)
        bad = np.flatnonzero(a != b)
        print(f"\n  {name}: {len(bad)} of {len(a)} entries differ from np.linspace")
        for i in bad[:4]:
            print(f"    [{i}] pycocotools {a[i]!r} ({a[i].hex()})")
            print(f"         {args.other:<12} {b[i]!r} ({b[i].hex()})")

    # Decisive: same grid, same everything else.
    grid = (
        np.asarray(ref.params.iouThrs, dtype=np.float64),
        np.asarray(ref.params.recThrs, dtype=np.float64),
    )
    forced, applied = load(args.other, args.gt, args.dt, args.iou_type, grid=grid)
    if not applied:
        print(f"\n  {args.other} did not take the grid override; skipping the isolation run")
        return
    describe(ref, forced, f"{args.other}, forced onto pycocotools' grid", cat_names)


if __name__ == "__main__":
    main()
