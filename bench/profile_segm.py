"""Split the segm setup cost into reading vs rasterising.

`profile_phases.py` says segm spends ~80% of engine time in "extract +
rasterise", which is two very different jobs glued together: pulling polygon
coordinates out of Python lists, and turning them into RLE masks. They want
opposite fixes, so guessing which one dominates is a good way to optimise the
wrong thing.

The experiment substitutes the ground-truth segmentation with forms of varying
cost while keeping everything else identical:

  bbox            no segmentation read at all      -> dict-reading floor
  segm/rle        pre-encoded compressed RLE       -> floor + string decode
  segm/polygon    the original polygons            -> floor + polygon read
                                                      + rasterisation

so `polygon - rle` is the polygon-specific cost and `rle - bbox` is what
reading and decoding an already-rasterised mask costs.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

import ultrafast_pycocotools as ufc
from ultrafast_pycocotools import _ufcoco
from ultrafast_pycocotools import mask as mask_util


def build_engine(ev, gts, dts, img_sizes, iou_type: str, repeat: int) -> float:
    p = ev.params
    sigmas = getattr(p, "kpt_oks_sigmas", np.zeros(0))
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        _ufcoco.Evaluator(
            gts, dts, img_sizes, p.imgIds, p.catIds,
            [float(x) for x in p.iouThrs], [float(x) for x in p.recThrs],
            [int(m) for m in p.maxDets],
            [[float(a[0]), float(a[1])] for a in p.areaRng],
            True, iou_type, [float(s) for s in np.asarray(sigmas).ravel()], True, 0.02,
        )
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--dt", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    gt = ufc.COCO(str(args.gt), verbose=False)
    with open(args.dt) as f:
        dt = gt.loadRes(json.load(f))

    ev = ufc.COCOeval(gt, dt, "segm", print_function=lambda *_: None)
    p = ev.params
    p.imgIds = [int(i) for i in np.unique(p.imgIds)]
    p.catIds = [int(c) for c in np.unique(p.catIds)]
    p.maxDets = sorted(p.maxDets)
    gts, dts, img_sizes = ev._prepare()

    n_poly = sum(1 for a in gts if isinstance(a.get("segmentation"), list))
    n_pts = sum(
        sum(len(r) for r in a["segmentation"]) // 2
        for a in gts
        if isinstance(a.get("segmentation"), list)
    )
    print(f"gt annotations      : {len(gts)} ({n_poly} polygon, {n_pts} vertices total)")
    print(f"dt annotations      : {len(dts)}")

    t_bbox = build_engine(ev, gts, dts, img_sizes, "bbox", args.repeat)
    t_poly = build_engine(ev, gts, dts, img_sizes, "segm", args.repeat)

    # Same masks, already rasterised: isolates the polygon-specific cost.
    gts_rle = []
    for a in gts:
        b = copy.copy(a)
        b["segmentation"] = gt.annToRLE(a)
        gts_rle.append(b)
    t_rle = build_engine(ev, gts_rle, dts, img_sizes, "segm", args.repeat)

    # Identical list structure and vertex count, but every vertex collapsed to
    # one point. Reading costs the same; rasterisation collapses from
    # O(5 * perimeter) traced points to O(vertices), i.e. ~nothing. So this
    # measures "read the polygon out of Python" on its own.
    gts_flat = []
    for a in gts:
        b = copy.copy(a)
        seg = a.get("segmentation")
        if isinstance(seg, list):
            b["segmentation"] = [[1.0] * len(r) for r in seg]
        gts_flat.append(b)
    t_flat = build_engine(ev, gts_flat, dts, img_sizes, "segm", args.repeat)

    print()
    print(f"bbox        (dict reading floor)  : {t_bbox:7.3f}s")
    print(f"segm / pre-encoded RLE            : {t_rle:7.3f}s")
    print(f"segm / zero-perimeter polygons    : {t_flat:7.3f}s")
    print(f"segm / original polygons          : {t_poly:7.3f}s")
    print()
    print(f"  read + decode RLE masks         : {t_rle - t_bbox:7.3f}s")
    print(f"  read polygons out of Python     : {t_flat - t_bbox:7.3f}s")
    print(f"  rasterise polygons              : {t_poly - t_flat:7.3f}s")
    if n_pts:
        print(f"  per vertex, reading             : {(t_flat - t_bbox) / n_pts * 1e9:7.1f} ns")


if __name__ == "__main__":
    main()
