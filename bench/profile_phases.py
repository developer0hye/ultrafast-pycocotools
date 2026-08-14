"""Split our evaluation wall time into its phases.

`evaluate()` is one call from Python but three very different jobs inside:
reading annotations out of Python dicts, rasterising segmentations, and the
IoU/match/accumulate loop. Optimising without knowing the split is guessing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import ultrafast_pycocotools as ufc
from ultrafast_pycocotools import _ufcoco


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--dt", type=Path, required=True)
    ap.add_argument("--iou-type", default="segm")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    t = time.perf_counter()
    gt = ufc.COCO(str(args.gt), verbose=False)
    t_load = time.perf_counter() - t

    t = time.perf_counter()
    with open(args.dt) as f:
        dets = json.load(f)
    dt = gt.loadRes(dets)
    t_res = time.perf_counter() - t

    ev = ufc.COCOeval(gt, dt, args.iou_type, print_function=lambda *_: None)
    p = ev.params
    p.imgIds = [int(i) for i in np.unique(p.imgIds)]
    p.catIds = [int(c) for c in np.unique(p.catIds)]
    p.maxDets = sorted(p.maxDets)

    t = time.perf_counter()
    gts, dts, img_sizes = ev._collect()
    t_prepare = time.perf_counter() - t

    sigmas = getattr(p, "kpt_oks_sigmas", np.zeros(0))
    build_times, run_times = [], []
    for _ in range(args.repeat):
        t = time.perf_counter()
        engine = _ufcoco.Evaluator(
            gts, dts, img_sizes, p.imgIds, p.catIds,
            [float(x) for x in p.iouThrs], [float(x) for x in p.recThrs],
            [int(m) for m in p.maxDets],
            [[float(a[0]), float(a[1])] for a in p.areaRng],
            True, p.iouType, [float(s) for s in np.asarray(sigmas).ravel()], True, 0.02,
        )
        build_times.append(time.perf_counter() - t)
        t = time.perf_counter()
        engine.run(False)
        run_times.append(time.perf_counter() - t)

    print(f"iouType              : {args.iou_type}")
    print(f"gt annotations       : {len(gts)}")
    print(f"dt annotations       : {len(dts)}")
    print(f"COCO(gt) load        : {t_load:7.3f}s")
    print(f"loadRes(dt)          : {t_res:7.3f}s")
    print(f"_prepare (python)    : {t_prepare:7.3f}s")
    print(f"extract + rasterise  : {min(build_times):7.3f}s  (best of {args.repeat})")
    print(f"iou + match + accum  : {min(run_times):7.3f}s  (best of {args.repeat})")


if __name__ == "__main__":
    main()
