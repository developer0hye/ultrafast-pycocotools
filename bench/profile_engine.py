"""Direct per-phase measurement of the evaluation engine.

`profile_phases.py` and `profile_segm.py` infer costs by differencing whole-run
timings — useful for a first cut, but it cannot see inside the engine and it
cannot see overlap. This reads the timers the engine itself keeps, so every
number is attributed rather than deduced.

Two kinds of number come back and they are not comparable:

* **extraction** is wall-clock on two threads that run concurrently (reading
  annotations out of Python needs the GIL, rasterising them does not), so
  ``read`` and ``rasterise`` overlap and do not sum. ``read_blocked`` is the
  slice of reading spent waiting for the rasteriser — large means the
  rasteriser is the bottleneck, ~0 means the GIL-bound reading is.
* **evaluation** phases are summed across rayon workers, so they exceed the
  wall-clock. Comparing the sum against wall-clock is what shows whether a
  phase actually parallelised.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import ultrafast_pycocotools as ufc
from ultrafast_pycocotools import _ufcoco


def bar(frac: float, width: int = 28) -> str:
    filled = int(round(max(0.0, min(1.0, frac)) * width))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--dt", type=Path, required=True)
    ap.add_argument("--iou-type", default="segm")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    t = time.perf_counter()
    gt = ufc.COCO(str(args.gt), verbose=False)
    t_load_gt = time.perf_counter() - t

    t = time.perf_counter()
    with open(args.dt) as f:
        dets = json.load(f)
    dt = gt.loadRes(dets)
    t_load_dt = time.perf_counter() - t

    ev = ufc.COCOeval(gt, dt, args.iou_type, print_function=lambda *_: None)
    p = ev.params
    p.imgIds = [int(i) for i in np.unique(p.imgIds)]
    p.catIds = [int(c) for c in np.unique(p.catIds)]
    p.maxDets = sorted(p.maxDets)

    t = time.perf_counter()
    gts, dts, img_sizes = ev._collect()
    t_prepare = time.perf_counter() - t

    sigmas = getattr(p, "kpt_oks_sigmas", np.zeros(0))
    best = None
    for _ in range(args.repeat):
        t = time.perf_counter()
        engine = _ufcoco.Evaluator(
            gts, dts, img_sizes, p.imgIds, p.catIds,
            [float(x) for x in p.iouThrs], [float(x) for x in p.recThrs],
            [int(m) for m in p.maxDets],
            [[float(a[0]), float(a[1])] for a in p.areaRng],
            True, p.iouType, [float(s) for s in np.asarray(sigmas).ravel()], True, 0.02,
        )
        t_build = time.perf_counter() - t
        t = time.perf_counter()
        engine.run(False)
        t_run = time.perf_counter() - t
        if best is None or t_build + t_run < best[0]:
            best = (t_build + t_run, t_build, t_run, engine.timings())
    _, t_build, t_run, tm = best

    print(f"iouType             : {args.iou_type}")
    print(f"gt / dt annotations : {len(gts)} / {len(dts)}")
    print()
    print("python side (wall)")
    print(f"  COCO(gt) load          {t_load_gt:8.3f}s")
    print(f"  loadRes(dt)            {t_load_dt:8.3f}s")
    print(f"  _prepare               {t_prepare:8.3f}s")
    print()
    print(f"extraction (wall {t_build:.3f}s; read and rasterise overlap)")
    for label, key in (
        ("gt_read", "gt_read"),
        ("dt_read", "dt_read"),
        ("rasterise", "rasterise"),
    ):
        v = tm[key]
        print(f"  {label:22s} {v:8.3f}s  {bar(v / max(t_build, 1e-9))}")
    print(
        f"  {'read blocked':22s} {tm['read_blocked']:8.3f}s"
        "  (reader waiting on the rasteriser)"
    )
    print()
    print(f"evaluation (wall {t_run:.3f}s; phases are summed over workers)")
    total_cpu = tm["iou_cpu"] + tm["match_cpu"] + tm["accumulate_cpu"]
    for name, key in (
        ("group index", "group_index"),
        ("iou", "iou_cpu"),
        ("match", "match_cpu"),
        ("accumulate", "accumulate_cpu"),
    ):
        v = tm[key]
        print(f"  {name:22s} {v:8.3f}s  {bar(v / max(total_cpu, 1e-9))}")
    if t_run > 0:
        print(f"  parallel speedup       {total_cpu / t_run:8.2f}x  (cpu sum / wall)")


if __name__ == "__main__":
    main()
