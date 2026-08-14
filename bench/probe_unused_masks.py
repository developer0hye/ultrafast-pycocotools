"""Measure the ceiling on skipping masks that no IoU ever consumes.

`count_unused_masks.py` says a quarter of segmentations are rasterised for
nothing. IoU is only computed inside an (image, category) cell that has both
ground truth and detections, so a mask on the empty side of a cell is never
read by anything: an unmatched ground truth counts as one false negative
whether its pixels are known or not, and a detection in a category the image
does not contain is a false positive on sight. Skipping those is provably
invisible to the result.

But "a quarter of the masks" is not "a quarter of the time", and the difference
decides whether the optimisation is worth building. Rasterisation already runs
on a worker thread behind the GIL-bound reading, so cutting it may buy nothing
at all. What skipping would really save is *reading* those segmentations out of
Python — and knowing which ones to skip requires the grouping, which requires a
first pass. Two passes to save part of one.

Rather than build that to find out, this measures the ceiling: strip the
segmentation from exactly the annotations a perfect implementation would skip
and time the engine build. Whatever that saves is the most the idea can ever be
worth, and a real two-pass version would come in under it.

Each arm runs in its own process. The first version of this ran all three in
one, and the stripped arm came out 63% *slower* — four copies of half a million
dicts do not fit the same way one does. Measuring less work as more work is the
experiment failing, not the idea.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_arm(gt_path: str, dt_path: str, mode: str, repeat: int) -> dict:
    """Time the engine build for one arm. Runs as a child process."""
    import ultrafast_pycocotools as ufc
    from ultrafast_pycocotools import _ufcoco

    gt = ufc.COCO(gt_path, verbose=False)
    with open(dt_path) as f:
        dt = gt.loadRes(json.load(f))
    iou_type = "bbox" if mode == "bbox" else "segm"
    ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *_: None)
    p = ev.params
    p.imgIds = [int(i) for i in np.unique(p.imgIds)]
    p.catIds = [int(c) for c in np.unique(p.catIds)]
    p.maxDets = sorted(p.maxDets)
    gts, dts, img_sizes = ev._collect()

    gt_cells: dict = defaultdict(int)
    dt_cells: dict = defaultdict(int)
    for a in gts:
        gt_cells[(a["image_id"], a["category_id"])] += 1
    for a in dts:
        dt_cells[(a["image_id"], a["category_id"])] += 1

    # Standing in for a skip. Deleting the key instead was the obvious move and
    # it was wrong: with no segmentation the reader falls back to the box, and a
    # box is *more* expensive to rasterise than a mask. COCO's RLE runs down
    # columns, so a 400-wide rectangle costs ~800 runs while a compact blob's
    # encoded string decodes to far fewer. Measured: rasterise 0.175s -> 0.350s.
    # A 1x1 empty mask keeps the same read path and the same variant, and is as
    # close to free as reading a segmentation can be.
    empty = ufc.mask.encode(np.zeros((1, 1), order="F", dtype=np.uint8))

    def rebuild(anns: list[dict], other: dict) -> list[dict]:
        """Fresh dicts in every arm, so only the segmentation's size differs."""
        out = []
        for a in anns:
            b = copy.copy(a)
            if mode == "strip" and other.get((a["image_id"], a["category_id"]), 0) == 0:
                b["segmentation"] = empty
            out.append(b)
        return out

    n_skip = (sum(1 for a in gts if dt_cells.get((a["image_id"], a["category_id"]), 0) == 0)
              + sum(1 for a in dts if gt_cells.get((a["image_id"], a["category_id"]), 0) == 0))
    gts, dts = rebuild(gts, dt_cells), rebuild(dts, gt_cells)

    sigmas = getattr(p, "kpt_oks_sigmas", np.zeros(0))
    best, best_t = float("inf"), None
    for _ in range(repeat):
        t = time.perf_counter()
        engine = _ufcoco.Evaluator(
            gts, dts, img_sizes, p.imgIds, p.catIds,
            [float(x) for x in p.iouThrs], [float(x) for x in p.recThrs],
            [int(m) for m in p.maxDets],
            [[float(a[0]), float(a[1])] for a in p.areaRng],
            True, iou_type, [float(s) for s in np.asarray(sigmas).ravel()], True, 0.02,
        )
        el = time.perf_counter() - t
        if el < best:
            best, best_t = el, engine.timings()
        del engine
    return {
        "mode": mode, "build": best, "timings": best_t,
        "n_ann": len(gts) + len(dts), "n_skip": n_skip,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--mode", choices=["now", "strip", "bbox"])
    args = ap.parse_args()

    if args.mode:  # child
        print("RESULT " + json.dumps(build_arm(args.gt, args.dt, args.mode, args.repeat)))
        return

    arms = {}
    for mode in ("now", "strip", "bbox"):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--gt", args.gt, "--dt", args.dt,
             "--repeat", str(args.repeat), "--mode", mode],
            capture_output=True, text=True,
        )
        line = next((x for x in proc.stdout.splitlines() if x.startswith("RESULT ")), None)
        if line is None:
            raise SystemExit(f"{mode} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        arms[mode] = json.loads(line[7:])

    a = arms["now"]
    print(f"{a['n_ann']:,} annotations, {a['n_skip']:,} ({a['n_skip'] / a['n_ann']:.1%}) "
          f"never reach an IoU")
    print()
    label = {"now": "segm, as-is", "strip": "segm, unused masks stripped",
             "bbox": "bbox (no masks at all)"}
    keys = ["gt_read", "dt_read", "rasterise", "read_blocked"]
    head = f"{'arm':30} {'build':>8} " + " ".join(f"{k:>13}" for k in keys)
    print(head)
    print("-" * len(head))
    for mode in ("now", "strip", "bbox"):
        r = arms[mode]
        t = r["timings"]
        cells = " ".join(f"{t.get(k, 0.0):13.3f}" for k in keys)
        print(f"{label[mode]:30} {r['build']:8.3f} {cells}")

    saving = arms["now"]["build"] - arms["strip"]["build"]
    print()
    print(f"ceiling on the optimisation: {saving:.3f}s "
          f"({saving / arms['now']['build']:.1%} of the engine build)")
    print("An upper bound only — a real two-pass version still walks the")
    print("annotations twice, so it lands below this.")


if __name__ == "__main__":
    main()
