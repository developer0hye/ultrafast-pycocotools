"""How many segmentation masks are rasterised but never compared?

IoU is only computed inside an (image, category) group that has both ground
truth and detections. A ground truth in a group with no detections still
counts toward the recall denominator, and a detection in a group with no
ground truth is still a false positive — but neither needs its *mask* for
that, only its area. Same for detections past `maxDets`, which the matcher
never sees.

If that fraction is large, the cheapest possible optimisation is to not
rasterise them.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import ultrafast_pycocotools as ufc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--dt", type=Path, required=True)
    ap.add_argument("--max-dets", type=int, default=100)
    args = ap.parse_args()

    gt = ufc.COCO(str(args.gt), verbose=False)
    with open(args.dt) as f:
        dt = gt.loadRes(json.load(f))

    ev = ufc.COCOeval(gt, dt, "segm", print_function=lambda *_: None)
    p = ev.params
    p.imgIds = [int(i) for i in np.unique(p.imgIds)]
    p.catIds = [int(c) for c in np.unique(p.catIds)]
    gts, dts, _ = ev._collect()

    gt_groups: dict = defaultdict(int)
    dt_groups: dict = defaultdict(int)
    for a in gts:
        gt_groups[(a["image_id"], a["category_id"])] += 1
    for a in dts:
        dt_groups[(a["image_id"], a["category_id"])] += 1

    gt_used = sum(n for k, n in gt_groups.items() if dt_groups.get(k))
    dt_used = sum(
        min(n, args.max_dets) for k, n in dt_groups.items() if gt_groups.get(k)
    )
    print(f"{'':22}{'total':>10}{'needs a mask':>14}{'wasted':>10}")
    print(f"{'ground truth':22}{len(gts):10,}{gt_used:14,}{len(gts) - gt_used:10,}")
    print(f"{'detections':22}{len(dts):10,}{dt_used:14,}{len(dts) - dt_used:10,}")
    total = len(gts) + len(dts)
    used = gt_used + dt_used
    print()
    print(f"masks rasterised for nothing: {total - used:,} of {total:,} "
          f"({(total - used) / max(total, 1):.1%})")


if __name__ == "__main__":
    main()
