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
    ap.add_argument("--run-storage", action="store_true",
                    help="Count uint32 RLE storage for matched groups and the maxDets subset")
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

    if args.run_storage:
        # Valid compressed COCO RLE ends each integer with a byte in [48, 79].
        # annToRLE canonicalizes polygons and uncompressed runs first. This
        # diagnostic counts run payload only, excluding headers/allocator slack.
        terminal = bytes(int(48 <= value < 80) for value in range(256))

        def run_bytes(handle, ann):
            counts = handle.annToRLE(ann)['counts']
            if isinstance(counts, str):
                counts = counts.encode('utf-8')
            return counts.translate(terminal).count(b'\x01') * 4

        gt_bytes = sum(run_bytes(gt, ann) for ann in gts
                       if dt_groups.get((ann['image_id'], ann['category_id'])))
        grouped = defaultdict(list)
        for ann in dts:
            if gt_groups.get((ann['image_id'], ann['category_id'])):
                grouped[(ann['image_id'], ann['category_id'])].append(ann)
        all_dt_bytes = capped_dt_bytes = 0
        for group in grouped.values():
            if not all(np.isfinite(ann['score']) for ann in group):
                raise ValueError('Storage diagnostic requires finite prediction scores')
            ordered = sorted(group, key=lambda ann: ann['score'], reverse=True)
            for rank, ann in enumerate(ordered):
                n = run_bytes(dt, ann)
                all_dt_bytes += n
                if rank < args.max_dets:
                    capped_dt_bytes += n
        print(f'GT decoded run payload in nonempty joins: {gt_bytes:,} bytes')
        print(f'DT decoded run payload in nonempty joins: {all_dt_bytes:,} bytes')
        print(f'DT decoded run payload after maxDets:     {capped_dt_bytes:,} bytes')
        print(f'Additional payload avoidable at maxDets: {all_dt_bytes - capped_dt_bytes:,} bytes')


if __name__ == "__main__":
    main()
