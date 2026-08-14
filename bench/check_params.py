"""Compare the evaluation grid (IoU / recall thresholds, area ranges) bitwise.

If two implementations disagree on the thresholds themselves, every downstream
metric difference is explained before you look at a single line of matching
code. numpy's ``linspace`` is not the same as ``start + i * step``, so this is
a real trap, not a hypothetical one.
"""

from __future__ import annotations

import sys

import numpy as np


def show(name: str, a: np.ndarray, b: np.ndarray) -> None:
    same = np.array_equal(a, b)
    print(f"{name}: identical={same}")
    if not same:
        print(f"  max |diff| = {np.max(np.abs(a - b)):.3e}")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  [{i}] pycocotools={x!r} ({float(x).hex()})")
                print(f"       other      ={y!r} ({float(y).hex()})")


def main() -> None:
    gt_path = sys.argv[1]
    other = sys.argv[2] if len(sys.argv) > 2 else "hotcoco"

    from pycocotools.cocoeval import Params

    p = Params(iouType="bbox")

    if other == "hotcoco":
        from hotcoco import COCO, COCOeval
    elif other == "faster":
        from faster_coco_eval import COCO
        from faster_coco_eval import COCOeval_faster as COCOeval
    else:
        from ultrafast_pycocotools import COCO, COCOeval

    gt = COCO(gt_path)
    ev = COCOeval(gt, gt, "bbox")

    show("iouThrs", np.asarray(p.iouThrs, np.float64), np.asarray(ev.params.iouThrs, np.float64))
    show("recThrs", np.asarray(p.recThrs, np.float64), np.asarray(ev.params.recThrs, np.float64))
    print("areaRng pycocotools:", [list(map(float, a)) for a in p.areaRng])
    print("areaRng other      :", [list(map(float, a)) for a in ev.params.areaRng])
    print("maxDets pycocotools:", list(p.maxDets))
    print("maxDets other      :", list(ev.params.maxDets))

    # The naive construction, for reference.
    naive = np.array([0.5 + 0.05 * i for i in range(10)])
    show("linspace vs naive", np.asarray(p.iouThrs, np.float64), naive)


if __name__ == "__main__":
    main()
