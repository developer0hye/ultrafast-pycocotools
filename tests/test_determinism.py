"""Results must not depend on how many threads rayon happens to use.

This is the failure mode that would quietly destroy the bit-exactness claim:
everything passes on the dev machine, then CI runs on a 2-core box, a
reduction lands in a different order, and AP moves in the last few digits.
Comparing against pycocotools on one machine cannot catch it — only running
ourselves at different widths can.

Thread count is a property of the process (rayon builds its pool once), so
each width runs in a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

RUNNER = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import ultrafast_pycocotools as ufc

    gt_path, dt_path, iou_type = sys.argv[1], sys.argv[2], sys.argv[3]
    gt = ufc.COCO(gt_path, verbose=False)
    dt = gt.loadRes(dt_path)
    ev = ufc.COCOeval(gt, dt, iou_type, print_function=lambda *a, **k: None)
    ev.run()
    out = {
        k: np.ascontiguousarray(ev.eval[k], dtype=np.float64).tobytes().hex()
        for k in ("precision", "recall", "scores")
    }
    out["stats"] = np.asarray(ev.stats, dtype=np.float64).tobytes().hex()
    # per_instance is the one path that combines per-thread results with a
    # reduce rather than writing into disjoint slots, so it is the one that
    # could actually reorder. Sorted before hashing so only the *content* is
    # compared, not the concatenation order, which is not part of the contract.
    dets, gts = ev.per_instance(iou_thr=0.5)
    rows = sorted(zip(dets["dt_id"].tolist(), dets["gt_id"].tolist(), dets["iou"].tolist()))
    out["per_instance"] = str(rows)
    out["per_instance_n"] = f"{len(rows)}/{len(gts['gt_id'])}"
    print(json.dumps(out))
    """
)


def run_with_threads(n: int, gt_path, dt_path, iou_type: str) -> dict:
    env = dict(os.environ, RAYON_NUM_THREADS=str(n))
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER, str(gt_path), str(dt_path), iou_type],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_thread_count_does_not_change_results(synthetic, iou_type):
    gt_path, dt_path = synthetic
    one = run_with_threads(1, gt_path, dt_path, iou_type)
    many = run_with_threads(8, gt_path, dt_path, iou_type)
    for key in ("precision", "recall", "scores", "stats", "per_instance", "per_instance_n"):
        assert one[key] == many[key], f"{iou_type}/{key} changed with thread count"
