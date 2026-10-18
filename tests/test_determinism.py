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
    # outcomes() promises category, image, score order, so it is compared as
    # returned, and so is everything detection_metrics derives from it.
    raw = ev._engine.outcomes(0, 100)
    out["outcomes"] = "".join(np.ascontiguousarray(raw[k]).tobytes().hex() for k in sorted(raw))
    m = ev.detection_metrics(iou_thr=[0.5, 0.75], score_thr="best_f1")
    out["detection_metrics"] = "".join(
        np.ascontiguousarray(v).tobytes().hex()
        for group in ("per_category", "micro", "macro") for v in m[group].values()
    ) + m["score_thr"].tobytes().hex()
    # DetectionMetrics fills outcome units and sweeps thresholds in parallel.
    from ultrafast_pycocotools.metrics import DetectionMetrics
    gt_json, dt_json = json.load(open(gt_path)), json.load(open(dt_path))
    def arrays(image_ids):
        preds, targets = [], []
        for i in image_ids:
            g = [a for a in gt_json["annotations"] if a["image_id"] == i]
            d = [x for x in dt_json if x["image_id"] == i]
            targets.append({"image_id": i, "boxes": np.array([a["bbox"] for a in g]).reshape(-1, 4),
                            "labels": np.array([a["category_id"] for a in g], dtype=np.int64),
                            "iscrowd": np.array([a["iscrowd"] for a in g], dtype=np.int64)})
            preds.append({"boxes": np.array([x["bbox"] for x in d]).reshape(-1, 4),
                          "scores": np.array([x["score"] for x in d]),
                          "labels": np.array([x["category_id"] for x in d], dtype=np.int64)})
        return preds, targets
    image_ids = sorted(im["id"] for im in gt_json["images"])
    metric = DetectionMetrics(iou_thrs=[0.5, 0.75], score_thr="best_f1", box_format="xywh")
    batches = [metric.update(*arrays(image_ids[i:i + 50])) for i in range(0, len(image_ids), 50)]
    def digest(r):
        return "".join(np.ascontiguousarray(v).tobytes().hex()
                       for group in ("per_category", "micro", "macro") for v in r[group].values()) + \
            r["score_thr"].tobytes().hex()
    out["batch_metrics"] = "".join(digest(r) for r in batches) + digest(metric.compute())
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
    for key in ("precision", "recall", "scores", "stats", "per_instance", "per_instance_n",
                "outcomes", "detection_metrics", "batch_metrics"):
        assert one[key] == many[key], f"{iou_type}/{key} changed with thread count"
