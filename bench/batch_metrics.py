"""Time ``metrics.DetectionMetrics`` per batch and its ``best_f1`` sweep.

Each (arm, thread count) runs in a fresh process; arms and thread orders
alternate every round so host drift affects them alike. An arm is a directory
put first on ``sys.path`` (a checkout's ``python/`` with the compiled module
copied in), or ``.`` for the installed package.

    python bench/batch_metrics.py --arm base=/tmp/base/python --arm new=. \\
        --rounds 5 --out bench/results/batch-metrics.json

Batches come from the committed COCO slice (``tests/data``); the sweep runs on
seeded synthetic outcomes at COCO scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "tests" / "data" / "coco_subset_gt.json"
DT = ROOT / "tests" / "data" / "coco_subset_dt.json"


def _worker(reps: int) -> dict:
    import numpy as np
    from ultrafast_pycocotools import _detection_metrics
    from ultrafast_pycocotools.metrics import DetectionMetrics

    gt, dt = json.loads(GT.read_text()), json.loads(DT.read_text())
    ids = sorted(im["id"] for im in gt["images"])

    def arrays(image_ids):
        preds, targets = [], []
        for i in image_ids:
            g = [a for a in gt["annotations"] if a["image_id"] == i]
            d = [x for x in dt if x["image_id"] == i]
            targets.append({
                "boxes": np.array([a["bbox"] for a in g], float).reshape(-1, 4),
                "labels": np.array([a["category_id"] for a in g], int),
                "iscrowd": np.array([a["iscrowd"] for a in g], int),
            })
            preds.append({
                "boxes": np.array([x["bbox"] for x in d], float).reshape(-1, 4),
                "scores": np.array([x["score"] for x in d], float),
                "labels": np.array([x["category_id"] for x in d], int),
            })
        return preds, targets

    out = {}
    configs = {"fixed": {}, "best_f1": {"score_thr": "best_f1"},
               "10_iou": {"iou_thrs": np.linspace(0.5, 0.95, 10)}}
    for n in (16, len(ids)):
        preds, targets = arrays(ids[:n])
        for name, config in configs.items():
            metric = DetectionMetrics(box_format="xywh", **config)
            for _ in range(10):
                metric.reset()
                metric.update(preds, targets)
            wall, cpu = [], []
            for _ in range(reps):
                metric.reset()
                w, c = time.perf_counter(), time.process_time()
                metric.update(preds, targets)
                wall.append(time.perf_counter() - w)
                cpu.append(time.process_time() - c)
            out[f"update/{n}img/{name}"] = {"wall_ms": float(np.median(wall)) * 1e3,
                                           "cpu_ms": float(np.median(cpu)) * 1e3}

    rng = np.random.default_rng(0)
    for n_cats, n_thrs in ((80, 10), (1203, 1)):
        d = 750_000
        cats = rng.integers(0, n_cats, d)
        score = np.round(rng.random(d), 3)
        outcome = rng.choice([0, 1, 2], size=(d, n_thrs), p=[0.05, 0.35, 0.6]).astype(np.uint8)
        npig = np.bincount(cats, minlength=n_cats) // 2 + 1
        for optimize in ("macro", "micro"):
            w, c = time.perf_counter(), time.process_time()
            _detection_metrics.summarize_outcomes(
                cats, score, outcome, npig, np.arange(n_cats), np.linspace(0.5, 0.95, n_thrs),
                score_thr="best_f1", optimize=optimize)
            out[f"sweep/{d}det/{n_cats}cat/{n_thrs}iou/{optimize}"] = {
                "wall_ms": (time.perf_counter() - w) * 1e3, "cpu_ms": (time.process_time() - c) * 1e3}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", default=[], help="name=path; repeatable")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--threads", default="1,2")
    ap.add_argument("--out")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.reps)))
        return

    import numpy as np

    arms = dict(a.split("=", 1) for a in args.arm) or {"installed": "."}
    threads = [int(t) for t in args.threads.split(",")]
    runs: dict = {}
    for r in range(args.rounds):
        names = list(arms) if r % 2 == 0 else list(arms)[::-1]
        for t in (threads if r % 2 == 0 else threads[::-1]):
            for name in names:
                env = dict(os.environ, RAYON_NUM_THREADS=str(t))
                if arms[name] != ".":
                    env["PYTHONPATH"] = os.pathsep.join(
                        [str(Path(arms[name]).resolve())] + [p for p in [env.get("PYTHONPATH")] if p])
                proc = subprocess.run([sys.executable, __file__, "--worker", "--reps", str(args.reps)],
                                      env=env, capture_output=True, text=True, check=True)
                result = json.loads(proc.stdout.strip().splitlines()[-1])
                for key, value in result.items():
                    runs.setdefault(f"{name}/{t}t/{key}", []).append(value)
    summary = {
        key: {m: float(np.median([v[m] for v in values])) for m in ("wall_ms", "cpu_ms")}
        for key, values in runs.items()
    }
    report = {
        "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (GT, DT)},
        "arms": arms, "rounds": args.rounds, "reps": args.reps, "median": summary, "runs": runs,
    }
    for key, value in summary.items():
        print(f"{key:55s} wall {value['wall_ms']:9.3f} ms  cpu {value['cpu_ms']:9.3f} ms")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
