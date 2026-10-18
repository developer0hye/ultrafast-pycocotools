"""Time ``metrics.DetectionMetrics`` per batch and its ``best_f1`` sweep.

Each (arm, thread count) runs in a fresh process; arms and thread orders
alternate every round so host drift affects them alike. An arm is a directory
put first on ``sys.path`` (a checkout's ``python/`` with its own compiled
module copied in), or ``.`` for the installed package.

    python bench/batch_metrics.py --arm base=/tmp/base/python --arm new=. \\
        --label base=COMMIT --label new=COMMIT \\
        --rounds 5 --out bench/results/batch-metrics.json

Batches:

* ``coco``: the committed COCO slice (``tests/data``), 16 and 93 images.
* ``detector``: the slice's ground truth with 100 seeded synthetic detections
  per image, as a detector's raw output looks: half near ground truth (mostly
  the right class), half background, unsorted.

The sweep runs on seeded synthetic outcomes at COCO scale. ``--comparators``
also times pycocotools (``COCO`` + ``loadRes`` + ``evaluate`` +
``accumulate`` per batch) and a plain NumPy per-image IoU and greedy match at
IoU 0.5. Neither computes the same thing (pycocotools computes AP; the NumPy
baseline has no crowd or ignore rules); they give the scale of what a batch
costs otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "tests" / "data" / "coco_subset_gt.json"
DT = ROOT / "tests" / "data" / "coco_subset_dt.json"


def _batches():
    import numpy as np

    gt, dt = json.loads(GT.read_text()), json.loads(DT.read_text())
    ids = sorted(im["id"] for im in gt["images"])
    cats = sorted(c["id"] for c in gt["categories"])
    by_image = {i: [a for a in gt["annotations"] if a["image_id"] == i] for i in ids}

    def target(i):
        g = by_image[i]
        return {
            "boxes": np.array([a["bbox"] for a in g], float).reshape(-1, 4),
            "labels": np.array([a["category_id"] for a in g], int),
            "iscrowd": np.array([a["iscrowd"] for a in g], int),
        }

    def coco(image_ids):
        preds = []
        for i in image_ids:
            d = [x for x in dt if x["image_id"] == i]
            preds.append({
                "boxes": np.array([x["bbox"] for x in d], float).reshape(-1, 4),
                "scores": np.array([x["score"] for x in d], float),
                "labels": np.array([x["category_id"] for x in d], int),
            })
        return preds, [target(i) for i in image_ids]

    def detector(image_ids, per_image=100):
        rng = np.random.default_rng(0)
        preds, targets = [], []
        for i in image_ids:
            t = target(i)
            gb, gl = t["boxes"], t["labels"]
            n_near = per_image // 2 if len(gl) else 0
            src = rng.integers(0, max(len(gl), 1), n_near)
            near = gb[src] * (1 + rng.normal(0, 0.15, (n_near, 4))) if len(gl) else np.zeros((0, 4))
            near_l = np.where(rng.random(n_near) < 0.8, gl[src] if len(gl) else 0, rng.choice(cats, n_near))
            n_far = per_image - n_near
            far = np.column_stack([rng.random(n_far) * 500, rng.random(n_far) * 400,
                                   10 + rng.random((n_far, 2)) * 150])
            scores = np.concatenate([0.3 + 0.7 * rng.random(n_near), 0.6 * rng.random(n_far)])
            preds.append({"boxes": np.vstack([near, far]), "scores": np.round(scores, 4),
                          "labels": np.concatenate([near_l, rng.choice(cats, n_far)]).astype(int)})
            targets.append(t)
        return preds, targets

    return {
        "coco/16img": coco(ids[:16]),
        "coco/93img": coco(ids),
        "detector/16img": detector(ids[:16]),
        "detector/64img": detector(ids[:64]),
    }


def _time(fn, reps, warmup=10):
    import numpy as np

    for _ in range(warmup):
        fn()
    wall, cpu = [], []
    for _ in range(reps):
        w, c = time.perf_counter(), time.process_time()
        fn()
        wall.append(time.perf_counter() - w)
        cpu.append(time.process_time() - c)
    return {"wall_ms": float(np.median(wall)) * 1e3, "cpu_ms": float(np.median(cpu)) * 1e3}


def _worker(reps: int) -> dict:
    import numpy as np
    from ultrafast_pycocotools import _detection_metrics
    from ultrafast_pycocotools.metrics import DetectionMetrics

    out = {}
    ten = np.linspace(0.5, 0.95, 10)
    configs = {"fixed": {}, "best_f1": {"score_thr": "best_f1"}, "10_iou": {"iou_thrs": ten},
               "agnostic_10_iou": {"class_agnostic": True, "iou_thrs": ten}}
    for batch, (preds, targets) in _batches().items():
        for name, config in configs.items():
            metric = DetectionMetrics(box_format="xywh", **config)

            def update():
                metric.reset()
                metric.update(preds, targets)

            out[f"update/{batch}/{name}"] = _time(update, reps)

    rng = np.random.default_rng(0)
    for n_cats, n_thrs in ((80, 10), (1203, 1)):
        d = 750_000
        cats = rng.integers(0, n_cats, d)
        score = np.round(rng.random(d), 3)
        outcome = rng.choice([0, 1, 2], size=(d, n_thrs), p=[0.05, 0.35, 0.6]).astype(np.uint8)
        npig = np.bincount(cats, minlength=n_cats) // 2 + 1
        for optimize in ("macro", "micro"):
            out[f"sweep/{d}det/{n_cats}cat/{n_thrs}iou/{optimize}"] = _time(
                lambda: _detection_metrics.summarize_outcomes(
                    cats, score, outcome, npig, np.arange(n_cats), np.linspace(0.5, 0.95, n_thrs),
                    score_thr="best_f1", optimize=optimize),
                1, warmup=0)
    return out


def _iou_xywh(a, b):
    import numpy as np

    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    w = np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0])
    h = np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(w, 0, None) * np.clip(h, 0, None)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(inter > 0, inter / union, 0.0)


def _numpy_greedy(preds, targets, thr=0.5):
    """Per image: IoU matrix, then each detection in score order takes the
    best free ground truth of its label at IoU >= thr."""
    import numpy as np

    tp = fp = fn = 0
    for p, t in zip(preds, targets):
        order = np.argsort(-p["scores"], kind="stable")
        iou = _iou_xywh(p["boxes"][order], t["boxes"])
        iou = np.where(p["labels"][order][:, None] == t["labels"][None], iou, -1.0)
        taken = np.zeros(len(t["labels"]), bool)
        for row in iou:
            row = np.where(taken, -1.0, row)
            g = int(row.argmax()) if row.size else -1
            if g >= 0 and row[g] >= thr:
                taken[g] = True
                tp += 1
            else:
                fp += 1
        fn += int((~taken).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _pycocotools(preds, targets):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets, labels = [], [], [], set()
    for i, (p, t) in enumerate(zip(preds, targets)):
        images.append({"id": i})
        for b, c, z in zip(t["boxes"].tolist(), t["labels"].tolist(), t["iscrowd"].tolist()):
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c, "bbox": b,
                         "area": b[2] * b[3], "iscrowd": z})
        dets.extend({"image_id": i, "category_id": c, "bbox": b, "score": s}
                    for b, c, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()))
        labels.update(t["labels"].tolist())
        labels.update(p["labels"].tolist())
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in sorted(labels)]}
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.evaluate()
        ev.accumulate()
    return np.asarray(ev.eval["precision"])


def _comparators(reps: int) -> dict:
    out = {}
    for batch, (preds, targets) in _batches().items():
        if batch.endswith("16img"):
            out[f"numpy_greedy/{batch}"] = _time(lambda: _numpy_greedy(preds, targets), max(reps // 10, 5))
            out[f"pycocotools/{batch}"] = _time(lambda: _pycocotools(preds, targets), 5, warmup=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", default=[], help="name=path; repeatable")
    ap.add_argument("--label", action="append", default=[],
                    help="name=text recorded for an arm instead of its path, e.g. a commit")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--threads", default="1,2")
    ap.add_argument("--comparators", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--worker", choices=["metrics", "comparators"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        work = _worker if args.worker == "metrics" else _comparators
        print(json.dumps(work(args.reps)))
        return

    import numpy as np

    arms = dict(a.split("=", 1) for a in args.arm) or {"installed": "."}
    threads = [int(t) for t in args.threads.split(",")]
    runs: dict = {}

    def run(kind, env, prefix):
        proc = subprocess.run([sys.executable, __file__, "--worker", kind, "--reps", str(args.reps)],
                              env=env, capture_output=True, text=True, check=True)
        for key, value in json.loads(proc.stdout.strip().splitlines()[-1]).items():
            runs.setdefault(f"{prefix}/{key}", []).append(value)

    for r in range(args.rounds):
        names = list(arms) if r % 2 == 0 else list(arms)[::-1]
        for t in (threads if r % 2 == 0 else threads[::-1]):
            for name in names:
                env = dict(os.environ, RAYON_NUM_THREADS=str(t))
                if arms[name] != ".":
                    env["PYTHONPATH"] = os.pathsep.join(
                        [str(Path(arms[name]).resolve())] + [p for p in [env.get("PYTHONPATH")] if p])
                run("metrics", env, f"{name}/{t}t")
        if args.comparators:
            run("comparators", dict(os.environ), "comparator")
    summary = {
        key: {m: float(np.median([v[m] for v in values])) for m in ("wall_ms", "cpu_ms")}
        for key, values in runs.items()
    }
    report = {
        "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (GT, DT)},
        "arms": {**arms, **dict(label.split("=", 1) for label in args.label)},
        "rounds": args.rounds, "reps": args.reps, "median": summary, "runs": runs,
    }
    for key, value in summary.items():
        print(f"{key:60s} wall {value['wall_ms']:10.3f} ms  cpu {value['cpu_ms']:10.3f} ms")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
