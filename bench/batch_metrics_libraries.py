"""Compare per-batch detection metrics across libraries.

Two tasks on COCO val2017 with the saved YOLO26n predictions of the published
evidence archive (``--inputs``: ``instances_val2017.json`` and
``detect-predictions.json``):

* ``batch``: score one batch of 16 images from scratch (new metric object,
  add the batch, compute), as a per-batch log line in a training loop does.
* ``stream``: add all 5,000 images in batches of 16, then compute once, as a
  validation loop does. Run once per process; the peak RSS the run adds over
  the RSS before it is recorded (Linux ``VmHWM``, reset first).

Each library receives its own input format, built before timing: NumPy arrays
or tensors where the API takes them, ``supervision.Detections`` for
supervision. The COCO-API libraries take annotation dictionaries, so turning a
batch's arrays into them is part of their timed work, as it is for anyone who
uses them per batch.

The libraries do not compute the same metric (see the report); the values
each reports are saved next to its timings. Usage::

    python bench/batch_metrics_libraries.py --inputs DIR --work DIR --out OUT.json \\
        [--impl NAME ...] [--rounds 3]

Every (implementation, task, config) runs in its own process; rounds
alternate the implementation order.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BATCH = 16
PACKAGES = ("ultrafast-pycocotools", "torch", "torchmetrics", "supervision", "ultralytics",
            "faster-coco-eval", "hotcoco", "pycocotools", "numpy")


def load_images(inputs: Path, work: Path):
    """Per-image arrays: COCO ``xywh`` boxes, category ids, crowd flags, areas."""
    cache = work / "val2017-images.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    gt = json.loads((inputs / "instances_val2017.json").read_text())
    dt = json.loads((inputs / "detect-predictions.json").read_text())
    ids = sorted(im["id"] for im in gt["images"])
    g_by = {i: [] for i in ids}
    d_by = {i: [] for i in ids}
    for a in gt["annotations"]:
        g_by[a["image_id"]].append(a)
    for x in dt:
        d_by[x["image_id"]].append(x)
    images = []
    for i in ids:
        g, d = g_by[i], d_by[i]
        images.append({
            "image_id": i,
            "gt_boxes": np.array([a["bbox"] for a in g], float).reshape(-1, 4),
            "gt_labels": np.array([a["category_id"] for a in g], np.int64),
            "gt_crowd": np.array([a["iscrowd"] for a in g], np.int64),
            "gt_area": np.array([a["area"] for a in g], float),
            "dt_boxes": np.array([x["bbox"] for x in d], float).reshape(-1, 4),
            "dt_scores": np.array([x["score"] for x in d], float),
            "dt_labels": np.array([x["category_id"] for x in d], np.int64),
        })
    work.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(images))
    return images


def xyxy(boxes):
    return np.column_stack([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]]) if len(boxes) else boxes.reshape(0, 4)


def coco_dicts(images):
    """A batch as COCO ground-truth and result dictionaries."""
    imgs, anns, dets, cats = [], [], [], set()
    for im in images:
        i = im["image_id"]
        imgs.append({"id": i, "width": 640, "height": 640})
        for b, c, z, a in zip(im["gt_boxes"].tolist(), im["gt_labels"].tolist(),
                              im["gt_crowd"].tolist(), im["gt_area"].tolist()):
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c, "bbox": b,
                         "area": a, "iscrowd": z})
        dets.extend({"image_id": i, "category_id": c, "bbox": b, "score": s}
                    for b, c, s in zip(im["dt_boxes"].tolist(), im["dt_labels"].tolist(),
                                       im["dt_scores"].tolist()))
        cats.update(im["gt_labels"].tolist())
        cats.update(im["dt_labels"].tolist())
    return {"images": imgs, "annotations": anns,
            "categories": [{"id": c, "name": str(c)} for c in sorted(cats)]}, dets


# ---------------------------------------------------------------------------
# Implementations. ``prepare`` runs before timing; ``run`` is timed and takes
# a list of batches (one batch for the batch task).
# ---------------------------------------------------------------------------


class Ufcoco:
    configs = {"f1": {}, "best_f1": {"score_thr": "best_f1"}}

    def __init__(self, config, tensors=False):
        self.config = self.configs[config]
        self.tensors = tensors

    def prepare(self, images):
        if self.tensors:
            import torch

            convert = torch.from_numpy
        else:
            def convert(x):
                return x
        preds = [{"boxes": convert(im["dt_boxes"]), "scores": convert(im["dt_scores"]),
                  "labels": convert(im["dt_labels"])} for im in images]
        targets = [{"boxes": convert(im["gt_boxes"]), "labels": convert(im["gt_labels"]),
                    "iscrowd": convert(im["gt_crowd"]), "area": convert(im["gt_area"])} for im in images]
        return preds, targets

    def run(self, batches):
        from ultrafast_pycocotools.metrics import DetectionMetrics

        metric = DetectionMetrics(box_format="xywh", iou_thrs=0.5, batch_results=False, **self.config)
        for preds, targets in batches:
            metric.update(preds, targets)
        r = metric.compute()
        return {"micro_p": r["micro"]["precision"], "micro_r": r["micro"]["recall"],
                "micro_f1": r["micro"]["f1"], "macro_p": r["macro"]["precision"],
                "macro_r": r["macro"]["recall"], "macro_f1": r["macro"]["f1"],
                "score_thr": r["score_thr"]}


class TorchMetrics:
    def __init__(self, backend):
        self.backend = backend

    def prepare(self, images):
        import torch

        t = torch.from_numpy
        preds = [{"boxes": t(im["dt_boxes"]).float(), "scores": t(im["dt_scores"]).float(),
                  "labels": t(im["dt_labels"])} for im in images]
        targets = [{"boxes": t(im["gt_boxes"]).float(), "labels": t(im["gt_labels"]),
                    "iscrowd": t(im["gt_crowd"]), "area": t(im["gt_area"]).float()} for im in images]
        return preds, targets

    def run(self, batches):
        from torchmetrics.detection import MeanAveragePrecision

        metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox", backend=self.backend)
        for preds, targets in batches:
            metric.update(preds, targets)
        with contextlib.redirect_stdout(io.StringIO()):
            r = metric.compute()
        return {"map": float(r["map"]), "map_50": float(r["map_50"])}


class Supervision:
    def __init__(self, averaging):
        self.averaging = averaging

    def prepare(self, images):
        import supervision as sv

        preds = [sv.Detections(xyxy=xyxy(im["dt_boxes"]), confidence=im["dt_scores"],
                               class_id=im["dt_labels"]) for im in images]
        targets = [sv.Detections(xyxy=xyxy(im["gt_boxes"]), class_id=im["gt_labels"]) for im in images]
        return preds, targets

    def run(self, batches):
        from supervision.metrics import AveragingMethod, F1Score

        metric = F1Score(averaging_method=getattr(AveragingMethod, self.averaging.upper()))
        for preds, targets in batches:
            metric.update(preds, targets)
        r = metric.compute()
        return {f"{self.averaging}_f1_50": float(r.f1_50), f"{self.averaging}_f1_50_95": float(r.f1_scores.mean())}


class Ultralytics:
    """``DetectionValidator.update_metrics``' path: per image ``box_iou`` and
    ``match_predictions``, ``DetMetrics.update_stats``, then ``process``."""

    def prepare(self, images):
        import torch

        t = torch.from_numpy
        cats = sorted({c for im in images for c in im["gt_labels"].tolist() + im["dt_labels"].tolist()})
        index = {c: k for k, c in enumerate(cats)}
        self.names = {k: str(c) for c, k in index.items()}
        out = []
        for im in images:
            out.append({
                "gt": t(xyxy(im["gt_boxes"])).float(),
                "gt_cls": t(np.array([index[c] for c in im["gt_labels"].tolist()], np.int64)).float(),
                "dt": t(xyxy(im["dt_boxes"])).float(),
                "conf": t(im["dt_scores"]).float(),
                "dt_cls": t(np.array([index[c] for c in im["dt_labels"].tolist()], np.int64)).float(),
                "name": str(im["image_id"]),
            })
        return out

    def run(self, batches):
        import torch
        from ultralytics.engine.validator import BaseValidator
        from ultralytics.utils.metrics import DetMetrics, box_iou

        class Iouv:
            iouv = torch.linspace(0.5, 0.95, 10)

        metrics = DetMetrics(names=self.names)
        for batch in batches:
            for im in batch:
                if len(im["gt"]) and len(im["dt"]):
                    iou = box_iou(im["gt"], im["dt"])
                    tp = BaseValidator.match_predictions(Iouv, im["dt_cls"], im["gt_cls"], iou).cpu().numpy()
                else:
                    tp = np.zeros((len(im["dt"]), 10), dtype=bool)
                cls = im["gt_cls"].numpy()
                metrics.update_stats({"tp": tp, "conf": im["conf"].numpy(), "pred_cls": im["dt_cls"].numpy(),
                                      "target_cls": cls, "target_img": np.unique(cls), "im_name": im["name"]})
        with contextlib.redirect_stdout(io.StringIO()):
            metrics.process()
        box = metrics.box
        return {"macro_p": float(box.mp), "macro_r": float(box.mr), "macro_f1": float(np.mean(box.f1)),
                "map50": float(box.map50), "map": float(box.map)}


class CocoApi:
    """pycocotools, faster-coco-eval or hotcoco on COCO dictionaries."""

    def __init__(self, name):
        self.name = name

    def prepare(self, images):
        return images

    def run(self, batches):
        gt, dets = {"images": [], "annotations": [], "categories": []}, []
        cats = set()
        for images in batches:
            g, d = coco_dicts(images)
            gt["images"].extend(g["images"])
            base = len(gt["annotations"])
            for a in g["annotations"]:
                a["id"] += base
            gt["annotations"].extend(g["annotations"])
            cats.update(c["id"] for c in g["categories"])
            dets.extend(d)
        gt["categories"] = [{"id": c, "name": str(c)} for c in sorted(cats)]
        with contextlib.redirect_stdout(io.StringIO()):
            if self.name == "pycocotools":
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                coco = COCO()
                coco.dataset = gt
                coco.createIndex()
                ev = COCOeval(coco, coco.loadRes(dets), "bbox")
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
                return {"map": float(ev.stats[0]), "map_50": float(ev.stats[1])}
            if self.name == "faster-coco-eval":
                from faster_coco_eval import COCO, COCOeval_faster

                coco = COCO(gt)
                ev = COCOeval_faster(coco, coco.loadRes(dets), "bbox", extra_calc=True,
                                     print_function=lambda *a: None)
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
                m = ev.extended_metrics
                return {"map": float(ev.stats[0]), "map_50": float(ev.stats[1]),
                        "macro_p": float(m["precision"]), "macro_r": float(m["recall"])}
            import hotcoco

            coco = hotcoco.COCO(gt)
            ev = hotcoco.COCOeval(coco, coco.load_res(dets), "bbox")
            ev.evaluate()
            ev.accumulate()
            f = ev.f_scores()
            ev.summarize()
            return {"map": float(ev.stats[0]), "map_50": float(ev.stats[1]), "f1_50": float(f["F1_50"])}


IMPLS = {
    "ufcoco": lambda c: Ufcoco(c),
    "ufcoco-torch": lambda c: Ufcoco(c, tensors=True),
    "torchmetrics-pycocotools": lambda c: TorchMetrics("pycocotools"),
    "torchmetrics-faster": lambda c: TorchMetrics("faster_coco_eval"),
    "supervision": lambda c: Supervision(c),
    "ultralytics": lambda c: Ultralytics(),
    "faster-coco-eval": lambda c: CocoApi("faster-coco-eval"),
    "hotcoco": lambda c: CocoApi("hotcoco"),
    "pycocotools": lambda c: CocoApi("pycocotools"),
}
# The configurations each implementation is timed with.
CONFIGS = {"ufcoco": ("f1", "best_f1"), "ufcoco-torch": ("f1", "best_f1"),
           "supervision": ("micro", "macro")}


def _status_kb(field: str) -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    raise KeyError(field)


def worker(args) -> dict:
    images = load_images(args.inputs, args.work)
    impl = IMPLS[args.impl](args.config)
    if args.task == "batch":
        native = impl.prepare(images[:BATCH])
        impl.run([native])  # warm-up: imports and first-call costs
        wall, cpu, value = [], [], None
        deadline = time.perf_counter() + args.budget
        while len(wall) < args.max_reps and (len(wall) < 3 or time.perf_counter() < deadline):
            w, c = time.perf_counter(), time.process_time()
            value = impl.run([native])
            wall.append(time.perf_counter() - w)
            cpu.append(time.process_time() - c)
        return {"wall_ms": float(np.median(wall)) * 1e3, "cpu_ms": float(np.median(cpu)) * 1e3,
                "reps": len(wall), "value": value}
    batches = [images[i:i + BATCH] for i in range(0, len(images), BATCH)]
    native = [impl.prepare(b) for b in batches]
    # Linux: writing 5 to clear_refs resets the peak (VmHWM) to the current
    # RSS, so the peak read afterwards is the run's own.
    Path("/proc/self/clear_refs").write_text("5")
    rss_before = _status_kb("VmRSS")
    w, c = time.perf_counter(), time.process_time()
    value = impl.run(native)
    wall, cpu = time.perf_counter() - w, time.process_time() - c
    peak = _status_kb("VmHWM")
    return {"wall_ms": wall * 1e3, "cpu_ms": cpu * 1e3, "value": value,
            "peak_rss_mb": peak / 1024, "rss_before_mb": rss_before / 1024,
            "added_peak_mb": (peak - rss_before) / 1024}


def _wait_for_load(limit) -> float:
    """The 1-minute load average once it is below ``limit`` (or now)."""
    deadline = time.time() + 1800
    while limit is not None and os.getloadavg()[0] >= limit and time.time() < deadline:
        time.sleep(10)
    return os.getloadavg()[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="cache directory for the per-image arrays")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--impl", nargs="*", default=list(IMPLS))
    ap.add_argument("--task", nargs="*", default=["batch", "stream"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1, help="RAYON / OMP / torch threads")
    ap.add_argument("--budget", type=float, default=3.0, help="seconds of timed batch repetitions")
    ap.add_argument("--max-reps", type=int, default=300)
    ap.add_argument("--max-load", type=float, default=None,
                    help="before each process, wait (up to 30 min) for the 1-minute load average to drop below this")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        args.impl, args.task = args.impl[0], args.task[0]
        print(json.dumps(worker(args)))
        return

    load_images(args.inputs, args.work)
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    runs: dict = {}
    for r in range(args.rounds):
        order = args.impl if r % 2 == 0 else args.impl[::-1]
        for impl in order:
            for task in args.task:
                for config in CONFIGS.get(impl, ("",)):
                    cmd = [sys.executable, __file__, "--worker", "--inputs", str(args.inputs),
                           "--work", str(args.work), "--impl", impl, "--task", task, "--config", config,
                           "--budget", str(args.budget), "--max-reps", str(args.max_reps)]
                    load_before = _wait_for_load(args.max_load)
                    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                    load_after = os.getloadavg()[0]
                    key = f"{impl}{'/' + config if config else ''}/{task}"
                    if proc.returncode != 0:
                        runs.setdefault(key, []).append({"error": proc.stderr.strip().splitlines()[-1]})
                        print(f"{key}: FAILED {proc.stderr.strip().splitlines()[-1]}", flush=True)
                        continue
                    result = json.loads(proc.stdout.strip().splitlines()[-1])
                    result.update(load_before=load_before, load_after=load_after)
                    runs.setdefault(key, []).append(result)
                    print(f"round {r} {key:40s} wall {result['wall_ms']:10.3f} ms  cpu {result['cpu_ms']:10.3f} ms",
                          flush=True)
    summary = {}
    for key, values in runs.items():
        ok = [v for v in values if "error" not in v]
        if not ok:
            continue
        summary[key] = {m: float(np.median([v[m] for v in ok])) for m in ("wall_ms", "cpu_ms")}
        for m in ("peak_rss_mb", "rss_before_mb", "added_peak_mb"):
            if m in ok[0]:
                summary[key][m] = float(np.median([v[m] for v in ok]))
        summary[key]["value"] = ok[0]["value"]
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    report = {"batch_images": BATCH, "threads": args.threads, "rounds": args.rounds,
              "versions": versions, "median": summary, "runs": runs}
    for key, value in summary.items():
        print(f"{key:40s} wall {value['wall_ms']:10.3f} ms  cpu {value['cpu_ms']:10.3f} ms  {value['value']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
