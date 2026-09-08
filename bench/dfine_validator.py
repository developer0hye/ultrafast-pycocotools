"""Replay real D-FINE-seg validation inputs with balanced, fresh-process timing.

Run with the patched D-FINE-seg environment (plus psutil). See docs/dfine-seg.md.
Preparation uses the training Loader and Trainer.get_preds_and_gt unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import urllib.request

import psutil

BACKENDS = ("faster_coco_eval", "ultrafast")
CASES = ("bbox", "bbox_segm")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot():
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    return dict(
        utc=datetime.now(timezone.utc).isoformat(),
        cpu_percent=psutil.cpu_percent(),
        load_average=list(os.getloadavg()),
        available_memory_bytes=memory.available,
        memory_percent=memory.percent,
        swap_used_bytes=swap.used,
        swap_in_bytes=swap.sin,
        swap_out_bytes=swap.sout,
    )


def spread(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def torch_setup(threads):
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    return torch


def prepare(args):
    torch = torch_setup(args.threads)
    from huggingface_hub import HfApi, hf_hub_download
    from omegaconf import OmegaConf
    from dfine_seg.dl.dataset import Loader
    from dfine_seg.dl.train import Trainer
    from dfine_seg.model.dfine import build_model

    args.out.mkdir(parents=True, exist_ok=True)
    dataset = args.out / "dataset"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    coco = json.loads(args.annotations.read_text())
    images = sorted(coco["images"], key=lambda x: x["id"])[: args.images]
    if len(images) != args.images:
        raise ValueError("Not enough source images")
    ids = {item["id"] for item in images}
    annotations = [a for a in coco["annotations"] if a["image_id"] in ids]
    subset = {**coco, "images": images, "annotations": annotations}
    # Loader requires both split manifests; only the val loader is executed.
    for split in ("train", "val"):
        write_json(dataset / f"{split}.json", subset)

    def download(item):
        path = dataset / "images" / item["file_name"]
        if not path.is_file():
            url = (
                "https://s3.amazonaws.com/images.cocodataset.org/val2017/"
                + item["file_name"]
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        data = response.read()
                    path.with_suffix(".part").write_bytes(data)
                    path.with_suffix(".part").replace(path)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(attempt + 1)
        return {
            "image_id": item["id"],
            "file_name": item["file_name"],
            "sha256": digest(path),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        image_manifest = []
        for entry in pool.map(download, images):
            image_manifest.append(entry)
            if len(image_manifest) % 50 == 0:
                print(f"Downloaded {len(image_manifest)}/{len(images)}", flush=True)

    repo = "ArgoSA/D-FINE-seg"
    revision = args.revision or HfApi().model_info(repo).sha
    weights = hf_hub_download(repo, "dfine_seg_s_coco.pt", revision=revision)
    print(f"Checkpoint {revision}: {weights}", flush=True)
    cfg = OmegaConf.load(args.dfine / "dfine_seg/config/default.yaml")
    cfg.task = "segment"
    cfg.model_name = "s"
    cfg.train.root = str(args.out)
    cfg.train.data_path = str(dataset)
    cfg.train.path_to_save = str(args.out / "unused")
    cfg.train.coco_dataset = True
    cfg.train.label_to_name = {
        i: c["name"]
        for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))
    }
    cfg.train.img_size = [640, 640]
    cfg.train.keep_ratio = False
    cfg.train.debug_img_processing = False
    loader = Loader(dataset, (640, 640), 1, 0, cfg, debug_img_processing=False)
    val_loader = loader._build_dataloader_impl(loader._make_dataset("val"))
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = build_model(
        "s",
        len(coco["categories"]),
        True,
        trainer.device,
        img_size=(640, 640),
        pretrained_model_path=weights,
        task="segment",
    )
    trainer.ema_model = None
    trainer.amp_enabled = False
    trainer.is_main = False
    trainer.keep_ratio = False
    trainer.num_labels = len(coco["categories"])
    trainer.conf_thresh = 0.5
    trainer.to_visualize_eval = False
    seen = []

    def batches():
        for index, batch in enumerate(val_loader):
            if index % 25 == 0:
                print(f"Inference {index}/{len(images)}", flush=True)
            seen.extend(str(p) for p in batch[2])
            yield batch

    start, cpu = time.perf_counter(), time.process_time()
    gt, preds = trainer.get_preds_and_gt(batches())
    elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
    if len(gt) != len(images) or [Path(p).name for p in seen] != [
        i["file_name"] for i in images
    ]:
        raise AssertionError("Validation loader skipped or reordered images")
    inputs = args.out / "inputs.pt"
    torch.save(
        dict(gt=gt, preds=preds, label_to_name=dict(cfg.train.label_to_name)), inputs
    )
    provenance = dict(
        image_selection="First N val2017 image IDs in ascending numeric order; no exclusions",
        images=image_manifest,
        source_annotations_sha256=digest(args.annotations),
        subset_annotations_sha256=digest(dataset / "val.json"),
        checkpoint_repo=repo,
        checkpoint_revision=revision,
        checkpoint_sha256=digest(weights),
        checkpoint_file="dfine_seg_s_coco.pt",
        inputs_sha256=digest(inputs),
        dfine_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.dfine, text=True
        ).strip(),
        inference_and_data_wall_seconds=elapsed,
        inference_and_data_cpu_seconds=cpu_elapsed,
        inference_device="cpu",
        inference_threads=args.threads,
        batch_size=1,
        amp=False,
        img_size=[640, 640],
        keep_ratio=False,
        conf_thresh=0.5,
        annotations=len(annotations),
        crowd_annotations=sum(a.get("iscrowd", 0) for a in annotations),
        gt_objects=sum(len(g["labels"]) for g in gt),
        raw_bbox_predictions=sum(len(p["all_labels"]) for p in preds),
        thresholded_predictions=sum(len(p["labels"]) for p in preds),
        gt_masks=sum(len(g.get("masks_rle", [])) for g in gt),
        predicted_masks=sum(len(p.get("masks_rle", [])) for p in preds),
        annotation_policy="Unmodified D-FINE-seg Loader: crowds omitted, polygons rasterized; "
        "Validator does not receive original COCO area/iscrowd fields",
    )
    write_json(args.out / "inputs.json", provenance)
    print(f"Prepared {len(gt)} images in {elapsed:.3f}s: {inputs}", flush=True)


def plain(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def worker(args):
    torch = torch_setup(args.threads)
    import numpy as np
    from dfine_seg.dl.validator import Validator

    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if args.case == "bbox":
        for sample in data["gt"] + data["preds"]:
            for key in ("masks", "masks_rle", "masks_size", "mask_probs"):
                sample.pop(key, None)
    phases, captured = {}, {}
    start, cpu = time.perf_counter(), time.process_time()
    validator = Validator(**data, coco_backend=args.backend, mask_batch_size=150)
    phases["prepare_update"] = dict(
        wall=time.perf_counter() - start, cpu=time.process_time() - cpu
    )

    def wrap(obj, method, label, capture=False):
        original = getattr(obj, method)

        def measured(*a, **kw):
            t, c = time.perf_counter(), time.process_time()
            result = original(*a, **kw)
            phases[label] = dict(
                wall=time.perf_counter() - t, cpu=time.process_time() - c
            )
            if capture:
                captured[label] = result
            return result

        setattr(obj, method, measured)

    wrap(validator.torch_metric, "compute", "bbox_map", True)
    if validator.use_masks:
        wrap(validator.torch_metric_mask, "compute", "segm_map", True)
    wrap(validator, "_compute_main_metrics", "f1_iou")
    wrap(validator, "_cleanup_torchmetrics", "cleanup")
    if args.diagnostic:
        validator.torch_metric.extended_summary = True
        if validator.use_masks:
            validator.torch_metric_mask.extended_summary = True
    t, c = time.perf_counter(), time.process_time()
    metrics = validator.compute_metrics(extended=True)
    phases["compute_metrics"] = dict(
        wall=time.perf_counter() - t, cpu=time.process_time() - c
    )
    phases["total"] = {
        k: phases["prepare_update"][k] + phases["compute_metrics"][k]
        for k in ("wall", "cpu")
    }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(
        backend=args.backend,
        case=args.case,
        phases=phases,
        metrics=plain(metrics),
        coco_stats={
            name: plain(
                {
                    k: v
                    for k, v in values.items()
                    if k not in ("ious", "precision", "recall", "scores")
                }
            )
            for name, values in captured.items()
        },
        peak_rss_MiB=rss / (1024**2 if sys.platform == "darwin" else 1024),
    )
    if args.diagnostic:
        arrays = {
            f"{name}_{key}": value.cpu().numpy()
            for name, values in captured.items()
            for key, value in values.items()
            if key in ("precision", "recall", "scores")
        }
        np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    write_json(args.out, result)
    print(f"{args.case} {args.backend}: {phases['total']['wall']:.4f}s", flush=True)


def compare(a, b):
    import numpy as np

    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, (list, float, int)):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=False)
    else:
        assert a == b


def run(args):
    import numpy as np

    args.out.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "RAYON_NUM_THREADS": str(args.threads),
        "OMP_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    source_root = Path(importlib.util.find_spec("dfine_seg").origin).parent
    result = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        method="One excluded fresh-process warmup per case/backend, then alternating backend "
        "and case order. No timing-based exclusions. Imports and input file loading "
        "are excluded; Validator construction/update, compute_metrics(extended=True), "
        "F1/IoU and cleanup are included. Separate extended-summary curve diagnostics "
        "are excluded. Whole-host telemetry includes worker startup and serialization.",
        dfine_source_sha256={
            str(p.relative_to(source_root)): digest(p)
            for p in sorted(source_root.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".yaml")
        },
        evaluation_device="cpu",
        mask_batch_size=150,
        conf_thresh=0.5,
        iou_thresh=0.5,
        rounds=args.rounds,
        command=[sys.executable, *sys.argv],
        script_sha256=digest(__file__),
        input_provenance=json.loads(args.inputs.with_suffix(".json").read_text()),
        inputs_sha256=digest(args.inputs),
        platform=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        logical_cpus=psutil.cpu_count(),
        physical_cpus=psutil.cpu_count(logical=False),
        memory_bytes=psutil.virtual_memory().total,
        versions={
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "torchvision",
                "torchmetrics",
                "numpy",
                "faster-coco-eval",
                "ultrafast-pycocotools",
                "psutil",
            )
        },
        threads=args.threads,
        runs=[],
    )
    assert result["inputs_sha256"] == result["input_provenance"]["inputs_sha256"]
    psutil.cpu_percent()
    result["baseline"] = []
    for _ in range(5):
        time.sleep(1)
        result["baseline"].append(snapshot())

    def save():
        write_json(args.out / "results.json", result)

    def execute(case, backend, round_index, diagnostic=False):
        label = f"{'diagnostic' if diagnostic else 'warmup' if round_index < 0 else f'round-{round_index + 1}'}-{case}-{backend}"
        path = args.out / f"{label}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--inputs",
            str(args.inputs),
            "--out",
            str(path),
            "--threads",
            str(args.threads),
            "--case",
            case,
            "--backend",
            backend,
        ]
        if diagnostic:
            command.append("--diagnostic")
        telemetry = [snapshot()]
        with path.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            while process.poll() is None:
                time.sleep(1)
                telemetry.append(snapshot())
        if process.returncode:
            raise RuntimeError(f"Worker failed: {path.with_suffix('.log')}")
        measurement = json.loads(path.read_text())
        result["runs"].append(
            dict(
                label=label,
                round=round_index,
                warmup=round_index < 0,
                diagnostic=diagnostic,
                case=case,
                backend=backend,
                measurement=measurement,
                telemetry=telemetry,
            )
        )
        save()
        print(f"{label}: {measurement['phases']['total']['wall']:.4f}s", flush=True)
        return measurement

    for round_index in range(-1, args.rounds):
        for case in CASES if round_index % 2 == 0 else CASES[::-1]:
            pair = {
                b: execute(case, b, round_index)
                for b in (BACKENDS if round_index % 2 == 0 else BACKENDS[::-1])
            }
            for key in ("metrics", "coco_stats"):
                compare(pair[BACKENDS[0]][key], pair[BACKENDS[1]][key])
    result["summary"] = {}
    result["paired_total_speedup"] = {}
    for case in CASES:
        result["summary"][case] = {}
        for backend in BACKENDS:
            runs = [
                r["measurement"]
                for r in result["runs"]
                if r["case"] == case and r["backend"] == backend and not r["warmup"]
            ]
            for r in runs[1:]:
                compare(r["metrics"], runs[0]["metrics"])
                compare(r["coco_stats"], runs[0]["coco_stats"])
            result["summary"][case][backend] = dict(
                phases={
                    p: {
                        k: spread([r["phases"][p][k] for r in runs])
                        for k in ("wall", "cpu")
                    }
                    for p in runs[0]["phases"]
                },
                peak_rss_MiB=spread([r["peak_rss_MiB"] for r in runs]),
            )
        pairs = [
            {
                r["backend"]: r["measurement"]["phases"]["total"]
                for r in result["runs"]
                if r["case"] == case and r["round"] == index
            }
            for index in range(args.rounds)
        ]
        result["paired_total_speedup"][case] = {
            k: spread([p[BACKENDS[0]][k] / p[BACKENDS[1]][k] for p in pairs])
            for k in ("wall", "cpu")
        }
    save()
    result["curve_parity"] = {}
    for case in CASES:
        for backend in BACKENDS:
            execute(case, backend, -1, diagnostic=True)
        with np.load(args.out / f"diagnostic-{case}-{BACKENDS[0]}.npz") as a, np.load(
            args.out / f"diagnostic-{case}-{BACKENDS[1]}.npz"
        ) as b:
            assert a.files == b.files
            result["curve_parity"][case] = {}
            for key in a.files:
                compare(a[key].tolist(), b[key].tolist())
                result["curve_parity"][case][key] = dict(
                    shape=list(a[key].shape),
                    max_abs_difference=float(np.max(np.abs(a[key] - b[key]))),
                    byte_identical=a[key].tobytes() == b[key].tobytes(),
                )
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["parity_atol"] = 1e-12
    save()
    print(f"PASS: {args.out / 'results.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dfine", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=int, default=500)
    p.add_argument("--revision")
    for mode in ("worker", "run"):
        s = sub.add_parser(mode)
        s.add_argument("--inputs", type=Path, required=True)
        if mode == "worker":
            s.add_argument("--backend", choices=BACKENDS, required=True)
            s.add_argument("--case", choices=CASES, required=True)
            s.add_argument("--diagnostic", action="store_true")
        else:
            s.add_argument("--rounds", type=int, default=6)
    for s in sub.choices.values():
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (
        args.threads < 1
        or getattr(args, "rounds", 1) < 1
        or getattr(args, "images", 1) < 1
    ):
        parser.error("threads, rounds and images must be positive")
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
