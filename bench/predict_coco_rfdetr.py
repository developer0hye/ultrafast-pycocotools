"""RF-DETR predictions over COCO val2017, in COCO results format.

A second, architecturally unrelated detector. YOLO is anchor-based with NMS;
RF-DETR is a DETR variant that emits a fixed set of queries with no NMS at all.
Their score distributions and box priors have nothing in common, so agreeing
with pycocotools on both is a stronger statement than agreeing on either.

RF-DETR already predicts in COCO's 1-90 category space, unlike ultralytics'
contiguous 0-79 — so there is no remapping here on purpose.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", default="base", choices=["nano", "small", "medium", "base"])
    ap.add_argument("--device", default="7")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.001)
    args = ap.parse_args()

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)

    import numpy as np
    from PIL import Image

    import rfdetr

    factory = {
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
        "base": rfdetr.RFDETRBase,
    }[args.variant]
    model = factory()

    ann = json.loads(args.ann.read_text())
    id_of = {Path(im["file_name"]).name: im["id"] for im in ann["images"]}
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() == ".jpg")
    files = [p for p in files if p.name in id_of]
    print(f"{len(files)} images, RF-DETR {args.variant}", flush=True)

    dets: list[dict] = []
    t0 = time.time()
    times: list[float] = []
    for start in range(0, len(files), args.batch):
        chunk = files[start : start + args.batch]
        ts = time.time()
        images = [Image.open(p).convert("RGB") for p in chunk]
        results = model.predict(images, threshold=args.threshold)
        if not isinstance(results, list):
            results = [results]
        for path, det in zip(chunk, results):
            image_id = int(id_of[path.name])
            xyxy = np.asarray(det.xyxy, dtype=np.float64)
            conf = np.asarray(det.confidence, dtype=np.float64)
            cls = np.asarray(det.class_id, dtype=np.int64)
            for i in range(len(conf)):
                x0, y0, x1, y1 = xyxy[i]
                dets.append({
                    "image_id": image_id,
                    "category_id": int(cls[i]),
                    "bbox": [
                        round(float(x0), 3),
                        round(float(y0), 3),
                        round(float(x1 - x0), 3),
                        round(float(y1 - y0), 3),
                    ],
                    "score": float(conf[i]),
                })
        times.append(time.time() - ts)
        done = start + len(chunk)
        avg = sum(times) / len(times)
        print(
            f"[{done}/{len(files)}] {len(dets)} dets  elapsed {time.time() - t0:6.0f}s  "
            f"batch avg {avg:5.2f}s  ETA {avg * (len(files) - done) / args.batch / 60:5.1f}min",
            flush=True,
        )
        if len(times) == 1:
            print(f"  -> whole run about {avg * len(files) / args.batch / 60:.1f} min", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dets))
    print(
        f"wrote {args.out}: {len(dets)} detections "
        f"({len(dets) / max(len(files), 1):.1f} per image), "
        f"{args.out.stat().st_size / 1e6:.1f} MB, total {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
