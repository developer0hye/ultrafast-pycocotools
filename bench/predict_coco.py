"""Run a real detector over COCO val2017 and write COCO-format predictions.

Every benchmark and parity number so far used detections derived from the
ground truth — jittered boxes with quantised scores. That is deliberately hard
in one way (thousands of exact score ties, which is where stable sorting
matters) and unrealistic in others: the boxes are correlated with the ground
truth, the scores are uniform, and AP is meaningless.

Real model output is the opposite stress case. Scores are float32 with almost
no ties, boxes come from a detector's own prior, and there are far more of them
per image. If the two agree on both, the equivalence claim is not an artefact
of one generator.

Writes the standard COCO results format, so the output drops straight into
`COCO.loadRes`.

    python bench/predict_coco.py --model yolo11m.pt --images /data/coco/val2017 \
        --ann /data/coco/annotations/instances_val2017.json --out preds.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Ultralytics' class indices are the contiguous 0-79 COCO subset; the dataset's
# own category ids run 1-90 with gaps where categories were removed. Getting
# this mapping wrong produces predictions that never match anything and an AP
# of zero that looks like a bug in the evaluator.
COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="5")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="COCO protocol keeps everything; the evaluator does the cutting",
    )
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--segm", action="store_true", help="also emit RLE masks")
    args = ap.parse_args()

    from ultralytics import YOLO

    ann = json.loads(args.ann.read_text())
    id_of = {Path(im["file_name"]).name: im["id"] for im in ann["images"]}
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() == ".jpg")
    files = [p for p in files if p.name in id_of]
    print(f"{len(files)} images, model {args.model}", flush=True)

    if args.segm:
        from pycocotools import mask as mask_util
        import numpy as np

    model = YOLO(args.model)
    dets: list[dict] = []
    t0 = time.time()
    times: list[float] = []

    for start in range(0, len(files), args.batch):
        chunk = files[start : start + args.batch]
        ts = time.time()
        results = model.predict(
            [str(p) for p in chunk],
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_det,
            verbose=False,
            retina_masks=args.segm,
        )
        for path, res in zip(chunk, results):
            image_id = id_of[path.name]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xywh = boxes.xyxy.cpu().numpy()
            xywh[:, 2] -= xywh[:, 0]
            xywh[:, 3] -= xywh[:, 1]
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            masks = None
            if args.segm and res.masks is not None:
                masks = res.masks.data.cpu().numpy().astype(np.uint8)
            for i in range(len(confs)):
                det = {
                    "image_id": int(image_id),
                    "category_id": COCO80_TO_COCO91[classes[i]],
                    "bbox": [round(float(v), 3) for v in xywh[i]],
                    "score": float(confs[i]),
                }
                if masks is not None and i < len(masks):
                    rle = mask_util.encode(np.asfortranarray(masks[i]))
                    rle["counts"] = rle["counts"].decode("ascii")
                    det["segmentation"] = rle
                dets.append(det)

        times.append(time.time() - ts)
        done = start + len(chunk)
        avg = sum(times) / len(times)
        eta = avg * (len(files) - done) / args.batch
        print(
            f"[{done}/{len(files)}] {len(dets)} dets  elapsed {time.time() - t0:6.0f}s  "
            f"batch avg {avg:5.2f}s  ETA {eta / 60:5.1f}min",
            flush=True,
        )
        if len(times) == 1:
            total = avg * len(files) / args.batch
            print(f"  -> whole run about {total / 60:.1f} min", flush=True)

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
