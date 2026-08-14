"""torchvision Mask R-CNN / Keypoint R-CNN predictions in COCO results format.

Two things this covers that YOLO and RF-DETR do not.

**Keypoints.** The OKS path has only ever been tested against keypoints this
repo generated itself. Generated keypoints are uniform noise inside a box;
real ones cluster on a body, are frequently occluded, and come with per-joint
scores. Keypoint R-CNN gives the real distribution, and OKS is the one
similarity function with its own arithmetic (three separate divisions, a
per-sigma variance, and a fall-back branch when nothing is visible).

**Real instance masks.** YOLO-seg composes masks from a handful of prototypes,
so its shapes are smooth and low-rank. Mask R-CNN predicts a per-RoI mask and
pastes it back, giving ragged boundaries and holes — much closer to what the
RLE code was written for.

torchvision is also the library that actually calls pycocotools in its own
reference scripts, which makes it the most realistic consumer to imitate.

Its COCO-trained detection models label with the dataset's own 91-category
indexing, so unlike ultralytics there is no remapping here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["segm", "keypoints"], required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--score-thr", type=float, default=0.01)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from pycocotools import mask as mask_util
    from torchvision.transforms import functional as TF

    if args.task == "segm":
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    else:
        weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_LEGACY
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)
    model.eval().to(args.device)

    ann = json.loads(args.ann.read_text())
    id_of = {Path(im["file_name"]).name: im["id"] for im in ann["images"]}
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() == ".jpg")
    files = [p for p in files if p.name in id_of]
    print(f"{len(files)} images, {args.task}, {type(model).__name__}", flush=True)

    dets: list[dict] = []
    t0 = time.time()
    times: list[float] = []
    for start in range(0, len(files), args.batch):
        chunk = files[start : start + args.batch]
        ts = time.time()
        batch = [
            TF.to_tensor(Image.open(p).convert("RGB")).to(args.device) for p in chunk
        ]
        with torch.inference_mode():
            outputs = model(batch)

        for path, out in zip(chunk, outputs):
            image_id = int(id_of[path.name])
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            masks = out.get("masks")
            kps = out.get("keypoints")
            if masks is not None:
                masks = masks.cpu().numpy()
            if kps is not None:
                kps = kps.cpu().numpy()

            for i in range(len(scores)):
                if scores[i] < args.score_thr:
                    continue
                x0, y0, x1, y1 = boxes[i]
                det = {
                    "image_id": image_id,
                    "category_id": int(labels[i]),
                    "bbox": [
                        round(float(x0), 3),
                        round(float(y0), 3),
                        round(float(x1 - x0), 3),
                        round(float(y1 - y0), 3),
                    ],
                    "score": float(scores[i]),
                }
                if masks is not None:
                    # Mask R-CNN emits soft masks; 0.5 is the threshold its own
                    # reference evaluation uses.
                    binary = (masks[i, 0] >= 0.5).astype(np.uint8)
                    rle = mask_util.encode(np.asfortranarray(binary))
                    rle["counts"] = rle["counts"].decode("ascii")
                    det["segmentation"] = rle
                if kps is not None:
                    # COCO's keypoint results format wants a flat
                    # [x, y, v] * 17 with v = 1 for every predicted joint.
                    flat: list[float] = []
                    for x, y, _v in kps[i]:
                        flat.extend([round(float(x), 2), round(float(y), 2), 1.0])
                    det["keypoints"] = flat
                    det["category_id"] = 1  # person
                dets.append(det)

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
