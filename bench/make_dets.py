"""Build a detection result file from a COCO ground-truth file.

Real annotations, synthetic scores: this gives a realistic PR curve (real box
statistics, real crowd regions, real polygon shapes) without needing a trained
model, and it is seeded so every implementation sees identical input.

Recipe per ground-truth annotation:
  * with probability ``--recall`` emit a jittered copy (a true positive whose
    IoU spans the whole 0.5:0.95 sweep, so every threshold is exercised);
  * a fraction of those get the wrong category (classification errors);
  * plus ``--fp-ratio`` random boxes per image as false positives.

Scores are quantised to three decimals on purpose: that produces thousands of
exact ties, which is where an unstable sort silently changes the greedy match
and moves AP.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recall", type=float, default=0.75)
    ap.add_argument("--wrong-class", type=float, default=0.12)
    ap.add_argument("--fp-ratio", type=float, default=2.0, help="extra false positives per image")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--score-decimals",
        type=int,
        default=3,
        help="rounding of detection scores; 3 gives thousands of exact ties "
        "(the interesting case), 12 gives effectively none",
    )
    ap.add_argument("--segm", action="store_true", help="also emit RLE segmentation")
    ap.add_argument("--keypoints", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    with open(args.gt) as f:
        gt = json.load(f)
    print(f"loaded {args.gt.name} in {time.time() - t0:.1f}s", flush=True)

    rng = random.Random(args.seed)
    cat_ids = [c["id"] for c in gt["categories"]]
    imgs = {im["id"]: im for im in gt["images"]}

    dets: list[dict] = []
    if args.segm:
        from pycocotools import mask as mask_util

    for ann in gt["annotations"]:
        if rng.random() > args.recall:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        jx = rng.uniform(-0.2, 0.2)
        jy = rng.uniform(-0.2, 0.2)
        nx, ny = x + w * jx, y + h * jy
        nw, nh = w * rng.uniform(0.7, 1.35), h * rng.uniform(0.7, 1.35)
        cat = ann["category_id"]
        if rng.random() < args.wrong_class:
            cat = rng.choice(cat_ids)
        det = {
            "image_id": ann["image_id"],
            "category_id": cat,
            "bbox": [round(nx, 2), round(ny, 2), round(nw, 2), round(nh, 2)],
            "score": round(rng.random() ** 0.6, args.score_decimals),
        }
        if args.segm:
            im = imgs[ann["image_id"]]
            poly = [[nx, ny, nx, ny + nh, nx + nw, ny + nh, nx + nw, ny]]
            rles = mask_util.frPyObjects(poly, im["height"], im["width"])
            rle = mask_util.merge(rles)
            rle["counts"] = rle["counts"].decode("ascii")
            det["segmentation"] = rle
        if args.keypoints and "keypoints" in ann:
            kp = list(ann["keypoints"])
            for i in range(0, len(kp), 3):
                if kp[i + 2] > 0:
                    kp[i] = round(kp[i] + rng.uniform(-8, 8), 1)
                    kp[i + 1] = round(kp[i + 1] + rng.uniform(-8, 8), 1)
                    kp[i + 2] = 1.0
            det["keypoints"] = kp
        dets.append(det)

    n_fp = int(len(gt["images"]) * args.fp_ratio)
    img_list = gt["images"]
    for _ in range(n_fp):
        im = rng.choice(img_list)
        bw = rng.uniform(10, min(im["width"], 400))
        bh = rng.uniform(10, min(im["height"], 400))
        bx = rng.uniform(0, max(im["width"] - bw, 1))
        by = rng.uniform(0, max(im["height"] - bh, 1))
        det = {
            "image_id": im["id"],
            "category_id": rng.choice(cat_ids),
            "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
            "score": round(rng.random() ** 1.8, args.score_decimals),
        }
        if args.segm:
            poly = [[bx, by, bx, by + bh, bx + bw, by + bh, bx + bw, by]]
            rles = mask_util.frPyObjects(poly, im["height"], im["width"])
            rle = mask_util.merge(rles)
            rle["counts"] = rle["counts"].decode("ascii")
            det["segmentation"] = rle
        if args.keypoints:
            kp = []
            for _ in range(17):
                kp.extend([round(rng.uniform(bx, bx + bw), 1), round(rng.uniform(by, by + bh), 1), 1.0])
            det["keypoints"] = kp
        dets.append(det)

    rng.shuffle(dets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dets, f)
    print(
        f"wrote {args.out} : {len(dets)} detections, "
        f"{args.out.stat().st_size / 1e6:.1f} MB, total {time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
