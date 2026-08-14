"""Deterministic synthetic COCO ground truth + detections.

Used for both the parity suite and the benchmarks. Everything is seeded, so
two runs on two machines produce byte-identical JSON and any metric difference
is the evaluator's, not the data's.

The generator deliberately exercises the paths where implementations usually
diverge:

* ``iscrowd=1`` ground truth (crowd IoU divides by the *detection* area, and a
  crowd box may absorb several detections);
* areas that straddle the small/medium/large boundaries exactly, so an
  off-by-epsilon in the area filter shows up;
* duplicate scores, so any non-stable sort reorders ties and changes the greedy
  match;
* images with ground truth but no detections, detections but no ground truth,
  and neither;
* polygons with duplicate consecutive vertices, which make ``rleFrPoly``
  divide by zero.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


def _polygon(x: float, y: float, w: float, h: float, rng: random.Random) -> list[float]:
    """A closed polygon inscribed in the box, occasionally degenerate."""
    n = rng.randint(3, 8)
    cx, cy = x + w / 2.0, y + h / 2.0
    pts: list[float] = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        r = rng.uniform(0.55, 1.0)
        pts.append(round(cx + math.cos(a) * (w / 2.0) * r, 2))
        pts.append(round(cy + math.sin(a) * (h / 2.0) * r, 2))
    if rng.random() < 0.05 and len(pts) >= 4:
        # Repeat a vertex: rleFrPoly then divides by a zero edge length.
        pts.extend(pts[-2:])
    return pts


def _uncompressed_rle_box(x: float, y: float, w: float, h: float, H: int, W: int) -> dict:
    """A rectangle as an *uncompressed* RLE, the form COCO uses for crowds.

    Built directly rather than by encoding a raster, so it is exact and the
    generator stays dependency-free. Counts are column-major and start with a
    run of zeros (possibly of length zero), which is the RLE convention.
    """
    x0, x1 = int(max(0, math.floor(x))), int(min(W, math.ceil(x + w)))
    y0, y1 = int(max(0, math.floor(y))), int(min(H, math.ceil(y + h)))
    counts: list[int] = []
    cur_val, cur_run = 0, 0
    for col in range(W):
        if x0 <= col < x1 and y1 > y0:
            segs = [(0, y0), (1, y1 - y0), (0, H - y1)]
        else:
            segs = [(0, H)]
        for val, n in segs:
            if n <= 0:
                continue
            if val == cur_val:
                cur_run += n
            else:
                counts.append(cur_run)
                cur_val, cur_run = val, n
    counts.append(cur_run)
    assert sum(counts) == H * W
    return {"size": [H, W], "counts": counts}


def _keypoints(
    x: float,
    y: float,
    w: float,
    h: float,
    rng: random.Random,
    k: int = 17,
    all_invisible: bool = False,
) -> list[float]:
    """Keypoint triplets, occasionally with nothing visible at all.

    ``all_invisible`` is not decoration. An instance with ``num_keypoints ==
    0`` is *ignored* by the evaluator, and OKS takes a completely different
    branch when no ground-truth keypoint is visible — it measures distance to
    a doubled bounding box instead of to the keypoints. Drawing visibility
    independently per keypoint makes that branch essentially unreachable
    ((1/5)^17), so it has to be chosen deliberately.
    """
    kp: list[float] = []
    for _ in range(k):
        v = 0 if all_invisible else rng.choice([0, 1, 2, 2, 2])
        kp.extend([
            round(x + rng.random() * w, 1) if v else 0.0,
            round(y + rng.random() * h, 1) if v else 0.0,
            float(v),
        ])
    return kp


def build(
    n_images: int,
    n_cats: int,
    gt_per_image: int,
    dt_per_image: int,
    seed: int = 0,
    with_keypoints: bool = False,
) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    images, categories, annotations, dets = [], [], [], []

    for c in range(1, n_cats + 1):
        categories.append({
            "id": c,
            "name": f"cat{c}",
            "supercategory": "super" if c % 2 else "other",
        })
    if with_keypoints:
        for c in categories:
            c["keypoints"] = [f"kp{i}" for i in range(17)]
            c["skeleton"] = [[i, i + 1] for i in range(1, 17)]

    ann_id = 1
    for i in range(1, n_images + 1):
        # Non-uniform image sizes so mask h/w mismatches would be caught.
        w_img = rng.choice([640, 800, 1024, 427])
        h_img = rng.choice([480, 600, 768, 640])
        images.append({
            "id": i,
            "file_name": f"{i:012d}.jpg",
            "width": w_img,
            "height": h_img,
        })

        # 5% of images have no ground truth, 5% no detections.
        n_gt = 0 if rng.random() < 0.05 else rng.randint(1, gt_per_image)
        n_dt = 0 if rng.random() < 0.05 else rng.randint(1, dt_per_image)

        gt_boxes = []
        for _ in range(n_gt):
            # Sizes chosen to land on both sides of 32^2 and 96^2, and exactly
            # on them often enough to matter.
            side = rng.choice([32.0, 96.0, rng.uniform(8, 300)])
            bw = min(side, w_img - 1.0)
            bh = min(side, h_img - 1.0)
            bx = rng.uniform(0, max(w_img - bw - 1, 0.1))
            by = rng.uniform(0, max(h_img - bh - 1, 0.1))
            bx, by, bw, bh = (round(v, 2) for v in (bx, by, bw, bh))
            cat = rng.randint(1, n_cats)
            iscrowd = 1 if rng.random() < 0.08 else 0
            ann = {
                "id": ann_id,
                "image_id": i,
                "category_id": cat,
                "bbox": [bx, by, bw, bh],
                "area": round(bw * bh, 4),
                "iscrowd": iscrowd,
                # COCO segmentation is a *list of rings*, even for one ring.
                "segmentation": [_polygon(bx, by, bw, bh, rng)],
            }
            if with_keypoints:
                ann["keypoints"] = _keypoints(
                    bx, by, bw, bh, rng, 17, all_invisible=rng.random() < 0.08
                )
                ann["num_keypoints"] = sum(1 for j in range(2, 51, 3) if ann["keypoints"][j] > 0)
            if iscrowd:
                # Crowd ground truth is stored as uncompressed RLE in COCO,
                # which is a different code path from polygons.
                ann["segmentation"] = _uncompressed_rle_box(bx, by, bw, bh, h_img, w_img)
            annotations.append(ann)
            gt_boxes.append((bx, by, bw, bh, cat))
            ann_id += 1

        for _ in range(n_dt):
            if gt_boxes and rng.random() < 0.65:
                # Jittered copy of a ground truth -> a plausible true positive.
                bx, by, bw, bh, cat = rng.choice(gt_boxes)
                j = rng.uniform(-0.25, 0.25)
                bx += bw * j
                by += bh * j
                bw *= rng.uniform(0.75, 1.3)
                bh *= rng.uniform(0.75, 1.3)
                if rng.random() < 0.15:
                    cat = rng.randint(1, n_cats)
            else:
                bw = rng.uniform(8, 250)
                bh = rng.uniform(8, 250)
                bx = rng.uniform(0, max(w_img - bw - 1, 0.1))
                by = rng.uniform(0, max(h_img - bh - 1, 0.1))
                cat = rng.randint(1, n_cats)
            bx = max(0.0, min(bx, w_img - 1.0))
            by = max(0.0, min(by, h_img - 1.0))
            bw = max(1.0, min(bw, w_img - bx))
            bh = max(1.0, min(bh, h_img - by))
            bx, by, bw, bh = (round(v, 2) for v in (bx, by, bw, bh))
            # Quantised scores create many exact ties -> exposes unstable sorts.
            score = round(rng.random(), 3)
            det = {
                "image_id": i,
                "category_id": cat,
                "bbox": [bx, by, bw, bh],
                "score": score,
            }
            if with_keypoints:
                det["keypoints"] = _keypoints(bx, by, bw, bh, rng, 17)
            dets.append(det)

    gt = {
        "info": {"description": "ultrafast-pycocotools synthetic", "version": "1.0"},
        "licenses": [{"id": 1, "name": "none"}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return gt, dets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=5000)
    ap.add_argument("--cats", type=int, default=80)
    ap.add_argument("--gt-per-image", type=int, default=12)
    ap.add_argument("--dt-per-image", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keypoints", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("bench/data"))
    args = ap.parse_args()

    t0 = time.time()
    gt, dt = build(
        args.images,
        args.cats,
        args.gt_per_image,
        args.dt_per_image,
        args.seed,
        args.keypoints,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "_kp" if args.keypoints else ""
    gt_path = args.out / f"gt{suffix}_{args.images}.json"
    dt_path = args.out / f"dt{suffix}_{args.images}.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f)
    with open(dt_path, "w") as f:
        json.dump(dt, f)
    print(
        f"wrote {gt_path} ({gt_path.stat().st_size / 1e6:.1f} MB, "
        f"{len(gt['annotations'])} anns) and {dt_path} "
        f"({dt_path.stat().st_size / 1e6:.1f} MB, {len(dt)} dets) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
