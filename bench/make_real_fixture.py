"""Carve a small, committable slice of real COCO out of instances_val2017.

The parity suite's strongest test runs against real annotations — real
polygons with hundreds of vertices, real crowd regions stored as uncompressed
RLE, real areas that land on the small/medium/large boundaries. Synthetic data
approximates all of that and gets some of it wrong: our generator draws
polygons with 3-8 vertices, COCO's have dozens and often several rings.

But `bench/data/` is gitignored, so that test skipped everywhere except the
machine that happened to have the 20 MB annotation file. A test that only runs
in one place is a test you find out about later.

So this picks a few dozen images that between them cover the awkward cases and
writes them to `tests/data/`, small enough to commit. Selection is greedy over
coverage, not random, so a smaller file buys more.

Run once when regenerating the fixture:

    python bench/make_real_fixture.py --gt path/to/instances_val2017.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

def score_image(anns: list[dict]) -> tuple[float, ...]:
    """Rank an image by awkward cases *per byte*.

    Weighted rather than lexicographic: sorting by crowd count first picks the
    handful of images with the most crowd regions, which are also the biggest
    files, and spends the whole budget on them. Dividing by the annotation
    payload keeps the selection paying for coverage instead of for size.

    Note there is no "area exactly on a scale boundary" term. Real COCO areas
    come from polygon integration and are never exactly 1024 or 9216 — that
    case only exists in the synthetic fixture, which constructs it on purpose.
    """
    crowd = sum(1 for a in anns if a.get("iscrowd"))
    multi_ring = sum(
        1 for a in anns if isinstance(a.get("segmentation"), list) and len(a["segmentation"]) > 1
    )
    tiny = sum(1 for a in anns if a["area"] < 32.0**2)
    large = sum(1 for a in anns if a["area"] > 96.0**2)
    vertices = sum(
        len(r) // 2
        for a in anns
        if isinstance(a.get("segmentation"), list)
        for r in a["segmentation"]
    )
    value = 8.0 * crowd + 3.0 * multi_ring + 1.0 * min(tiny, 6) + 1.0 * min(large, 6)
    cost = max(vertices, 1) + 4.0 * len(anns)
    return (value / cost, value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True, help="instances_val2017.json")
    ap.add_argument("--out", type=Path, default=Path("tests/data"))
    ap.add_argument("--images", type=int, default=90)
    ap.add_argument("--crowd-images", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()

    data = json.loads(args.gt.read_text())
    by_img: dict = defaultdict(list)
    for a in data["annotations"]:
        by_img[a["image_id"]].append(a)

    # Crowd regions get their own quota rather than competing on
    # coverage-per-byte. They are the highest-value case — a whole branch of
    # the matcher, plus the uncompressed-RLE decode path — and they only occur
    # in busy images, so a pure efficiency ranking starves them: the first cut
    # of this fixture kept exactly two.
    crowd_first = sorted(
        (i for i in by_img if any(a.get("iscrowd") for a in by_img[i])),
        key=lambda i: sum(1 for a in by_img[i] if a.get("iscrowd")),
        reverse=True,
    )
    keep_ids = set(crowd_first[: args.crowd_images])
    ranked = sorted(by_img, key=lambda i: score_image(by_img[i]), reverse=True)
    for img_id in ranked:
        if len(keep_ids) >= args.images:
            break
        keep_ids.add(img_id)
    # A few images with no annotations at all: the "neither ground truth nor
    # detections" path is otherwise unreachable.
    empty = [im["id"] for im in data["images"] if im["id"] not in by_img][:3]
    keep_ids.update(empty)

    images = [im for im in data["images"] if im["id"] in keep_ids]
    anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]
    kept_cats = {a["category_id"] for a in anns}
    gt = {
        "info": {
            "description": "COCO val2017 subset for parity testing",
            "source": "https://cocodataset.org — annotations are CC BY 4.0",
        },
        "licenses": data.get("licenses", []),
        # Every category is kept, including ones with no annotations here:
        # a category present in params but absent from the data is exactly the
        # "-1 sentinel" case.
        "categories": data["categories"],
        "images": images,
        "annotations": anns,
    }

    # Detections derived the same way bench/make_dets.py derives them, kept
    # here so the fixture is one self-contained pair.
    rng = random.Random(args.seed)
    cat_ids = [c["id"] for c in data["categories"]]
    dets: list[dict] = []
    for a in anns:
        if rng.random() > 0.75:
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        cat = a["category_id"] if rng.random() > 0.12 else rng.choice(cat_ids)
        dets.append({
            "image_id": a["image_id"],
            "category_id": cat,
            "bbox": [
                round(x + w * rng.uniform(-0.2, 0.2), 2),
                round(y + h * rng.uniform(-0.2, 0.2), 2),
                round(w * rng.uniform(0.7, 1.35), 2),
                round(h * rng.uniform(0.7, 1.35), 2),
            ],
            # Three decimals: ties are where stable sorting matters.
            "score": round(rng.random() ** 0.6, 3),
        })
    for im in images:
        for _ in range(2):
            bw = rng.uniform(10, min(im["width"], 300))
            bh = rng.uniform(10, min(im["height"], 300))
            dets.append({
                "image_id": im["id"],
                "category_id": rng.choice(cat_ids),
                "bbox": [
                    round(rng.uniform(0, max(im["width"] - bw, 1)), 2),
                    round(rng.uniform(0, max(im["height"] - bh, 1)), 2),
                    round(bw, 2),
                    round(bh, 2),
                ],
                "score": round(rng.random() ** 1.8, 3),
            })
    rng.shuffle(dets)

    args.out.mkdir(parents=True, exist_ok=True)
    gt_path = args.out / "coco_subset_gt.json"
    dt_path = args.out / "coco_subset_dt.json"
    gt_path.write_text(json.dumps(gt, separators=(",", ":")))
    dt_path.write_text(json.dumps(dets, separators=(",", ":")))

    n_crowd = sum(1 for a in anns if a.get("iscrowd"))
    n_multi = sum(
        1 for a in anns if isinstance(a.get("segmentation"), list) and len(a["segmentation"]) > 1
    )
    n_small = sum(1 for a in anns if a["area"] < 32.0**2)
    n_large = sum(1 for a in anns if a["area"] > 96.0**2)
    print(f"{gt_path}  {gt_path.stat().st_size / 1e3:6.1f} kB")
    print(f"{dt_path}  {dt_path.stat().st_size / 1e3:6.1f} kB")
    print(
        f"  {len(images)} images, {len(anns)} annotations, {len(dets)} detections\n"
        f"  {n_crowd} crowd, {n_multi} multi-ring polygons, "
        f"{n_small} small / {n_large} large by COCO's area scale"
    )


if __name__ == "__main__":
    main()
