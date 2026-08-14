"""Break down the time inside COCO() and loadRes().

`profile_engine.py` covers everything from `_prepare()` onward, which is where
the interesting algorithms are — and so that is where the optimisation went.
But a user's wall clock starts earlier, and on a segmentation result set
`loadRes` alone costs more than the whole evaluation. None of it is engine
work: it is Python loops over half a million dicts, and nobody has looked.

Splits each phase so the next optimisation is aimed rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ultrafast_pycocotools as ufc
from ultrafast_pycocotools import _ufcoco


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    args = ap.parse_args()

    rows: list[tuple[str, float]] = []

    def phase(name: str, fn):
        t = time.perf_counter()
        out = fn()
        rows.append((name, time.perf_counter() - t))
        return out

    gt_raw = phase("GT  json parse (Rust)", lambda: _ufcoco.load_json(args.gt))
    gt = ufc.COCO(verbose=False)
    gt.dataset = gt_raw
    phase("GT  createIndex", gt.createIndex)

    dt_anns = phase("DT  json parse (Rust)", lambda: _ufcoco.load_json(args.dt))
    n = len(dt_anns)
    keys = sorted(dt_anns[0].keys())

    # The same phases loadRes runs, timed one at a time.
    def check():
        ids = [a["image_id"] for a in dt_anns]
        assert set(ids) & set(gt.getImgIds()) == set(ids)

    phase("DT  image_id assert", check)

    has_bbox = "bbox" in dt_anns[0] and dt_anns[0]["bbox"] != []

    def derive():
        if has_bbox:
            for idx, ann in enumerate(dt_anns):
                bb = ann["bbox"]
                if "segmentation" not in ann:
                    x1, x2, y1, y2 = bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = idx + 1
                ann["iscrowd"] = 0
        else:
            for idx, ann in enumerate(dt_anns):
                ann["area"] = ufc.mask.area(ann["segmentation"])
                if "bbox" not in ann:
                    ann["bbox"] = ufc.mask.toBbox(ann["segmentation"])
                ann["id"] = idx + 1
                ann["iscrowd"] = 0

    phase(f"DT  derive fields ({'bbox branch' if has_bbox else 'mask branch'})", derive)

    res = ufc.COCO(verbose=False)
    res.dataset = {
        "info": {}, "images": list(gt.dataset["images"]),
        "categories": list(gt.dataset["categories"]), "annotations": dt_anns,
    }
    phase("DT  createIndex", res.createIndex)

    ev = ufc.COCOeval(gt, res, "segm", print_function=lambda *_: None)
    phase("_collect", ev._collect)

    total = sum(t for _, t in rows)
    print(f"{n:,} detections, keys = {keys}")
    print()
    width = max(len(k) for k, _ in rows)
    for name, t in rows:
        bar = "#" * round(28 * t / max(x for _, x in rows))
        print(f"  {name:<{width}}  {t:6.3f}s  {t / total:5.1%}  {bar}")
    print(f"  {'total':<{width}}  {total:6.3f}s")


if __name__ == "__main__":
    main()
