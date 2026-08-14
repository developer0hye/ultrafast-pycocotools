"""Shared fixtures for the parity suite.

The suite compares against a real ``pycocotools`` install rather than against
recorded golden numbers. Goldens would freeze in whatever we happened to
produce the day they were written; comparing live means a regression on either
side shows up as a failure here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

pycocotools = pytest.importorskip("pycocotools", reason="parity needs the reference")

DATA = ROOT / "bench" / "data"
REAL_GT = DATA / "instances_val2017.json"
REAL_DT = DATA / "dt_val2017_bbox.json"
REAL_KP_GT = DATA / "person_keypoints_val2017.json"

# A committed slice of real COCO val2017. Synthetic data approximates real
# annotations and gets some of it wrong — our generator draws polygons with 3-8
# vertices where COCO's have dozens across several rings, and it cannot
# reproduce the exact shapes that make `rleFrPoly` interesting. This runs
# everywhere, including a fresh clone and CI; see bench/make_real_fixture.py.
FIXTURE = ROOT / "tests" / "data"
COCO_SUBSET_GT = FIXTURE / "coco_subset_gt.json"
COCO_SUBSET_DT = FIXTURE / "coco_subset_dt.json"


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> tuple[Path, Path]:
    """A small synthetic dataset that hits the awkward paths on purpose.

    Crowd regions, areas sitting exactly on the small/medium/large
    boundaries, duplicate scores, images with only ground truth or only
    detections, and polygons with repeated vertices.
    """
    from make_dataset import build

    gt, dt = build(n_images=180, n_cats=12, gt_per_image=9, dt_per_image=25, seed=7)
    d = tmp_path_factory.mktemp("synthetic")
    gt_path, dt_path = d / "gt.json", d / "dt.json"
    gt_path.write_text(json.dumps(gt))
    dt_path.write_text(json.dumps(dt))
    return gt_path, dt_path


@pytest.fixture(scope="session")
def synthetic_kp(tmp_path_factory) -> tuple[Path, Path]:
    from make_dataset import build

    gt, dt = build(
        n_images=120, n_cats=1, gt_per_image=6, dt_per_image=12, seed=11, with_keypoints=True
    )
    d = tmp_path_factory.mktemp("synthetic_kp")
    gt_path, dt_path = d / "gt.json", d / "dt.json"
    gt_path.write_text(json.dumps(gt))
    dt_path.write_text(json.dumps(dt))
    return gt_path, dt_path


@pytest.fixture(scope="session")
def real_coco() -> tuple[Path, Path]:
    """The committed slice of real COCO val2017. Always available."""
    assert COCO_SUBSET_GT.exists(), (
        f"{COCO_SUBSET_GT} is missing; regenerate with bench/make_real_fixture.py"
    )
    return COCO_SUBSET_GT, COCO_SUBSET_DT


@pytest.fixture(scope="session")
def real_pair() -> tuple[Path, Path]:
    """Full COCO val2017, when someone has it locally.

    Skips rather than fails: this is a bonus on top of `real_coco`, not the
    only real-data coverage. It used to be the only one, which meant the
    suite's strongest test ran on exactly one machine.
    """
    if not (REAL_GT.exists() and REAL_DT.exists()):
        pytest.skip("full COCO val2017 not present; run bench/make_dets.py")
    return REAL_GT, REAL_DT


def subset_gt(path: Path, n_images: int, tmp: Path) -> Path:
    """First ``n_images`` images of a COCO file, annotations included.

    Keeps the real annotation statistics (real polygons, real crowd regions)
    while keeping the test fast.
    """
    data = json.loads(path.read_text())
    keep = {im["id"] for im in data["images"][:n_images]}
    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": [im for im in data["images"] if im["id"] in keep],
        "annotations": [a for a in data["annotations"] if a["image_id"] in keep],
        "categories": data["categories"],
    }
    p = tmp / f"gt_{n_images}.json"
    p.write_text(json.dumps(out))
    return p


def subset_dt(path: Path, gt_path: Path, tmp: Path) -> Path:
    gt = json.loads(gt_path.read_text())
    keep = {im["id"] for im in gt["images"]}
    dets = [d for d in json.loads(path.read_text()) if d["image_id"] in keep]
    p = tmp / "dt_subset.json"
    p.write_text(json.dumps(dets))
    return p
