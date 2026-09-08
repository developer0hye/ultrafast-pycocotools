# ultrafast-pycocotools

[![Library CI](https://github.com/developer0hye/ultrafast-pycocotools/actions/workflows/ci.yml/badge.svg)](https://github.com/developer0hye/ultrafast-pycocotools/actions/workflows/ci.yml)

COCO evaluation in Rust, with a Python API compatible with `pycocotools`.

Use it to reduce evaluation time while keeping the reference metrics. The test
suite compares the complete `precision`, `recall`, and `scores` arrays and the
summary statistics **byte for byte**, including score ties, crowd annotations,
custom evaluation parameters, and real COCO fixtures.

Benchmarks cover public pretrained detector outputs on COCO and a separate
synthetic scalability workload on the public Objects365 dataset. Both compare
complete evaluation arrays against pycocotools.

**Status:** 0.1.0, alpha. The installation instructions below build from source.
Validate your application's
parameters and subclass behavior before replacing its reference evaluator.

## Installation

Requirements: Python 3.9+, a recent stable [Rust toolchain](https://rustup.rs/),
and a working native compiler toolchain. NumPy is installed as a dependency.

Clone the repository and build the package:

```bash
git clone https://github.com/developer0hye/ultrafast-pycocotools.git
cd ultrafast-pycocotools
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead.
The source build uses Maturin and compiles the Rust extension; Rust is needed at
build time, not when importing an already built wheel.

## Reproduce without downloads

After installing the test/reference dependencies, run the complete synthetic
check with one command:

```bash
python -m pip install ".[test]"
python bench/reproduce.py quick --out bench/out/quick --verify-published quick
```

It generates data, runs both scorers, and verifies input hashes and every
evaluation-array byte. No images, model weights, or GPU are needed.
See [the reproduction guide](docs/reproducibility.md) for Objects365 and public
detector predictions, expected hashes, resource requirements, and output files.

## Quick start

This example uses the small COCO fixture included in the repository. Replace
the two paths with your annotation and prediction JSON files for a real run.

```python
from ultrafast_pycocotools import COCO, COCOeval

gt = COCO("tests/data/coco_subset_gt.json")
dt = gt.loadRes("tests/data/coco_subset_dt.json")
evaluator = COCOeval(gt, dt, "bbox")

evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()

print("AP:", evaluator.stats[0])
```

`evaluator.run()` is a convenience method for the last three calls. Standard
COCO evaluation modes are `"bbox"`, `"segm"`, and `"keypoints"`.

### Use with an existing framework

Prefer direct imports when you own the evaluation code. If a framework imports
`pycocotools` internally, register the replacement **before importing that
framework**:

```python
from ultrafast_pycocotools import init_as_pycocotools

init_as_pycocotools()

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
```

This changes the import mapping for the entire Python process. Use separate
processes when comparing the reference package and the replacement.

## Scaling with input size

![Evaluation time and peak memory versus GT plus prediction count](docs/assets/scaling.png)

Measured on nested Objects365 subsets with identical inputs for both scorers.
Every point passes byte-level parity for all evaluation arrays.
[Method, counts and reproduction](docs/scaling.md) ·
[SVG](docs/assets/scaling.svg) · [PDF](docs/assets/scaling.pdf) ·
[Raw measurements](bench/results/scaling.json).

## Public benchmarks

Both cases produced **byte-identical precision, recall, scores and summary
statistics** against pycocotools 2.0.11. Timings below include GT indexing,
result loading, evaluation, accumulation and summarization; they exclude input
JSON parsing, detector inference and output serialization.

| Workload | Images | Predictions | pycocotools | ultrafast | Speedup | Peak RSS, reference / ultrafast |
|---|---:|---:|---:|---:|---:|---:|
| COCO val2017 · public YOLO11m | 5,000 | 431,145 | 29.87 s | 2.61 s | 11.46× | 1.28 / 0.70 GiB |
| Objects365 v2 val · synthetic predictions | 80,000 | 1,090,984 | 520.26 s | 12.43 s | 41.86× | 22.62 / 2.37 GiB |

Measured with Python 3.12.3, NumPy 2.4.4 and ultrafast-pycocotools 0.1.0 on an
AMD EPYC 9554 host, with **two CPU cores available to each scorer**, two Rayon
threads and one OpenBLAS thread. These are single-run observations on a shared
host, not timing guarantees. Peak RSS covers the whole process, including
parsed inputs and result serialization.

YOLO11m uses public pretrained weights and scores 50.695 AP on these saved COCO
predictions with both backends. Objects365 uses reproducible, seeded jittered
GT boxes plus false positives: **it measures evaluator scalability, not trained
detector accuracy**. No GPU is used during either scorer comparison.

[Reproduce the inputs and evaluation](docs/reproducibility.md) ·
[Raw results, SHA-256 hashes and provenance](bench/results/public_benchmarks.json).
The raw file records per-phase timings, all summary statistics, input and array
hashes, source hashes, versions, CPU affinity and exact synthetic parameters.

## Compatibility and intentional differences

The public COCO, COCOeval, and mask APIs are checked against pycocotools.
Compatibility tests cover query ordering, result loading, RLE formats,
evaluation parameters, and subclass overrides. Full-array comparison matters:
rounded AP can hide differences in individual precision cells.

Some implementation details intentionally differ:

- `evaluate()` performs matching and accumulation internally; `accumulate()`
  exposes the result through the standard API.
- Per-image `evalImgs` records are not stored by default. Use
  `COCOeval(gt, dt, "bbox", store_eval_imgs=True)` if your integration reads them.
- IoU and annotation lookup structures are built lazily when accessed.
- Ground-truth annotation dictionaries are not rewritten in place.
- `COCO(path, verbose=False)` suppresses loader progress output.

The evaluator follows pycocotools' treatment of `iscrowd`, including its handling
of the annotation `ignore` field. Applications that rely on mutation side
effects or unusual evaluation parameters should run their own parity checks.
See [the detailed compatibility notes](docs/implementation-notes.md#drop-in-호환).

## Additional diagnostics

After evaluation, the same matching results support per-category statistics,
precision–recall curves, match inspection, and a confusion matrix:

```python
print(evaluator.stats_as_dict)
print(evaluator.per_category_stats())
curve = evaluator.pr_curve(cat_id=1, iou_thr=0.5)
matches = evaluator.matches(iou_thr=0.5)
matrix = evaluator.confusion_matrix()
```

Boundary IoU is available as an additional evaluation mode. It is an extension,
not a standard pycocotools metric. See the [implementation notes](docs/implementation-notes.md#확장-기능)
for custom thresholds, area ranges, and diagnostic output formats.

## Development and verification

[Automated CI](docs/ci.md) builds and tests Linux, macOS and Windows, checks
NumPy 1/2 and multiple Python versions, and runs Rust tests in debug and release.
The required `CI` status blocks `main` updates when any test job fails.

Install the test dependencies and run the reference-comparison suite:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
cargo test -p ufcoco-core
```

The repository includes synthetic inputs and a small real COCO fixture. Tests
requiring the complete dataset skip when those local files are absent; see
[fixture documentation](tests/data/README.md). Exact parity claims refer to
the tested inputs and configurations, not a proof over every possible input.

Use [bench/compare_saved_predictions.py](bench/compare_saved_predictions.py) to
save full evaluation arrays and compare your own predictions. For changes to
matching or accumulation, retain byte-level parity tests rather than checking
only the rounded summary.

## License and credits

[BSD-2-Clause](LICENSE). COCO evaluation algorithms and API compatibility are
based on [pycocotools](https://github.com/cocodataset/cocoapi), by Piotr Dollár
and Tsung-Yi Lin, under BSD-2-Clause. Implementation design and numerical
compatibility decisions are described in [DESIGN.md](DESIGN.md).
