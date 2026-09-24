# ultrafast-pycocotools

[![CI](https://github.com/developer0hye/ultrafast-pycocotools/actions/workflows/ci.yml/badge.svg)](https://github.com/developer0hye/ultrafast-pycocotools/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ultrafast-pycocotools)](https://pypi.org/project/ultrafast-pycocotools/)
[![Python](https://img.shields.io/pypi/pyversions/ultrafast-pycocotools)](https://pypi.org/project/ultrafast-pycocotools/)
[![License](https://img.shields.io/pypi/l/ultrafast-pycocotools)](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/LICENSE)

COCO evaluation in Rust, with a drop-in `pycocotools` API.

- **Same results.** The complete `precision`, `recall` and `scores` arrays match
  pycocotools **byte for byte** on every tested input, not just the rounded AP.
- **Fast.** On COCO val2017, **18–54× faster than pycocotools**, 7–9× faster
  than faster-coco-eval and 1.8–2.6× faster than hotcoco.
- **Small.** **58–86% lower peak memory** than pycocotools, and the lowest of all four
  evaluators on every task.
- **Drop-in.** Same `COCO` / `COCOeval` API, bbox, segmentation, keypoints and
  LVIS, with prebuilt wheels for Linux, macOS and Windows (CPython 3.8–3.14).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/developer0hye/ultrafast-pycocotools/main/docs/assets/readme-benchmark-dark.svg">
  <img alt="COCO val2017 evaluation time and peak memory for pycocotools, faster-coco-eval, hotcoco and ultrafast-pycocotools" src="https://raw.githubusercontent.com/developer0hye/ultrafast-pycocotools/main/docs/assets/readme-benchmark-light.svg">
</picture>

**Status:** alpha. Validate your application's parameters and subclass behavior
before replacing its reference evaluator.

## Installation

```bash
pip install ultrafast-pycocotools
```

Wheels need no Rust compiler; NumPy is installed automatically. Other platforms
build from source with a stable [Rust toolchain](https://rustup.rs/):
`pip install git+https://github.com/developer0hye/ultrafast-pycocotools`.

## Quick start

```python
from ultrafast_pycocotools import COCO, COCOeval

gt = COCO("instances_val2017.json")
dt = gt.loadRes("predictions.json")
evaluator = COCOeval(gt, dt, "bbox")  # or "segm", "keypoints"
evaluator.run()                       # evaluate() + accumulate() + summarize()

print(evaluator.stats_as_dict)
```

For LVIS, pass `lvis_style=True` to use the federated protocol and official
metric names (`AP`, `APr`, `AR@300`, ...). See the
[LVIS guide](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/lvis.md).

### Use inside an existing framework

If a framework imports `pycocotools` internally, register the replacement
**before importing that framework**:

```python
from ultrafast_pycocotools import init_as_pycocotools

init_as_pycocotools()  # `import pycocotools` now resolves to ultrafast
```

This changes the import mapping for the whole Python process.

## Benchmarks

COCO val2017 (all 5,000 images) with saved YOLO26n, YOLO26n-seg and
YOLO26n-pose predictions. Times include JSON loading, evaluation, accumulation
and summary; inference is excluded.

| Task | pycocotools 2.0.11 | faster-coco-eval 1.8.0 | hotcoco 1.0.1 | **ultrafast 0.1.11** |
| --- | ---: | ---: | ---: | ---: |
| bbox | 47.55 s · 1,925 MB | 7.51 s · 1,781 MB | 2.33 s · 2,307 MB | **0.89 s · 272 MB** |
| segm | 49.38 s · 2,191 MB | 15.94 s · 2,629 MB | 5.30 s · 3,403 MB | **2.33 s · 838 MB** |
| keypoints | 9.83 s · 603 MB | 4.54 s · 603 MB | 0.99 s · 642 MB | **0.54 s · 256 MB** |
| Full arrays = pycocotools | reference | ✗ (bbox/segm precision ≤ 2.2e-16) | ✗ (`scores` differ) | **✓ byte-identical** |

Wall time and peak RSS, median of 6 fresh processes, Intel Core i5-10400,
2 threads (pycocotools is single-threaded), measured 2026-09-24.
[Full report, method and raw data](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/benchmarks/i5-10400-v0111.md)
· [All benchmarks, other hosts and historical results](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/benchmarks/README.md)

## Used by

- [**RF-DETR**](https://github.com/roboflow/rf-detr): optional `ufcoco`
  backend for COCO bbox and mask evaluation, merged in
  [roboflow/rf-detr#1449](https://github.com/roboflow/rf-detr/pull/1449).
  [Benchmark](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/rfdetr.md)

Proposed integrations under review:
[Ultralytics](https://github.com/ultralytics/ultralytics/pull/26101) ([validation](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/ultralytics-pr26101-validation.md)) ·
[torchvision](https://github.com/pytorch/vision/pull/9666) ·
[TorchMetrics](https://github.com/Lightning-AI/torchmetrics/pull/3500) ·
[SAHI](https://github.com/obss/sahi/pull/1452) ·
[SAM 3](https://github.com/facebookresearch/sam3/pull/620) ·
[RT-DETR](https://github.com/lyuwenyu/RT-DETR/pull/689) ·
[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2/pull/170)

## Compatibility

The public `COCO`, `COCOeval` and mask APIs are tested against pycocotools,
including query ordering, result loading, RLE formats, custom parameters,
crowd annotations and subclass overrides. A few internals intentionally differ:

- Per-image `evalImgs` are not stored by default; pass `store_eval_imgs=True`
  if your code reads them.
- Box-only results omit the derived polygon `segmentation`; pass
  `derive_segmentation=True` if you read that field.
- `COCO(annotation_dict)` borrows the dictionary instead of deep-copying it, and
  ground-truth annotations are not rewritten in place.

Exact-parity claims cover the tested inputs and configurations. See the
[compatibility notes](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/implementation-notes.md#drop-in-compatibility).

## Extras

Per-category statistics, precision–recall curves, match inspection, a confusion
matrix and Boundary IoU are available after evaluation:

```python
evaluator.per_category_stats()
evaluator.pr_curve(cat_id=1, iou_thr=0.5)
evaluator.matches(iou_thr=0.5)
evaluator.confusion_matrix()
```

See the [implementation notes](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/implementation-notes.md#extensions).

## Development

```bash
git clone https://github.com/developer0hye/ultrafast-pycocotools.git
cd ultrafast-pycocotools
pip install -e ".[test,lvis-test]"
python bench/fetch_lvis_fixture.py
python -m pytest -q
cargo test -p ufcoco-core
```

`python bench/reproduce.py quick --out bench/out/quick --verify-published quick`
runs a synthetic end-to-end parity check with no downloads. See the
[reproduction guide](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/reproducibility.md),
[CI](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/ci.md) and
[release process](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/publishing.md).
Documentation, comments and examples are written in English.

## License and credits

[BSD-2-Clause](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/LICENSE).
The evaluation algorithm and API follow
[pycocotools](https://github.com/cocodataset/cocoapi) by Piotr Dollár and
Tsung-Yi Lin (BSD-2-Clause). LVIS verification uses the official
[LVIS API](https://github.com/lvis-dataset/lvis-api). Design decisions are in
[DESIGN.md](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/DESIGN.md).
