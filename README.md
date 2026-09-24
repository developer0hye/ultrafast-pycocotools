# ultrafast-pycocotools

[![CI](https://github.com/developer0hye/ultrafast-pycocotools/actions/workflows/ci.yml/badge.svg)](https://github.com/developer0hye/ultrafast-pycocotools/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ultrafast-pycocotools)](https://pypi.org/project/ultrafast-pycocotools/)
[![Python](https://img.shields.io/pypi/pyversions/ultrafast-pycocotools)](https://pypi.org/project/ultrafast-pycocotools/)
[![License](https://img.shields.io/pypi/l/ultrafast-pycocotools)](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/LICENSE)

A drop-in replacement for `pycocotools` that computes COCO-style AP/AR for
object detection, instance segmentation and keypoint detection. Ground-truth
indexing, IoU/OKS matching and precision–recall accumulation run in a
multithreaded Rust core (Rayon), exposed to Python through PyO3.

- **Bit-exact metrics.** The complete `precision`, `recall` and `scores`
  arrays are **bit-identical** to pycocotools on every tested input, not only
  the rounded AP/AR summary.
- **Fast.** On COCO val2017, **18–54× lower wall-clock time than pycocotools**,
  7–9× lower than faster-coco-eval and 1.8–2.6× lower than hotcoco.
- **Memory-efficient.** **58–86% lower peak RSS** than pycocotools, and the
  lowest of all four evaluators on every task.
- **Drop-in.** The same `COCO` / `COCOeval` API for `bbox`, `segm` and
  `keypoints`, plus the LVIS federated protocol. Prebuilt wheels for Linux,
  macOS and Windows (CPython 3.8–3.14).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/developer0hye/ultrafast-pycocotools/main/docs/assets/readme-benchmark-dark.svg">
  <img alt="COCO val2017 evaluation time and peak memory for pycocotools, faster-coco-eval, hotcoco and ultrafast-pycocotools" src="https://raw.githubusercontent.com/developer0hye/ultrafast-pycocotools/main/docs/assets/readme-benchmark-light.svg">
</picture>

**Status:** alpha. Before replacing the reference evaluator in your pipeline,
check parity on your own evaluation parameters and `COCOeval` subclasses.

## Installation

```bash
pip install ultrafast-pycocotools
```

Prebuilt wheels need no Rust compiler; NumPy is installed as a dependency.
Other platforms build from source with a stable [Rust toolchain](https://rustup.rs/):
`pip install git+https://github.com/developer0hye/ultrafast-pycocotools`.

## Quick start

```python
from ultrafast_pycocotools import COCO, COCOeval

gt = COCO("instances_val2017.json")    # ground-truth annotations
dt = gt.loadRes("detections.json")     # detection results in COCO format
evaluator = COCOeval(gt, dt, "bbox")   # or "segm", "keypoints"
evaluator.run()                        # evaluate() + accumulate() + summarize()

print(evaluator.stats_as_dict)         # AP, AP50, AP75, APs, APm, APl, AR@1, ...
```

For LVIS, pass `lvis_style=True` to apply the federated annotation protocol
(negative and not-exhaustive category lists, 300 detections per image) and the
official metric names (`AP`, `APr`, `APc`, `APf`, `AR@300`, ...). See the
[LVIS guide](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/lvis.md).

### Use inside an existing framework

If a training or validation framework imports `pycocotools` internally,
register the replacement **before importing that framework**:

```python
from ultrafast_pycocotools import init_as_pycocotools

init_as_pycocotools()  # `import pycocotools` now resolves to ultrafast
```

This patches `sys.modules` for the whole Python process.

## Benchmarks

COCO val2017 (all 5,000 images) with cached YOLO26n, YOLO26n-seg and
YOLO26n-pose detections: 733,070 boxes, 724,953 instance masks and 134,663
pose instances. Model inference is excluded; the timed region covers JSON
parsing, ground-truth indexing, matching, accumulation and summarization.

| Task | pycocotools 2.0.11 | faster-coco-eval 1.8.0 | hotcoco 1.0.1 | **ultrafast 0.1.11** |
| --- | ---: | ---: | ---: | ---: |
| bbox | 47.55 s · 1,925 MB | 7.51 s · 1,781 MB | 2.33 s · 2,307 MB | **0.89 s · 272 MB** |
| segm | 49.38 s · 2,191 MB | 15.94 s · 2,629 MB | 5.30 s · 3,403 MB | **2.33 s · 838 MB** |
| keypoints | 9.83 s · 603 MB | 4.54 s · 603 MB | 0.99 s · 642 MB | **0.54 s · 256 MB** |
| Bit-identical to pycocotools | reference | ✗ (`precision` ≤ 2.2e-16 off on bbox/segm) | ✗ (`scores` differ on bbox/segm) | **✓ all tasks** |

Wall-clock time · peak RSS; median of 6 runs, each in a fresh process, on an
Intel Core i5-10400 with a 2-thread pool (pycocotools is single-threaded),
measured 2026-09-24.
[Full report, method and raw data](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/benchmarks/i5-10400-v0111.md)
· [All benchmarks, other hosts and historical results](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/benchmarks/README.md)

## Used by

- [**RF-DETR**](https://github.com/roboflow/rf-detr): optional `ufcoco`
  backend for bbox and mask mAP during training and validation, merged in
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

The public `COCO`, `COCOeval` and `mask` (RLE) APIs are tested against
pycocotools, covering annotation query order, `loadRes`, compressed and
uncompressed RLE, custom `Params` (IoU thresholds, area ranges, `maxDets`),
`iscrowd` handling, score ties and subclass overrides. A few internals
intentionally differ:

- Per-image matching records (`evalImgs`) are not stored by default; pass
  `store_eval_imgs=True` if your code reads them.
- Box-only detections omit the polygon `segmentation` derived from each box;
  pass `derive_segmentation=True` if you read that field.
- `COCO(annotation_dict)` borrows the dictionary instead of deep-copying it, and
  ground-truth annotations are not rewritten in place.

Bit-exactness claims cover the tested inputs and configurations. See the
[compatibility notes](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/implementation-notes.md#drop-in-compatibility).

## Extras

The same matching results provide per-category AP, precision–recall curves,
per-detection TP/FP matches and a confusion matrix. Boundary IoU is available
as an additional evaluation mode (an extension, not a pycocotools metric):

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
runs an end-to-end parity check on synthetic data, with no dataset download
or model weights. See the
[reproduction guide](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/reproducibility.md),
[CI](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/ci.md) and
[release process](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/publishing.md).
Documentation, comments and examples are written in English.

## License and credits

[BSD-2-Clause](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/LICENSE).
The evaluation protocol and API follow
[pycocotools](https://github.com/cocodataset/cocoapi) by Piotr Dollár and
Tsung-Yi Lin (BSD-2-Clause). LVIS verification uses the official
[LVIS API](https://github.com/lvis-dataset/lvis-api). Design decisions are in
[DESIGN.md](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/DESIGN.md).
