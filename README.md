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
- **Fast.** On COCO val2017, **25–91× lower wall-clock time than pycocotools**,
  10–15× lower than faster-coco-eval and 2.5–4.4× lower than hotcoco.
- **Memory-efficient.** **59–86% lower peak RSS** than pycocotools, and the
  lowest of all four evaluators on every task.
- **Drop-in.** The same `COCO` / `COCOeval` API for `bbox`, `segm` and
  `keypoints`, plus the LVIS federated protocol. Prebuilt wheels for Linux,
  macOS and Windows (CPython 3.8–3.14).
- **Per-batch precision, recall and F1.** Scored from the same matching as
  AP, per batch or accumulated over a validation loop: **0.26 ms** for a
  16-image batch and **0.09 s** for all of COCO val2017, **52–780× and 27–580×
  less time** than hotcoco, Ultralytics, TorchMetrics, supervision,
  faster-coco-eval and pycocotools
  ([comparison](#per-batch-precision-recall-and-f1)).

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

| Task | pycocotools 2.0.11 | faster-coco-eval 1.8.0 | hotcoco 1.0.1 | **ultrafast 0.1.12** |
| --- | ---: | ---: | ---: | ---: |
| bbox | 46.54 s · 1,926 MB | 7.40 s · 1,781 MB | 2.26 s · 2,308 MB | **0.51 s · 262 MB** |
| segm | 50.88 s · 2,192 MB | 16.39 s · 2,630 MB | 5.43 s · 3,404 MB | **1.66 s · 556 MB** |
| keypoints | 10.05 s · 604 MB | 4.66 s · 603 MB | 1.01 s · 643 MB | **0.41 s · 249 MB** |
| Bit-identical to pycocotools | reference | ✗ (`precision` ≤ 2.2e-16 off on bbox/segm) | ✗ (`scores` differ on bbox/segm) | **✓ all tasks** |

Wall-clock time · peak RSS; median of 6 runs, each in a fresh process, on an
Intel Core i5-10400 with a 2-thread pool (pycocotools is single-threaded),
measured 2026-09-30.
[Full report, method and raw data](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/benchmarks/i5-10400-v0112.md)
· [All benchmarks, other hosts and historical results](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/benchmarks/README.md)

## Used by

- [**RF-DETR**](https://github.com/roboflow/rf-detr): optional `ufcoco`
  backend for bbox and mask mAP during training and validation
  (`TrainConfig.eval_backend="ufcoco"`), merged in
  [roboflow/rf-detr#1449](https://github.com/roboflow/rf-detr/pull/1449) and
  released in RF-DETR 1.11.0.
  [Benchmark](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/rfdetr.md)
- [**SAHI**](https://github.com/obss/sahi): optional backend for
  `sahi coco evaluate` (`pip install "sahi[ultrafast]"`, then
  `--backend ultrafast`), merged in
  [obss/sahi#1452](https://github.com/obss/sahi/pull/1452) and released in
  SAHI 0.12.7.

Proposed integrations under review:
[Ultralytics](https://github.com/ultralytics/ultralytics/pull/26101) ([validation](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/ultralytics-pr26101-validation.md)) ·
[torchvision](https://github.com/pytorch/vision/pull/9666) ·
[TorchMetrics](https://github.com/Lightning-AI/torchmetrics/pull/3500) ·
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
per-detection TP/FP matches, a confusion matrix, and precision, recall and F1
at a score threshold or at the best-F1 cut. Boundary IoU is available
as an additional evaluation mode (an extension, not a pycocotools metric):

```python
evaluator.per_category_stats()
evaluator.pr_curve(cat_id=1, iou_thr=0.5)
evaluator.matches(iou_thr=0.5)
evaluator.confusion_matrix()
evaluator.detection_metrics(iou_thr=0.5, score_thr="best_f1")
```

See the [implementation notes](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/implementation-notes.md#extensions).

## Per-batch precision, recall and F1

For training and validation loops, `DetectionMetrics` scores batches of
per-image arrays or tensors (or COCO data) with the same matching as AP:

```python
from ultrafast_pycocotools.metrics import DetectionMetrics

metric = DetectionMetrics(iou_thrs=0.5, score_thr="best_f1")  # box_format="xyxy"
for images, targets in loader:
    batch = metric.update(model(images), targets)  # [{"boxes", "scores", "labels"}, ...]
total = metric.compute()  # per_category / micro / macro, and the chosen score_thr
```

- `update()` returns the batch's metrics; `compute()` returns them over every
  batch, bit-identical to one evaluation over all images. Only a category id,
  a score and one byte per IoU threshold are kept per detection, never boxes
  or masks (`batch_results=False` skips the per-batch summary).
- Precision, recall and F1 per category, micro and macro, at any IoU
  thresholds, with every detection, a score threshold, or the score cut that
  maximizes F1 (`score_thr="best_f1"`). Zero denominators give 0.
- Boxes (`xyxy` or `xywh`) or masks, class-aware or class-agnostic. COCO's
  crowd regions are honored, and its area ranges and `maxDets` are optional.
- NumPy arrays, lists and PyTorch tensors; CPU tensors are read in place.
  `COCOeval.detection_metrics()` computes the same metrics after
  `evaluate()`, with that evaluation's parameters.

COCO val2017 with YOLO26n predictions, one thread, on an Intel Core i5-10400;
each library computes its own metric (P/R/F1, F1 or AP), so the times are
the cost of getting each library's numbers per batch:

| Library | 16-image batch | All 5,000 images in batches of 16 | Peak memory added by the 5,000 images |
| --- | ---: | ---: | ---: |
| **ultrafast-pycocotools** | **0.26 ms** | **0.09 s** | **34 MB** |
| hotcoco 1.1.0 | 13 ms | 2.3 s | 578 MB |
| Ultralytics 8.4.172 | 20 ms | 2.9 s | 145 MB |
| TorchMetrics 1.9.0 (faster-coco-eval backend) | 30 ms | 9.1 s | 1,582 MB |
| supervision 0.30.6 | 35 ms | 10.3 s | 74 MB |
| faster-coco-eval 1.8.0 | 170 ms | 28.3 s | 1,623 MB |
| pycocotools 2.0.11 | 197 ms | 44.7 s | 1,728 MB |
| TorchMetrics 1.9.0 (pycocotools backend) | 200 ms | 49.7 s | 1,697 MB |

[What each library computes, method and raw data](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/batch-metrics-comparison.md)
· [Design](https://github.com/developer0hye/ultrafast-pycocotools/blob/main/docs/detection-metrics-design.md)

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
