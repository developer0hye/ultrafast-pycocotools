# Implementation notes

Historical measurements and design decisions. These include different synthetic
prediction recipes, thread counts and timing scopes; they are not the public
reproduction run. See [README](../README.md), the [0.1.1 efficiency report](efficiency.md)
and [public_benchmarks.json](../bench/results/public_benchmarks.json) for the
versioned benchmark recipes and results. Test counts below describe development
snapshots; the current suite and CI are authoritative.

## ultrafast-pycocotools

A Rust implementation of COCO evaluation with a compatible Python API. On the
tested inputs, AP and the complete `precision`, `recall` and `scores` arrays
match pycocotools byte for byte. Tests compare bytes rather than using a tolerance.

The documented installation builds from source and needs a Rust toolchain.
An already built wheel does not require Rust at runtime. See
[Try it locally](#try-it-locally).

```bash
git clone https://github.com/developer0hye/ultrafast-pycocotools
cd ultrafast-pycocotools
pip install maturin && maturin develop --release
```

```python
from ultrafast_pycocotools import COCO, COCOeval

gt = COCO("instances_val2017.json")
dt = gt.loadRes("detections.json")
ev = COCOeval(gt, dt, "bbox")
ev.run()          # evaluate() + accumulate() + summarize()
```

For code that already imports pycocotools, register the replacement first:

```python
from ultrafast_pycocotools import init_as_pycocotools
init_as_pycocotools()          # Before importing torchvision, detectron2 or mmdetection.
```

## Why use it?

| Symptom | Cause | Relevant section |
|---|---|---|
| Evaluation takes longer than an epoch | Python image-by-category loops in `evaluate()` | [Speed and memory](#speed-and-memory) |
| Large-dataset evaluation runs out of memory | `evalImgs` retains K×A×I dictionaries and NumPy arrays | [Speed and memory](#speed-and-memory) |
| A faster backend changes the fifth or sixth decimal of AP | Reconstructed threshold grids differ from `np.linspace` | [Numerical equivalence](#numerical-equivalence) |
| A replacement breaks subclasses or postprocessing | Missing public methods or different return types | [Drop-in compatibility](#drop-in-compatibility) |
| Installing a native package on Windows fails to compile | Native compiler/toolchain requirements | Use a compatible wheel when available, or install the build toolchain |
| Tied scores produce different AP across implementations | Greedy matching depends on stable ordering | [Numerical equivalence](#numerical-equivalence) |

Reading guide:

- For basic evaluation, start with the installation and usage examples.
- For reproducible paper results or AP regression checks, read numerical equivalence.
- For `COCOeval` subclasses or direct `evalImgs`/`ious` access, read compatibility.
- For custom metrics and error analysis, read extensions.
- For implementation changes, read [`DESIGN.md`](../DESIGN.md).

## Numerical equivalence

Tests compare the bytes of complete arrays, including `eval['precision']`,
rather than just the 12 summary statistics. A COCO precision tensor has nearly
a million doubles. Averaging can hide differences in many curve cells and still
produce the same rounded AP.

Historical comparison using 431,145 YOLO11m predictions:

| Implementation | Different precision cells | Largest cell error | Largest summary error |
|---|---:|---:|---:|
| ultrafast-pycocotools | 0 | 0.0e+00 | 0.0e+00 |
| faster-coco-eval 1.7.2 | 5,744 | 2.2e-16 | 0.0e+00 |
| hotcoco 0.5.0 | 6,069 | 8.5e-01 | 3.9e-05 |

Zero summary error does not imply identical arrays. For faster-coco-eval,
5,744 of 969,600 precision cells differed by one ULP, but averaging produced
the same 12 summary doubles. Early checks in this repository also missed that
difference until array digests were added.

A difference of 1e-5 disappears when AP is reported to three decimal places,
but individual hotcoco cells differed by as much as 0.85 in this workload:

| Difference magnitude | Cells | Cause |
|---|---:|---|
| Approximately 2e-16, one ULP | 5,744 | Missing `np.spacing(1)` |
| 1e-6 to 1e-2 | 84 | One-ULP threshold-grid differences |
| Greater than 1e-2 | 241 | One-ULP threshold-grid differences |

`bench/diagnose_divergence.py` reproduces the threshold mechanism:

```text
pycocotools recThrs[70] = 0x1.6666666666667p-1  (0.7000000000000001)
hotcoco     recThrs[70] = 0x1.6666666666666p-1  (0.7)
recall 7/10            = 0x1.6666666666666p-1  (0.7)

7/10 >= pycocotools threshold: False -> no sample    -> precision 0.0
7/10 >= hotcoco threshold:     True  -> sample curve -> precision 0.85
```

For a category with ten GT annotations, recall can land exactly on 7/10.
The grid difference then determines whether that sampled precision is 0 or
0.85. Categories with few annotations are especially sensitive.

### Sources of numerical differences

**1. Reconstructing threshold grids changes rounding.**

Pycocotools uses `np.linspace` and warns about alternative grid construction:

```python
# np.arange causes trouble.  the data point on arange is slightly larger than the true value
self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
```

`np.linspace(0.5, 0.95, 10)` differs from `[0.5 + 0.05*i]` in two of ten
entries. `np.linspace(0, 1, 101)` differs in ten entries:

```text
iouThrs[7]  linspace 0.85               (0x1.b333333333333p-1)
            naive    0.8500000000000001 (0x1.b333333333334p-1)
recThrs     0.35 0.41 0.47 0.57 0.69 0.70 0.82 0.83 0.94 0.95
            These ten entries differ by one ULP.
```

Accumulation uses `np.searchsorted(rc, recThrs, side='left')`, with strict `<`
comparisons. Recall is `tp/npig` and often lands exactly on values such as 0.35.
The grids can therefore select different precision samples. This explained why
hotcoco could match AP@0.50 and AP@0.75, whose grid entries agree, while differing
in the AP@[0.50:0.95] average.

Ultrafast constructs grids in Python with `np.linspace` and passes them into
Rust without reconstruction.

**2. Unstable score sorting can change matches.**

Pycocotools uses `np.argsort(-scores, kind='mergesort')`. Tied scores retain
annotation order, and the earlier detection can claim a GT first.

For predictions rounded to three decimal places, hotcoco's AR differed by
6.5e-7 in the historical test. Removing ties reduced the AR difference to about
1e-16, showing that tie ordering had affected matching itself.

Ultrafast preserves stable ordering and the collection order of
`loadAnns(getAnnIds(...))`, so both implementations use the same definition of
annotation order.

**3. Precision requires the epsilon term.**

Pycocotools computes `pr = tp / (fp + tp + np.spacing(1))`. Omitting the epsilon
changes points where `fp + tp == 1` by one ULP. Ultrafast retains it as
`f64::EPSILON`.

### Verification coverage

The suite combines direct comparisons against the installed pycocotools package
with Rust unit tests and API invariants. It does not rely only on golden files.
Historical snapshots recorded 139 Python tests and 57 Rust tests; the suite has
since grown. See the [current CI coverage](ci.md).

### Mutation testing checks whether tests detect broken behavior

A passing suite is not sufficient evidence of sensitivity. The mutation harness
intentionally changes the core and checks whether tests fail:

```bash
python bench/mutation_check.py
# Historical result: caught 21/21
```

Almost half of the initial 21 mutations survived. Examples:

| Surviving mutation | Why the initial tests missed it |
|---|---|
| Transpose `bb_iou` output | The intended transpose fixture was itself symmetric: `[[1,0],[0,0]]` |
| Remove sign extension in `from_str` | The round-trip corpus contained no negative deltas |
| Remove the precision envelope | Fixtures had only one or two detections, so curves were too simple |
| Remove the crowd rematching exception | Test detections matched different GTs and never exercised multiple detections absorbed by one crowd |
| Remove the OKS mean division | There was no Rust keypoint test |

Fixtures were extended to catch every mutation. Sign extension needed only the
run sequence `vec![10, 5, 3, 2]`; real masks frequently contain decreasing run
lengths, but the original corpus happened to contain increasing sequences.

The historical 57-test Rust suite used small, manually verifiable cases so that
failures identify a broken rule rather than only a changed AP:

```rust
assert_eq!(res.precision[0], 1.0 / (1.0 + EPS));
assert_ne!(res.precision[0], 1.0, "the epsilon was dropped");
```

Covered rules include hardware-compatible `c_i32` behavior (`NaN -> INT_MIN`),
tie resolution by annotation ID, multiple detections absorbed by a crowd,
matching exactly at an IoU threshold, right-to-left precision-envelope
propagation, `0.0` versus `-1.0` sentinels, and OKS division order. Checking
matched IDs matters because TP/FP totals can agree while assignments differ.

The Python suite expanded across several snapshots; historical category counts
below are not an additive current total:

- **Mask API, 47 cases:** `encode`, `decode`, `merge`, `area`, `toBbox`, `iou`
  and `frPyObjects`; 1×1, empty and full masks; out-of-image and duplicate-vertex
  polygons; and crowd flags. Duplicate vertices exercise division by zero and
  NaN-to-integer conversion inside `rleFrPoly`.
- **Real COCO, eight cases:** the committed 93-image, approximately 440 kB fixture
  runs on fresh clones and CI. Earlier tests silently skipped without a local
  20 MB annotation file. Real multi-ring polygons and arbitrary uncompressed
  crowd RLEs cover shapes that synthetic data only approximates.
- **Fixture coverage, eight cases:** assert that difficult cases actually exist.
  Removing all crowds would leave parity tests green on easier data. These
  checks found that independently sampled keypoint visibility made instances
  with `num_keypoints == 0` effectively absent: their probability was `(1/5)^17`.
- **Evaluation parity, 16 cases:** full-array bytes for synthetic bbox/segm,
  real COCO subsets, keypoints, `useCats=0`, custom area/maxDets/IoU settings,
  image/category subsets, no detections, all-tied scores, omitted derived
  polygons and complete `evalImgs` records.
- **Drop-in compatibility, 34 cases:** described below.
- **JSON loading, 16 cases:** compare float bits with `json.load` on committed
  real COCO data. These tests caught one-ULP differences in serde_json's default
  float parser.
- **Extension APIs, 15 cases:** invariants such as per-class AP averaging back
  to mAP and confusion-matrix counts agreeing with AP accounting.
- **Determinism, two cases:** compare result bytes with one and eight Rayon
  threads, which a single reference comparison on one thread count cannot test.

## Drop-in compatibility

Identical AP does not ensure existing application code works unchanged.
Returning `str` instead of `bytes`, or omitting an overridable method, can break
an otherwise numerically correct replacement.

`bench/audit_api.py` enumerates the reference public API instead of relying on
a hand-maintained list. A historical audit reported zero incompatibilities
after adding `COCOeval.computeIoU`, `computeOks` and `evaluateImg`. Without a
working override path, subclasses can silently stop affecting evaluation.

The original 34 compatibility cases checked:

| Behavior | Reason |
|---|---|
| All enumerated public methods and attributes exist | Also catches APIs not anticipated by maintainers |
| Signatures retain arguments without adding required ones | Additional optional keyword arguments are allowed |
| Order for 13 `getAnnIds` filter combinations | Order resolves score ties |
| `loadAnns` returns the stored dictionary itself, checked with `is` | Callers can mutate that dictionary |
| RLE `counts` is `bytes`, not `str` | Return-type compatibility |
| Character-identical `summarize()` output | Consumers may parse the text |
| Values from `computeIoU`, `computeOks`, `evaluateImg`, `ious` and `_gts` | Subclass and diagnostic access |
| torchvision-style `CocoEvaluator` loop and NumPy Nx7 `loadRes` | Real caller patterns |
| A subclass `evaluateImg` override is actually called | Avoid silently ignored customization |
| Complete execution after `init_as_pycocotools()` and a reference-style import | Framework import path |
| Explicit assertions for intentional differences | Detect unintended changes in either direction |

Diagnostic structures are built on first access. Historical before/after
compatibility measurements were bbox 0.1043 → 0.1059 s and segmentation
0.1999 → 0.1979 s, with unchanged memory: within measurement noise.

## Speed and memory

These historical measurements use `bench/compare.py`. On shared machines,
wall time alone can confuse contention with implementation efficiency:

- Include `--threads 1` to compare implementations without varying core counts.
- Report CPU time beside wall time; contention can inflate wall time without
  increasing CPU time.
- Interleave implementations across repeats so drift does not affect only the
  implementation measured later.
- Record host load and measurement spread. Single measurements around 0.1 s
  can vary by 15%.

**Single-thread YOLO11m predictions:** five runs at 15–48% host load.

| Implementation | Best wall | Median wall | Spread | Best CPU | Wall speedup | CPU speedup | Peak RSS | Equivalence |
|---|---|---|---|---|---|---|---|---|
| pycocotools | 22.135 s | 22.478 s | 22% | 21.672 s | 1.0× | 1.0× | 1.29 GB | Reference |
| faster-coco-eval | 3.527 s | 3.559 s | 4% | 3.516 s | 6.3× | 6.2× | 1.24 GB | One-ULP array differences |
| hotcoco | 1.168 s | 1.349 s | 33% | 1.156 s | 18.9× | 18.7× | 1.25 GB | Maximum summary difference 3.9e-05 |
| ultrafast-pycocotools | 0.548 s | 0.551 s | 16% | 0.531 s | 40.4× | 40.8× | 641 MB | Bit-identical |

Ultrafast was 2.1× faster than hotcoco in this single-thread case. Similar wall
and CPU speedups, 40.4× and 40.8×, suggest limited contention distortion.
Parallel results depend more strongly on available cores.

The following tables use default parallel settings and the best of three runs
per implementation in separate processes. `eval` means
`evaluate + accumulate + summarize`; it excludes loading.

### Verification with real model outputs

Synthetic detections are derived from GT, with correlated boxes and rounded
scores that deliberately create ties. Real model outputs provide a different
stress case: float32 scores with fewer ties, detector-generated boxes and often
more predictions per image. Both are needed to avoid relying on one generator.

Historical COCO val2017 runs used several detector architectures
(`bench/predict_coco*.py`):

| Model | Architecture | iouType | Detections | Per image | Measured AP | Published AP reference |
|---|---|---|---|---:|---:|---|
| YOLO11m | Anchor-free with NMS | bbox | 431,145 | 86.2 | 0.507 | 51.5 |
| YOLO11m-seg | Prototype masks | segm | 431,006 | 86.2 | — | — |
| RF-DETR base | DETR queries, no NMS | bbox | 1,500,000 | 300.0 | 0.532 | Approximately 53–54 |
| Mask R-CNN | Two-stage, per-RoI masks | segm | 171,031 | 34.2 | 0.346 | 34.6 |
| Keypoint R-CNN | Two-stage, OKS | keypoints | 74,143 | 14.8 | 0.600 | 61.1 |

The measured values provided a pipeline sanity check, with Mask R-CNN matching
34.6 AP exactly. They are not all exact reproductions of published model AP.
All five evaluator comparisons were bit-identical. Keypoints are particularly
useful because OKS uses `exp()`, where NumPy and Rust/platform math libraries
can differ.

### Verification across two machines

Agreement with pycocotools on one machine does not prove cross-platform
agreement. C `long` width in `rleToString` and platform implementations of OKS
`exp` were two potential sources of differences.

| Platform | Reference build |
|---|---|
| Windows 11 / MSVC / AMD64 | 32-bit C `long` |
| Ubuntu 24.04 / glibc 2.39 / EPYC 9554 | GCC with 64-bit C `long` |

`bench/cross_platform.py` compared complete precision/recall/scores digests for
identical predictions on both machines. Bbox, segmentation and keypoint outputs
agreed across those machines for each of the four implementations. Their
reference-comparison conclusions also agreed: ultrafast was bit-identical on
both, while hotcoco had summary differences around 3.7e-05 to 3.9e-05.

**COCO val2017:** 5,000 images / 36,781 GT annotations / 37,504 detections.

| iouType | Implementation | Evaluation | Speedup | Peak RSS | Reported equivalence |
|---|---|---|---|---|---|
| bbox | pycocotools 2.0.11 | 6.494 s | 1.0× | 644 MB | Reference |
| bbox | faster-coco-eval 1.7.2 | 1.839 s | 3.5× | 636 MB | Bit-identical on this historical case |
| bbox | hotcoco 0.5.0 | 0.149 s | 43.4× | 477 MB | Maximum absolute difference 1.0e-05 |
| bbox | ultrafast-pycocotools | 0.104 s | 62.3× | 244 MB | Bit-identical |
| segm | pycocotools 2.0.11 | 7.484 s | 1.0× | 636 MB | Reference |
| segm | faster-coco-eval 1.7.2 | 3.704 s | 2.0× | 685 MB | Bit-identical on this historical case |
| segm | hotcoco 0.5.0 | 0.220 s | 34.0× | 482 MB | Maximum absolute difference 2.0e-06 |
| segm | ultrafast-pycocotools | 0.200 s | 37.4× | 341 MB | Bit-identical |

**Objects365 validation:** 80,000 images / 1,240,587 GT annotations /
1,170,984 detections / 365 categories.

| Implementation | Evaluation | Speedup | Peak RSS | Reported equivalence |
|---|---|---|---|---|
| pycocotools 2.0.11 | 384.7 s | 1.0× | 24.89 GB | Reference |
| faster-coco-eval 1.7.2 | 157.6 s | 2.4× | 28.81 GB | Bit-identical on this historical case |
| hotcoco 0.5.0 | 4.22 s | 91.1× | 10.74 GB | Maximum absolute difference 1.8e-06 |
| ultrafast-pycocotools | 2.75 s | 139.8× | 2.41 GB | Bit-identical |

For this Objects365 case, ultrafast used roughly one tenth of pycocotools' RSS
and about 22% of hotcoco's. Much of pycocotools' memory is `evalImgs`:
category × area range × image dictionaries containing T×D arrays. Ultrafast
accumulates within each category/area range without retaining that global
intermediate representation. `store_eval_imgs=True` restores those records.

A historical memory profile attributed 89% of retained memory to Python
annotation dictionaries: GT 867 MB, DT 828 MB and Rust engine 197 MB. These
components and whole-process RSS have different scopes and are not additive.
Keeping real dictionaries supports frameworks that directly inspect and mutate
`coco.anns[id]`.

Since 0.1.1, `loadRes` omits bbox-derived polygons by default, avoiding both their
allocation time and retained memory. `annToRLE`, `annToMask` and `showAnns`
construct bbox geometry only when needed:

```python
dt = gt.loadRes("detections.json")  # Time and memory improvements apply by default.
```

`test_default_result_loading_preserves_bbox_and_segmentation_metrics` compares
both bbox and segmentation arrays against pycocotools. Code that reads a derived
`ann['segmentation']` field directly can request `derive_segmentation=True`.
See the [0.1.1 measurements](efficiency.md); the allocation breakdown above
predates that default.

The Rust annotation loader also matches `json.load` float bits, checked on real
COCO files in `tests/test_json_loader.py`. A historical 269 MB Objects365 file
loaded in 2.52 s versus 4.07 s for the reference path.

## Profiling tools

The repository includes the tools used to identify optimization opportunities:

```bash
# Enumerate differences from the pycocotools public API.
python bench/audit_api.py

# Read internal engine phase timers.
python bench/profile_engine.py --gt gt.json --dt dt.json --iou-type segm

# Count Rust allocations and measure process memory.
maturin develop --release --features alloc-stats
python bench/profile_memory.py --gt gt.json --dt dt.json --iou-type segm
```

`profile_engine.py` reports:

- **Extraction:** `read` needs the GIL; `rasterise` does not. They overlap and
  must not be added. A large `read_blocked` suggests a rasterization bottleneck;
  zero indicates that reading is slower in that measurement.
- **Evaluation:** phase durations are summed across workers. Comparison with
  wall time indicates parallel overlap, reported as parallel speedup.

`profile_memory.py` reports per-phase RSS deltas, optional Python allocation
tracking with `tracemalloc`, Rust live/peak bytes and allocation counts. Without
`alloc-stats`, Rust values are `n/a` rather than a misleading zero.

| Profiling finding | Change | Historical effect |
|---|---|---|
| Python string creation for each annotation-field lookup | Cache keys with `intern!` | Removed 14.4 million strings on Objects365 |
| Alternating reads and rasterization | Pipeline through a worker | Rasterization overlapped reading |
| Function-call overhead from `PyFloat_AS_DOUBLE` under `abi3` | Per-version wheels | Polygon reading 0.098 → 0.067 s |
| `Vec<f64>` allocation for each bbox | Direct fixed-array reads | Removed 2.4 million allocations on Objects365 |
| Reallocate match buffers for every area range | Reuse buffers within a category | Objects365 evaluation allocations 21.76 → 6.66 million |
| Spare capacity in RLE `cnts` | `shrink_to_fit` | Reduced up to 2× mask storage to required capacity |
| DT reading waits for the GT rasterizer to drain | Share one worker across GT and DT | Segmentation extraction 0.127 → 0.115 s |
| Category imbalance limited 12-core parallel speedup to 4.5× | Parallelize images within categories | Segmentation evaluation 0.052 → 0.037 s |

## Extensions

Additional diagnostics use the same matches as AP, keeping their accounting
consistent with the displayed evaluation result:

```python
ev.run()

ev.stats_as_dict            # {"AP": ..., "AP_50": ..., "AR_small": ...}
ev.per_category_stats()     # Per-class AP/AP50/AP75/AR.
ev.pr_curve(cat_id=1, iou_thr=0.5)   # {"recall", "precision", "score"}
ev.matches(iou_thr=0.5)     # Columns: image_id/category_id/dt_id/gt_id/score/iou.
ev.confusion_matrix()       # Includes a background row and column.
ev.mean_iou()               # Mean IoU of matched pairs, describing localization quality.
```

`matches()` returns column-wise NumPy arrays instead of a list of dictionaries.
Hundreds of thousands of matching records can otherwise cost more to materialize
than evaluation itself.

Custom evaluation grids are supported and covered by reference comparisons:

```python
ev.params.areaRng    = [[0, 1e10], [0, 500], [500, 5000], [5000, 1e10]]
ev.params.areaRngLbl = ["all", "tiny", "mid", "big"]
ev.params.maxDets    = [3, 25, 300]
ev.params.iouThrs    = np.linspace(0.3, 0.9, 7)
```

The additional `"boundary"` IoU mode implements Boundary IoU (Cheng et al.,
CVPR 2021), using the minimum of mask and boundary IoU. It is useful for
boundary-sensitive segmentation comparisons and is not a pycocotools metric.

## Intentional differences

These differences concern execution and side effects rather than the tested
COCO metric values:

| Difference | Reason |
|---|---|
| `evaluate()` performs matching and accumulation; `accumulate()` exposes the result | Avoid retaining K×A×I intermediate dictionaries while preserving the call sequence |
| Normal evaluation does not rewrite annotation dictionaries | Reference preparation can replace segmentation with RLE and overwrite ignore flags in place |
| `ious`, `_gts` and `_dts` are created on access | Avoid materialization costs when diagnostics are unused |
| `frPyObjects` accepts a list of boxes and single-object forms | Support documented forms where the reference can call `len()` on a float and raise `TypeError` |
| `COCO(..., verbose=False)` suppresses progress output | Additional convenience argument |
| Box-only results omit derived polygons by default | Reduce time and memory; explicit materialization remains available |

**COCO ignore behavior.** The reference preparation code overwrites the explicit
ignore flag with the crowd flag:

```python
gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']     # Unconditionally replaces the previous value.
```

An annotation's standalone `ignore` field therefore has no effect in this COCO
path, which can surprise callers with datasets such as CrowdHuman. Ultrafast
reproduces that behavior for equivalence. Explicit LVIS mode has its own ignore
semantics; see the [LVIS guide](lvis.md).

## Try it locally

The following commands use Windows virtual-environment paths; on Linux/macOS,
replace `.venv/Scripts/` with `.venv/bin/`:

```bash
git clone https://github.com/developer0hye/ultrafast-pycocotools
cd ultrafast-pycocotools
python -m venv .venv
.venv/Scripts/pip install -e ".[test]" maturin
.venv/Scripts/python -m maturin develop --release

# Generate detections from local real COCO annotations.
.venv/Scripts/python bench/make_dets.py --gt path/to/instances_val2017.json \
    --out bench/data/dt.json --segm

.venv/Scripts/python -m pytest tests/ -q
```

An early development snapshot reported `61 passed in 5.11s`; current counts
and timings depend on the revision and available optional fixtures.

Benchmark command:

```bash
.venv/Scripts/python bench/run_impl.py --impl ufcoco \
    --gt bench/data/instances_val2017.json --dt bench/data/dt.json --iou-type bbox
```

## Further reading

- [`DESIGN.md`](../DESIGN.md): data structures, parallelism, SIMD policy and
  numerical-equivalence rules.
- [pycocotools](https://github.com/cocodataset/cocoapi): reference implementation,
  especially `common/maskApi.c` and `PythonAPI/pycocotools/cocoeval.py`.
- [faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval): C++ implementation
  with LVIS/CrowdPose extensions.
- [hotcoco](https://github.com/derekallman/hotcoco): Rust implementation with TIDE,
  calibration, OBB and Open Images features.

## Verification principle

A speedup that preserves AP must be checked against complete array bytes,
not inferred from matching rounded summaries. The tests enforce that contract
for their covered inputs and configurations.

## License

BSD-2-Clause. The COCO algorithms are based on pycocotools by Piotr Dollár and
Tsung-Yi Lin, also under BSD-2-Clause.
