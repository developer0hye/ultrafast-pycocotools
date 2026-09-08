# Design notes

This document is for contributors and maintainers. [`README.md`](README.md)
explains what the library does; this document explains why it is implemented
this way. Many choices that could look simpler in another implementation are
constrained by numerical equivalence.

Profiling numbers below are historical development measurements, with the input
and timing scope described in each section. For the current default behavior,
see the [0.1.1 efficiency report](docs/efficiency.md).

## Where to start

| Intended change | Read first |
|---|---|
| Arithmetic optimization: vectorization, reassociation, FMA | [Equivalence rules](#three-equivalence-rules) |
| Sorting changes, including `sort_unstable` | [Rule 2](#rule-2-preserve-stable-ordering) |
| Generate thresholds in Rust | [Rule 3](#rule-3-the-caller-constructs-the-grids) |
| Reduce memory on large datasets | [Data structures](#data-structures-dense-tables-do-not-scale-to-objects365) |
| Understand evaluation order | [Loop structure](#loop-structure-fused-evaluation-and-accumulation) |
| Add a metric | [Extension points](#extension-points) |

## Three equivalence rules

Speed and equivalence are both requirements. These rules address differences
observed and measured in other implementations.

### Rule 1: do not reassociate floating-point arithmetic

Rust does not enable fast-math by default. LLVM cannot freely reassociate
expressions or fuse them into FMA instructions. The release profile in
`Cargo.toml` uses `opt-level`, `lto` and `codegen-units`, which preserve values.

| Operation | Example | Allowed? |
|---|---|---|
| Integer SIMD | RLE run counting and decoding | Yes: exact by construction |
| Element-wise floating-point SIMD | Independent `(d, g)` pairs in `bb_iou` | Yes: preserve scalar operation order |
| Exact-operation reduction | Backward precision `max` | Yes: `max` introduces no rounding |
| Horizontal floating-point sum | OKS `sum(exp(-e))` | No: changing order can change the result |
| `f64::mul_add` | Anywhere in reference arithmetic | No: removes intermediate rounding |

Specific requirements:

- Keep `EPS` in `pr = tp / (fp + tp + EPS)`, where
  `EPS = np.spacing(1) = f64::EPSILON`. Removing it changes the first curve point,
  where `fp + tp == 1`, by one ULP. The inspected faster-coco-eval C++ and hotcoco
  paths omit this term.
- Keep the divisions in OKS sequential:
  `(dx*dx + dy*dy) / vars[t] / (area + EPS) / 2.0`.
  Combining their denominators changes rounding.
- `rle_iou` computes the exact run intersection only when bbox IoU is strictly
  positive. This is reference behavior, not merely an optimization: pairs that
  fail the bbox check must return exactly `0.0` without inspecting the masks.

### Rule 2: preserve stable ordering

Pycocotools uses `kind='mergesort'`, so tied detection scores retain annotation
order. The greedy matcher depends on that order: an earlier detection can take
a GT that would otherwise match a later detection.

- Use `Vec::sort_by` or `sort_by_key`. The exception is `a.sort_unstable()` in
  `rle_fr_poly`, where equal integer values are indistinguishable.
- `cmp_desc_score` produces the same permutation as ascending `-score`, including
  NumPy's treatment of `-0.0` and `0.0` as equal and placement of NaNs last.
- Input order is part of the contract. Collection preserves the order of
  `loadAnns(getAnnIds(...))`, and `group.rs` retains it. Additional sorting,
  deduplication or filtering can change what annotation order means.

### Rule 3: the caller constructs the grids

Python constructs `iouThrs` and `recThrs` with `np.linspace` and passes them to
the engine. Do not regenerate them in Rust.

`np.linspace(0.5, 0.95, 10)` differs from `[0.5 + 0.05*i]` in two of ten entries
by one ULP. For `np.linspace(0, 1, 101)`, ten entries differ. Accumulation uses
`searchsorted(rc, recThrs, side='left')`, so these differences can select a
different precision value when recall lands exactly on a threshold. That is
common because recall is `tp/npig`. This explained the measured COCO AP
difference of 1.7e-6 in hotcoco.

## Data structures: dense tables do not scale to Objects365

Pycocotools effectively builds a dense image-by-category table. COCO's
5,000 × 80 table is manageable. Objects365 validation has
80,000 × 365 = 29.2 million cells: even empty 24-byte `Vec` headers would use
about 700 MB before storing any annotations.

`group.rs` stores only existing pairs. It stably sorts annotation indices into
a CSR-style representation:

```text
order:      annotation indices, stably sorted by (group, image, category)
runs:       (image_slot, start, len), a contiguous (group, image) range
group_runs: each group's [start, end) range in runs
```

Storage is O(annotations), with O(1) group lookup. `RunJoin` merge-walks GT and DT
runs in image order and never visits images where both are empty. Pycocotools
visits those images and creates `None`; those entries contribute to no metric,
so skipping them preserves observable results.

## Loop structure: fused evaluation and accumulation

Pycocotools computes all IoUs, creates K×A×I result dictionaries, and then
accumulates them. The intermediate dictionaries account for much of its memory.

Our outer loop is over categories, completing IoU, matching and accumulation
within each category:

```text
for k in categories (parallel with Rayon):
    work = prepare_category(k)            # Per-image IoU matrices for this category
    for a in areaRanges:
        matches = work.map(evaluate_img)  # Matches for this (k, a)
        for m in maxDets:
            accumulate_slice(...)        # Accumulate immediately, then discard
```

Each worker retains one category's working data. IoU matrices are reused across
area ranges, preserving the reference amount of IoU computation.

The tradeoff is that `evalImgs` is not materialized by default. With
`store_eval_imgs=True`, `materialise_eval_imgs()` reconstructs reference-style
dictionaries and restores `None` for absent images, preserving positional indices.

## Parallelism

Rayon distributes categories across workers. Each task writes only its own
output slots, so changing the thread count does not change the result. Tests
verify this property. `accumulate_slice` writes category-local buffers. Each completed category is
copied into the final tensors under a short lock, then its local arrays are
released. This avoids retaining a second full set of result tensors.

Reading Python annotations requires the GIL and is single-threaded. Geometry is
collected in chunks of 4,096 annotations; rasterization runs with the GIL
released through `py.detach()`. Peak storage is one chunk plus completed RLEs,
instead of all polygons plus all RLEs.

## Rejected optimization: skip unused masks

IoU is computed only within an `(image, category)` cell. Masks in cells with GT
or DT on only one side are never compared. Unmatched GT still contributes an FN,
and predictions in an absent category contribute FPs, without needing pixels.
Yet both implementations convert every annotation to RLE. For YOLO11m-seg on
COCO val2017, 119,564 of 467,787 masks (25.6%) were converted and never used.

Skipping that conversion would leave AP, `evalImgs` and `matches` unchanged.
The optimization was rejected because its estimated upper bound was only 1.8%
of total evaluation time.

`bench/probe_unused_masks.py` estimates the upper bound by replacing those
segmentations with empty 1×1 masks and measuring engine construction:

```text
                              build   gt_read  dt_read  rasterise  read_blocked
segm, as-is                   0.288     0.055    0.200      0.182         0.026
segm, unused masks stripped   0.257     0.062    0.167      0.144         0.014
bbox (no masks at all)        0.070     0.011    0.042      0.000         0.000
```

The difference is about 0.030 s against 1.655 s for single-thread segmentation
evaluation, or 1.8%. A real implementation would first need grouping to identify
empty cells, requiring another annotation traversal and reducing the benefit.
Rasterization already overlaps reading (`read_blocked` is 0.026 s), so most of
the remaining saving is only a portion of Python detection-field extraction.

### Two mistakes in the experiment design

1. Only the stripped arm initially copied dictionaries. Removing 25% of the work
   appeared to make it 70% slower: 120,000 new dictionaries had different memory
   locality from the JSON-created objects and caused cache misses. Both arms
   were changed to make identical copies.
2. Deleting `segmentation` did not simulate skipping conversion. The reader falls
   back to bbox geometry, whose RLE can cost more than the original mask. COCO
   RLE is column-major: a rectangle 400 pixels wide can require about 800 runs,
   whereas a compact blob may decode from a much shorter string. Rasterization
   doubled from 0.175 to 0.350 s. An empty 1×1 mask retained the same read path
   and geometry variant while eliminating the geometry work.

When reducing work makes a benchmark slower, check the measurement design
before rejecting the idea. Both mistakes above had that symptom.

## SIMD policy

Use SIMD only where it preserves rule 1. Profiling established these priorities:

1. **JSON parsing: rejected.** Loading was the largest wall-time component
   (`bench/profile_loadres.py`: 51.7% for segmentation results), but Python object
   construction accounted for about 82% of loader time. `bench/profile_json.py`
   separates reading, parsing and object construction:

   | Input | Read | Parse, discard output | Python object construction | Standard library total |
   |---|---|---|---|---|
   | 163 MB DT | 0.050 s | +0.087 s (11%) | +0.647 s (82%) | 1.098 s |
   | 269 MB Objects365 | 0.086 s | +0.238 s (14%) | +1.420 s (81%) | 2.946 s |

   Even an infinitely fast tokenizer would save at most 14% in these cases.
   Most remaining work creates a `PyDict` and roughly ten numeric objects per
   annotation. Keeping `coco.anns[id]` as a real, mutable dictionary is necessary
   for existing framework access. The loader was already 1.3–1.7× faster than
   the standard library.
2. **The `bb_iou` inner loop:** safe element-wise arithmetic, useful for crowded
   images and `useCats=0`.
3. **Accumulation sorting: rejected.** Inlining each score in its sort entry
   removed two pointer dereferences from the comparator
   (`matches[i].dt_scores[d]`), but RF-DETR bbox evaluation with 1.5 million
   detections changed only from 1.490 to 1.447 s, within measurement noise.
   Three maxDets slices cost only 0.017 s more than one. Neither sorting nor
   repeated work justified a radix sort. Inline scores were retained because
   they removed the `scores_sorted` buffer and a full traversal.
4. **RLE run merging:** data-dependent branches offer little SIMD benefit, so
   keep the scalar implementation. Avoid unnecessary merges instead, as below.

Every SIMD kernel must include a bit-identical comparison against the scalar
reference. Tests, rather than contributor memory, must enforce the policy.

## Pairs that do not need an exact IoU

After its bbox prefilter, `rle_iou` normally merges both run lists completely.
The matcher only needs to compare the result with thresholds, usually starting
at 0.5. In the profiled workload, 77% of surviving pairs ultimately fell below
0.5.

A mask lies inside its tight bounding box, so its intersection cannot exceed
the box intersection:

```text
i <= m = min(area_dt, area_gt, box_inter)
IoU = i / (a + b - i) is increasing in i
=> IoU <= m / (a + b - m)          (for crowd: m / area_dt)
```

If this upper bound is already below the threshold, merging cannot change any
matching decision, so the engine uses `0.0` and skips the merge. On COCO val2017
with YOLO11m-seg, this eliminated 63.5% of pairs, or 82% of the below-threshold
work. No measured pair violated the upper bound, consistent with the derivation.

Disable this shortcut when `min_thr <= 0`. At zero threshold the matcher can
rank even small IoUs to choose the best GT. Replacing a true IoU of 0.37 with
0.0 could then change the winning match.

Historical single-thread segmentation measurements:

| Measurement | Before | After |
|---|---|---|
| `iou` phase, summed over workers | 0.637 s | 0.591 s |
| YOLO11m-seg total evaluation | 1.655 s | 1.512 s |
| Mask R-CNN total evaluation | 1.047 s | 0.977 s |

The overall benefit was about 7% in the Mask R-CNN case. Run merging was not the
main remaining cost: removing it entirely left 0.388 s of IoU work. That time
covered `to_bbox` for 467,000 RLEs, plus `area`, `bb_iou` and allocations.
Division accounted for an estimated 0.104 s, obtained by comparing `area` and
`toBbox` on identical inputs.

Further savings would require computing and caching `to_bbox`/`area` during
rasterization. With a single rasterization worker, moving this work would grow
the serial section and offset gains in the parallel section. Parallelizing the
rasterizer with Rayon is therefore a prerequisite for that candidate change.

## Loading becomes the dominant cost

Historical end-to-end measurements for COCO val2017 with YOLO11m-seg and default
thread settings:

| Phase | pycocotools | ufcoco |
|---|---|---|
| Load GT and DT | 1.845 s (7%) | 1.880 s (75%) |
| Evaluate, accumulate, summarize | 24.116 s (93%) | 0.574 s (25%) |
| Total | 25.961 s | 2.454 s |

After a 42× evaluation speedup, loading represented three quarters of elapsed
time. About 82% of loader work constructed annotation dictionaries. That cost
comes from the compatibility requirement that `coco.anns[id]` remain a real
Python dictionary, rather than from the evaluation algorithm.

Passing a path to `loadRes` uses the Rust JSON loader (0.789 s versus 1.098 s
for the standard library on a 163 MB input). Passing an already parsed list,
as many evaluation harnesses do, cannot benefit from that loader.

## Extension points

Start new metrics from `Evaluator::matches()` so that they use the same matches
as AP. A confusion matrix with different matching rules can contradict the AP
shown beside it.

`MatchRecord` contains `(image, category, dt_id, gt_id, score, iou)`. It supports
TP/FP/FN lists, per-image diagnostics, mean IoU and calibration.

Add a `GeomStore` variant for a new IoU type, following `Boundaries`: provide
an extraction path and a corresponding `compute_iou` match arm.

## Measurement methods

Profile before optimizing. The profiling tools are included with the code.

```bash
python bench/make_dets.py --gt <gt.json> --out bench/data/dt.json [--segm]

# Compare implementations in separate processes; this development harness takes the best of three runs.
python bench/run_impl.py --impl {pycocotools,faster,hotcoco,ufcoco} \
    --gt <gt.json> --dt bench/data/dt.json --iou-type {bbox,segm} --json-out out.json
python bench/summarize.py bench/out/*.json

# Read the engine's internal phase timers.
python bench/profile_engine.py --gt <gt.json> --dt bench/data/dt.json --iou-type segm

# Measure memory with the Rust global allocator instrumentation.
maturin develop --release --features alloc-stats
python bench/profile_memory.py --gt <gt.json> --dt bench/data/dt.json --iou-type segm

python bench/check_params.py <gt.json> {hotcoco,faster,ufcoco}   # Compare threshold-grid bits.
```

Measurement rules:

- Include single-thread results (`bench/compare.py --threads 1`). Parallel
  implementations compete for available cores on shared hosts. A measurement
  on a desktop at 37% background load can primarily reflect idle-core count.
  Single-thread measurements are easier to compare across machines.
- Inspect CPU time alongside wall time. Contention increases wall time without
  necessarily increasing CPU time; a changed ratio can reveal interference.
- Compare arrays, not just summaries. Earlier checks incorrectly described
  faster-coco-eval as bit-identical based on its 12 statistics. Array digests
  exposed 5,744 cells differing by one ULP. Averaging 969,600 cells can round
  those differences to the same summary double. `bench/run_impl.py` includes
  array digests for this reason.
- Use a separate process for each implementation. Shared caches and allocator
  state contaminate measurements and invalidate comparisons of peak RSS.
- Repeat short measurements. The historical development harness uses the best
  of at least three runs; 0.1–0.3 s measurements showed about 15% variation.
  Buffer reuse once appeared 13% slower in a single run but proved to be noise
  across five runs. Published reports state their own sample counts and
  aggregation method; the 0.1.1 COCO report uses medians.
- Distinguish indirect estimates from direct measurements. `profile_segm.py`
  varies inputs and infers costs from total-time differences; it cannot see
  overlap inside the engine. `profile_engine.py` reads internal timers and
  provides direct phase attribution.

### Interpreting different kinds of measurements

`profile_engine.py` reports extraction and evaluation differently:

- Extraction measures wall time on two overlapping threads. Python annotation
  reading needs the GIL; rasterization does not. They run concurrently, so
  `read` and `rasterise` must not be added. `read_blocked` helps identify the
  bottleneck.
- Evaluation reports time summed over worker threads and can exceed wall time.
  The sum describes where worker time goes; its ratio to wall time, shown as
  parallel speedup, indicates how effectively the phase overlaps across workers.

RSS deltas from `profile_memory.py` include allocator slack and fragmentation.
They can exceed the Rust/Python allocation measurements and are not additive
with them. RSS is the process memory relevant to an OOM.

Without the `alloc-stats` feature, the Rust column displays `n/a`, not zero.
Zero would incorrectly imply that the engine allocates nothing.

`bench/make_dataset.py` generates fixed-seed synthetic data with difficult cases:
crowds, exact small/medium/large area boundaries, tied scores, GT-only images,
DT-only images, and polygons with duplicate vertices.

## Changes driven by profiling

| Observation | Cause | Change |
|---|---|---|
| Dictionary lookup dominated extraction | `get_item("image_id")` created a Python string on each call | Cache keys with `intern!`; removed 14.4 million creations on Objects365 |
| Extraction and rasterization used 81% of segmentation setup | Reading and rasterization alternated | Pipeline a worker through a bounded channel; rasterization overlaps reading (`read_blocked` 0.000 s in that measurement) |
| Polygon vertex reads cost 92 ns | Under `abi3`, `PyFloat_AS_DOUBLE` is a function call rather than a macro | Use per-version wheels; measured 0.098 → 0.067 s |
| One heap allocation per bbox | `Vec<f64>::extract` | Read directly into a fixed array; removed 2.4 million allocations on Objects365 |
| Evaluation made 21.76 million allocations | Match buffers reallocated per area range | Reuse within a category; reduced to 6.66 million and 2.22 → 0.83 s for that phase |
| Mask memory was twice the estimate | Spare capacity in `cnts` built with `Vec::push` | Apply `shrink_to_fit` |
| Objects365 `loadRes` used 1,148 MB | Four-corner polygon created per box prediction | Rasterize boxes on demand; historical `derive_segmentation=False` measurement saved 320 MB and 2.1 s; omission is the default in 0.1.1 |

Dropping `abi3` is a packaging decision: wheels must be built for each Python
version. The limited API turns `PyFloat_AS_DOUBLE` and `PyList_GET_ITEM` into
function calls, whose cost is visible in the hot annotation-reading loop.
Pycocotools also distributes per-version wheels for its Cython extension.

Consequently, `cargo build -p ufcoco-py` must find a Python interpreter through
an activated virtual environment or `PYO3_PYTHON`. The algorithms and core tests
have no PyO3 dependency, so `cargo test -p ufcoco-core` does not need that setup.

## Design principle

Many changes can make code faster; fewer preserve every output bit. Most unusual
choices in this implementation follow from that constraint.
