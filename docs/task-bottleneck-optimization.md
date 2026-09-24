# Per-task bottleneck optimization (unreleased)

Each task was profiled on the headline workload, then its three largest
operations were optimized for time and memory together. The complete
`precision`, `recall` and `scores` arrays remain byte-identical to pycocotools
2.0.11 in every configuration below.

- Baseline: `main` at `f03c4ca064f34fe06162e8b6ff0d0517adc9162f`
  (the 0.1.11 code; the baseline wheel was rebuilt locally).
- Optimized: branch `perf/task-bottlenecks` at
  `79253f525941323594513ee65c9c45b66eeeeddb` (headline table). The per-phase
  tables were measured at `152f936936ba9c1d1b2cd4fe26a28fbd51d72ec0`, before
  the last change (item 7 below), which affects only the segm IoU phase.
- Both wheels were built on the same host with `maturin build --release` and
  identical flags. Per-phase memory attribution uses separate
  `--features alloc-stats` builds of the same two commits.

## Results

COCO val2017 (5,000 images) from files, median of 6 fresh processes per build.
[`bench/ab_builds.py`](../bench/ab_builds.py) alternates the two builds and the
thread order every round, so host drift affects both alike; the output digests
of every run were identical. Wall and CPU time cover loading, evaluation,
accumulation and summary.

| Task | Threads | Wall s (main → branch) | CPU s | Peak RSS MB |
| --- | ---: | ---: | ---: | ---: |
| bbox | 2 | 0.850 → 0.495 (-41.7%) | 1.233 → 0.780 | 273.4 → 262.4 (-4.0%) |
| bbox | 1 | 1.211 → 0.751 (-38.0%) | 1.211 → 0.751 | 266.3 → 256.7 (-3.6%) |
| segm | 2 | 2.316 → 1.616 (-30.2%) | 3.923 → 2.843 | 838.2 → 555.8 (-33.7%) |
| segm | 1 | 3.344 → 2.770 (-17.2%) | 3.897 → 2.769 | 833.1 → 549.0 (-34.1%) |
| keypoints | 2 | 0.541 → 0.421 (-22.1%) | 0.708 → 0.685 | 257.1 → 249.1 (-3.1%) |
| keypoints | 1 | 0.707 → 0.674 (-4.6%) | 0.706 → 0.674 | 257.1 → 249.0 (-3.2%) |

The run is in [ab-summary.json](../bench/results/task-bottlenecks-20260924/ab-summary.json)
(2026-09-24, median host CPU 8.3%). The earlier measurement of `152f936` with
`hotcoco_benchmark.py`, one build after the other, gave the same picture
(segm -26.8% / -11.7%); it is in [summary.json](../bench/results/task-bottlenecks-20260924/summary.json).

With one thread, file parsing stays sequential and segmentation masks are
decoded on the evaluation thread instead of a separate rasteriser thread, so
single-thread gains are smaller.

## Top three operations per task

The phases come from [`bench/profile_tasks.py`](../bench/profile_tasks.py)
(median of 5 fresh processes, 2 threads). "Memory" is the growth of the process
high-water mark (VmHWM) during that phase. Engine sub-phases are CPU time
summed over workers.

### bbox

| Operation | Wall s | Memory MB |
| --- | ---: | ---: |
| Load detections (`loadRes`) | 0.276 → 0.177 | 136.4 → 133.1 |
| Engine: matching + accumulation (CPU 0.250 + 0.460 → 0.151 + 0.130) | 0.394 → 0.173 | 37.7 → 34.2 |
| Instance extraction | 0.099 → 0.070 | 34.5 → 31.4 |

### segm

| Operation | Wall s | Memory MB |
| --- | ---: | ---: |
| Mask extraction | 1.060 → 0.586 | 378.2 → 108.1 |
| Engine: IoU, matching, accumulation | 0.701 → 0.673 | 33.9 → 17.4 |
| Load detections (`loadRes`) | 0.499 → 0.394 | 363.1 → 366.5 |

The IoU sub-phase got heavier (0.602 → 1.042 CPU s at `152f936`) because mask
decoding moved into it; it replaced the extraction-time decode (`dt_read`
0.528 → 0.172 s), and matching and accumulation fell. Item 7 then reduced the
IoU sub-phase to 0.87 CPU s (three `profile_engine.py` runs): of the 1.04 s,
0.46 s was decoding, 0.35 s mask boxes and areas, and the rest run merging. With compact segmentation
inputs this relocation is not free for the diagnostic APIs: `matches()`,
`confusion_matrix()` and `per_instance` re-run `compute_iou`, so each call now
decodes the detection masks again (about 0.4 CPU s per call here) instead of
reusing masks decoded at extraction. `profile_engine.py`'s `rasterise` timer is
0 for this route, and `gt_read`/`dt_read` include the parallel span work.

Detection loading grew 3.4 MB from the parallel parser's chunk columns. It
also makes one small allocation per segmentation record (725,047 versus 83
sequentially): each record gets a fresh `serde_json` deserializer, whose scratch
stack allocates when skipping the nested `segmentation` object.

### keypoints

| Operation | Wall s | Memory MB |
| --- | ---: | ---: |
| Load detections (`loadRes`) | 0.238 → 0.158 | 164.6 → 159.6 |
| Instance extraction | 0.183 → 0.183 | 33.5 → 33.7 |
| Engine | 0.062 → 0.037 | 6.9 → 3.7 |

Keypoint extraction is unchanged. About 75% of it is converting retained
coordinates with Rust's correctly rounded `f64` parser, measured by decoding
40,000 real spans in isolation; a hand-written Clinger fast path was slower
and was not kept.

What remains of bbox and keypoint peak memory is mostly the owned JSON
snapshot (99 MB and 154 MB) and its columns, which are retained by design so
that annotation views stay exact after the file changes.

## Changes

1. **Slot lookups.** Compact extraction resolves image/category slots once
   per row with a multiply-rotate hasher (the maps are probed, never
   iterated). Bbox extraction looks them up directly without a slot column.
2. **Lazy segmentation masks.** Segmentation spans are recorded at load.
   Extraction parses only selected spans in parallel and keeps compressed
   `counts` strings, including the `\\`-escaped form, as references into the
   snapshot. The engine decodes each one inside `compute_iou`, where each mask
   is used once, so decoded run arrays no longer coexist. Polygon spans use a
   direct decoder with the serde visitor's values, and the polygon rasteriser
   streams its trace (the original is kept as a bit-exact test reference).
3. **Accumulation.** After matching, each image keeps one outcome byte per
   (detection, threshold), and its match table stays in scratch unless
   `evalImgs` or `per_instance` need it. Outcomes are gathered once per area
   range. Precision curves are built from true-positive positions only: between
   true positives precision never rises, so every suffix maximum is attained at
   a true positive. Recall samples compare integers with counts found using the
   same division.
4. **Matching.** A detection whose largest IoU is below a threshold skips that
   threshold's scan, which could not choose a ground truth.
5. **Parallel result parsing.** Result arrays of 4 MiB or more are split at
   candidate record starts and parsed on the Rayon pool. A chunk is accepted
   only if its last record ends exactly at the next chunk's start, so every
   accepted start is a real boundary; anything else re-runs the sequential
   parser, which decides the outcome and its error.
6. **Narrower columns.** Row IDs are stored in 32 bits (with a 64-bit side
   column when needed), pose spans as 32-bit pairs, and detection instances
   omit crowd/ignore columns that only ground truth uses.

7. **Mask boxes and areas.** The IoU kernel computes each mask's box and area
   in one pass, splitting run indices with an exact reciprocal division
   (`floor(t * floor(2^32 / h) / 2^32)` is the quotient or one less; one
   remainder check corrects it). The counts decoder is specialized for plain
   and escaped text and reserves one slot per character, so it never
   reallocates.

Tried and reverted: parsing bbox/score numbers from raw tokens (slower than
serde's parser), an incremental `rleToBbox` (no measurable change), and a
custom decimal fast path for keypoints (slower than the standard library).

Measured and not adopted:
- `fast-float2` 0.2.4 and `lexical-parse-float` 1.0.6 for keypoint coordinates:
  bit-identical to the standard library on all 2.04M tokens of 40,000 real
  spans, but only 27 → 23 ns per token, about 0.02 s of keypoint evaluation.
  Under the dependency checklist in [AGENTS.md](../AGENTS.md) that does not
  justify a new crate.
- A strict hand-written result-record parser: 270 versus 280–340 ns per record
  for serde on real detections. Number conversion dominates both, and the gain
  does not justify a second JSON validator.

## Agreement and tests

- The three tasks' full arrays match the saved pycocotools 2.0.11 oracle
  byte for byte at 1 and 2 threads.
- New tests cover escaped RLE counts, parallel parsing against sequential
  semantics (including record-boundary look-alikes inside strings and six kinds
  of malformed input), IDs beyond 32 bits, and precision envelopes with tied
  scores, unusual recall grids and `maxDets` including 0.
- Rust tests compare the streaming rasteriser with the original on 4,000
  polygons, and the polygon and number decoders with serde.
- Unsorted or NaN recall thresholds already differed from pycocotools on
  `main`; the branch reproduces `main`'s outputs for them exactly (checked on 42
  configurations), and this change does not alter that behavior.
- Locally 421 tests pass, including the LVIS, full COCO val2017 and
  faster-coco-eval protocol tests; the RF-DETR integration test is skipped
  here and runs in CI.

## Workload and host

- Inputs: the [PR #26101 evidence archive](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz)
  (YOLO26n, YOLO26n-seg and YOLO26n-pose predictions; 733,070 boxes, 724,953
  masks, 134,663 poses). SHA-256 hashes are in the
  [condensed results](../bench/results/task-bottlenecks-20260924/summary.json).
- Intel Core i5-10400 (6 cores / 12 threads), 31.2 GiB RAM, Linux 7.0.0-31,
  glibc 2.39, CPython 3.12, NumPy 2.4.4.
- Measured 2026-09-24 02:38–02:48 UTC with the desktop session running. Median
  host CPU during timed runs was 8–9%. Swap traffic during a whole run set was
  at most 6.8 MB.

## Reproduce

```sh
python bench/hotcoco_benchmark.py --inputs INPUTS --out OUT \
  --backends ufcoco --modes files --threads 2 --rounds 6   # and --threads 1
python bench/profile_tasks.py --inputs INPUTS --task bbox --runs 5 --json-out bbox.json
python bench/summarize_task_bottlenecks.py --runs RUNS --profiles PROFILES --out summary.json
```

`hotcoco_benchmark.py` expects one output directory per build and thread count
(`base-t2`, `new-t2`, ...) for the summary script; `profile_tasks.py` needs an
`alloc-stats` build for the Rust columns.
