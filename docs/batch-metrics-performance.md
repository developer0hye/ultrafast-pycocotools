# Per-batch detection metrics: performance

Measurements of `metrics.DetectionMetrics` and the precision/recall/F1 summary
([design](detection-metrics-design.md)). No change here alters a result: each
new path is compared bit for bit with the one it replaces, as listed under
[What changed](#what-changed).

## Results

Before: `7760ced` (box batches as engine columns, NumPy summary). After:
`b7c87c2`, with the engine changes of `c23f1fc`. One `update()` per batch,
wall time (CPU time), ms, one thread unless stated:

| Batch | Score threshold, IoU | Before | After | NumPy greedy | pycocotools |
| --- | --- | ---: | ---: | ---: | ---: |
| COCO slice, 16 images | none, 0.5 | 0.531 (0.543) | **0.087** (0.097) | 0.933 | 77.2 |
| COCO slice, 16 images | `best_f1`, 0.5 | 0.806 (0.795) | **0.102** (0.112) | | |
| COCO slice, 16 images | none, 0.5:0.95 | 0.670 (0.665) | **0.109** (0.118) | | |
| COCO slice, 93 images | none, 0.5 | 1.449 (1.410) | **0.337** (0.352) | | |
| Detector output, 16 images | none, 0.5 | 0.981 (0.957) | **0.216** (0.226) | 3.416 | 186.4 |
| Detector output, 16 images | `best_f1`, 0.5 | 1.428 (1.394) | **0.302** (0.312) | | |
| Detector output, 16 images | none, 0.5:0.95 | 1.269 (1.225) | **0.292** (0.302) | | |
| Detector output, 16 images | class-agnostic, 0.5:0.95 | 1.063 (1.064) | **0.297** (0.297) | | |
| Detector output, 64 images | none, 0.5 | 2.744 (2.715) | **0.734** (0.693) | | |
| Detector output, 64 images | `best_f1`, 0.5 | 4.119 (4.084) | **1.027** (1.006) | | |
| Detector output, 64 images | class-agnostic, 0.5:0.95 | 3.592 (3.592) | **1.056** (1.057) | | |

Two threads, same batches: COCO 16 images 0.512 → 0.094; detector 16 images
0.881 → 0.187; detector 64 images 2.579 → 0.564.

The comparators compute different things and give scale only:
pycocotools builds `COCO` handles, runs `loadRes`, `evaluate()` and
`accumulate()` for the batch (AP, the usual way to score a batch with it);
"NumPy greedy" is a plain per-image IoU matrix and a greedy match in score
order at IoU 0.5 without crowd or ignore rules, as a training loop might do by
hand.

`compute()` with `score_thr="best_f1"` over 750,000 accumulated detections
(seeded synthetic outcomes; scores rounded to three decimals, so many ties),
ms:

| Categories | IoU thresholds | Optimize | Before, 1 thread | After, 1 thread | After, 2 threads |
| ---: | ---: | --- | ---: | ---: | ---: |
| 80 | 10 | macro | 1,080 | **172** | 130 |
| 80 | 10 | micro | 311 | **130** | 106 |
| 1,203 | 1 | macro | 227 | **50** | 48 |
| 1,203 | 1 | micro | 92 | **49** | 47 |

## What changed

In `b7c87c2`:

- `update()` with boxes reads the per-image dicts natively
  (`_ufcoco.box_batch`): NumPy arrays of common dtypes are read in place,
  anything else goes through NumPy's conversion, and matching follows in the
  same call. Tests compare it with COCO input bit for bit, and each dtype and
  layout with float64 / int64 input.
- The summary (counts, ratios, the best-F1 sweep) runs in
  `ufcoco_core::metrics`, compared bit for bit with the NumPy implementation it
  replaced (`tests/reference_detection_metrics.py`). The sweep sorts once on
  integer keys for all IoU thresholds, reads the sorted columns sequentially,
  and runs thresholds in parallel.

In `c23f1fc` (shared with the AP path; see
[DESIGN.md](../DESIGN.md#small-and-multi-threshold-cells)):

- A branch-free IoU kernel for cells of 40 pairs or more, vectorized by the
  compiler and built for AVX2 at run time ([SIMD
  policy](../DESIGN.md#simd-policy)). In a development microbenchmark of the
  kernel alone (random boxes, cells of 12×14 to 300×10 pairs) it took about
  2.5 ns per pair against 7–8 ns for the scalar loop, whose overlap tests were
  mispredicted; below 40 pairs the scalar loop stays faster.
- Candidate ground truths per detection when there are several IoU
  thresholds; integer-key sorting of each cell's detections, skipped when they
  arrive in score order; stable counting sorts in `Grouping::build`; one-sided
  cells without IoU or matching; preallocated output ranges in `outcomes()`.

The step before (`f9f2490` → `421ff6e`, measured the same way, raw data in
[ab-v1.json](../bench/results/batch-metrics-20261018/ab-v1.json)) took the
16-image COCO batch from 1.325 to 0.536 ms by passing boxes to the engine as
columns, and the macro sweep from 9,315 to 919 ms by searching candidate cuts.

## AP path

`c23f1fc` changes the engine every `COCOeval` uses, so the AP path was
measured too, with [`bench/ab_builds.py`](../bench/ab_builds.py): COCO
val2017 with the saved YOLO26n, YOLO26n-seg and YOLO26n-pose predictions of
the [evidence
archive](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz)
(`instances_val2017.json` sha256 `e8c7f790…8c810b6f`,
`person_keypoints_val2017.json` `788e2dae…370d2b14`,
`detect-predictions.json` `95075681…bac7c4be`, `segment-predictions.json`
`5299053c…42c182de`, `pose-predictions.json` `15701515…0c83b885`), file
inputs, 6 rounds in fresh processes, `7760ced` and `b7c87c2` as wheels built
with `maturin build --release`. Median wall time, CPU time and peak RSS:

| Task | Threads | Wall s | CPU s | Peak RSS MB |
| --- | ---: | ---: | ---: | ---: |
| bbox | 2 | 0.516 → 0.476 (−7.7%) | 0.814 → 0.771 | 261.8 → 262.1 |
| bbox | 1 | 0.775 → 0.742 (−4.3%) | 0.775 → 0.742 | 256.1 → 256.3 |
| segm | 2 | 1.777 → 1.677 (−5.6%) | 3.099 → 2.984 | 555.1 → 555.4 |
| segm | 1 | 2.930 → 2.845 (−2.9%) | 2.930 → 2.843 | 548.4 → 548.5 |
| keypoints | 2 | 0.410 → 0.406 (−1.1%) | 0.655 → 0.653 | 248.4 → 249.0 |
| keypoints | 1 | 0.645 → 0.633 (−1.9%) | 0.645 → 0.633 | 248.5 → 248.9 |

The `precision`, `recall`, `scores` and `stats` digests were identical in
every run of both builds. Median host CPU during the run was 12% (maximum
35%). Raw data: [ap-path-rows.json](../bench/results/batch-metrics-20261018/ap-path-rows.json).

## Method

- [`bench/batch_metrics.py`](../bench/batch_metrics.py), 5 rounds; each (arm,
  thread count) runs in a fresh process, and arm and thread order alternate
  every round. Per process: median of 300 `update()` calls after 10 warm-up
  calls, and one call of each sweep; values above are medians over rounds.
  The comparators ran once per round in their own process.
- Each arm is its own `maturin build --release` (or `maturin develop
  --release` for the tree under test) on the same host and flags.
- Batches: the committed COCO slice, `tests/data/coco_subset_gt.json`
  (sha256 `95c3b461…879c32`) and `coco_subset_dt.json` (`b697823a…685766`),
  as per-image NumPy `xywh` arrays. "Detector output" keeps the slice's ground
  truth and replaces the detections with 100 seeded synthetic ones per image:
  half near ground truth (80% the right class), half background, not sorted
  by score.
- Host: Intel Core i5-10400 (6 cores, 12 threads), Linux 7.0.0-38, CPython
  3.12.4, NumPy 2.5.4, rustc 1.97.1. Load average 1.0–2.7 during the run
  (median 2.2, including the benchmark's own one or two threads).
- Raw data: [ab-v2.json](../bench/results/batch-metrics-20261018/ab-v2.json);
  its `arms` entries name the commits (the run recorded local paths, replaced
  afterwards; `--label` now records them directly).

## What did not move

- `update()` with masks and `update_coco()` take the `COCO` route and were not
  changed or measured here.
- Class-agnostic batches of up to 64 images run on one thread (one category,
  one unit of work), so a second thread does not help them.
- Reading the per-image dicts is 18% of a 16-image COCO update and 13% of a
  16-image detector-output one (timers around each stage of `box_batch`,
  after this change). A columnar `update` over whole-batch arrays would avoid
  it and is not implemented. Matching and the per-cell work in `outcomes()`
  remain the largest part.
- The micro sweep over a single IoU threshold was already one sort; it gained
  less than the macro sweep.
