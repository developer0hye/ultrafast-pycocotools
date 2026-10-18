# Per-batch detection metrics: box columns and the best-F1 sweep

Two changes to the precision/recall/F1 extension
([design](detection-metrics-design.md)), measured against the code before
them. Neither changes a result: the tests compare the new paths bit for bit
with COCO input and with evaluating every score cut.

1. **Box batches skip the `COCO` layer.** `DetectionMetrics.update()` used to
   build an annotation dictionary per box, a `COCO` handle, `loadRes`, a
   `COCOeval` and its accumulated curves. A profile of a 93-image batch put the
   engine (`run` plus `outcomes`) at about 1 ms of 8 ms of profiled time; the
   rest was that Python layer. Boxes now go to the engine as columns
   (`_ufcoco.box_outcomes`), which fills the same instances and returns the
   outcomes without the curves.
2. **The macro best-F1 sweep evaluates only candidate cuts.** It used to build
   a (detections x categories) count matrix per IoU threshold. A running sum
   now bounds every cut's macro F1 within a proven rounding margin, and only
   cuts near the maximum are evaluated with the fixed-threshold arithmetic.
   Scores are sorted once for all IoU thresholds, which also helps the micro
   sweep.

## Method

- Baseline: `f9f2490dc0a34388b28632e53bc4352c3f650fbc` (Python package).
  Optimized: the working tree committed as the box-column change on
  `feat/batch-detection-metrics`, on top of
  `3252e45ecd3e1eeff6c670ca02c0fa16d6629530` (the sweep change).
- Both arms load the same compiled module (`maturin develop --release` of the
  optimized tree; the baseline only lacks `box_outcomes`, which it does not
  call), so the arms differ only in the code under test.
- [`bench/batch_metrics.py`](../bench/batch_metrics.py), 5 rounds; each
  (arm, thread count) is a fresh process, and arm and thread order alternate
  every round. Per process: median of 300 `update()` calls after 10 warm-up
  calls, and one call of each sweep. Values below are the medians over rounds.
- Batches: the committed COCO slice, `tests/data/coco_subset_gt.json`
  (sha256 `95c3b461…879c32`) and `coco_subset_dt.json` (`b697823a…685766`),
  given as per-image NumPy `xywh` arrays. 16 images are 133 ground truth and
  136 detections; 93 images are 785 and 780.
- Sweep: seeded synthetic outcomes, 750,000 detections with scores rounded to
  three decimals (many ties), 5% ignored, 35% true positives.
- Host: Intel Core i5-10400 (6 cores, 12 threads), Linux 7.0.0-38, CPython
  3.12.4, NumPy 2.5.4, rustc 1.97.1; load average below 1.3 during the run.
  Thread count is `RAYON_NUM_THREADS`.
- Raw data: [ab.json](../bench/results/batch-metrics-20261018/ab.json).

## Results

`update()` for one batch, wall time (CPU time in parentheses), ms:

| Batch | Score threshold | Threads | Before | After |
| --- | --- | ---: | ---: | ---: |
| 16 images | none | 1 | 1.325 (1.301) | 0.536 (0.519) |
| 16 images | none | 2 | 1.241 (1.333) | 0.500 (0.525) |
| 16 images | `best_f1` | 1 | 1.548 (1.518) | 0.783 (0.765) |
| 16 images | none, 10 IoU thresholds | 1 | 1.422 (1.434) | 0.651 (0.633) |
| 93 images | none | 1 | 4.363 (4.281) | 1.398 (1.355) |
| 93 images | none | 2 | 4.167 (4.391) | 1.313 (1.385) |
| 93 images | `best_f1` | 1 | 6.052 (6.010) | 1.761 (1.716) |
| 93 images | none, 10 IoU thresholds | 1 | 4.767 (4.682) | 1.666 (1.620) |

`best_f1` over 750,000 accumulated detections (`compute()` at COCO scale), one
thread, ms:

| Categories | IoU thresholds | Optimize | Before | After |
| --- | ---: | --- | ---: | ---: |
| 80 | 10 | macro | 9,315 | 919 |
| 80 | 10 | micro | 718 | 244 |
| 1,203 | 1 | macro | 8,788 | 189 |
| 1,203 | 1 | micro | 76 | 76 |

## What did not move

- Two threads change little: the remaining per-batch time is mostly Python
  (per-image array checks and the summary), and the engine call on a 16-image
  batch is about 0.14 ms.
- The micro sweep with one IoU threshold is unchanged; it was already a single
  sort and cumulative sum.
- `update()` with masks and `update_coco()` still take the `COCO` route and
  were not measured here.
