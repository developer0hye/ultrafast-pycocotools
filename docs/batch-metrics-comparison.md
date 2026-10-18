# Per-batch detection metrics: comparison with other libraries

How long other libraries take to score detection batches, and what each of
them computes, next to `metrics.DetectionMetrics`
([design](detection-metrics-design.md)). The libraries do not compute the same
metric, so the timings say what it costs to get each library's own numbers and
the [Values](#values) section says how those numbers differ.

## Libraries

Read from the installed sources of the versions below (the paths are inside
each package); each claim was checked against that code, not only the
documentation.

| Library | API used | Precision / recall / F1 | Score threshold | Matching | Accumulates |
| --- | --- | --- | --- | --- | --- |
| ultrafast-pycocotools 0.1.14 | `metrics.DetectionMetrics` | at any IoU grid; per category, micro, macro | none, fixed, or the best-F1 cut (exact, one shared cut) | COCO: score order, crowd regions, area ranges, `maxDets` | one outcome byte per detection and IoU threshold |
| TorchMetrics 1.9.0 | `detection.MeanAveragePrecision` | not provided; `extended_summary` returns COCO's precision/recall curve arrays | — | COCO, through pycocotools or faster-coco-eval | every box, score, label, crowd flag and area (`detection/mean_ap.py`, `add_state` lists) |
| supervision 0.30.6 | `metrics.F1Score` (`Precision`, `Recall` alike) | F1 at IoU 0.50:0.95 (fixed grid); macro, micro, weighted | none: every prediction | highest IoU first, no crowd handling (`metrics/utils/matching.py`) | every `Detections`; `compute()` evaluates all, small, medium and large |
| Ultralytics 8.4.172 | `DetMetrics` with `BaseValidator.match_predictions`, as `DetectionValidator.update_metrics` does | P, R, F1 at IoU 0.5 | the confidence maximizing the mean F1 after interpolating onto 1,000 points and smoothing (`utils/metrics.py:892`) | highest IoU first, unique pairs, no crowd handling | per-detection true-positive matrices, scores and classes |
| faster-coco-eval 1.8.0 | `COCOeval_faster(extra_calc=True).extended_metrics` | macro P and R at IoU 0.5 | the best macro F1 over every distinct score; ties go to the lowest | COCO; precision divides by every detection of the class (`core/faster_eval_api.py`) | — (COCO objects) |
| hotcoco 1.1.0 | `COCOeval.f_scores()` | F1 at 0.50:0.95, 0.5, 0.75 | each category's own best threshold, then averaged | COCO | — (COCO objects) |
| pycocotools 2.0.11 | `COCOeval` | not provided (AP/AR) | — | COCO | — |

Macro averages also differ: this package averages the categories with ground
truth, as COCO does for AR; supervision averages the union of ground-truth and
predicted classes (`metrics/f1_score.py:376`); Ultralytics the classes in the
targets (`utils/metrics.py:847`).

## Results

COCO val2017 with the saved YOLO26n predictions (733,070 detections) of the
[evidence
archive](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz).
One thread (`RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` = 1),
median of 3 fresh processes per row.

**Batch**: score one batch of 16 images (2,242 detections) from scratch:
new metric object, add the batch, compute.

| Library | Wall ms | CPU ms | vs ufcoco |
| --- | ---: | ---: | ---: |
| **ufcoco**, all detections | **0.257** | 0.258 | 1× |
| **ufcoco**, best-F1 cut | **0.353** | 0.353 | |
| ufcoco, CPU tensors | 0.380 | 0.380 | |
| hotcoco | 13.4 | 13.2 | 52× |
| Ultralytics | 19.9 | 19.9 | 77× |
| TorchMetrics, faster-coco-eval backend | 29.6 | 38.1 | 115× |
| supervision | 34.9 | 34.9 | 136× |
| faster-coco-eval | 169.5 | 178.2 | 660× |
| pycocotools | 196.7 | 196.7 | 765× |
| TorchMetrics, pycocotools backend | 200.4 | 200.3 | 780× |

**Stream**: add all 5,000 images in batches of 16, then compute once. The
added peak memory is the process's peak RSS during that work minus its RSS
before it.

| Library | Wall s | CPU s | Added peak MB | vs ufcoco |
| --- | ---: | ---: | ---: | ---: |
| **ufcoco**, all detections | **0.085** | 0.085 | **34** | 1× |
| **ufcoco**, best-F1 cut | **0.127** | 0.127 | 57 | |
| ufcoco, CPU tensors | 0.124 | 0.124 | 36 | |
| hotcoco | 2.31 | 2.31 | 578 | 27× |
| Ultralytics | 2.93 | 2.93 | 145 | 34× |
| TorchMetrics, faster-coco-eval backend | 9.11 | 13.37 | 1,582 | 107× |
| supervision | 10.32 | 10.32 | 74 | 121× |
| faster-coco-eval | 28.30 | 32.57 | 1,623 | 332× |
| pycocotools | 44.72 | 44.72 | 1,728 | 525× |
| TorchMetrics, pycocotools backend | 49.70 | 49.70 | 1,697 | 583× |

## Values

What each reported on the full stream (IoU 0.5 where a library picks one):

| Library | Reported |
| --- | --- |
| ufcoco, all detections | micro P 0.0449, R 0.8525, F1 0.0853; macro F1 0.0851 |
| ufcoco, best-F1 cut (macro) | at score 0.1915: macro P 0.6519, R 0.5221, F1 0.5709; micro F1 0.6129 |
| supervision | F1 at IoU 0.5: micro 0.0806, macro 0.0820 |
| Ultralytics | at its best-F1 confidence: macro P 0.6493, R 0.4728, F1 0.5424 |
| faster-coco-eval | at its best macro-F1 cut: macro P 0.6130, R 0.5993 |
| hotcoco | F1 at IoU 0.5 with per-category thresholds: 0.5899 |
| pycocotools, faster-coco-eval, hotcoco | AP 0.4081, AP50 0.5688 |
| TorchMetrics (both backends) | AP 0.4080, AP50 0.5686 |

The differences follow from the definitions above. Crowd regions are ignored
in COCO matching and counted as ordinary ground truth by supervision and
Ultralytics, which match by highest IoU rather than score order. Ultralytics
picks its confidence on a smoothed, interpolated curve; faster-coco-eval
counts every detection of a class and breaks ties toward the lowest threshold;
hotcoco lets each category choose its threshold, which can only raise F1.
TorchMetrics' AP differs from the others in the fourth decimal; its boxes were
given as float32 tensors, the likely cause, which was not investigated
further. This package's AP through `COCOeval` is
bit-identical to pycocotools; its precision, recall and F1 come from the same
matching.

## Method

- [`bench/batch_metrics_libraries.py`](../bench/batch_metrics_libraries.py),
  every (implementation, task, configuration) in its own process, 3 rounds
  with alternating order, each process started only once the 1-minute load
  average was below 1.2 (`--max-load`; recorded per run). Batch rows are the
  median of up to 300 repetitions within 3 s after a warm-up call.
- Inputs are built before timing in each library's own format: NumPy arrays
  (or CPU tensors) for this package, tensors for TorchMetrics and Ultralytics,
  `supervision.Detections` for supervision. The COCO-API libraries take
  annotation dictionaries, so converting a batch's arrays into them is part of
  their timed work.
- This package ran with `batch_results=False` and one `compute()` call, the
  same pattern as the others.
- Versions: torch 2.14.1 (CPU), torchmetrics 1.9.0, supervision 0.30.6,
  ultralytics 8.4.172, faster-coco-eval 1.8.0, hotcoco 1.1.0, pycocotools
  2.0.11, NumPy 2.5.2, CPython 3.12.3
  ([requirements](../bench/batch-metrics-libraries-requirements.txt)); each
  the newest release at least two weeks old on 2026-10-18.
- Inputs: `instances_val2017.json` sha256 `e8c7f790…8c810b6f`,
  `detect-predictions.json` `95075681…bac7c4be`.
- Host: Intel Core i5-10400, Linux 7.0.0-38. ufcoco built at `638d377`
  (released as 0.1.14) with `maturin develop --release`.
- Raw data: [libraries.json](../bench/results/batch-metrics-20261018/libraries.json).

```sh
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r bench/batch-metrics-libraries-requirements.txt
python bench/batch_metrics_libraries.py --inputs INPUT_DIRECTORY --work WORK_DIRECTORY \
    --rounds 3 --max-load 1.2 --out libraries.json
```
