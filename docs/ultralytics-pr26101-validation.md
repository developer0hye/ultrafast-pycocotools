# Ultralytics PR #26101: independent real-data validation

All 94 fresh-process measurements and diagnostics completed. The strict
comparison passes for all returned validator metrics, fitness and complete
precision/recall/score arrays. Detection, segmentation and pose full-validation
metrics are exactly equal; LVIS metric differences are at most 5.56e-17.
Whole-validation speed improves in all three COCO tasks; pose peak RSS increases
by 24.7 MiB (0.9%), despite lower memory in evaluator-only replay.

This report addresses the [item-by-item review checklist](ultralytics-review-checklist.md)
for replacing faster-coco-eval in the shared Ultralytics validator. It separates
real-dataset accuracy, evaluation performance, whole-validation performance and
platform installation evidence.

## Exact software and workload

- Reference Ultralytics: `4e6701fe01eb49d719923029d8a30226d8a2c987`.
- Measured replacement Ultralytics: `0d4068533176b3b692837801afc7b5a3fd9fd7ec`.
- Updated PR head: `0451a3c9282461e74cbaeeb5d03991be9162cdaf`; its only change from
  the measured commit installs the two evaluators in the regular CI matrix.
  Runtime and test source files are identical.
- Evaluators: faster-coco-eval 1.8.0 and ultrafast-pycocotools 0.1.7.
- Ultrafast release source: `b52e8527cb772393880289ee4eb0fb30f29a7d15`.
- Ubuntu 24.04, Intel i5-10400 (6 cores / 12 threads), 31.24 GiB visible RAM,
  RTX 3070 8 GiB; Python 3.12.3, torch 2.13.0+cu126,
  torchvision 0.28.0+cu126, NumPy 2.4.4.
- YOLO26n, YOLO26n-seg and YOLO26n-pose weights from the official v8.4.0 assets.
- All 5,000 official COCO val2017 images for each task. Pose also includes images
  with no keypoint annotations; it does not use the shorter positive-image-only
  pose manifest. Evaluation uses the original instances/keypoints JSON.
- GPU FP32, image size 640, batch 16, two data-loader workers, rectangular
  batching, confidence 0.001, IoU 0.7, max_det 300. Torch, OpenMP, BLAS and Rayon
  use two threads. COCO scoring retains the validator's maxDets=100 protocol.

Glenn also reports [independent three-task validation](https://github.com/ultralytics/ultralytics/pull/26101#issuecomment-5587037002)
on RTX PRO 6000, batch 8 and a different Torch/NumPy environment. His values are
separate evidence; model scores from those different prediction-generation
settings are not compared across hosts as an evaluator regression test. Each
comparison here uses its own identical prediction files.

The small CI smoke tests now use independent original COCO train2017 annotations
for the existing eight COCO8/COCO8-pose validation images. They compare real model
predictions with the previous evaluator and check cached reuse. All eight focused
integration/parity cases passed locally on the replacement commit. The original
synthetic cases remain useful for crowd, ties, ignored/federated categories and
other deliberately constructed regression cases.

## Accuracy and complete-array parity

| Task | Images / predictions | Main AP, before = after | Fitness | Max metric difference | Max complete-array difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Detection | 5,000 / 733,070 | 0.4081073392 | 0.4241725330 | 0 | 2.22e-16 |
| Segmentation (bbox + mask) | 5,000 / 724,953 | 0.3436595381 | 0.3631488382 | 0 | 2.22e-16 |
| Pose (bbox + keypoints) | 5,000 / 134,663 | 0.5638424803 | 0.5894217506 | 0 | 0 |
| LVIS 100 bbox | 100 / 29,980 | 0.3676645003 | 0.3676645003 | 5.55e-17 | 2.22e-16 |
| LVIS 93 annotated (bbox + mask) | 93 / 14,449 | 0.1408811229 | 0.1617998796 | 2.78e-17 | 2.22e-16 |

The raw JSON includes every returned metric and fitness, all array shapes and
per-array maximum differences. All three calls preserve cached metrics exactly;
complete arrays are element-for-element equal across cold/cached reuse within
each backend. The cross-backend tolerance remains 1e-12, with zero relative tolerance.

## Whole-validation performance

Four alternating fresh processes per backend; medians. The nested evaluator
interval is part of the whole-validation interval, not an additional cost.

| Task | Whole validation (s) | Evaluator inside validation (s) | Peak process RSS (MiB) | Fresh process including startup/exit (s) |
| --- | ---: | ---: | ---: | ---: |
| Detection | 37.782 → 31.609 | 7.221 → 0.837 | 3707.7 → 2248.2 | 41.253 → 34.969 |
| Segmentation (bbox + mask) | 153.892 → 139.950 | 20.564 → 6.762 | 5141.7 → 4436.6 | 158.411 → 144.404 |
| Pose (bbox + keypoints) | 35.915 → 32.330 | 5.180 → 1.684 | 2726.0 → 2750.7 | 39.297 → 35.730 |

## Cold and cached evaluator replay

Six alternating fresh processes per backend; one cold call and two cached calls
in each process. Each cell is before → after. Diagnostic profiling is excluded.

| Task | Cold (s) | Cached (s) | Peak process RSS (MiB) | Fresh process including all 3 calls (s) |
| --- | ---: | ---: | ---: | ---: |
| Detection | 7.681 → 0.849 | 7.009 → 0.779 | 2887.0 → 1370.3 | 26.054 → 6.737 |
| Segmentation (bbox + mask) | 21.844 → 5.836 | 20.915 → 5.049 | 4293.2 → 3789.2 | 69.878 → 22.077 |
| Pose (bbox + keypoints) | 5.251 → 1.817 | 5.049 → 1.538 | 1794.4 → 1719.8 | 20.541 → 10.144 |
| LVIS 100 bbox | 0.687 → 0.289 | 0.597 → 0.256 | 1381.6 → 1035.2 | 4.498 → 3.421 |
| LVIS 93 annotated (bbox + mask) | 1.084 → 0.466 | 1.037 → 0.437 | 1390.6 → 1042.5 | 5.745 → 3.943 |

## Observed variation and background load

These ranges retain every timing sample. Host CPU utilization includes the
benchmark itself and other processes; it does not isolate background CPU cost.

| Task | Cold min–max, reference / replacement (s) | Sampled host CPU min–max (%) | Available host RAM min–max (GiB) |
| --- | ---: | ---: | ---: |
| Detection | 7.666–7.715 / 0.843–0.858 | 0.0–79.2 | 26.08–29.51 |
| Segmentation (bbox + mask) | 21.809–21.951 / 5.786–6.031 | 0.0–78.7 | 24.73–29.41 |
| Pose (bbox + keypoints) | 5.174–5.314 / 1.794–1.890 | 0.0–28.6 | 26.91–29.34 |
| LVIS 100 bbox | 0.677–0.706 / 0.280–0.293 | 0.0–28.6 | 28.27–29.30 |
| LVIS 93 annotated (bbox + mask) | 1.080–1.108 / 0.453–0.479 | 0.0–28.6 | 27.52–29.10 |

The raw data also includes whole-validation and cached-call ranges, CPU time,
sampled process-tree memory, hardware/software versions and each exact command.
GPU utilization was not sampled, so whole-validation timings are observations
under this workload and host state, not an isolated-GPU throughput claim.

## Measurement boundaries

Every measured run is a fresh Python process. Four full-validation rounds and
six evaluator-replay rounds use alternating reference/replacement order, reversed
on odd rounds. Every sample is retained. Models and data were downloaded first;
excluded initial validations warmed the filesystem and label caches.

**Whole validation** times the actual task validator call, including model and
data-loader setup, GPU inference, postprocessing, native Ultralytics metrics,
prediction serialization and COCO evaluation. A wrapper times the existing
`eval_json` call separately. Imports before the validator call are excluded from
that interval but included in the separately reported fresh-process wall time.
Each full-validation run must produce byte-identical prediction JSON.

**Evaluator replay** loads those fixed predictions into the real validator and
calls its shared `coco_evaluate` method once with cold ground truth, then twice
with the identical cached COCO object. These intervals include annotation/result
loading, native preparation, evaluation, accumulation and summary. They exclude
inference and prediction-file generation. Full returned metric dictionaries and
fitness are retained for every call.

**Array diagnostics** run separately with a Python profile observer that captures
the actual evaluator's complete precision, recall and scores arrays after
`summarize`. Profiling and array-copy costs are excluded from performance tables.
The unchanged comparison gate is absolute tolerance 1e-12, zero relative tolerance;
cached reuse additionally requires element-for-element equality within each
backend. No runtime backend selector or fallback is added to Ultralytics.

Process peak RSS comes from `getrusage`, including imports and retained input.
Half-second samples record whole-host CPU utilization, available RAM and worker
process-tree RSS. Other processes are not stopped and background load is not
subtracted. Median and min/max describe observed conditions, not an isolated
machine. The first full detection pair also overlapped a brief preliminary
array comparison; it remains in the raw data and the summary.

## LVIS scope and the old backend's mask failure

The pinned official LVIS example has 100 images, 1,230 categories and annotations
in 93 images. Its published prediction file contains boxes only. The primary
compatibility comparison therefore evaluates that unchanged 100-image box file,
including rare/common/frequent AP and the validator's bbox fitness policy.

For additional mask coverage, real YOLO26n-seg predictions on those same images
are mapped with the LVIS project's pinned COCO-to-synset mapping. It maps 78 COCO
categories; two unmapped categories are excluded. This produces 15,117 actual
predicted masks. Ground-truth masks remain the independent original LVIS polygons.
These COCO-trained predictions are a compatibility workload, not a full-taxonomy
LVIS model accuracy claim.

On the original 100-image mask input, faster-coco-eval 1.8.0 raises
`unordered_map::at`; 1.7.2 also raises `_Map_base::at`. The failure occurs during
mask preparation; source inspection points to missing image-size entries for
some zero-GT images. Ultrafast
0.1.7 completes both bbox and mask evaluation and two cached repeats. The failed
reference logs are retained; this is not called successful 100-image mask parity.
An explicitly labelled 93-image annotated subset additionally compares complete
bbox/mask arrays and metrics through both validators. The 100-image box comparison
and the original 100-image replacement mask result are preserved separately.

## Saved evidence

- [Complete result JSON, metric dictionaries, arrays summary and raw samples](../bench/results/ultralytics_pr26101_20260909.json).
- [Public 0.1.7 wheel hashes and native installation/test verification](../bench/results/release_v017_verification.json).
- [Raw predictions, annotations, full arrays, logs, frozen measured scripts and test reports](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz).
- [Archive SHA-256](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz.sha256). The archive also contains a per-file `SHA256SUMS` manifest. All 245 archived files were verified; [GitHub asset digests](../bench/results/ultralytics_pr26101_artifacts.json) match the local archive.

Backend-specific `stats_as_dict` aliases differ between the two libraries. The
raw evidence records shared and exclusive names; all actual validator-returned
metric keys are required to match, as are the complete arrays from which those
summaries are computed. No summary-alias equality is claimed.

## Reproduce

Prepare two checkouts at the exact commits above, named `reference/` and
`replacement/` under a chosen benchmark directory. Use a Python 3.12 environment
with the recorded Torch/CUDA and evaluator versions. Copy
`bench/ultralytics_metric.py`, `bench/ultralytics_review.py`, and
`bench/fetch_lvis_fixture.py` into that directory; copy
`bench/results/lvis_fixture_sources.json` to its `results/` directory.

`bench/prepare_ultralytics_review.py --root /path/to/benchmark` verifies SHA-256
for the official images, annotations, YOLO labels and weights before extracting
only validation data. The annotation source and prediction hashes, all 5,000
image hashes, source revisions and native-library hashes accompany the results.

The archive's `requirements-measured.txt` records the complete environment.
Run the following from the benchmark directory after input preparation. The
three excluded full validations warm the filesystem and label caches:

```bash
export PYTHONPATH="$PWD/reference"
export YOLO_AUTOINSTALL=false RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
python ultralytics_metric.py --mode full --task detect --model yolo26n.pt \
  --data data/coco.yaml --gt data/coco/annotations/instances_val2017.json \
  --output results/warmup-detect-reference.json
python ultralytics_metric.py --mode full --task segment --model yolo26n-seg.pt \
  --data data/coco.yaml --gt data/coco/annotations/instances_val2017.json \
  --output results/warmup-segment-reference.json
python ultralytics_metric.py --mode full --task pose --model yolo26n-pose.pt \
  --data data/coco-pose.yaml --gt data/coco/annotations/person_keypoints_val2017.json \
  --output results/warmup-pose-reference.json
```

Generate the additional LVIS mask input and run the measured comparisons:

```bash
python fetch_lvis_fixture.py --out data \
  --coco-seg-pred results/warmup-segment-reference/predictions.json
python ultralytics_review.py --root .
python /path/to/ultrafast-pycocotools/bench/summarize_ultralytics_review.py \
  results/formal --output review-results.json
```

The summarizer fails on missing processes, mismatched input hashes, different
metric keys, metric/array differences above tolerance, or changed arrays after
cached reuse. Detailed commands, software versions and raw outputs are retained
for independent replay.
