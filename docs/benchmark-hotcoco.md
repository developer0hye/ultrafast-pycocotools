# hotcoco 1.0.0 comparison

This compares the public **hotcoco 1.0.0** and
**ultrafast-pycocotools 0.1.10** wheels on identical saved COCO predictions.
The separate [pose input optimization](pose-input-optimization.md) on main is
not part of the released 0.1.10 wheel measured here. No model inference runs
during this benchmark, and the RTX 3070 does not execute the metric.

## Results

Measured on 2026-09-10. ultrafast uses less peak RSS in all 24 configurations.
It is faster in all bbox/mask configurations and all 12 server configurations.
On M2, hotcoco is faster for pose file input: at pool size 2, ultrafast takes
0.551 s versus 0.436 s (26.4% longer), while using 72.5% less peak RSS.
The M2 pose list route favors ultrafast. These measurements cover COCO bbox,
segmentation and keypoints, not other datasets or hotcoco features.

For file input at pool size 2, ultrafast's wall-time speed ratios are **2.18×
bbox / 2.42× mask on M2**, and **2.73× bbox / 2.34× mask / 1.27× pose on the
server**. These ratios describe loading plus metric computation, not full model
validation or inference.

### Apple M2

All values are medians. Each pair is hotcoco → ultrafast.

| Task | Input | Pool size | Wall (s) | CPU (s) | Peak RSS (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| bbox | files | 1 | 1.580 → 0.748 | 1.579 → 0.748 | 2804.8 → 295.5 |
| bbox | files | 2 | 1.162 → 0.532 | 1.616 → 0.770 | 2811.5 → 301.1 |
| bbox | list | 1 | 3.589 → 2.102 | 3.588 → 2.101 | 2859.1 → 961.8 |
| bbox | list | 2 | 3.175 → 1.898 | 3.623 → 2.131 | 2866.5 → 967.0 |
| segm | files | 1 | 5.122 → 1.916 | 5.103 → 2.191 | 4004.0 → 1010.5 |
| segm | files | 2 | 3.397 → 1.406 | 5.263 → 2.266 | 4004.0 → 1014.9 |
| segm | list | 1 | 7.507 → 4.556 | 7.477 → 4.758 | 4380.2 → 2876.4 |
| segm | list | 2 | 5.772 → 3.845 | 7.627 → 4.857 | 4409.0 → 2881.5 |
| keypoints | files | 1 | 0.492 → 0.578 | 0.488 → 0.576 | 1200.5 → 332.7 |
| keypoints | files | 2 | 0.436 → 0.551 | 0.489 → 0.567 | 1208.0 → 332.6 |
| keypoints | list | 1 | 1.966 → 1.762 | 1.965 → 1.762 | 1316.1 → 819.5 |
| keypoints | list | 2 | 1.921 → 1.747 | 1.973 → 1.763 | 1324.0 → 819.4 |

### i5-10400 / RTX 3070 server

All values are medians. Each pair is hotcoco → ultrafast.

| Task | Input | Pool size | Wall (s) | CPU (s) | Peak RSS (MB) |
| --- | --- | ---: | ---: | ---: | ---: |
| bbox | files | 1 | 2.820 → 1.143 | 2.820 → 1.143 | 2306.3 → 268.2 |
| bbox | files | 2 | 2.232 → 0.818 | 2.898 → 1.178 | 2310.5 → 275.2 |
| bbox | list | 1 | 6.154 → 2.819 | 6.154 → 2.819 | 2703.4 → 716.0 |
| bbox | list | 2 | 5.564 → 2.514 | 6.226 → 2.879 | 2707.3 → 721.8 |
| segm | files | 1 | 7.720 → 3.183 | 7.719 → 3.708 | 3402.5 → 835.0 |
| segm | files | 2 | 5.313 → 2.269 | 7.882 → 3.828 | 3406.9 → 840.3 |
| segm | list | 1 | 11.827 → 6.616 | 11.826 → 6.995 | 4337.4 → 1807.2 |
| segm | list | 2 | 9.352 → 5.263 | 11.939 → 7.108 | 4341.7 → 1811.2 |
| keypoints | files | 1 | 1.074 → 0.788 | 1.074 → 0.788 | 645.9 → 292.4 |
| keypoints | files | 2 | 0.963 → 0.757 | 1.073 → 0.786 | 645.9 → 292.2 |
| keypoints | list | 1 | 3.208 → 2.623 | 3.207 → 2.623 | 916.9 → 567.8 |
| keypoints | list | 2 | 3.123 → 2.602 | 3.234 → 2.631 | 922.7 → 567.8 |


## Hardware and concurrent load

| Host | CPU cores (physical / logical) | RAM | OS | CPython |
| --- | ---: | ---: | --- | --- |
| Apple M2 | 8 / 8 | 16 GiB | macOS 26.6.2 arm64 | 3.12.13 |
| RTX 3070 server | 6 / 12 (i5-10400) | 31.24 GiB usable | Linux 7.0.0-28 x86_64, glibc 2.39 | 3.12.3 |

The server has an RTX 3070 with 8 GiB VRAM; evaluation uses the CPU.
Both hosts were measured as they were, with other processes present.
Across telemetry samples from timed processes, median whole-host CPU usage was
26.9% on M2 and 8.3% on the server. Available RAM ranged from 5.23–8.60 GB
on M2 and 26.41–31.31 GB on the server (decimal GB).

The M2 host swap-out counter increased during 37 of 144 timed processes.
Across its entire experiment, host swap-out increased by 30.77 MB and swap-in
by 1.127 GB. The server showed no swap-in/out counter changes. All samples are
retained, so the M2 figures include this observed contention and paging rather
than representing an idle-machine limit. Telemetry includes child startup and
verification; short boundary samples reaching 0% or 100% do not establish
sustained background load. Per-configuration min/median/max and all individual
measurements are in the [machine-readable summary](../bench/results/hotcoco-20260910/summary.json).

## Workload and method

All three tasks use every one of the 5,000 COCO val2017 images, including images
without annotations. Predictions are the unchanged outputs of YOLO26n,
YOLO26n-seg and YOLO26n-pose from the
[published validation evidence](ultralytics-pr26101-validation.md): 733,070 bbox,
724,953 segmentation and 134,663 pose detections. Input hashes accompany the
results. These are not the different inputs used in hotcoco's own published
performance table.

- CPython 3.12 and NumPy 2.4.4 on both hosts; pycocotools 2.0.11 is the oracle.
  [Exact benchmark dependencies](../bench/hotcoco-requirements.txt) are installed
  into isolated environments. Existing benchmark environments are preserved.
- Two input routes: prediction filename passed directly to `loadRes`, and
  prediction JSON parsed into a Python list before `loadRes`. GT is loaded from
  a file in both routes. Both include JSON parsing in the measured loading time.
  The Python prediction list remains alive during evaluation.
  In hotcoco 1.0.0, `loadRes` delegates to the same native `load_res` method.
- Six fresh processes per backend/task/input/pool configuration, with backend
  order alternated and thread order reversed each round. One separate
  array-export warmup per configuration and a pycocotools oracle run are excluded
  from performance summaries. No samples are removed because of their timing,
  memory use, host load, or array disagreement.
- Rayon/OpenMP limits are 1 and 2 threads; BLAS/NumExpr/Accelerate limits are 1.
  These are pool settings, not hard process-wide thread/CPU quotas. In particular,
  ultrafast's segmentation producer can overlap a worker even at pool size 1;
  this is visible in CPU time exceeding wall time. No CPU affinity is applied.
  OS file caches are warmed by the preceding
  runs; this is not a cold-storage benchmark.
- Wall and process CPU time include GT/DT loading, evaluator construction,
  `evaluate`, `accumulate` and `summarize`. Imports, inference and the later
  public-array conversion/hash/export are excluded from the scoring clock.
  Evaluator-only time is also retained in the raw measurements.
- Peak RSS includes imports and input/evaluator storage. It is captured before
  diagnostic array conversion/hash/export. MB means 1,000,000 bytes. Host CPU,
  available RAM and swap counters are sampled throughout each child process,
  including its startup and verification work.
- Standard COCO parameters are retained: bbox/mask maxDets `[1,10,100]`, pose
  `[20]`, all standard thresholds and area ranges. The recorded parameters are
  compared against the oracle, as are complete precision/recall/scores arrays.

Interpret each host separately: the CPU, OS, Python patch release and platform
wheels differ. Host telemetry includes the benchmark and other processes; it
does not identify which process caused CPU load or paging.

## Sampled-score compatibility

All 348 child processes completed successfully: 288 timed runs and 60 separate
oracle/diagnostic runs. Every timed output matched its backend's diagnostic
array hashes. Both hosts used identical input hashes, package versions and
executed benchmark scripts. COCO parameters matched the oracle throughout.

Against pycocotools 2.0.11, ultrafast's complete precision, recall and scores
arrays were byte-identical and AP/AR differences were zero in every tested
configuration. hotcoco's AP/AR agreed within absolute tolerance 1e-12; the
largest observed difference was 1.61e-14. Its precision differences were at
most 2.22e-16, and recall was identical. The following sampled-score results
were the same on both hosts, both input routes and both pool settings:

| Task | Unequal scores entries per array | Maximum absolute score difference |
| --- | ---: | ---: |
| bbox | 180 | 0.09508 |
| segmentation | 60 | 0.03574 |
| keypoints | 0 | 0 |

The benchmark checks complete arrays in addition to AP/AR. The pinned hotcoco
wheel differs in some sampled `scores` entries; these are not merely differences
in floating-point rounding. The performance comparison must not be read as a
claim that all public output arrays are interchangeable.

The [minimal reproduction](../bench/hotcoco_scores_repro.py) uses two images:
one small GT box with a matching score-0.5 detection, and one image without GT
with a score-0.9 detection outside the small-area range. All three libraries
report AP 0.5. At IoU 0.5, recall 0, small area and maxDets 100:

| Library | Sampled score | Interpolated precision |
| --- | ---: | ---: |
| pycocotools 2.0.11 | 0.9 | 0.9999999999999998 |
| hotcoco 1.0.0 | 0.5 | 1.0 |
| ultrafast-pycocotools 0.1.10 | 0.9 | 0.9999999999999998 |

This is consistent with hotcoco's v1.0.0
[empty/ignored-cell omission](https://github.com/derekallman/hotcoco/blob/079b718721e59ab4b8f38929b65aa01acdc762d9/crates/hotcoco/src/detection/matching.rs#L486):
the ignored detection still occupies a score rank in pycocotools, even though
it contributes neither TP nor FP. The reproduction reports observed outputs;
it does not modify either evaluator to make them agree.

## Reproduce

Download the input files from the
[existing evidence archive](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz).
Use the directory containing `instances_val2017.json`,
`person_keypoints_val2017.json`, `detect-predictions.json`,
`segment-predictions.json` and `pose-predictions.json`:

```sh
uv venv --python 3.12 .venv-hotcoco
uv pip install --python .venv-hotcoco/bin/python -r bench/hotcoco-requirements.txt
.venv-hotcoco/bin/python bench/hotcoco_benchmark.py \
  --inputs INPUT_DIRECTORY --out bench/out/hotcoco \
  --threads 1 2 --rounds 6
.venv-hotcoco/bin/python bench/hotcoco_scores_repro.py \
  --out bench/out/hotcoco-score-reproduction.json
```

Each run preserves its command, log, package/runtime/native-binary hashes,
parameters, metrics, timings and telemetry. Diagnostic runs additionally save
the full arrays; timed runs check their hashes against the same backend's
diagnostic run. `results.json` contains all repetitions and their min/median/max,
including the measured compatibility differences.

## Evidence and verification

- [Condensed results and generated tables](../bench/results/hotcoco-20260910/)
- [Full raw evidence archive](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.10/hotcoco-comparison-20260910.tar.gz): every child JSON/log, diagnostic NPZ, host telemetry, package freezes, source hashes and both minimal-reproduction outputs.
- [Archive SHA-256 and size](../bench/results/hotcoco-20260910/evidence.json)
- [Two-host summarizer](../bench/summarize_hotcoco.py) rechecks all diagnostic arrays, parameters, process counts, timed-run stability and medians against the raw results.

The archive contains the exact executed scripts. The repository runner received
only a docstring/help clarification about pool limits after measurement.
Raw macOS child JSON retains legacy `NaN` CPU-load fields; use the accompanying
psutil telemetry for host load. The condensed summary is strict JSON.

To verify the archived results after extracting them:

```sh
cd hotcoco-evidence-20260910
shasum -a 256 -c SHA256SUMS
/path/to/.venv-hotcoco/bin/python bench/summarize_hotcoco.py \
  --m2 m2 --server server --out /tmp/hotcoco-verified
```
