# Apple M2 desktop benchmark — 2026-09-08

On this busy Apple M2 desktop, ultrafast 0.1.6 scored the synthetic 5,000-image
bbox workload in **0.136 s median** with a requested two-thread pool limit.
The observed median ratios were **84.9× versus pycocotools** and
**10.2× versus faster-coco-eval**. These figures apply to this workload and
recorded desktop state. Large transient slowdowns occurred during the run;
the experiment does not establish idle-machine performance or a stable speedup.

[Raw results, all repetitions and system telemetry](../bench/results/apple_m2_20260908.json)

The [i5-10400 / RTX 3070 server report](benchmark-rtx3070.md) repeats this
workload with identical source, measurement scripts and input files, and
includes a table of both hosts' observations.

## Environment and workload

- Apple M2: 8 CPU cores (4 performance, 4 efficiency), 16 GiB RAM, arm64.
- macOS 26.6.2 (25G83), AC power, low-power mode disabled. Thermal status
  was unavailable from `pmset`; core placement and clock frequency were not measured.
- Python 3.12.13; NumPy 2.4.4; pycocotools 2.0.11;
  faster-coco-eval 1.8.0; ultrafast-pycocotools 0.1.6; psutil 7.2.2.
- Local release build of commit `5d5da8c8dcc85ffd1be2f48541adbdc79adeb97f`,
  Rust 1.98.0, two build jobs, no custom RUSTFLAGS. Measurement scripts add
  phase CPU-time recording; their SHA-256 hashes are in the result file.
- Measurement window: 2026-09-08 **21:10:23–21:14:25 KST**, including warmups.
- Deterministic synthetic inputs: 5,000 images, 80 categories, 30,812 ground-truth
  annotations and 98,155 detections; seed 0, GT limit 12 and detection limit 40
  per image. Images are metadata only; no detector inference is performed.
- Standard bbox COCO parameters, all images/categories, maxDets `[1, 10, 100]`.
  This is a separate synthetic case from the public COCO detector and Objects365
  benchmarks. Its sparse category/image structure affects relative performance.

## Method and measured results

Each backend/thread setting had one excluded warmup and six measured runs,
all in fresh processes: **42 executions, 36 included measurements**.
The six backend permutations balance execution position; the order of thread
settings alternates each round. All runs are retained, including slow runs.
No other applications were stopped and no competing tests or builds were
started by this benchmark during measurement.

`RAYON_NUM_THREADS` and `OMP_NUM_THREADS` were set to 1 or 2;
`OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `MKL_NUM_THREADS` to 1.
These are requested pool settings, not enforced process-wide CPU quotas.
No CPU affinity was set. In particular, faster-coco-eval CPU time exceeded
wall time even in the one-thread setting; this is not evidence that every
backend used exactly one runnable thread.

Wall and process CPU times sum the same phases: JSON/GT loading and indexing,
result loading, evaluate, accumulate and summarize. Imports, evaluator
construction, bookkeeping and output serialization are outside this timer.
Peak RSS is the whole child-process high-water mark, including serialization;
it is not evaluator-only allocation or macOS total memory footprint.

All ranges below are **minimum–maximum**, with no outlier removal.

| Requested threads | Backend | Wall s: median (range) | CPU s: median | Peak RSS MiB: median (range) |
|---:|---|---:|---:|---:|
| 1 | pycocotools | 10.953 (10.516–15.745) | 10.815 | 859.5 (859.4–859.9) |
| 1 | faster-coco-eval | 1.411 (1.307–2.881) | 1.835 | 645.8 (645.3–646.8) |
| 1 | ultrafast | 0.214 (0.199–0.618) | 0.211 | 91.9 (91.6–96.0) |
| 2 | pycocotools | 11.560 (10.680–22.316) | 11.266 | 827.7 (810.6–861.5) |
| 2 | faster-coco-eval | 1.385 (1.324–1.580) | 1.793 | 645.9 (644.9–646.2) |
| 2 | ultrafast | 0.136 (0.128–0.150) | 0.199 | 92.3 (92.1–92.4) |

The JSON also contains means, sample standard deviations, every phase time and
individual RSS observations. Ratios above divide backend medians; they are
not confidence intervals or a correction for background load.

## Background load and interpretation

The five baseline samples before warmup ranged from **37.4% to 61.4% CPU**.
During measured child processes, the approximately one-second samples ranged
from **17.4% to 75.6%**
(median 32.2%). They include the benchmark itself,
imports and serialization, and do not isolate background CPU use.
Reported available RAM ranged from
**3.33 to 4.32 GiB**;
swap usage remained approximately **1.46 GiB**.
The host already had compressed memory. Raw psutil swap counters are retained
for inspection and are not used to attribute swap I/O to the benchmark.

Round 4 illustrates the disturbance: the two-thread-setting pycocotools run
rose to **22.316 s wall / 21.585 s CPU**, while the one-thread ultrafast run
rose to **0.618 s wall / 0.505 s CPU**. Both CPU and wall time changed, so
subtracting scheduler delay would not recover an idle-machine result.
Core placement, clock changes and cache effects were not measured and no
specific cause is assigned. Execution-order balancing reduces systematic
order bias but cannot eliminate rapid load changes between sequential runs.
Subsecond ultrafast scoring is also shorter than the telemetry interval.

The measured speed ranking is consistent across all rounds, but precise ratios
and thread-scaling estimates need a separate measurement on an idle host.
The unusually slow observations remain in all ranges and summary statistics.

## Correctness and retained evidence

- Every warmup and measured comparison passed byte equality for the complete
  `precision`, `recall`, `scores` and `stats` arrays between ultrafast and pycocotools.
- Each backend's measured array hashes were stable across repetitions. The
  archived results allow comparison across both thread settings as well.
- Faster-coco-eval satisfied absolute tolerance `1e-12`, `rtol=0`, throughout;
  maximum observed absolute difference was `2.220446049250313e-16`.
- The separate 64-image quick check also matched the repository's published
  input and reference-array hashes with `--verify-published quick`.
- 36 targeted tests for reproduction verification and compact file loading
  passed before the timed benchmark.

Input SHA-256:

```text
GT:   b23701d6733ce49a2344160bcbf30ea804eba99b80c0bc2a845e1a43f1bd13a5
Pred: aa0c5b1a53ddc0d3cef46f6ad272af4980b38f68fcfbe88a33d34b4682f93df4
```

The tracked result JSON includes all individual measurements, telemetry,
comparisons, quick-check evidence, package/build details and summary statistics.
Full arrays and process logs remain locally under
`bench/out/m2-20260908-interleaved/`; generated input JSON files are under
`bench/out/m2-20260908-inputs/`. These bulk artifacts are gitignored.

## Reproduce

Use a new output directory for each run. On this host, install into the existing
virtual environment and run:

```bash
.venv/bin/python -m pip install '.[test]' 'numpy==2.4.4' \
  'pycocotools==2.0.11' 'faster-coco-eval==1.8.0' 'psutil==7.2.2'
.venv/bin/python bench/make_dataset.py --images 5000 --cats 80 \
  --gt-per-image 12 --dt-per-image 40 --seed 0 --out bench/out/local-inputs
.venv/bin/python bench/interleaved.py \
  --gt bench/out/local-inputs/gt_5000.json \
  --pred bench/out/local-inputs/dt_5000.json \
  --out bench/out/local-benchmark --threads 1 2 --rounds 6 --input-mode files
```

System CPU/memory telemetry requires permission to read host statistics.
See the [general reproduction guide](reproducibility.md) for timing scope and
other workloads.
