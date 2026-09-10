# Closing the hotcoco performance gap

This report compares hotcoco 1.0.0 with an **unreleased ultrafast source build**
based on `d4b8c12c75052c7c7bd3acc71daca4310354b0f9` and the pose numeric-decoding
change described below. It follows the [parallel pose loader](pose-parallel-loading.md),
whose remaining measured exception was M2 pose file input at pool size 1.
The previous main build and this candidate both report package version 0.1.10; the candidate's source
and native binary hashes distinguish it from the published wheel. The
[released-wheel comparison](benchmark-hotcoco.md) remains historical evidence.

## What changed

Pose snapshots already contain JSON spans validated by serde_json. The new
internal decoder reads flat coordinate tokens directly into final x/y storage,
using Rust's standard `f64` parser for retained values. The standard converter
targets nearest IEEE 754 values with ties rounded to even
([Rust implementation notes](https://dev-doc.rust-lang.org/stable/src/core/num/imp/dec2flt/mod.rs.html)).
There is no approximate coordinate representation and OKS arithmetic is unchanged.

Detection visibility and coordinates outside the stable evaluation cap have no
numerical consumer, but still require validation. For an already validated JSON
number with no exponent and at most 300 bytes, its magnitude is below 10^300,
which is below `f64::MAX`; numeric type and this bound establish finiteness
without converting the value. Longer tokens and exponent forms still undergo
conversion. Unrecognized structures or types fall back to the original checked
serde visitor. This decoder is private and is **not a standalone JSON validator**.
Ground-truth visibility and Python object input conversion are unchanged.

The first experiment converted every coordinate with the standard parser and
was approximately tied with hotcoco. Its measurements and decoder are retained
as exploratory evidence. Final measurements use the bounded validation path
for unused values.

## Complete comparison

On 2026-09-10, **all 24 configurations had lower median wall time and peak RSS**
than hotcoco. The candidate was faster in **all 144 matched repetition pairs**.
Across configurations the speed ratio was 1.03–2.69× and peak RSS was
36.3–89.5% lower. The narrowest margin was M2 pose file input at pool size 1:
0.506 → 0.490 seconds, with all six pairs favoring the candidate.

| Host | Task | Input | Pool size | Wall (s), hotcoco → candidate | Peak RSS (MB), hotcoco → candidate | Speed ratio | Faster pairs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| m2 | bbox | files | 1 | 1.630 → 0.788 | 2804.4 → 295.5 | 2.07× | 6/6 |
| m2 | bbox | files | 2 | 1.191 → 0.541 | 2811.4 → 301.3 | 2.20× | 6/6 |
| m2 | bbox | list | 1 | 3.751 → 2.203 | 2859.0 → 961.8 | 1.70× | 6/6 |
| m2 | bbox | list | 2 | 3.299 → 1.969 | 2866.2 → 967.6 | 1.68× | 6/6 |
| m2 | segm | files | 1 | 5.338 → 2.060 | 4570.8 → 1014.7 | 2.59× | 6/6 |
| m2 | segm | files | 2 | 3.517 → 1.497 | 4577.8 → 1014.4 | 2.35× | 6/6 |
| m2 | segm | list | 1 | 7.790 → 4.730 | 4585.5 → 2877.0 | 1.65× | 6/6 |
| m2 | segm | list | 2 | 5.952 → 3.973 | 4524.8 → 2882.4 | 1.50× | 6/6 |
| m2 | keypoints | files | 1 | 0.506 → 0.490 | 1200.4 → 268.3 | 1.03× | 6/6 |
| m2 | keypoints | files | 2 | 0.457 → 0.378 | 1207.8 → 268.3 | 1.21× | 6/6 |
| m2 | keypoints | list | 1 | 2.062 → 1.820 | 1316.3 → 810.9 | 1.13× | 6/6 |
| m2 | keypoints | list | 2 | 2.017 → 1.802 | 1324.4 → 811.1 | 1.12× | 6/6 |
| server | bbox | files | 1 | 2.815 → 1.141 | 2306.1 → 268.3 | 2.47× | 6/6 |
| server | bbox | files | 2 | 2.221 → 0.825 | 2309.8 → 275.3 | 2.69× | 6/6 |
| server | bbox | list | 1 | 6.160 → 2.828 | 2703.2 → 716.1 | 2.18× | 6/6 |
| server | bbox | list | 2 | 5.556 → 2.503 | 2707.4 → 721.8 | 2.22× | 6/6 |
| server | segm | files | 1 | 7.671 → 3.217 | 3402.0 → 834.8 | 2.38× | 6/6 |
| server | segm | files | 2 | 5.308 → 2.293 | 3406.5 → 840.3 | 2.31× | 6/6 |
| server | segm | list | 1 | 11.829 → 6.593 | 4337.0 → 1807.7 | 1.79× | 6/6 |
| server | segm | list | 2 | 9.408 → 5.232 | 4341.6 → 1812.5 | 1.80× | 6/6 |
| server | keypoints | files | 1 | 1.061 → 0.665 | 645.6 → 258.8 | 1.60× | 6/6 |
| server | keypoints | files | 2 | 0.966 → 0.522 | 645.6 → 258.8 | 1.85× | 6/6 |
| server | keypoints | list | 1 | 3.209 → 2.595 | 916.6 → 568.0 | 1.24× | 6/6 |
| server | keypoints | list | 2 | 3.115 → 2.567 | 922.4 → 568.0 | 1.21× | 6/6 |


Times and RSS are medians of six runs per backend/configuration; MB is decimal.
The speed ratio is hotcoco wall time divided by candidate wall time. Faster-pair
counts compare corresponding alternating repetitions; all individual samples,
min/median/max, CPU time and output hashes are retained in the
[machine-readable summary](../bench/results/hotcoco-goal-20260910/summary.json).
The scope is this COCO workload and the configurations in the table, not a
universal guarantee for arbitrary inputs, machines or other hotcoco features.

## Protocol and host load

The [comparison harness](../bench/hotcoco_benchmark.py) uses all 5,000 COCO
val2017 images and the same saved YOLO26n predictions as the earlier reports:
733,070 bbox, 724,953 mask and 134,663 pose detections. Both hosts use identical
input hashes. Inference is excluded; the RTX 3070 does not execute the metric.

- Apple M2: 8 cores, 16 GiB RAM, macOS 26.6.2 arm64, CPython 3.12.13,
  rustc 1.98.0.
- Server: Intel i5-10400, 6 physical / 12 logical cores, 31.24 GiB usable RAM,
  Linux 7.0.0-28 x86_64, CPython 3.12.3, rustc 1.97.1, RTX 3070 8 GiB.
- NumPy 2.4.4, pycocotools 2.0.11 and psutil 7.2.2. hotcoco uses its public
  1.0.0 wheel; ultrafast is an ordinary portable release build from source,
  without allocation instrumentation or native-CPU/fast-math flags.
- Prediction filenames are passed directly to `loadRes` in the files route.
  The list route includes Python JSON parsing and keeps that list alive during
  evaluation. GT is a compact file input in both cases. The original snapshots
  remain available for lazy public annotation access and changed evaluation caps.
- Wall/process CPU include loading, construction, evaluate/accumulate/summarize.
  Imports, inference and later public-array conversion/hash/export are excluded.
  RSS is sampled before array diagnostics and includes imports and input storage.
- Rayon/OpenMP pool sizes are 1 and 2, with BLAS limits set to 1. These are pool
  settings, not hard process-wide CPU/thread quotas. No CPU affinity is applied.
  File caches are warm. Fresh processes, alternating backend order and reversed
  pool order balance repetitions; separate warmup/oracle runs are excluded.
- Both hosts have other processes present. No timing, memory or load outliers
  are discarded. Whole-host CPU, available RAM and swap counters include the
  benchmark and other processes, including startup and verification intervals.

During the final comparison, median whole-host CPU usage was 22.5% on M2 and
8.3% on the server. Available RAM ranged from 5.06–8.50 GB and 26.28–31.25 GB,
respectively. M2's swap-out counter increased during 23 of 144 timed-process
intervals; across the whole experiment, swap-out increased by 11.96 MB and
swap-in by 1.044 GB. Server swap counters did not change. All samples remain
included; host counters do not identify which process caused paging.

## Correctness and audit

The new pure-Rust numeric oracle compares approximately 200,000 finite float and
independently generated decimal spellings against serde_json with
`float_roundtrip`, bit for bit. Four tests also cover signed zero, integer cast
boundaries, half-way decimals, subnormals, overflow, booleans, malformed geometry
and long unused numeric values. This oracle is now required by the Rust CI job.
The final local Python suite passed 383 tests with 10 optional skips. The focused
77-case pose suite and numeric oracle also ran on the server.

| Requirement | Evidence |
| --- | --- |
| Same full workload | Both hosts' input hashes match the published inputs; 5,000 images for each task and exact prediction counts were rechecked. |
| Every configuration faster and lower RSS | 24/24 median comparisons pass; every one of the 144 matched timing pairs favors the candidate. |
| Complete execution | 348 successful processes: 288 timed, 48 warmups and 12 pycocotools oracles. No exclusions. |
| Candidate correctness | Every diagnostic precision/recall/scores array is byte-identical to pycocotools, with zero AP/AR differences; every timed curve hash is stable and every timed candidate summary matches its oracle. |
| Consistent runtime | All timed native/runtime hashes and parameters match their diagnostic runs; both hosts executed identical benchmark scripts with matching package versions. |
| Numeric conversion | Four Rust oracle tests, including the approximately 200,000-value corpus, pass locally in debug and release and on the server in debug; optimized tests are required in CI. |
| Reproducible evidence | [Verifier](../bench/verify_hotcoco_goal.py), complete raw archive, source snapshot, input hashes and checksums are provided. |

hotcoco's AP/AR agrees within absolute tolerance 1e-12 on these predictions,
but its known bbox/mask sampled-score differences remain visible in the
retained full-array comparisons. This report does not equate all hotcoco output
arrays with pycocotools. See the [minimal reproduction](benchmark-hotcoco.md#sampled-score-compatibility).

## Reproduce

Use the unchanged input files from the
[existing evidence archive](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz).
Install the candidate source into an isolated Python 3.12 environment alongside
hotcoco 1.0.0, pycocotools 2.0.11, NumPy 2.4.4 and psutil 7.2.2. Use a separate
`CARGO_TARGET_DIR` for this build. Do not substitute the public ultrafast 0.1.10
wheel for the candidate.

```sh
python bench/hotcoco_benchmark.py \
  --build-description 'hotcoco 1.0.0 wheel versus unreleased pose numeric candidate' \
  --inputs INPUTS --out NEW_RESULTS --threads 1 2 --rounds 6
cargo test --locked --release -p ufcoco-py --test pose_numbers
python -m pytest -q tests/test_keypoint_storage.py
```

[Full raw evidence](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.10/hotcoco-goal-evidence-20260910.tar.gz)
includes every child JSON/log, diagnostic NPZ, exact scripts and candidate source,
host telemetry and both exploratory attempts.
[Archive SHA-256 and size](../bench/results/hotcoco-goal-20260910/evidence.json).
Raw macOS JSON retains legacy `NaN` CPU-load fields; use the psutil samples for
host load. The committed summary is strict JSON.

The verifier fails if any of the 24 configurations lacks a lower median wall
time or peak RSS, or if required processes, source hashes or candidate accuracy
checks disagree. It also rechecks all full diagnostic arrays and raw summaries:

```sh
cd hotcoco-goal-evidence-20260910
shasum -a 256 -c SHA256SUMS
python bench/verify_hotcoco_goal.py --m2 m2 --server server --out NEW_SUMMARY
```
