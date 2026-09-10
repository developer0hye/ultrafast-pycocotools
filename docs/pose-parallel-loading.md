# Parallel pose loading and capped coordinate storage

The later [numeric-decoding comparison](hotcoco-performance-goal.md) closes the remaining measured M2 pool-size-1 exception. This report retains the earlier source build and its original measurements.

This follow-up optimizes **pose file input** after the earlier
[compact coordinate storage change](pose-input-optimization.md).
It indexes keypoint JSON byte ranges while loading the immutable snapshot,
then decodes independent detection rows into disjoint final storage with Rayon.
Coordinates outside the stable per-image/category `maxDets` selection are still
validated, but their x/y payload is not retained. Scalar rows remain available
for accumulation and diagnostics, including detection-only images and zero caps.
Ground-truth visibility, custom joint counts, and OKS arithmetic are unchanged.
No new API or opt-in flag is required.

The candidate is an **unreleased source build**, based on main commit
`7aa096398ae25df5d2b1a7b44fbaaceb58de9da1`, with the changes in this report.
Both baseline and candidate identify as package version 0.1.10; native binary
hashes and source snapshots distinguish them. The baseline already includes
the previous pose storage optimization. The separate
[hotcoco release comparison](benchmark-hotcoco.md) continues to report the
public 0.1.10 wheel; its historical measurements are unchanged.

## Results

On 2026-09-10, six alternating fresh-process pairs on all 5,000 COCO val2017
images reduced file-route wall time by **24.9% on Apple M2** and **22.7% on the
i5-10400 server**. Peak RSS fell by 9.2 MB (3.3%) and 9.9 MB (3.7%), respectively.
Process CPU time fell by about 3%; most of the wall-time improvement comes from
using two workers for independent number conversion. Python prediction-list
end-to-end time changed by less than 1%. List-route peak RSS changed by less
than 0.2%, so this is a file-route improvement rather than a general list-memory
reduction.

| Host | Input | Wall (s), before → after | CPU (s), before → after | Peak RSS (MB), before → after |
| --- | --- | ---: | ---: | ---: |
| m2 | files | 0.517 → 0.388 | 0.532 → 0.514 | 277.4 → 268.2 |
| m2 | list | 1.732 → 1.720 | 1.748 → 1.735 | 810.9 → 811.0 |
| server | files | 0.701 → 0.542 | 0.728 → 0.707 | 269.0 → 259.1 |
| server | list | 2.572 → 2.564 | 2.599 → 2.589 | 567.7 → 568.4 |

| Host | Input | Pool size | Wall (s), hotcoco → candidate | Peak RSS (MB), hotcoco → candidate |
| --- | --- | ---: | ---: | ---: |
| m2 | files | 1 | 0.501 → 0.512 | 1200.3 → 268.3 |
| m2 | files | 2 | 0.458 → 0.392 | 1207.6 → 268.3 |
| m2 | list | 1 | 1.969 → 1.740 | 1316.2 → 811.0 |
| m2 | list | 2 | 1.933 → 1.728 | 1323.9 → 811.0 |
| server | files | 1 | 1.064 → 0.700 | 645.9 → 259.0 |
| server | files | 2 | 0.964 → 0.542 | 645.9 → 259.1 |
| server | list | 1 | 3.216 → 2.600 | 916.9 → 568.4 |
| server | list | 2 | 3.125 → 2.580 | 922.7 → 568.4 |


All values above are medians; MB is decimal. The first table compares the main
baseline and candidate with pool size 2. The second independently compares the
hotcoco 1.0.0 wheel and candidate in balanced runs at pool sizes 1 and 2.
At pool size 2 the candidate also beats hotcoco for M2 pose file input, which
was slower with the public ultrafast 0.1.10 wheel. At pool size 1, **hotcoco is
still about 2% faster on M2**. Do not read the results as a universal speedup.
Individual repetitions, min/median/max, CPU time, native hashes and host load
are in the [machine-readable results](../bench/results/pose-parallel-20260910/summary.json).

## Measurement boundaries and hardware

- Identical saved YOLO26n pose predictions: 134,663 detections, 11,004 GT
  annotations, every COCO val2017 image including images without annotations.
  No model inference is rerun. Input hashes match across hosts.
- Apple M2: 8 cores, 16 GiB RAM, macOS 26.6.2 arm64, CPython 3.12.13,
  rustc 1.98.0. Server: i5-10400, 6 cores / 12 threads, 31.24 GiB usable RAM,
  Linux 7.0.0-28 x86_64, CPython 3.12.3, rustc 1.97.1. The server's RTX 3070
  does not execute the CPU metric.
- NumPy 2.4.4 and pycocotools 2.0.11. Both source variants use ordinary portable
  release builds with the same compiler on each host, without allocation
  instrumentation, native-CPU flags or fast-math. Build directories and Python
  environments are isolated. Both package source snapshots are archived.
- Wall/process CPU include GT/DT loading, evaluator construction, evaluation,
  accumulation and summary. Imports, inference and later public-array
  conversion/hashing are excluded. Peak RSS is sampled before diagnostic exports.
- Files route passes prediction filenames to `loadRes`; list route includes
  Python JSON parsing of predictions and keeps that list alive. GT uses file
  input in both routes. Original file snapshots remain available for lazy public
  annotation access and reevaluation after parameter changes.
- Rayon/OpenMP pool size is 2 for before/after runs, and 1/2 for the hotcoco
  comparison. Pool size is not a hard process-wide CPU/thread quota. BLAS is
  limited to 1 thread, no CPU affinity is applied, and OS file caches are warm.
- Both PCs have other processes present. No samples are excluded for speed,
  memory, CPU load or paging. Before/after logs record whole-host CPU and available
  RAM; the separate hotcoco comparison also records swap counters. These are
  observed process measurements, not isolated-machine limits or a theoretical optimum.

During the matched hotcoco comparison, median whole-host CPU usage was 18.1% on
M2 and 8.3% on the server. Available RAM ranged from 6.31–7.29 GB and
29.75–31.13 GB respectively. Across the entire M2 comparison, host swap-in
increased by 26.61 MB and swap-out by 0.18 MB; none of the sampled timed-run
intervals showed a swap-out increase. The server counters did not change.
These counters include other processes and do not attribute paging to a backend.

## Correctness and remaining costs

All 48 final before/after processes and 116 hotcoco/oracle/diagnostic processes
completed successfully. All timed curve hashes were stable; the candidate's
full precision, recall and scores arrays were byte-identical to pycocotools,
and AP/AR differences were zero at both pool sizes, both input routes and both
hosts. The same candidate binaries were used for the before/after and hotcoco
experiments on each host.

The full final local suite passed **383 tests with 10 optional skips**; Rust core
passed 63 tests. The focused pose suite passed 77 cases on M2 and the server.
Tests include 3/17/25 joints, custom sigmas, zero/NaN visibility, alternate area
calculation, malformed discarded coordinates, cap 0 and later cap growth,
category-agnostic ties, complete `evalImgs`, `per_instance`, `computeOks`, and
annotation access after deleting or changing source files. Parallel decoding
uses checked Rust slices and immutable owned bytes; no borrowed Python objects
are accessed from workers.

The Python list route is still dominated by JSON parsing and Python object
construction. The file route retains the original JSON snapshot for API
compatibility and still validates all selected detection coordinates. The
number parser, first-pass scan and OKS computation remain meaningful costs;
this change does not establish a theoretical speed or memory limit.

## Reproduce and audit

Use the unchanged inputs from the
[published validation archive](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz).
Build the baseline commit and this candidate in separate Python 3.12 environments
and **separate `CARGO_TARGET_DIR` directories**, with NumPy 2.4.4, pycocotools
2.0.11 and psutil 7.2.2. Install the source packages with `uv pip install`; do
not substitute the published 0.1.10 wheel for the main baseline.

```sh
python bench/pose_input_benchmark.py \
  --baseline-python BASELINE/bin/python --candidate-python CANDIDATE/bin/python \
  --gt INPUTS/person_keypoints_val2017.json --dt INPUTS/pose-predictions.json \
  --out NEW_PAIR_RESULTS --repeat 6

# In the candidate environment, also install hotcoco==1.0.0:
python bench/hotcoco_benchmark.py \
  --build-description 'hotcoco 1.0.0 wheel versus unreleased pose candidate' \
  --inputs INPUTS --out NEW_HOTCOCO_RESULTS \
  --tasks keypoints --threads 1 2 --rounds 6
```

[Full evidence archive](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.10/pose-parallel-evidence-20260910.tar.gz)
contains raw logs/JSON, diagnostic NPZ arrays, source snapshots, executed scripts,
input/runtime hashes and exploratory measurements. Early serial prototypes are
kept separately and are not pooled into final measurements.
[Archive SHA-256 and size](../bench/results/pose-parallel-20260910/evidence.json).
The archived macOS child JSON preserves legacy `NaN` CPU-load fields; use the
psutil samples for host load. The committed summary is strict JSON.

After extraction, verify `SHA256SUMS` and regenerate the summary:

```sh
cd pose-parallel-evidence-20260910
shasum -a 256 -c SHA256SUMS
python bench/summarize_pose_focus.py --evidence . --out NEW_SUMMARY
```
