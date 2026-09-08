# Mask buffer allocation follow-up

This follows the [native non-bbox file evaluation change](nonbbox-optimization.md),
merged in [PR #3](https://github.com/developer0hye/ultrafast-pycocotools/pull/3).
Its two exact-head CI runs passed at `d2085fdf5e8824af1717da4014ac399bf777a74c`;
the merge is `94d47a0aad3321a4da49c39847e4e6084821d84e`.

The additional implementation is
`506c8213985bafcbd46b88120aa28ec1354c7213`. It writes segmentation masks into
their final native vector using indexed Rayon extension. It avoids a persistent
`Option<Rle>` field for every segmentation mask, intermediate header copies,
and a second destination allocation. Boundary masks keep separate mask/boundary
columns. Mixed/list input still shares the GT/DT rasterization worker, with the
GT vector shrunk after splitting. Matching arithmetic and output arrays do not
change. These builds still report package version 0.1.7 and are unreleased.

## Paired results

The baseline is the first optimization's runtime at
`b3b98166c73816c6fabb1bce31afd5cc6e3173d4`, not public v0.1.7. Both hosts use
six alternating fresh processes per version, two threads, matching local
compilers/dependencies and the same full COCO segmentation predictions. The
previous report describes the hardware and input hashes; raw records repeat
the environment and hashes. All background-load samples are retained.

| Scope | Time, before → after | Maximum process RSS, before → after |
| --- | ---: | ---: |
| M2 standalone file loading + segmentation evaluation | 1.642 → 1.632 s | 1098.1 → 1052.4 MB |
| 3070 server, actual Ultralytics cold bbox+mask replay | 3.513 → 3.522 s | 2512.4 → 2473.1 MiB |
| Same server, cached-GT replay | 2.782 → 2.762 s | Same process measurement above |

These are median measurements; M2 uses decimal MB and server RSS uses MiB.
The memory reductions are 45.8 MB (4.2%) and 39.3 MiB (1.6%). Timing changes
are small and mixed, all below 1%; no additional runtime speedup is claimed.
The larger speed and memory gains from public v0.1.7 are documented separately
in the first-stage report, rather than combining different benchmark rounds.

Every returned metric/fitness is exactly equal to the prior published result.
The two server diagnostics compare all 18 full precision/recall/score arrays
per process against the archived public v0.1.7 arrays: every difference is zero,
including both cached calls. The M2 direct evaluator's full arrays also match
the first optimization exactly. The complete Python suite passes 279 tests;
one optional RF-DETR collection is skipped locally.

## Whole validation and subsequent compatibility correction

The [whole-validation records](../bench/results/nonbbox_buffers_20260909/whole_validation.json)
compare public v0.1.7 source `b52e8527cb772393880289ee4eb0fb30f29a7d15` with
the buffer implementation at `506c8213985bafcbd46b88120aa28ec1354c7213`.
Each task has two alternating fresh-process pairs on the RTX 3070 server,
using the original 5,000-image inputs, models and frozen Ultralytics source.
Settings remain FP32, 640 pixels, batch 16, workers 2, confidence 0.001,
NMS IoU 0.7 and 300 model detections; COCO evaluation keeps the default cap of
100 for bbox/mask and 20 for keypoints.

| Task | Whole validation, before → after | Nested evaluator, before → after | Peak main-process RSS, before → after |
| --- | ---: | ---: | ---: |
| Segmentation | 139.523 → 135.514 s | 6.863 → 2.920 s | 4434.4 → 3417.1 MiB |
| Pose | 32.284 → 31.897 s | 1.665 → 0.959 s | 2747.1 → 2416.2 MiB |

Every prediction file hash and every returned metric/fitness matches the prior
published result. These medians use only two samples per version/task; retain
that limitation rather than treating the small whole-validation time differences
as a precise general speedup. The six-pair replay experiments provide the larger
timing sample. All eight full-process records and host telemetry are saved.

A subsequent review found that duplicate `segmentation`, `keypoints` or
`num_keypoints` JSON fields reached a stricter geometry parser instead of keeping
Python JSON's last value. Commit `41d0cad777263329880c3a9c75b129d897546dde`
makes these inputs use ordinary JSON loading, matching duplicate scalar-field
handling. Three new regression cases fail before the correction and pass after
it. The complete corrected suite passes **282 tests** (one optional RF-DETR
collection skipped). D-FINE backend and pretrained accuracy checks pass **11
tests** with both the buffer version and the corrected native version. The
whole-validation table above specifically measures the earlier buffer commit;
the final corrected version has a separate replay measurement.

## Allocation accounting and conditional limits

A separate build enables the existing `alloc-stats` allocator counters. Its
timings are instrumentation-affected and are excluded from the performance
comparison. `bench/profile_memory.py --file-inputs --json-out ...` now retains
the complete output arrays and records exact allocation counts on macOS as
well as Linux. The diagnostic measures this same 5,000-image input:

| Current representation's retained components | Bytes |
| --- | ---: |
| Immutable JSON snapshots, native source columns and complete output arrays | 410,395,109 |
| GT RLE run payload for nonempty joins | 34,928,080 |
| DT RLE run payload for nonempty joins | 290,920,280 |
| RLE headers, 761,734 × 32 bytes on this 64-bit target | 24,375,488 |
| Native instance columns, 761,734 × 35 bytes | 26,660,690 |
| **Subtotal** | **787,279,647** |
| **Measured retained Rust allocations** | **792,159,637** |
| **Measured peak Rust allocations** | **816,153,972** |

The retained/peak Rust allocation measurements are about 0.6%/3.7% above this
subtotal. However, **this is not an information-theoretic lower bound**. It
assumes the current immutable snapshot, decoded-RLE and column layouts, and
includes decoded DT payload beyond `maxDets`. The run-count diagnostic shows
another **20,557,564 bytes** of that payload could be omitted. Removing it
would lower the subtotal to 766,722,083 bytes; the measured native peak is
about 6.4% above that conditional subtotal. The optimization has not implemented
that extra filtering yet.

These numbers also do not make process RSS optimal. Python objects, allocation
size classes, fragmentation and allocator-retained pages contribute to RSS;
the normal-build M2 peak remains 1052.4 MB. A compressed or different geometry
representation could change the subtotals and trade computation for memory.
No hardware throughput or global time optimum is claimed.

## Final corrected source: comparison with public v0.1.7

The [final corrected report](../bench/results/nonbbox_buffers_20260909/final_corrected.json)
compares `b52e8527cb772393880289ee4eb0fb30f29a7d15` with
`41d0cad777263329880c3a9c75b129d897546dde`, including the duplicate-field
compatibility correction. Each host/task has six alternating fresh-process
pairs. Both versions use the same compiler and dependencies within each host.
This is the cumulative change, including PR #3, rather than the buffer-only
comparison at the top of this report.

| Scope | Median time, before → after | Median peak RSS, before → after |
| --- | ---: | ---: |
| M2, load files + segmentation evaluation | 4.720 → 2.500 s | 2728.6 → 1053.6 MB |
| M2, load files + keypoint evaluation | 1.265 → 0.775 s | 708.1 → 329.8 MB |
| 3070 server, Ultralytics cold bbox+mask replay | 5.863 → 3.524 s | 3787.2 → 2482.6 MiB |
| 3070 server, Ultralytics cold bbox+keypoint replay | 1.828 → 0.949 s | 1717.5 → 1360.6 MiB |

The server CPU is an Intel i5-10400 (6 cores / 12 threads), with 31.24 GiB RAM.
These replay evaluations run on the CPU; the GPU identifies the machine used
for the separate whole-validator experiment. Every metric and all complete
arrays match. The server's two baseline/candidate diagnostic NPZ files for
each task are also byte-identical to the public v0.1.7 archive, including all
three calls and both evaluation types. The report links those existing public
files and records their hashes. Four additional M2 diagnostics compare all
precision, recall and score arrays exactly.

During this work the shared managed Python installation changed build identity
while retaining version 3.12.13. Its cause was not established. The final M2
study therefore uses a private copied runtime, verifies all 1,898 functional
file hashes, and records the executable, shared-library, Python source and
native-extension hashes in every measured process. All 24 timed processes use
the same runtime hashes and Python build (Clang 21.1.4); macOS dyld confirms
that the loaded core library belongs to the private copy. Source and native
hashes remain constant within each variant. Provenance reads occur after timing
and peak-RSS capture.

The M2 host remains busy: sampled host CPU utilization has median 62.2% and
maximum 100%, with at least 4.07 GiB available RAM. Server samples have median
8.7% CPU and at least 25.75 GiB available RAM. These are within-run comparisons
under recorded contention, not isolated hardware throughput estimates. An
earlier unpinned M2 run with median 100% CPU remains in the report explicitly
marked superseded; it is not used for the final table. Earlier stage/buffer
studies remain separate and their absolute times must not be combined with
this final round. Missing legacy CPU readings are JSON `null` in published
reports; hashes of the original raw files remain unchanged.

## Experiments not selected

The evidence includes failed or intermediate attempts, rather than only the
selected version:

- A Python summary-view change on the real 93-image COCO-style LVIS mask case
  gave cold time 0.452 → 0.457 s and RSS 1036.8 → 1037.3 MiB. Cached time was
  0.431 → 0.428 s. Full arrays/metrics matched, but this did not establish a
  useful whole-call time/memory improvement. The patch was reverted and saved.
- An intermediate mask-buffer implementation reduced M2 RSS by 47.4 MB but
  increased load-plus-evaluation time from 1.524 to 1.551 s. The final direct
  extension removes its remaining per-chunk pair/copy step. Its separate
  paired results are shown above; the intermediate results remain in the data.

LVIS summary/federated preparation optimization remains open. Its large
complete output tensors are part of the public API, so reducing their shape
or omitting empty categories is not an acceptable memory shortcut.

## Evidence checklist

- [x] First-stage code merged after both exact-head CI runs passed.
- [x] Preserve immutable snapshots, mutable views, crowd and class-agnostic behavior.
- [x] Record six alternating process pairs on each host for the selected follow-up.
- [x] Preserve every metric and compare complete arrays, including cached GT reuse.
- [x] Count retained RLE payload, native live/peak allocations and remaining `maxDets` payload.
- [x] Retain rejected/intermediate experiments and avoid claiming a global optimum.
- [x] Validate the final correction with 282 Python tests and 11 D-FINE checks.
- [x] Repeat final paired measurements with a frozen M2 runtime and complete array checks.
- Follow-up exact-head CI and merge status: [PR #4](https://github.com/developer0hye/ultrafast-pycocotools/pull/4).
- [x] Complete eight separate whole-validator checks; do not infer whole-validation gains from replay.
- [ ] Publish a versioned release before changing the upstream dependency floor.

[Buffer experiment report](../bench/results/nonbbox_buffers_20260909/report.json) contains
all timed records, exact source/native/input hashes, arrays comparisons and
allocation accounting. The same directory includes telemetry, frozen scripts,
experimental patches and verification logs. Large NPZ files remain local under
`bench/out/nonbbox-memory` and are hash-identified in the reports. Final server
arrays are byte-identical to files in the linked public archive; the other
experimental NPZ files are not published release assets.
