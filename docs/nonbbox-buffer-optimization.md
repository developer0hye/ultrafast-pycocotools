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
- [ ] Finish follow-up live-head CI and merge.
- [ ] Complete separate whole-validator checks; do not infer whole-validation gains from replay.
- [ ] Publish a versioned release before changing the upstream dependency floor.

[Raw report](../bench/results/nonbbox_buffers_20260909/report.json) contains
all timed records, exact source/native/input hashes, arrays comparisons and
allocation accounting. The same directory includes telemetry, frozen scripts,
experimental patches and verification logs. Large NPZ files remain local under
`bench/out/nonbbox-memory` and are hash-identified in the report; they are not
yet published release assets.
