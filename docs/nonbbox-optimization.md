# Native segmentation and keypoint file evaluation

The first non-bbox optimization removes Python annotation materialization from
eligible file-backed segmentation, boundary, and keypoint evaluation. It also
avoids rasterizing and retaining masks in groups without an opposing annotation.
Public annotation access still produces mutable Python objects and disables the
snapshot path. In-memory inputs, subclasses, and LVIS retain their existing path.

This report measures unreleased commit
`b3b98166c73816c6fabb1bce31afd5cc6e3173d4` against the v0.1.7 release source
`b52e8527cb772393880289ee4eb0fb30f29a7d15`. Both packages still report version
0.1.7; these improvements are **not yet in the public 0.1.7 wheel**.

## Results

Each comparison uses six fresh processes per version, alternating execution
order. All samples are retained, including host CPU and available RAM telemetry.
Each host builds both versions with its own identical compiler and uses the same
Python/NumPy versions and two Rayon threads. Hosts were not otherwise isolated.
Compare versions within each row; the two hosts run different measurement scopes.

### RTX 3070 server: actual Ultralytics evaluator replay

Intel i5-10400, 6 cores/12 threads, 31.24 GiB RAM, Ubuntu 24.04,
Python 3.12.3, NumPy 2.4.4, Rust 1.97.1. The RTX 3070 is not used by this CPU
evaluator replay. The frozen Ultralytics source is
`0d4068533176b3b692837801afc7b5a3fd9fd7ec`, with Torch 2.13.0+cu126.

| Real COCO task | Cold evaluation, before → after | Cached evaluation, before → after | Peak process RSS, before → after |
| --- | ---: | ---: | ---: |
| Segment: bbox + mask | 5.899 → 3.482 s | 5.117 → 2.782 s | 3788.2 → 2512.9 MiB |
| Pose: bbox + keypoints | 1.873 → 0.948 s | 1.574 → 0.911 s | 1719.8 → 1363.0 MiB |

Cold time falls 41.0%/49.4%, and peak RSS falls 33.7%/20.7%, respectively.
Each process retains the actual predictions in `validator.jdict` and performs
one cold plus two calls using the same cached COCO GT object. Cached values are
the median of each process's two-call mean. Initial prediction loading,
inference, and process startup are outside the evaluator clock; their allocated
memory remains inside process RSS. This does not establish a whole-validation
speedup or a new whole-validation pose memory result.

Both tasks include all 5,000 val2017 images. Segment uses 724,953 predictions;
pose uses 134,663. Inputs are exactly those in the
[previous full-validation report](ultralytics-pr26101-validation.md), with the
original independent annotations. Every returned metric and fitness equals the
published 0.1.7 result. Four separate fresh-process diagnostics save all 18 arrays
per process (three calls × two IoU types × precision/recall/scores): all arrays
are exactly equal to the previously archived 0.1.7 arrays, including cached reuse.

### M2: standalone library file loading and evaluation

Apple M2, 16 GiB RAM, Python 3.12.13, NumPy 2.4.4, Rust 1.98.0.
These runs use the same real predictions and annotations. They include file
loading and a single evaluation, without Torch, Ultralytics, or retained `jdict`.
Memory below uses decimal MB, as reported by `bench/run_impl.py`.

| IoU type | Load + evaluate, before → after | Evaluation only, before → after | Peak process RSS, before → after |
| --- | ---: | ---: | ---: |
| Segmentation | 2.781 → 1.495 s | 2.426 → 1.140 s | 2727.8 → 1098.3 MB |
| Keypoints | 0.941 → 0.593 s | 0.738 → 0.389 s | 710.5 → 329.4 MB |

Load-plus-evaluate time falls 46.2%/37.0%; RSS falls 59.7%/53.6%.
Every timed run has matching summary values and complete-array digests.
Four additional diagnostics save and compare the actual full arrays with
`numpy.testing.assert_array_equal`; all comparisons pass.

## Implementation and correctness

Previously, non-bbox evaluation expanded the native immutable file snapshot into
Python dictionaries and indexes, then extracted geometry back into Rust. The new
path reads geometry directly from that snapshot with typed JSON visitors. It
retains a bounded mask queue and shared Rayon rasterizer, and stores keypoints
as native numeric arrays. Numeric parsing keeps float-roundtrip semantics.

Groups without an opposing annotation never call mask IoU. Their scalar fields
remain present for false-positive/false-negative accounting, but their masks use
empty internal placeholders. The membership test includes crowd/ignored GT and
uses image-only groups when `useCats=0`. It does not omit detections beyond
`maxDets`; that is a separate, unfinished optimization requiring the exact stable
score order. Mask field types and image metadata are still read before omitting
rasterization. Public diagnostic methods can materialize the original snapshot.

Validation for this revision:

- 279 Python tests pass; the optional RF-DETR integration collection is skipped.
- 62 Rust core tests pass in debug and release mode.
- Real full-array equality is verified separately on both hosts.
- Tests cover mixed file/dict inputs, filtering, cached reuse, public keypoint
  mutation, Boolean coordinates, duplicate top-level annotation arrays,
  class-agnostic evaluation, boundary evaluation, and per-instance records.
- `cargo clippy` completes with four pre-existing warnings in unchanged core
  and JSON-loader code. The stricter `-D warnings` invocation fails on those
  existing warnings; it is not reported as passing.

## Remaining work and limits

This is a measured improvement, not proof of theoretical optimality.

The current representation retains the immutable input snapshot so later API
access survives file changes/deletion. For this segmentation input, snapshots,
native scalar columns and complete output arrays alone occupy **410,395,109
bytes**, before decoded mask runs, engine columns, indexes, allocator overhead or
the application. For keypoints the corresponding subtotal is **172,816,706
bytes**, before keypoint coordinates and engine storage. These are conditional
storage subtotals for the current representation, not information-theoretic
lower bounds; a different representation could trade CPU work for memory.

The M2 native phase diagnostic attributes about 0.693 s to extraction and
0.453 s to evaluation for segmentation. Reading and rasterization overlap;
worker-summed IoU/matching/accumulation times must not be added to wall time.
An independent LVIS diagnostic identifies materialization and summary-array
selection as additional costs; its complete output arrays require 239,702,400
bytes in the tested COCO-style LVIS configuration.

- [x] Eliminate Python materialization for eligible mask/keypoint file inputs.
- [x] Avoid mask rasterization/storage where there is no opposing annotation.
- [x] Check complete arrays, mutable APIs, crowd and class-agnostic behavior.
- [x] Record balanced speed/memory comparisons and background load on both hosts.
- [ ] Reduce LVIS materialization and summary-copy costs.
- [ ] Evaluate native geometry caching and allocation capacity reductions.
- [ ] Omit masks beyond `maxDets` with a shared, proven stable ordering rule.
- [ ] Measure required geometry storage and remaining allocation overhead.
- [ ] Repeat whole-validator segmentation/pose measurements for a release candidate.
- [ ] Complete live-head CI and publish a versioned release before updating the
  upstream dependency floor again.

## Evidence and reproduction

[Raw report](../bench/results/nonbbox_20260909/report.json) includes all timed
metric records, versions/native hashes, array shapes/differences and artifact
SHA-256 hashes. The same directory contains the exact driver scripts,
per-process telemetry, diagnostic metadata, and verification logs. Large NPZ
files remain in `bench/out/nonbbox-opt/{m2,server}`; they are hash-identified in
the report but are not yet published as release assets.

For a standalone run, build each pinned source in a separate environment and
Cargo target directory, with matching Python and NumPy. Run the candidate's
benchmark script using each interpreter:

```sh
python bench/run_impl.py --impl ufcoco --file-inputs --threads 2 \
  --gt instances_val2017.json --dt segment-predictions.json --iou-type segm \
  --json-out result.json
```

Use `person_keypoints_val2017.json`, `pose-predictions.json` and `--iou-type
keypoints` for pose. Alternate fresh processes rather than importing both
builds into one process. The frozen replay drivers record their exact commands,
source and input paths; replace their workspace roots when reproducing elsewhere.
