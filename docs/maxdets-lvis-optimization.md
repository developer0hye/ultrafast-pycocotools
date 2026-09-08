# maxDets storage and LVIS preparation/summary optimization

This follows the native non-bbox loader and mask-buffer changes packaged for
0.1.8. The new runtime was measured at
`0a0de58d11f8b7404f5f2885ee72f7385e61d2e7`; version 0.1.9 packages these changes.
The within-host baseline is `41d0cad777263329880c3a9c75b129d897546dde`, whose
runtime is the same as 0.1.8. Server and earlier M2 source builds still report 0.1.7 in package metadata.
The final M2 build reports 0.1.9 at `26cec60`, which additionally guards unaligned
summary arrays. These are controlled source builds, not public 0.1.7 wheel
measurements. The server table predates that alignment guard.

## Changes and preserved behavior

- For eligible file-backed segmentation/boundary results, omit decoded mask
  payload outside the largest per-image/category `maxDets`. Scalar rows and
  original IDs remain available. Small slot grids are counted first and only
  overflowing groups are sorted; large sparse grids use the existing sparse
  grouping. Class-agnostic ties retain category-major ordering. Geometry and
  image metadata are still validated before discarding payload.
- LVIS result snapshots materialize only selected federated detections, without
  building every public annotation index. The official global per-image cap
  is applied before category/federated filtering. COCO-style LVIS retains its
  per-category cap. Public mutable views disable the snapshot path, and later
  evaluations read current parameters and metadata.
- LVIS summary selection gathers valid values directly from the original
  arrays. NumPy computes the mean of the identical ordered sequence; native
  code does not substitute a different floating-point reduction. Complete
  public output arrays retain their shapes. Strided arrays, duplicate selections
  and mutation between summaries have differential coverage; unusual dtypes
  unaligned arrays and array subclasses keep ordinary NumPy indexing.
- A regression test also exposed an older zero-cap diagnostic discrepancy:
  detection-only groups must retain an empty `evalImgs` record after truncation,
  rather than become `None`. The test fails on the baseline and passes now.

## Controlled M2 comparison

Apple M2, 8 logical cores, 16 GiB RAM, macOS arm64; Python 3.12.13 uses the
private frozen runtime from the preceding report, NumPy 2.4.4, Rust 1.98.0.
Both variants use the same interpreter, dependencies, compiler and two Rayon/
OpenMP threads. Each task has six alternating fresh-process pairs and a
separate full-array diagnostic pair. Times include file loading and evaluation;
RSS uses decimal MB and is captured immediately after evaluation, before
array hashing or file verification can allocate additional buffers. The final run contains no concurrent heavy work started by
this agent; existing unrelated processes remained active and telemetry is saved.

| Task and input | Load + evaluation, before → after | Summary only, before → after | Median peak RSS, before → after |
| --- | ---: | ---: | ---: |
| COCO segmentation, 5,000 images | 5.416 → 5.072 s | 3.18 → 3.12 ms | 1048.5 → 1016.7 MB |
| Official LVIS bbox example, 100 images | 0.1955 → 0.1390 s | 24.13 → 12.89 ms | 165.7 → 135.2 MB |
| COCO-style LVIS mask subset, 93 images | 0.3344 → 0.2601 s | 53.51 → 27.53 ms | 350.7 → 305.3 MB |

The final M2 host was heavily contended: CPU samples have median/maximum 100%,
with at least 3.21 GiB RAM available. Its wall-time medians describe that paired
run, not isolated throughput. The earlier lower-load counted-group round had
24.75% median CPU and COCO mask time 1.523 → 1.517 s (under 1% change); its RSS
was recorded after verification copies and is superseded by the table above.
All rounds remain in the raw data. The final M2 LVIS reductions are about
29%/22% in load-plus-evaluation time and 18%/13% in peak RSS. The less-contended
server result below is the stronger evidence for evaluator wall-time changes.
Every statistic and full precision/recall/score array matches the baseline.
The 93-image mask input uses the previously archived YOLO masks mapped into
LVIS categories; it is a compatibility workload, not full-taxonomy model accuracy.

## Server validation

Intel i5-10400 (6 cores / 12 threads), 31.24 GiB RAM, RTX 3070; Python
3.12.3, NumPy 2.4.4, Rust 1.97.1, two Rayon/OpenMP/BLAS threads. Frozen
Ultralytics source `0d406853` replays identical archived predictions in six
alternating process pairs, retaining the caller's prediction objects. Each
process runs one cold and two cached-GT calls. Cached medians first average
the two calls within a process; RSS uses MiB.

| CPU replay | Cold evaluator, before → after | Cached evaluator, before → after | Median peak RSS, before → after |
| --- | ---: | ---: | ---: |
| COCO bbox + mask, 5,000 images | 3.461 → 3.478 s | 2.776 → 2.791 s | 2470.1 → 2453.0 MiB |
| COCO-style LVIS bbox + mask, 93 images | 0.473 → 0.403 s | 0.439 → 0.376 s | 1039.8 → 1017.9 MiB |

COCO mask time rises by about 0.5%; the measured memory reduction is 17.1 MiB.
LVIS time falls about 15% cold / 14% cached, with a 22.0 MiB RSS reduction.
Every returned metric, fitness and evaluator statistic matches. The complete
server NPZ diagnostics are byte-identical to the existing public evidence
archive, including all cold/cached calls and both evaluation types. Host CPU
samples have median 9.7%, maximum 100%, and at least 27.00 GiB available RAM.
These evaluations run on the CPU. GPU inference is outside this follow-up's
replay clock; no whole-validation speedup is inferred from it.

## Native allocation accounting

A separate `alloc-stats` build retains the complete output arrays. Instrumented
wall times are excluded from performance comparisons. The same full COCO input
was used in the earlier buffer allocation diagnostic at `506c821`.

| Rust allocator counter | Earlier buffer implementation | New implementation |
| --- | ---: | ---: |
| Live bytes at end | 792,159,637 | 771,602,105 |
| Peak bytes | 816,153,972 | 795,596,264 |
| Total allocated bytes | 1,413,654,525 | 1,391,084,402 |
| Allocation count | 2,470,750 | 2,379,785 |

Run-storage accounting identifies 20,557,564 bytes of removable decoded DT
payload (290,920,280 → 270,362,716). Actual total live/peak counters drop by
20,557,532 / 20,557,708 bytes; the tiny differences are retained rather than
rounding the counters into exact payload equivalence. The previous report's
conditional subtotal after removing this payload is 766,722,083 bytes. The
new live/peak values are about 0.6%/3.8% above that subtotal. This assumes the
current snapshots, RLE representation, headers and output arrays; it is not a
global theoretical memory or throughput bound, and excludes Python/RSS slack.

## Retained experiments and validation

The first cap implementation sorted a second complete grouping. It saved
memory but slightly increased time. The counted-group implementation avoids
that overhead. LVIS preparation was measured separately before adding native
summary selection, so both stages remain inspectable. An earlier M2 combined
run overlapped this agent's D-FINE tests during some LVIS mask samples; it is
retained as superseded and excluded from the final table. A further harness
review moved RSS capture before array hash copies; earlier M2 RSS records
include that verification overhead and are explicitly superseded.

- [x] 310 local Python tests and the Rust core suite pass.
- [x] D-FINE backend/pretrained checks pass (11 tests on the first combined runtime).
- [x] Both LVIS protocols, official reference data, capped ties and mutable views are covered.
- [x] Compare complete M2 arrays and all summary statistics; retain each sample and environment hash.
- [x] Record exact allocation counters separately from normal-build timings.
- [x] Complete final server paired records, all-statistic checks and public NPZ hash comparisons.
- Exact-head CI and merge status: [PR #6](https://github.com/developer0hye/ultrafast-pycocotools/pull/6).
- Release installation/distribution verification: [0.1.9 release](https://github.com/developer0hye/ultrafast-pycocotools/releases/tag/v0.1.9).

[Raw records and provenance](../bench/results/maxdets_lvis_20260909/report.json)
include all samples, telemetry, exact source/native/runtime/input hashes and
separate allocation accounting. The directory retains the drivers, test logs
and superseded experiments. Server array hashes point to identical files in
the public archive; M2 NPZ files remain locally hash-identified.
