# Further default improvements in 0.1.2

Version 0.1.2 further reduces both evaluation time and peak memory on the tested
COCO workload, with normal API calls and no performance flags. Public annotation
dictionaries, query ordering and numerical outputs remain covered by tests.

| Package defaults, same inputs | 0.1.1 median time | 0.1.2 median time | 0.1.1 median peak RSS | 0.1.2 median peak RSS |
|---|---:|---:|---:|---:|
| COCO val2017 / public YOLO11m | 1.604 s | **1.461 s** | 601.7 MiB | **590.4 MiB** |
| Objects365 v2 / synthetic predictions | 8.191 s | 8.074 s | 2,164.4 MiB | **2,093.3 MiB** |

COCO time falls **8.9%** and peak RSS **1.9%** relative to 0.1.1. Objects365
peak RSS falls **3.3%**. Its median time is 1.4% lower, but measured ranges
overlap: 8.039–8.655 s for 0.1.1 and 7.889–8.846 s for 0.1.2. This is not
evidence of a reliable Objects365 speedup. Every run's complete precision,
recall, scores and stats hashes match the published pycocotools 2.0.11 reference.
[All samples, hashes and timing breakdowns](../bench/results/efficiency_v012.json).

## What changed

- Native loops build annotation indexes and prepare bbox metadata without
  repeated Python bytecode dispatch. The resulting dictionaries and lists remain
  ordinary mutable Python objects. Duplicate annotation IDs, insertion order,
  annotation identity, mapping overrides and Python numeric types are preserved.
- Validation no longer retains a detection-length image-ID list plus redundant
  sets. Temporary annotation-reference lists are released before result allocation.
- Workers copy completed category results into final C-order tensors immediately
  and release their local arrays. They no longer retain a second complete set of
  result tensors for a final assembly pass. A short lock protects publication;
  matching remains parallel and arithmetic/order are unchanged.

Garbage collection, Python input dictionaries and shared-host contention still
account for substantial costs. Reducing a native phase does not guarantee an
equal end-to-end reduction on every dataset. The report keeps per-phase and
whole-process measurements distinct.

## Measurement and reproduction

Measured on 2026-09-08, shared AMD EPYC 9554 host, Linux, Python 3.12.3,
NumPy 2.4.4, CPU affinity 0–1, two Rayon/OpenMP threads and one OpenBLAS thread.
There are three fresh processes per version and workload, with alternating
version order. Objects365's first pair and two later pairs were run in separate
batches; all samples are retained rather than selecting the fastest batch.

Time includes GT indexing, prediction-dictionary copies, result loading,
matching, accumulation and summarization. It excludes JSON parsing, inference
and output serialization. RSS is whole-process peak memory, including parsed
inputs and serialization. Use the same saved inputs and procedure from the
[0.1.1 report](efficiency.md#reproduce), comparing separately installed 0.1.1 and
0.1.2 packages. Run three separate processes per version, alternating their
order, and compare medians and full-array hashes.

[YOLO26n measurements](yolo26.md) independently compare 0.1.2 against pycocotools
and faster-coco-eval on newly generated real predictions. Historical 0.1.0
scaling plots remain labeled with their measured version.
