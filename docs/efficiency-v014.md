# Rust ownership and buffer improvements in 0.1.4

Version 0.1.4 further reduces temporary storage in the Rust evaluator. Normal
API calls use the changes automatically; complete output arrays retain their
pycocotools-compatible float64 values and ordering.

## Repeated measurements

| Input | 0.1.3 time | 0.1.4 time | 0.1.3 peak RSS | 0.1.4 peak RSS |
|---|---:|---:|---:|---:|
| YOLO26n / files | 0.662 s | **0.628 s** | 242.1 MiB | **223.2 MiB** |
| Objects365 / files | 3.612 s | **3.580 s** | 834.3 MiB | **807.1 MiB** |
| YOLO26n / in-memory | 1.886 s | **1.900 s** | 745.4 MiB | **740.9 MiB** |

YOLO26n file evaluation uses **5.3% less time and 7.8% less peak RSS** in this
batch. Time ranges are 0.654–0.663 s for 0.1.3 and 0.627–0.650 s for 0.1.4.
Objects365 peak RSS falls **3.3%**. Its 0.9% median time reduction is within
measurement variation: 3.545–3.632 s versus 3.552–3.686 s. This does not establish
an Objects365 speedup. In-memory YOLO26n time is essentially unchanged (overlapping
ranges), with a small **0.6%** reduction in whole-process peak RSS.

The earlier 0.1.3 report used a different measurement batch. Use the paired
0.1.3 baseline here to assess this change; comparing only with its historical
0.805 s figure would overstate the improvement. Shared-host measurements and
three samples do not establish a general performance guarantee.

## Implementation

- **Borrowed GT indices:** per-image work borrows slices from the sparse grouping
  index. Rust lifetimes tie these views to the evaluator, including through Rayon
  workers, so GT indices do not need a new owned vector for each image/category.
- **Borrowed bbox columns:** IoU reads indexed coordinates and crowd flags from
  the existing columns. It no longer gathers three temporary vectors per image.
  The contiguous and indexed paths share the same arithmetic helper.
- **Smaller detection records:** file-backed detections retain only independent
  fields. Sequential IDs, bbox-derived areas and false crowd flags are recreated
  exactly when constructing engine instances. Storage falls from **80 to 56 bytes
  per detection**, excluding its JSON snapshot. Ground truth retains its explicit
  IDs, areas and crowd flags. Public annotation access still returns the same
  derived fields.
- **Direct numeric deserialization:** a Serde visitor converts bbox coordinates
  directly to f64 instead of constructing tagged `serde_json::Number` values.
  Large integer literals still fall back to ordinary loading, preserving Python
  integer behavior. Round-trip float parsing remains enabled.
- **Borrowed accumulation order:** the largest maxDets limit reads the existing
  sorted slice. Only smaller limits need a filtered buffer.
- **Smaller recall workspace:** a forward pass records the first eligible index
  for each requested recall threshold. After the unchanged backward precision
  envelope, these indices sample the outputs. Scratch storage for this step is
  proportional to the recall grid size, rather than the detection count.
- **Stable binary partition:** GT ignore ordering uses two iterator passes instead
  of a comparison sort. Both partitions retain annotation order.

Score sorting remains stable. An explored in-place grouping sort with explicit
original-position tie breakers increased grouping time and was discarded.
Fewer allocations alone are not evidence of lower end-to-end latency.

## Measurement method

The comparison uses 0.1.3 and 0.1.4 in isolated installations on the same shared
AMD EPYC 9554 host, CPU affinity 4–5, two Rayon/OpenMP threads, one OpenBLAS thread,
Python 3.12.3 and NumPy 2.4.4. Each version/workload uses three fresh processes,
with alternating version order and no filesystem cache eviction.

File-input time includes JSON loading, native extraction/indexing, evaluation,
accumulation and summarization. In-memory time excludes JSON parsing and includes
prediction dictionary copies. Neither includes inference or output serialization.
Memory is whole-process peak RSS, including serialization. Compare versions
within each input mode; do not compare the two modes as if their timing scopes
were identical. Native phase counters are recorded separately from wall time.

All samples and input/output hashes are retained in
[efficiency_v014.json](../bench/results/efficiency_v014.json). Full precision,
recall, scores and stats hashes are checked against the published pycocotools
2.0.11 reference for each input. The [0.1.3 report](efficiency-v013.md) explains
the file-input API, materialization behavior and limits of theoretical memory
claims; those qualifications still apply.

## Reproduce and validate

Use the [0.1.3 file-input reproduction procedure](efficiency-v013.md#reproduce),
comparing separately installed 0.1.3 and 0.1.4 packages. The same command also
accepts `--input-mode in-memory` for the materialized input path. Use a new output
directory and a fresh process for each sample.

```bash
python bench/compare_saved_predictions.py \
  --backend ultrafast --input-mode files \
  --gt instances_val2017.json --pred predictions.json \
  --out bench/out/rust-run-1 --repeats 1
```

Regression checks cover indexed bbox permutations/repeated indices/crowds,
filtered detection IDs and derived fields, duplicate/custom recall thresholds,
and large integer fallback. The existing suite additionally covers COCO modes,
LVIS, public mutations, diagnostic views and multiple thread counts. CI builds
and tests the native package on Windows, Ubuntu and macOS.
