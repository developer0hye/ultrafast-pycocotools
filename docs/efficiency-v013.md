# Compact file evaluation in 0.1.3

Version 0.1.3 removes eager Python annotation allocation from the normal bbox
file-input path. On three tested workloads, evaluation becomes **2.6–2.9× faster**
and peak process memory falls **56–62%** versus 0.1.2. Both improvements apply
together, without a performance flag.

| Same files, package defaults | 0.1.2 time | 0.1.3 time | 0.1.2 peak RSS | 0.1.3 peak RSS |
|---|---:|---:|---:|---:|
| COCO val2017 / YOLO26n, 596,202 predictions | 2.108 s | **0.805 s** | 631.3 MiB | **242.4 MiB** |
| COCO val2017 / YOLO11m, 431,145 predictions | 1.544 s | **0.525 s** | 509.6 MiB | **201.2 MiB** |
| Objects365 v2 / synthetic predictions | 10.359 s | **3.768 s** | 1,887.6 MiB | **833.2 MiB** |

These are medians of three fresh processes per version and workload, alternating
version order. **Time includes JSON loading** as well as matching, accumulation
and summarization. It excludes inference and result serialization. RSS is the
whole-process peak, including serialization. Filesystem caches were not evicted.
All measurements used the same shared AMD EPYC 9554 host, CPU affinity 4–5,
two Rayon/OpenMP threads, one OpenBLAS thread, Python 3.12.3 and NumPy 2.4.4.

Every run's complete precision, recall, scores and stats hashes match the
published pycocotools 2.0.11 reference for the same inputs. All samples, input
hashes, phase times, output sizes and retained storage sizes are in
[efficiency_v013.json](../bench/results/efficiency_v013.json). Timing ranges are
retained: for example, YOLO26n ranges from 1.956–3.132 s in 0.1.2 and
0.629–0.818 s in 0.1.3. Shared-host timing varies; the report does not select
only the fastest run.

## YOLO26n against other backends

All three backends load the same files directly with package defaults. Each has
three fresh processes, the same CPU/thread limits and the same timing scope.
The other-backend runs followed the version comparison in a separate batch;
shared-host timing remains a limitation.

| Backend | Median time, JSON included | Median peak RSS |
|---|---:|---:|
| pycocotools 2.0.11 | 36.945 s | 1,485.1 MiB |
| faster-coco-eval 1.8.0 | 7.333 s | 1,535.5 MiB |
| ultrafast-pycocotools 0.1.3 | **0.805 s** | **242.4 MiB** |

This is **45.9× faster and 83.7% less peak RSS versus pycocotools**, or
**9.1× faster and 84.2% less peak RSS versus faster-coco-eval**. Ultrafast's four
output arrays match pycocotools byte for byte. Faster-coco-eval's precision
array differs by at most 2.22e-16; recall, scores and stats match byte for byte.
Every backend's arrays also match its previously recorded YOLO26n hashes.

![File-input evaluation medians and full run ranges](assets/compact-yolo26n.svg)

Regenerate the PNG, SVG and PDF with `python bench/plot_compact.py`. Both axes
start at zero; whiskers show each backend's minimum and maximum, not a claimed
confidence interval. These are one fixed workload's measurements, not evidence
of an exponential-versus-linear complexity change.

## What changed

1. File loading keeps an owned JSON snapshot and parses bbox fields directly
   into native columns. Unknown annotation fields remain available in the
   snapshot. Evaluation no longer needs hundreds of thousands of Python
   dictionaries, bbox lists and boxed scalar objects.
2. Full-selection evaluation shares immutable bbox coordinates with the input
   handle. It does not copy the coordinate column into a second engine buffer.
   Image/category subsets use filtered columns with the same ordering.
3. Public annotation access materializes ordinary mutable Python objects once.
   The handle then evaluates from those public objects, so edits are observed.
   Scalar arithmetic, stable sorting and output dtypes remain unchanged.

## API behavior and scope

```python
from ultrafast_pycocotools import COCO, COCOeval

gt = COCO("instances_val2017.json")
dt = gt.loadRes("predictions.json")
evaluator = COCOeval(gt, dt, "bbox")
evaluator.run()
```

This ordinary file-input sequence enables compact evaluation automatically.
Reading or assigning `dataset`, `anns`, `imgToAnns` or `catToImgs` materializes
annotations and disables compact evaluation for that handle. Annotation queries,
mask/plot helpers and diagnostic views may therefore trigger materialization.
Default `getImgIds()` and `getCatIds()` only need metadata. Pickle/deepcopy also
materialize annotations and preserve public index identity. Changing or deleting
the original file cannot change the handle: later reads use its owned snapshot.

Existing dict/list inputs keep their mutable representation. In a separate
three-process YOLO26n check, median scoring times were 1.828 s for 0.1.2 and
1.869 s for 0.1.3, with overlapping ranges and essentially unchanged peak RSS.
**This release does not demonstrate an in-memory speed or memory improvement.**
Those measurements exclude JSON parsing and must not be mixed with the table
above or the [historical 0.1.2 benchmark](yolo26.md).

Compact evaluation is currently limited to bbox evaluation using the standard
COCO/COCOeval classes. Segmentation, keypoints, LVIS and subclasses use the
existing materialized path. Unsupported JSON schemas and duplicate GT IDs fall
back to ordinary loading. Explicit `derive_segmentation=True` retains its
existing eager polygon behavior. Tests cover mixed file/dict inputs, filters,
crowds, tied scores, repeated evaluation, public mutations and snapshot lifetime.

## What the lower bounds actually say

For a previously unseen JSON file of B bytes, general exact validation must
inspect the input: reading/parsing is at least proportional to B. It cannot be
made constant-time by changing the evaluator. Stable ranking, IoU computation
and matching add input-dependent work; this report does not claim a universal
optimal matching algorithm or a hardware throughput limit.

The public dense float64 arrays also require storage. With T IoU thresholds,
R recall thresholds, K categories, A area ranges, M detection limits and S summary
values, their payload is `8 * (2*T*R*K*A*M + T*K*A*M + S)` bytes. Standard COCO
outputs require **15,590,496 bytes (14.87 MiB)**. That is only an output-storage
lower bound under the current API, not a lower bound equal to process RSS.

For YOLO26n, the current representation additionally retains 88,327,577 bytes
of JSON snapshots and 50,638,640 bytes of native annotation columns. Keeping
unknown fields and making later mutable API access independent of file changes
has a real cost. Metadata, engine scalar columns, sorting/matching workspace,
the Python runtime and allocator behavior add to peak RSS. Input compression,
a different interchange format or a narrower output/API contract could change
these costs and may introduce new time/memory tradeoffs.

**These measurements establish a substantial improvement, not attainment of an
absolute theoretical minimum.** The retained-byte breakdown identifies the
remaining costs without treating all of them as unavoidable mathematical limits.

## Reproduce

Install each revision into its own environment and use identical saved inputs.
The [YOLO26n instructions](yolo26.md#reproduce) generate the public-model
predictions; the [general reproduction guide](reproducibility.md) describes
COCO/Objects365 inputs and the deterministic synthetic prediction recipe.

```bash
python bench/compare_saved_predictions.py \
  --backend ultrafast --input-mode files \
  --gt instances_val2017.json --pred predictions.json \
  --out bench/out/my-file-run --repeats 1
```

Run three fresh processes per version with distinct output directories,
alternating version order. Pin the same CPU affinity and thread counts for all
backends. To reproduce full-array agreement on generated data without downloading
images or annotations:

```bash
python bench/reproduce.py quick --input-mode files --out bench/out/quick-files
```

Add `--include-faster` after installing faster-coco-eval 1.8.0 to compare all
three backends. This command exercises file loading on every backend; the
runner's default `--input-mode in-memory` remains unchanged for historical
reproduction. CI checks both input modes on Windows, Ubuntu and macOS.
