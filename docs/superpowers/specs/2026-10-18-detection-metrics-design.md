# Detection metrics: precision, recall and F1 per batch and accumulated

Design for [issue #28](https://github.com/developer0hye/ultrafast-pycocotools/issues/28)
("Expose per Batch Metrics": recall, precision and F1 for a batch of
predictions and ground truth at a given IoU threshold).

Status: design, 2026-10-18. Delivered as two pull requests (PR 1, PR 2 below).
Accumulated AP across batches is a separate, later design (see
[Out of scope](#out-of-scope)).

## Goals

1. Precision, recall and F1 from the same matching as AP, so they never
   contradict the AP printed beside them.
2. Inputs as in-memory arrays or tensors (one dict per image) **and** as COCO
   dictionaries, files or `COCO` objects.
3. A result for each batch **and** a result accumulated over all batches,
   without retaining boxes or masks between batches.
4. A fixed score threshold **and** the score threshold that maximizes F1.
5. Micro and macro averages, per-category values, and a class-agnostic mode.
6. COCO conventions (crowd ignore, area ranges, `maxDets` truncation) available
   but optional.
7. 0.1.x compatible: new methods and a new module only. Nothing existing
   changes its defaults, signature or return type.

## Measured motivation

`bench`-style probe on `tests/data/coco_subset` (16 images, 133 GT, 136
detections), 12-core Linux desktop, `maturin develop --release`, median of 50–100
runs:

| Path | Time |
| --- | ---: |
| pycocotools `evaluate()` + `accumulate()` | ~77 ms |
| `COCOeval.evaluate()` with default `Params` | 4–9 ms |
| `COCO(dict)` + `loadRes(list)` (part of the above) | ~0.25 ms |
| `COCOeval.evaluate()` with one IoU threshold, area `all`, one `maxDets` | 0.4–0.6 ms |

The fixed per-batch cost is allocating and filling the full
`T x R x K x A x M` precision/recall/scores tensors, not building annotation
dictionaries. Per-batch metrics therefore evaluate a narrowed `Params`
(the requested IoU thresholds, one area range, one `maxDets`). This needs no
new loader.

Two invariants were checked on the same fixture and become tests:

- Recall computed as `tp / (tp + fn)` with no score threshold is bit-identical
  to `eval['recall'][t, :, a, m]` (53 categories with ground truth).
- Counting only detections with `score >= s` after matching gives the same
  TP/FP/FN as matching only those detections (six thresholds).

## Definitions

Everything is counted from the engine's per-detection outcome at each IoU
threshold, which is exactly what accumulation reads: `IGNORED`,
`TRUE_POSITIVE` or `FALSE_POSITIVE`.

- **TP**: detections with outcome `TRUE_POSITIVE` and `score >= score_thr`.
- **FP**: detections with outcome `FALSE_POSITIVE` and `score >= score_thr`.
- **FN**: non-ignored ground truth minus TP (`npig - tp`). Each true positive
  consumes exactly one non-ignored ground truth, so this equals the count of
  non-ignored ground truth not matched by a kept detection.
- Ignored detections (matched to a crowd region, or unmatched and outside the
  area range) and ignored ground truth (crowd, or outside the area range) are
  in none of the three counts, as in AP.
- **Zero denominators give 0**: precision is 0 when `tp + fp == 0`, recall is 0
  when `tp + fn == 0`, F1 is 0 when `precision + recall == 0`.
- `precision = tp / (tp + fp)`, `recall = tp / (tp + fn)`,
  `f1 = 2 * precision * recall / (precision + recall)`, each as one `f64`
  operation sequence on integer counts. `EPS` is not added: it belongs to the
  reference PR curve, and adding it here would make `precision` disagree with
  `tp / (tp + fp)` for no reason.

### Why filtering by score after matching is exact

Matching processes detections of an (image, category) in stable descending
score order (`cmp_desc_score`), and each detection's match depends only on the
detections before it. The detections with `score >= s` are a prefix of that
order: ties at `s` are all kept and are contiguous, and NaN scores sort last and
fail `>= s`. A prefix is matched identically whether or not the rest of the
list exists, so filtering after matching equals matching only the kept
detections. `maxDets` truncation is also a prefix, and the intersection of two
prefixes is the shorter one, so the order of truncation and filtering does not
matter. This argument goes in a comment next to the filter, with the test
described under [Tests](#tests).

### Averages

- **Per category**: the counts and ratios above for every category that has
  non-ignored ground truth or a non-ignored detection in the evaluated data.
- **Micro**: sum TP, FP and FN over all categories, including categories with
  only false positives, then apply the ratios. Every false positive is counted.
- **Macro**: unweighted mean of the per-category precision, recall and F1 over
  categories with **at least one non-ignored ground truth**.
  - This matches COCO, where a category without ground truth has recall `-1`
    and is left out of AR, and Ultralytics, which averages over the classes in
    the targets. Macro recall with no score threshold therefore equals the
    mean of the valid `eval['recall']` entries.
  - False-positive-only categories still lower micro precision and are reported
    per category, so they are not hidden.
  - Macro F1 is the mean of per-category F1, not the harmonic mean of macro
    precision and macro recall (the scikit-learn and Ultralytics definition).
  - With no category having ground truth, macro values are 0.
- **Class-agnostic** (`class_agnostic=True` / `params.useCats = 0`): one
  category, so micro and macro coincide.

### Score threshold

- `score_thr=None`: every detection up to `maxDets` (the last point of the
  unsmoothed PR curve).
- `score_thr=<float>`: keep `score >= score_thr`.
- `score_thr="best_f1"`: choose one threshold per IoU threshold and report every
  value (per category, micro, macro) at it.
  - `optimize="macro"` (default) maximizes macro F1; `optimize="micro"`
    maximizes micro F1. One shared threshold is the operating point a deployed
    model uses; per-category thresholds are not offered.
  - Candidates are the distinct scores of non-ignored detections. A cut is never
    placed between tied scores.
  - Ties in F1 go to the **highest** threshold: the same F1 with fewer
    detections. With no non-ignored detections the threshold is `+inf`.
  - The sweep computes counts with cumulative sums and the ratios with the same
    function as a fixed threshold, so `score_thr="best_f1"` and
    `score_thr=<the chosen value>` give bit-identical results.
  - Unlike Ultralytics, F1 is not smoothed or interpolated on a confidence
    grid; values are exact at real score cuts.

## Components

### Rust: `Evaluator::outcomes` (`ufcoco-core`) and its binding (`ufcoco-py`)

A sibling of `Evaluator::per_instance`, built from the same
`prepare_category` / `evaluate_img` calls:

```rust
pub fn outcomes(&self, a_idx: usize, max_det: usize) -> Outcomes

pub struct Outcomes {
    /// Per kept detection, in category then image order then score order:
    pub cat_slot: Vec<u32>,
    pub img_id: Vec<i64>,
    pub score: Vec<f64>,
    /// D x T, detection-major: IGNORED / TRUE_POSITIVE / FALSE_POSITIVE.
    pub outcome: Vec<u8>,
    /// Per category slot: non-ignored ground truth in this area range.
    pub npig: Vec<u64>,
}
```

- Detections past `max_det` per (image, category) are omitted, as in
  `per_instance`.
- `max_det` must be at most `params.max_dets.last()` (the engine truncates to
  that before matching); the Python side validates it.
- The binding returns NumPy arrays (`outcome` as a `(D, T)` `uint8` array) and
  the outcome constants, with the GIL released.
- Unlike calling `per_instance` once per threshold, this matches once for all
  IoU thresholds.

### Python: `ultrafast_pycocotools/_detection_metrics.py`

A private module with one function used by both public entry points:

```python
def summarize_outcomes(cat_slot, score, outcome, npig, cat_ids, *,
                       score_thr, optimize) -> dict
```

It takes the outcome arrays (possibly concatenated over batches), applies the
score threshold or sweep, and builds the result described below. All
aggregation is NumPy (`np.bincount` over category slots, a stable `argsort`
for the sweep); no Python loop over detections.

### PR 1: `COCOeval.detection_metrics`

```python
ev.detection_metrics(
    iou_thr=0.5,            # float, or a sequence of values in params.iouThrs
    score_thr=None,         # None | float | "best_f1"
    optimize="macro",       # "macro" | "micro", only for "best_f1"
    area="all",
    max_dets=None,          # None = params.maxDets[-1]
)
```

- Requires `evaluate()`; calls `self._engine.outcomes(aind, mdet)`.
- An `iou_thr` that is not in `params.iouThrs` raises `ValueError` naming the
  grid and saying to set `params.iouThrs` before `evaluate()`. The lookup uses
  the same `np.isclose` rule as the other extensions.
- An unknown `area`, a `max_dets` above `params.maxDets[-1]`, an unknown
  `optimize` or a string `score_thr` other than `"best_f1"` raise `ValueError`.
- `confusion_matrix`, `matches` and `per_instance` are unchanged.

### PR 2: `ultrafast_pycocotools.metrics.DetectionMetrics`

```python
from ultrafast_pycocotools.metrics import DetectionMetrics

m = DetectionMetrics(
    iou_type="bbox",        # "bbox" | "segm" | "keypoints" (COCO input only)
    iou_thrs=(0.5,),        # caller's values, passed to the engine as given
    score_thr=None,         # None | float | "best_f1"
    optimize="macro",
    class_agnostic=False,
    box_format="xyxy",      # array input only: "xyxy" | "xywh"
    area_range=None,        # None = no area filtering, or (lo, hi)
    max_dets=None,          # None = no truncation, or an int
)

batch = m.update(preds, targets)     # this batch's metrics
batch = m.update_coco(gt, dt)        # same, from COCO data
total = m.compute()                  # accumulated over every update so far
m.reset()
```

Array input: `preds` and `targets` are equal-length lists with one dict per
image.

| Key | preds | targets | Shape |
| --- | --- | --- | --- |
| `boxes` | required for bbox | required for bbox | `(N, 4)` |
| `scores` | required | – | `(N,)` |
| `labels` | required | required | `(N,)` integer |
| `masks` | required for segm | required for segm | `(N, H, W)` bool |
| `iscrowd` | – | optional, default 0 | `(N,)` |
| `area` | – | optional, default box or mask area | `(N,)` |
| `image_id` | optional | optional | scalar |

- Values may be NumPy arrays, lists, or objects with `__array__`. Objects with
  `detach` (PyTorch tensors) are converted with `.detach().cpu().numpy()` by
  duck typing; `torch` is never imported and not a dependency.
- `xyxy` boxes are converted to `[x1, y1, x2 - x1, y2 - y1]`. `xywh` boxes are
  passed through unchanged, so they evaluate exactly like COCO input.
- Image ids: if no dict has `image_id`, ids are assigned from a counter that
  continues across updates. Mixing present and absent ids in one call is an
  error. An id seen in an earlier update raises `ValueError`, because the same
  image counted twice changes the accumulated metrics silently.
- Masks are encoded with `mask.encode` into RLE. The image's height and width
  come from the mask shape (targets first, then predictions); masks of
  different sizes within one image are an error.
- `keypoints` is supported through `update_coco` only.

COCO input: `update_coco(gt, dt)` takes `gt` as a `COCO` object, dictionary or
path, and `dt` as anything `COCO.loadRes` accepts. Its image ids join the same
duplicate check.

Per update, internally:

1. Build GT and DT handles (array input: a GT dictionary and an `Nx7` result
   array for bbox, which takes the existing `load_compact_array` route).
2. Evaluate with narrowed `Params`: `imgIds` of this batch, `catIds` = labels
   seen in the batch, `iouThrs = iou_thrs`, one area range (`area_range`, or
   `[0, inf]` when `None`), `maxDets = [max_dets]` or the batch's detection
   count (at least 1) when `None`, `useCats = not class_agnostic`.
3. Call `outcomes(0, max_det)`; drop detections ignored at every threshold;
   map category slots to global category ids.
4. Append `(category id, score, outcome rows)` to the accumulated store and add
   `npig` per category id.
5. Return `summarize_outcomes` over this batch's arrays.

`compute()` runs `summarize_outcomes` over the concatenated store. Stored
state is about `8 + 8 + T` bytes per kept detection (for example, 26 B at ten
thresholds, so the README benchmark's 733,070 boxes need under 20 MB), and no
geometry.

Area semantics follow COCO: with `area_range`, ground truth outside it is
ignored, and an unmatched detection outside it is ignored. GT `area` defaults
to the box area (`w * h`) or the mask pixel count.

Because matching is per image and the score cut is a prefix, the accumulated
counts at a fixed threshold equal one evaluation over all images, and the
`best_f1` sweep over the concatenated store equals the sweep over that single
evaluation.

### Convenience function

`metrics.detection_metrics(preds, targets, **config)` returns
`DetectionMetrics(**config).update(preds, targets)`.

## Result format

Plain dictionaries of NumPy arrays, like the other extensions:

```python
{
    "iou_thrs":    (T,) float64,
    "category_ids": (K,) int64,          # [-1] when class-agnostic
    "score_thr":   (T,) float64,          # chosen, given, or -inf for None
    "per_category": {"tp", "fp", "fn": (T, K) int64,
                     "precision", "recall", "f1": (T, K) float64},
    "micro": {"tp", "fp", "fn": (T,) int64,
              "precision", "recall", "f1": (T,) float64},
    "macro": {"precision", "recall", "f1": (T,) float64},
}
```

`COCOeval.detection_metrics` with a scalar `iou_thr` drops the `T` axis.

## Errors

- Input validation raises `ValueError` with the offending key and image index:
  mismatched lengths, wrong box shape, `scores` missing, an unknown
  `box_format`, `masks` missing for `segm`, keypoints through array input.
- Empty images (no predictions, no targets, or both) are valid.
- A NaN score is allowed; it sorts last and is never kept by a finite
  threshold, as in matching.

## Tests

PR 1 (`tests/test_extensions.py` or a new `tests/test_detection_metrics.py`):

- Per-category recall with `score_thr=None` is bit-identical to
  `eval['recall'][t, k, a, m]` for every valid `k`, for bbox, segm and
  keypoints, every area range and every `maxDets`.
- Prefix argument: for several thresholds (including one equal to a tied score),
  counts from `score_thr=s` equal counts from a separate evaluation of only the
  detections with `score >= s`. Use the synthetic dataset from
  `bench/make_dataset.py`, which has crowds, area boundaries and tied scores.
- `best_f1`: the result is bit-identical to calling with the chosen threshold,
  and no candidate threshold gives a higher F1.
- Micro sums equal the per-category sums; macro means equal the mean over
  categories with ground truth; the zero-denominator rules.
- `confusion_matrix` TP/FP/FN agree with `detection_metrics` counts.
- The error messages above.
- `cargo test -p ufcoco-core`: `outcomes` agrees with `per_instance` on the
  existing hand-built scenarios.

PR 2 (`tests/test_detection_metrics.py`):

- Split the fixture into batches. Accumulated `compute()` is bit-identical to
  `COCOeval.detection_metrics` on the whole fixture with the matching
  `Params`, for fixed thresholds, `best_f1` (micro and macro) and the
  class-agnostic mode.
- Each `update()` result is bit-identical to `COCOeval.detection_metrics` on
  that batch alone.
- Array input with `xywh` gives the same result as COCO input of the same data;
  `xyxy` input gives the same result as `xywh` input of the converted boxes.
- Segm array input (masks) agrees with COCO RLE input.
- Duplicate image ids raise; automatic ids continue across updates; empty
  images and empty batches work; `reset()` clears state.
- A PyTorch tensor test runs when `torch` is installed and is skipped otherwise.

## Documentation

- README "Extras": one example of each entry point.
- `docs/implementation-notes.md` "Extensions": definitions, the averaging and
  zero rules, the prefix argument and the memory cost.
- A short `docs/` report with per-batch timings (host, commit, build type,
  medians, one and two threads), linked from `docs/benchmarks/README.md`, for
  any speed claim made in the README.

## Compatibility

- New method, new module, new Rust method. No existing default, signature or
  return type changes; downstream integrations are unaffected. The upstream
  integration jobs still run on both pull requests.
- No new dependency.

## Out of scope

- **Accumulated AP across batches.** It can be made bit-identical to
  pycocotools on the full dataset by also storing image order and rebuilding
  the reference tie order at the end. Its exactness argument is larger than
  the ones here, and the issue asks for precision, recall and F1, so it is a
  separate design after PR 2.
- A dedicated Rust path that builds instances from NumPy columns without the
  `COCO` layer. Revisit only if PR 2's profile shows the `COCO` layer, not
  matching, dominating per-batch time.
- Per-category best-F1 thresholds and smoothed F1 curves.
