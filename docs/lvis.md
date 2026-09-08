# LVIS evaluation and framework integration

`COCOeval(..., lvis_style=True)` implements the federated LVIS bbox/segmentation
protocol. It uses the shared Rust matcher and accumulator; the protocol applies
verified-negative categories, ignores unverified predictions, and ignores
unmatched detections for non-exhaustively annotated categories. Its global
per-image top-300 cap is applied before category selection or filtering, with
stable score-tie ordering. LVIS GT `ignore` semantics are distinct from COCO's
crowd semantics and are enabled only in LVIS mode.

```python
from ultrafast_pycocotools import COCO, COCOeval

gt = COCO("lvis_val.json", verbose=False)  # also accepts a dict without copying it
dt = gt.loadRes("predictions.json")
evaluator = COCOeval(gt, dt, "bbox", lvis_style=True)
evaluator.run()
metrics = evaluator.stats_as_dict
print(metrics["AP"], metrics["APr"], metrics["AR@300"])
```

Use `"segm"` for mask evaluation. The input must provide each category's
`frequency` (`r`, `c`, `f`) and each selected image's `neg_category_ids` and
`not_exhaustive_category_ids`. Missing protocol metadata raises `ValueError`.
LVIS only supports category-aware bbox/segm evaluation; `params.maxDets` must
contain one positive image limit (default `[300]`). Ground-truth metadata is
preserved. This is a COCO-style interface to LVIS evaluation, not an import
replacement for the entire `lvis` package.

## Metric names

COCO preserves pycocotools' `stats` array order (12 detection metrics or 10
keypoint metrics). Pycocotools does not define a `stats_as_dict` naming standard.
For LVIS, the 13 primary names and array positions follow the official API:

`AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `APr`, `APc`, `APf`,
`AR@300`, `ARs@300`, `ARm@300`, `ARl@300`.

A custom image limit changes the `@300` suffix. Values use the 0–1 scale;
undefined categories/area groups use -1, as in the reference.

| Meaning | Conventional/LVIS name | Framework/legacy alias |
|---|---|---|
| Overall AP | `AP` | `AP_all` |
| AP at IoU 0.50 / 0.75 | `AP50`, `AP75` | `AP_50`, `AP_75` |
| AP by object size | `APs`, `APm`, `APl` | `AP_small`, `AP_medium`, `AP_large` |
| LVIS category frequency AP | `APr`, `APc`, `APf` | Same names |
| COCO recall at a detection cap | `AR@1`, `AR@10`, `AR@100` | Existing `AR_1`, `AR_10`, `AR_100` |

Aliases coexist in the dictionary; its length is not the `stats` array length.
Existing metric keys remain available.

The loader accepts both paths and in-memory COCO dictionaries. The constructor
accepts `lvis_style`, and the dictionary exposes the AP keys consumed by
[Ultralytics' detection validator](https://github.com/ultralytics/ultralytics/blob/88b030173dbbaff2f22925faffd0cb780500c1a9/ultralytics/models/yolo/detect/val.py#L590).
Tests exercise this evaluator call shape and metric lookup; they do not claim
an upstream Ultralytics merge or an end-to-end model validation test. Direct
imports are recommended in integrations that control backend selection.

## Official public example verification

The official API's public example contains 100 images, 977 GT annotations and
29,980 predictions. Both bbox and segmentation match the complete official
precision/recall arrays and all 13 summary metrics **byte for byte**. The
COCO-style result arrays retain a singleton maxDets axis; parity tests remove
only that axis before comparison. The official LVIS API does not expose the
COCO `scores` array, so no official LVIS scores-array comparison is claimed.
Synthetic regression cases also cover verified negatives, unverified categories,
non-exhaustive images, GT ignore flags, global caps, ties and category subsets.

| Official 100-image example | LVIS 0.5.3 total | Ultrafast 0.1.1 total | LVIS peak RSS | Ultrafast peak RSS |
|---|---:|---:|---:|---:|
| Bbox | 1.224 s | 0.238 s | 244.4 MiB | 240.1 MiB |
| Segmentation | 1.637 s | 0.275 s | 254.6 MiB | 243.7 MiB |

One fresh process per mode/backend, two CPU cores (affinity 4–5), same shared
host/software as the [efficiency report](efficiency.md). Reference and additional
backend runs were measured earlier in the same session. These totals include
JSON parsing, result loading and scoring, excluding imports and output
serialization. RSS includes whole-process imports and serialization. This timing
scope differs from the COCO table; the fixture is not full-LVIS scalability.
[Raw results and hashes](../bench/results/lvis_v011.json).

Faster-coco-eval 1.8.0 completes bbox in 0.464 s / 352.4 MiB, with numerical
agreement within absolute 1e-12 (rtol=0), but not byte identity. On this unchanged
public segmentation fixture it raises `IndexError: unordered_map::at` in
`calculateRleForAllAnnotations`. The failed run has no reported timing or metric
comparison; no third-party package patch was applied.

## Reproduce locally or in CI

```bash
python -m pip install ".[test,lvis-test]"
python bench/fetch_lvis_fixture.py
python -m pytest -q tests/test_lvis.py tests/test_result_loading.py

python bench/compare_lvis.py --backend lvis --iou-type bbox --out bench/out/lvis-reference
python bench/compare_lvis.py --backend ultrafast --iou-type bbox --out bench/out/lvis-ultrafast
```

Repeat with `--iou-type segm` and new output directories for masks. Each output
retains complete arrays, their SHA-256 hashes, input hashes, named metrics and
resource measurements. Compare `array_sha256` and input hashes in the two result
JSONs. For the optional third backend (Python 3.10+), install
`faster-coco-eval==1.8.0` and use `--backend faster-coco-eval`.

Fixture downloads are pinned to official source commit
`7d7f07def11da91f8b2710ce352c62a78fd5a7ad` and validated with
[recorded SHA-256 hashes](../bench/results/lvis_fixture_sources.json). No images
or GPUs are needed. Raw JSON files stay in ignored `bench/data`. Every
Windows/Ubuntu/macOS Python CI job downloads and verifies the fixture and runs
these reference tests. Official LVIS 0.5.3 uses the removed `np.float` spelling;
only tests/reference benchmark processes alias it to the identical built-in
`float` type. The production library does not modify NumPy.

Protocol references: official [evaluation](https://github.com/lvis-dataset/lvis-api/blob/7d7f07def11da91f8b2710ce352c62a78fd5a7ad/lvis/eval.py)
and [result loading](https://github.com/lvis-dataset/lvis-api/blob/7d7f07def11da91f8b2710ce352c62a78fd5a7ad/lvis/results.py).
