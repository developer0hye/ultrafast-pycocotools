# Evaluation input boundary fixes (0.1.10)

RF-DETR integration against upstream `51c3ed1f14d5dbcd2b74d3af12db0ff3c324253d` exposed a failure in joint bbox/mask evaluation with empty images. TorchMetrics can retain an image containing only its ID when that side has no masks. Version 0.1.9 tried to read dimensions for every image and raised `KeyError: height`. The evaluator now skips dimensionless images without selected annotations and collects dimensions from the other side when available. A selected nonempty image still requires dimensions. Segmentation arrays match pycocotools byte for byte; boundary arrays match a fully dimensioned counterpart.

The expert's Python ownership audit also led to a reproduced native crash: a coordinate object's `__float__` can clear its containing list during bbox or polygon extraction. Holding the GIL does not prevent this reentrant mutation. Both 0.1.9 reproducers terminated with SIGSEGV on macOS/CPython 3.12. Checked list access now raises `IndexError` instead. Each regression runs in a subprocess so a future native crash cannot terminate the test runner.

The extension explicitly requires the GIL until its Python extraction and blocking rasterization joins have been audited for free-threaded execution. This is not a claim of free-threaded support or of complete input-boundary security coverage.

Validation of the source candidate on macOS, CPython 3.12, NumPy 2.4.4:

- Library: 310 passed, 6 optional skips.
- RF-DETR new backend tests: 7 passed, 1 fixture doctest skipped; bbox, segmentation, combined evaluation, caps 100/500, tied scores, crowd annotations, empty images, incremental updates, pickle/reset, and complete precision/recall/score arrays.
- These are correctness fixes; no speedup is claimed.

Reproduce the focused regressions with `python -m pytest -q tests/test_empty_mask_images.py tests/test_reentrant_geometry.py`. CI tests the complete library suite and installed release wheels separately.
