# Benchmarks and validation reports

The [project README](../../README.md) shows one headline measurement: the
current release on a single host. Everything else is collected here. Each
report states its own version, host, inputs, method and raw evidence, so
results from different reports should not be combined into one comparison.

## Current headline

- [COCO val2017 on i5-10400, 0.1.11](i5-10400-v0111.md): pycocotools,
  faster-coco-eval, hotcoco and ultrafast on identical YOLO26n bbox,
  segmentation and pose predictions. This is the source of the README table.

## Comparisons with other evaluators

- [hotcoco 1.0.0 comparison (0.1.10)](../benchmark-hotcoco.md): Apple M2 and
  i5-10400, file and list inputs, pool sizes 1 and 2, sampled-score differences.
- [Closing the hotcoco performance gap (0.1.11 source build)](../hotcoco-performance-goal.md)
- [faster-coco-eval comparison (0.1.0)](../faster-coco-eval.md): COCO / YOLO11m
  and Objects365 synthetic workloads.
- [Scaling with input size (0.1.0)](../scaling.md): nested Objects365 subsets up
  to 80,000 images.

## Framework integrations

- [RF-DETR](../rfdetr.md): metric replay with RF-DETR Nano predictions.
- [Ultralytics PR #26101 validation (0.1.7)](../ultralytics-pr26101-validation.md)
  · [Reviewer checklist](../ultralytics-review-checklist.md)
  · [Initial evaluator replacement (0.1.6)](../ultralytics.md)
- [D-FINE-seg](../dfine-seg.md) · [Validator benchmark (0.1.6)](../benchmark-dfine-seg.md)
- [YOLO26n real predictions (0.1.2)](../yolo26.md)
- [LVIS evaluation and integration](../lvis.md)

## Other hosts (synthetic workload)

- [Apple M2 desktop (0.1.6)](../benchmark-apple-m2.md)
- [i5-10400 / RTX 3070 server (0.1.6)](../benchmark-rtx3070.md)

## Per-release optimization notes (historical)

These notes measure each change against the previous version. Later releases
include them, so their absolute numbers are superseded by the headline above.

| Version | Report |
| --- | --- |
| 0.1.1 | [Result loading without redundant polygons](../efficiency.md) |
| 0.1.2 | [Native index and metadata loops](../efficiency-v012.md) |
| 0.1.3 | [Compact bbox file loading](../efficiency-v013.md) |
| 0.1.4 | [Rust ownership and buffer improvements](../efficiency-v014.md) |
| 0.1.7 | [Mask encoding optimization](../mask-encoding-optimization.md) |
| 0.1.8 | [Native segmentation/keypoint file evaluation](../nonbbox-optimization.md) · [Mask buffer allocation](../nonbbox-buffer-optimization.md) |
| 0.1.9 | [maxDets storage and LVIS optimization](../maxdets-lvis-optimization.md) |
| 0.1.10 | [Evaluation input boundary fixes](../evaluation-boundary-fixes.md) |
| 0.1.11 | [Pose input storage](../pose-input-optimization.md) · [Parallel pose loading](../pose-parallel-loading.md) |
| unreleased | [Per-task bottleneck optimization](../task-bottleneck-optimization.md) · [NumPy result arrays](../numpy-results.md) |

## Reproducing

- [Reproduction guide](../reproducibility.md): synthetic check without
  downloads, Objects365 and public detector predictions, expected hashes.
- [Implementation notes](../implementation-notes.md): design decisions and
  older measurements with different recipes and timing scopes.
