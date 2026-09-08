# Ultralytics PR #26101 review evidence

This checklist tracks every acceptance requirement in Glenn Jocher's
[review](https://github.com/ultralytics/ultralytics/pull/26101#pullrequestreview-5142733326)
and [additional comment](https://github.com/ultralytics/ultralytics/pull/26101#issuecomment-5586442824).
A checked item requires saved, reproducible evidence; pending items are not claims of completion.

| Done | Requirement or concern | Disposition | Evidence |
| --- | --- | --- | --- |
| [x] | Missing CPython 3.8 macOS arm64 wheel | Valid finding, fixed in public 0.1.7. Native public binary installation: 249 passed, 17 optional skips. All 35 wheels and the sdist match publication hashes. | [Release verification](../bench/results/release_v017_verification.json) |
| [x] | Dependency floor must require the fixed distribution | Every versioned production/CI requirement is >=0.1.7; pushed to existing PR head 0451a3c9. | [PR commit](https://github.com/ultralytics/ultralytics/commit/0451a3c9282461e74cbaeeb5d03991be9162cdaf) |
| [x] | Prediction-derived GT is not accuracy evidence | Valid finding. Replaced it with original independent COCO train2017 annotations for the eight COCO8/pose images. | [Fixture provenance](https://github.com/ultralytics/ultralytics/blob/0451a3c9282461e74cbaeeb5d03991be9162cdaf/tests/fixtures/README.md) |
| [x] | Preserve useful synthetic regression cases | Five bbox/mask/keypoints/LVIS edge cases retained alongside three real-prediction cases. Normal CI now installs both packages so the cases execute. | [Tests](https://github.com/ultralytics/ultralytics/blob/0451a3c9282461e74cbaeeb5d03991be9162cdaf/tests/test_integrations.py) |
| [x] | Real detection accuracy | 5,000 images / 733,070 predictions; all 8 full runs return exactly identical metrics and fitness. Main AP 0.4081073392. | [Accuracy table](ultralytics-pr26101-validation.md#accuracy-and-complete-array-parity) |
| [x] | Real segmentation accuracy | 5,000 images / 724,953 predictions; all bbox/mask metrics and fitness exactly equal. Mask AP 0.3436595381. | [Accuracy table](ultralytics-pr26101-validation.md#accuracy-and-complete-array-parity) |
| [x] | Real pose accuracy | All 5,000 images / 134,663 predictions; all bbox/keypoints metrics and fitness exactly equal. Pose AP 0.5638424803. | [Accuracy table](ultralytics-pr26101-validation.md#accuracy-and-complete-array-parity) |
| [x] | LVIS compatibility | Official 100-image bbox case and explicitly labelled 93-image real-mask subset pass. Metric delta <=5.56e-17; arrays <=2.23e-16. Original 100-image mask failures retained separately. | [LVIS scope](ultralytics-pr26101-validation.md#lvis-scope-and-the-old-backends-mask-failure) |
| [x] | Exact source and package revisions | Both Ultralytics commits, public 0.1.7 source, native hashes, frozen executed scripts and complete package versions saved. The final PR change is CI-only. | [Raw results](../bench/results/ultralytics_pr26101_20260909.json) |
| [x] | Dataset and prediction hashes | Official URLs and SHA-256 for annotations, models, every image, manifests and predictions; archived inputs checked against measurement hashes. | [Archive and provenance](ultralytics-pr26101-validation.md#saved-evidence) |
| [x] | All returned metrics and fitness | Complete dictionaries saved for every call; identical key sets required. COCO deltas 0; LVIS <=5.56e-17. Replays checked against full-validation COCO metrics. | [Raw results](../bench/results/ultralytics_pr26101_20260909.json) |
| [x] | Complete precision, recall and scores arrays | Actual arrays captured separately from timing. Maximum delta 2.23e-16; pose arrays exactly equal. Full NPZ arrays and per-array shapes/deltas published. | [Accuracy and arrays](ultralytics-pr26101-validation.md#accuracy-and-complete-array-parity) |
| [x] | Alternating fresh-process wall times | 94 processes completed: 4 full rounds per COCO task, 6 replay rounds per task, separate diagnostics. Balanced order; all samples retained. | [Measurement boundaries](ultralytics-pr26101-validation.md#measurement-boundaries) |
| [x] | Peak memory and background load | Peak RSS plus sampled process-tree RSS, host CPU/RAM and hardware recorded. Pose whole-validation RSS increases 0.9%; this is explicitly retained. | [Variation and load](ultralytics-pr26101-validation.md#observed-variation-and-background-load) |
| [x] | Evaluator versus whole-validation time | Separate nested eval_json, full validator and fresh-process clocks. Whole validation improves 16.3% / 9.1% / 10.0% for detect / segment / pose. | [Whole-validation table](ultralytics-pr26101-validation.md#whole-validation-performance) |
| [x] | Cached GT repeated evaluation | Cold plus two calls reusing the identical COCO object in each replay. All metrics and complete arrays are identical across reuse within each backend. | [Cold/cached table](ultralytics-pr26101-validation.md#cold-and-cached-evaluator-replay) |
| [x] | No selector, new user argument or fallback shim | Reviewed full runtime diff: replacement stays in the shared validator owner. Instrumentation is outside production; no runtime selector or fallback added. | [PR diff](https://github.com/ultralytics/ultralytics/pull/26101/files) |
| [ ] | Exact live-head terminal-green CI | Head 0451a3c9 is pushed and cold-reviewed. Linux, Windows and Python 3.8 test jobs pass. CI is not terminal-green: the ARM benchmark failed downloading cityscapes8.zip with HTTP 500. Upstream rerun requires a maintainer; this is not reported as green. | [CI run](https://github.com/ultralytics/ultralytics/actions/runs/34245506499) |

## Additional finding: LVIS mask reference limitation

The official example has 100 images but annotations in only 93. Both
faster-coco-eval 1.7.2 and 1.8.0 throw a map lookup exception during segmentation
preparation on the original 100-image workload. ultrafast 0.1.7 completes it,
including two cached-ground-truth repeats. This is retained as a reference
limitation; it is not labelled a successful 100-image before/after mask parity
comparison. The official box predictions contain no masks. Additional mask
predictions come from YOLO26n-seg on the same 100 real images, with the official
COCO-to-LVIS synset mapping (78 mapped COCO categories). The ground-truth masks
remain the original LVIS annotations.

## Reviewer's independent follow-up

In [comment 5587037002](https://github.com/ultralytics/ultralytics/pull/26101#issuecomment-5587037002),
Glenn reports independent real detection/segmentation/pose validation at the
previous head: identical predictions and returned metrics, complete arrays within
2.23e-16, cold/cached evaluation, LVIS box compatibility, 100 Objects365-style
annotation lookups and 100 Comet-style mask decodes. He identifies the missing
CPython 3.8 macOS arm64 wheel as the remaining blocker. Release 0.1.7 and the
pushed dependency-floor change address that blocker. His measurements are
attributed to his separate hardware/software configuration; they are not mixed
with the new RTX 3070 timing samples.
