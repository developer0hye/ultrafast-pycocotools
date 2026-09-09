# Pose input storage and parsing experiment

This implements expert recommendation 3.1: write keypoint x/y pairs directly into final native storage, retain only GT visibility predicates (`v > 0`), and omit unused detection bboxes. File snapshots stream coordinates through the existing serde parser; Python inputs use checked element access without an intermediate vector. Ground-truth `num_keypoints` retains its separate ignore behavior. The pose kernel also stops allocating unused crowd flags (part of 3.2). OKS arithmetic and summation order are unchanged.

## Measured results

Apple M2 (8 cores, 16 GiB RAM), macOS 26.6.2, CPython 3.12.13, NumPy 2.4.4, rustc 1.98.0. Both variants used the same compiler, portable release flags and 2 Rayon threads. Baseline is `0080c20babadc5babe7d95e93e00ce1d266ea529`, the 0.1.10 safety fixes before this optimization.

COCO val2017: 5,000 images, 11,004 GT annotations, 134,663 saved YOLO26n pose detections. Six alternating fresh-process pairs per input route; ordinary release builds for timing, separate instrumented wheels for allocation counts. Peak RSS is captured before result hashing. End-to-end time includes GT/results loading, evaluation, accumulation and summary, and excludes inference. The list route includes Python JSON loading of predictions; GT remains a compact file snapshot in both routes.

| Input route | Wall seconds before → after | CPU seconds before → after | Peak RSS MB before → after |
| --- | --- | --- | --- |
| JSON files | 0.6037 → 0.5668 (-6.1%) | 0.6129 → 0.5806 (-5.3%) | 330.4 → 275.0 (-16.8%) |
| Python prediction list | 1.7707 → 1.7479 (-1.3%) | 1.7807 → 1.7559 (-1.4%) | 817.3 → 808.8 (-1.0%) |

These are observed medians, not guarantees across hardware or workloads. The small list-route end-to-end change is largely masked by JSON/Python-object work. Evaluation-only medians improve from 0.3919 to 0.3561 seconds for files and 0.1366 to 0.1201 seconds for lists. Host CPU/RAM samples, runtime/source hashes and individual timings are retained in the evidence directory.

The RTX 3070 server (Intel i5-10400, 6 cores/12 threads, 31.24 GiB RAM, Linux, CPython 3.12.3, NumPy 2.4.4, rustc 1.97.1) used the same inputs and 2 Rayon threads. This is CPU evaluation; the GPU did not execute the metric. Six fresh-process pairs produced:

| Server input route | Wall seconds before → after | CPU seconds before → after | Peak RSS MB before → after |
| --- | --- | --- | --- |
| JSON files | 0.7644 → 0.7072 (-7.5%) | 0.7930 → 0.7344 (-7.4%) | 290.3 → 266.8 (-8.1%) |
| Python prediction list | 2.4351 → 2.4106 (-1.0%) | 2.4625 → 2.4366 (-1.0%) | 565.8 → 565.8 (unchanged) |

Linux list-route peak RSS is unchanged despite lower native retained bytes: the process high-water mark can occur during Python JSON/list construction. The focused 72-case pose suite also passed on this server. CPU load and available RAM were sampled throughout each run; no GPU throughput improvement is claimed.

During file-input evaluation, native allocation events fall from 785,336 to 54,266 and newly allocated bytes from 216,530,024 to 59,717,308. Retained native bytes added by evaluation fall from 128,490,515 to 46,270,975 (82,219,540 fewer bytes). Allocation-event counters include vector growth; they are not a count of calls to only `malloc`. These native savings differ from total process RSS.

The first candidate reduced RSS but slightly increased file-route wall time (0.5883 → 0.5961 seconds). It retained an intermediate coordinate vector in the file parser. The final candidate removes that vector. Both attempts' raw results are retained; the first attempt is not evidence for a speedup.

## Correctness and reproduction

The full local library suite passed 382 tests with 6 optional skips; Rust core tests passed in debug and release modes. Complete precision, recall and score arrays from actual COCO predictions match pycocotools byte for byte. Every timed run also checks all curve hashes. The expanded pose tests cover 3/17/25 joints, custom sigmas, zero and NaN visibility, zero/custom thresholds, caps 1/20, list/tuple/NumPy/file inputs and the alternate area calculation. Existing tests retain public `computeOks`, mutable annotation views and compact-file equivalence coverage.

Run `python -m pytest -q tests/test_keypoint_storage.py` for the 72 focused cases. Use the normal release builds for timing:

```sh
python bench/pose_input_benchmark.py --baseline-python BASELINE/bin/python --candidate-python CANDIDATE/bin/python --gt person_keypoints_val2017.json --dt pose-predictions.json --out RESULTS
```

For allocation diagnostics, build separate wheels with `maturin build --release --features alloc-stats`, install them into separate environments, and run `RAYON_NUM_THREADS=2 python bench/pose_input_allocations.py --mode files --gt GT.json --dt DT.json --out counts.json` (also test `--mode list`). Do not use those instrumented builds for speed measurements.

[Raw results and complete arrays](../bench/results/pose-inputs-20260910/) include input SHA-256 values. Inputs come from the previously published [Ultralytics evidence archive](https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz); the archive and source model are unchanged.

## Expert recommendation checklist

| Recommendation | This cycle |
| --- | --- |
| 1.1 Native NumPy bbox columns | Not attempted; remains a separate API proposal. |
| 1.2 Bounded stable cap selection | Not attempted; existing ordering retained. |
| 1.3 Sparse internal IoU candidates | Not attempted. |
| 2.1 Segmentation-only compact inputs | Not attempted. |
| 2.2 Indexed masks and fused metadata | Not attempted. |
| 2.3 RLE arenas / direct strided encode | Not attempted. |
| 3.1 Direct compact keypoint storage | Implemented and measured here. |
| 3.2 OKS setup cleanup | Removed unused crowd allocation; sigma caching and split kernels remain unattempted. |
| 3.3 Pose geometry cap / keypoint-only compact inputs | Not attempted; all geometry remains available for changed parameters. |
| 4.1 One-byte verdicts | Not attempted. |
| 4.2 Reverse-count accumulation | Not attempted. |
| 4.3 Image-block matrix lifetime | Not attempted. |
| 5 Ownership / free-threading audit | Reentrant bbox/polygon crash reproduced and fixed in 0.1.10; extension explicitly requires the GIL. Full free-threading audit remains open. |

No claim of a theoretical speed or memory optimum is made. The remaining proposals need independent correctness and performance experiments.
