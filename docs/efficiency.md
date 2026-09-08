# Faster evaluation and lower memory, together

In 0.1.1, ordinary `gt.loadRes(predictions)` avoids allocating and retaining
four-corner polygons for box-only predictions. The evaluator and public mask
and plotting helpers derive the geometry when needed. This single change
reduces both time and memory; there is no separate speed or memory mode.
GT indexing and evaluation collection also avoid redundant traversals/lists.

| Same saved inputs, package defaults | 0.1.0 time | 0.1.1 time | Time reduction | 0.1.0 peak RSS | 0.1.1 peak RSS | RSS reduction |
|---|---:|---:|---:|---:|---:|---:|
| COCO val2017 / public YOLO11m | 2.544 s | 1.524 s | 40.1% | 712.7 MiB | 599.8 MiB | 15.9% |
| Objects365 v2 / synthetic predictions | 12.247 s | 8.429 s | 31.2% | 2,432.8 MiB | 2,165.5 MiB | 11.0% |

Every measurement matches the published pycocotools 2.0.11 `precision`,
`recall`, `scores` and `stats` array hashes exactly, on identical input hashes.
The COCO case has 5,000 images and 431,145 predictions; Objects365 has
80,000 images, 1,240,587 GT annotations and 1,090,984 synthetic predictions.
[Raw measurements, hashes and environment](../bench/results/efficiency_v011.json).

Measured on 2026-09-08 on a shared AMD EPYC 9554 host, Linux, Python 3.12.3,
NumPy 2.4.4, CPU affinity 0–1, Rayon/OMP two threads, OpenBLAS one thread.
COCO reports medians of three fresh processes per version with alternating
version order. Objects365 uses one fresh process per version. Timing includes
GT indexing, detection dictionary copies and result loading, evaluation,
accumulation and summarization. It excludes JSON parsing, inference and output
serialization. RSS is the whole-process high-water mark and includes parsed
inputs and serialization. Retained Python input dictionaries limit the RSS
reduction. These observations are not guaranteed timings or GPU validation speed.

The polygon-omission option already existed in 0.1.0. The gain here is from
making it the default and making `annToRLE`, `annToMask` and `showAnns` work with
that representation, alongside collection/indexing changes. It is not a claim
of a new matching algorithm. Real segmentation annotations are retained.
Code that directly needs `annotation['segmentation']` for box-only results can
request `derive_segmentation=True`; that explicitly restores materialized
annotation fields and their allocation cost. No metric option is changed.

## Reproduce

Follow the [public input instructions](reproducibility.md) to obtain COCO
annotations/public predictions or generate Objects365 predictions. Run the same
benchmark command in separate environments for each revision:

- 0.1.0 source: commit `0f96274336173b4f98daf5ccf7d763b251c1c4b6`.
- 0.1.1 source: the commit containing this report, recorded in its Git history.

Install the chosen revision and NumPy 2.4.4 in each environment. The benchmark
records the installed package version and always uses its default result loader.
Use separate output directories (each must not already exist):

```bash
python bench/compare_saved_predictions.py --backend ultrafast \
  --gt path/to/annotations.json --pred path/to/predictions.json \
  --out bench/out/version-011-run-1
```

For the reported resource limits, set `RAYON_NUM_THREADS=2`, `OMP_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=1` and restrict the process to two CPU cores. Repeat the
COCO command three times per version, alternating order. Compare input and
`array_sha256` fields of every `result.json` against the corresponding reference
in [public_benchmarks.json](../bench/results/public_benchmarks.json); use the
median `runs[0].total_scoring_seconds` and median `peak_rss_MiB` for this table.
Peak RSS is recorded on Linux/macOS; the portable benchmark reports `null` on
Windows, where the standard-library `resource` module is unavailable.

The older [scaling graphs](scaling.md) and [three-backend table](faster-coco-eval.md)
remain explicitly labeled 0.1.0 measurements. They are not relabeled as new runs.
