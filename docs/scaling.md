# Evaluation scaling

![Measured evaluation time and peak process memory against total GT and prediction count](assets/scaling.png)

[Vector SVG](assets/scaling.svg) · [Print-ready PDF](assets/scaling.pdf) ·
[Raw measurements](../bench/results/scaling.json)

## What varies

We take nested subsets of 1,000, 5,000, 10,000, 20,000, 40,000 and 80,000
Objects365 v2 validation images. The image IDs are sorted, shuffled with
Python `random.Random(20260908)`, then selected by prefix. All 365 categories
are retained at every size. Annotation and prediction order is preserved.
The synthetic predictions use the same fixed recipe as the
[reproduction guide](reproducibility.md#3-objects365-scalability-experiment).

The horizontal axis counts **GT annotations plus prediction boxes**, before
COCO's maxDets filtering. Increasing the dataset size increases both counts;
this experiment does not independently vary box density or category count.
Exact counts for each point are recorded below and in the raw JSON.

## Measurement controls

Each scorer runs in a fresh process, with CPU affinity `0,1`, two Rayon/OpenMP
threads, and one OpenBLAS thread, on the same AMD EPYC 9554 host. Input
preparation runs in a separate process and is excluded from scoring. The
scorer order alternates across subset sizes. Each backend is measured once
per size; no error bars or confidence intervals are implied.

The 80,000-image endpoint reuses the previously published measurement with the
same input SHA-256 hashes, library source, CPU affinity and thread budget.
All smaller points are newly measured. This avoids repeating the long full
reference run; the raw file identifies the reused endpoint explicitly.
The host is shared, so unrelated activity can influence timings.

Time includes GT indexing, result loading, matching, accumulation and
summarization, and excludes JSON parsing, inference and result serialization.
Memory is the absolute **whole-process peak RSS**, including parsed inputs and
output serialization. It is not allocator-only memory or a baseline-subtracted
increase. Both panels start at zero and use linear axes. Line segments connect
measured points; there are no fitted, smoothed or extrapolated curves.

All four evaluation outputs (`precision`, `recall`, `scores`, `stats`) must
match byte for byte at every size. These are synthetic detections, so their AP
is not a trained detector's accuracy. A lower curve demonstrates less time or
memory for this workload; it does **not** establish exponential versus linear
asymptotic complexity.

## Reproduce the measurements

Install the package and reference/plotting dependencies:

```bash
python -m pip install ".[test,plot]"
```

Obtain `zhiyuan_objv2_val.json` as described in the reproduction guide, then:

```bash
python bench/make_dets.py \
  --gt /path/to/zhiyuan_objv2_val.json \
  --out bench/data/objects365_predictions.json \
  --seed 1234 --recall 0.75 --wrong-class 0.12 --fp-ratio 2.0 --score-decimals 3
python bench/scale.py \
  --gt /path/to/zhiyuan_objv2_val.json \
  --pred bench/data/objects365_predictions.json \
  --out bench/out/scaling --threads 2 --cpus 0,1
python bench/plot_scaling.py \
  --results bench/out/scaling/scaling.json \
  --out bench/out/scaling/figure
```

Choose a new output directory for each run. Omit `--cpus` outside Linux or use
CPUs permitted on your host. The full series needs tens of GB of RAM and can
take more than 15 minutes with the reference backend. No images or GPU are
needed. To perform the same endpoint reuse as the published run, add
`--reuse-full bench/results/public_benchmarks.json`; that option checks the
endpoint input hashes before accepting cached results. A fully fresh run
should omit it, as the command above does.

To redraw the published figure without running benchmarks or downloading a
dataset:

```bash
python bench/plot_scaling.py
```

This reads the committed measurements and writes PNG, SVG and PDF assets.
The published rendering uses Matplotlib 3.10.8. Chart labels and endpoint
ratios are calculated from the JSON, not entered manually.

## Measured points

| Images | GT boxes | Predictions | Reference time | Ultrafast time | Reference peak RSS | Ultrafast peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 15,690 | 13,718 | 5.21 s | 0.17 s | 0.41 GiB | 0.24 GiB |
| 5,000 | 77,222 | 67,850 | 26.40 s | 0.74 s | 1.52 GiB | 0.32 GiB |
| 10,000 | 154,418 | 136,040 | 58.01 s | 1.36 s | 2.93 GiB | 0.46 GiB |
| 20,000 | 310,333 | 273,157 | 120.20 s | 3.01 s | 5.74 GiB | 0.73 GiB |
| 40,000 | 621,752 | 546,763 | 239.78 s | 5.93 s | 11.37 GiB | 1.29 GiB |
| 80,000 | 1,240,587 | 1,090,984 | 520.26 s | 12.43 s | 22.62 GiB | 2.37 GiB |
