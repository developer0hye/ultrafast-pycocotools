# Reproduce the evaluation results

The benchmark is CPU-only once annotation and prediction JSON files exist.
The fully synthetic check needs no downloads, images, model weights, or GPU.
Every run writes input SHA-256 hashes, package versions, per-phase timings,
peak memory where supported, full evaluation arrays, and a comparison report.

## 1. Install

From a checkout of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[test]"
```

A recent stable Rust toolchain is needed to build the extension. On Windows,
activate with `.\.venv\Scripts\Activate.ps1`. The published measurements used
Python 3.12, pycocotools 2.0.11 and NumPy 2.4.4; for that reference environment:

```bash
python -m pip install "pycocotools==2.0.11" "numpy==2.4.4"
```

## 2. No-download synthetic check

```bash
python bench/reproduce.py quick --out bench/out/quick --verify-published quick
```

This generates 64 images' worth of COCO-format metadata, eight categories,
and synthetic detections with seed 0. It includes crowd regions, tied scores,
area boundaries and empty-image cases. No actual image files are created or read.
The generator is [make_dataset.py](../bench/make_dataset.py).

The command runs both evaluators in separate processes and exits nonzero if
input hashes, array shapes, or any precision/recall/scores/stats bytes disagree.
With `--verify-published quick`, inputs and reference array hashes must also
match the small reference case committed in this repository.

A successful run ends with:

```text
PASS: precision, recall, scores and stats are byte-identical.
```

Output directories must be new. To repeat, choose a different `--out`; previous
measurements are never silently overwritten.

## 3. Objects365 scalability experiment

Obtain the **Objects365 v2 validation annotation JSON**,
`zhiyuan_objv2_val.json`, from the [official dataset download page](https://www.objects365.org/download.html).
The provider may require registration. Only the annotation JSON is needed;
do not download the images for this benchmark. Dataset terms and attribution
remain those of the Objects365 Consortium.

```bash
python bench/reproduce.py objects365 \
  --gt /path/to/zhiyuan_objv2_val.json \
  --out bench/out/objects365 \
  --threads 2 \
  --verify-published objects365
```

This is **synthetic prediction evaluation on real public annotations**, not the
accuracy of a trained Objects365 detector. The fixed recipe is:

| Parameter | Value |
|---|---:|
| Seed | 1234 |
| Retain/jitter probability | 0.75 |
| Wrong-category probability | 0.12 |
| Extra false-positive boxes | 160,000 total; two per image on average |
| Score decimal places | 3 |

[make_dets.py](../bench/make_dets.py) generates predictions from the annotation
JSON. Three-decimal scores deliberately exercise stable tie handling. Generated
files stay in the output directory and are not committed. The published input
has 80,000 images, 1,240,587 annotations and 365 categories; this recipe generates
1,090,984 detections. Do not substitute an Objects365 v1 or training annotation.

`--verify-published objects365` checks the original annotation file and generated
prediction hashes **before scoring**, and the reference array hashes afterward.
It fails rather than labeling different inputs as reproduction. A harmless JSON
reformat also changes the file hash; omit this flag to evaluate another input,
but report it as a separate experiment. Exact expected hashes and raw timings
are in [public_benchmarks.json](../bench/results/public_benchmarks.json).

The measured reference run took about 8.7 minutes and peaked at 22.6 GiB RSS.
Allow additional RAM for your operating system and other workloads.
Run the small check first. For the published Linux CPU affinity, add `--cpus 0,1`;
omit that option on other platforms or choose CPUs permitted on your system.

## 4. Public detector predictions on COCO

The real-detector case uses the public pretrained **YOLO11m** checkpoint and
all 5,000 COCO val2017 images. Obtain COCO images and annotations from the
[official COCO website](https://cocodataset.org/#download). Prediction generation
is optional and requires the detector dependency; evaluator benchmarking does not.

```bash
python -m pip install ultralytics
python bench/predict_coco.py \
  --model yolo11m.pt --device 0 --batch 32 --imgsz 640 --conf 0.001 --max-det 300 \
  --images /path/to/coco/val2017 \
  --ann /path/to/instances_val2017.json \
  --out bench/data/preds_yolo11m_bbox.json
python bench/reproduce.py predictions \
  --gt /path/to/instances_val2017.json \
  --pred bench/data/preds_yolo11m_bbox.json \
  --out bench/out/yolo11m --threads 2
```

The original public weights URL, counts and input hashes are recorded in
[the benchmark results](../bench/results/public_benchmarks.json). Inference can
vary with framework versions and hardware, so regenerated predictions need not
have the same file hash. Use `--verify-published yolo11m` only when using the exact
recorded inputs. Regardless of input hash, each comparison scores identical
predictions with both backends. Public detector weights and dataset files are
not bundled here.

## Evaluation parameters

All three recipes evaluate bounding boxes over every image and category in the
annotation JSON, including images with no detections. They use standard COCO
IoU thresholds 0.50–0.95 in steps of 0.05, 101 recall thresholds, maxDets
`[1, 10, 100]`, and the standard all/small/medium/large area ranges. Prediction
generation may retain up to 300 boxes per image; the evaluator applies its own
per-image, per-category limit of 100 for AP. Both backends receive the same
JSON inputs and parameters.

## Reading the results

`comparison.json` includes each backend's `result.json` and a byte-equality verdict.
`pycocotools/arrays.npz` and `ultrafast/arrays.npz` contain the full arrays for
independent inspection. Timings separate GT indexing, result loading,
`evaluate`, `accumulate` and `summarize`; input JSON parsing is recorded separately.

The scorer total excludes detector inference, distributed prediction gathering,
and application-specific F1 reporting. Threads are explicitly set through
RAYON_NUM_THREADS/OMP_NUM_THREADS, with OPENBLAS_NUM_THREADS=1. A single run
checks parity and gives an observed timing; use `--repeats 3` for a spread and
[compare.py](../bench/compare.py) for interleaved comparisons. Hardware, workload,
affinity and library versions affect time and memory. Metric parity and speed
are separate claims.
