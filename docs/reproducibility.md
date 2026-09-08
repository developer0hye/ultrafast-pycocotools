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
PASS: precision, recall, scores and stats are byte-identical (pycocotools vs ultrafast).
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

## Include faster-coco-eval

With Python 3.10+, install `faster-coco-eval==1.8.0` and add `--include-faster` to any reproduction
command to score the same inputs with all three implementations. Ultrafast's
byte-equality gate remains strict. Faster-coco-eval's numerical differences
are reported separately, with absolute tolerance 1e-12 and zero relative
tolerance. See the [comparison report](faster-coco-eval.md) for results and
full commands.

## Measurements on a busy desktop

Use [interleaved.py](../bench/interleaved.py) to retain raw measurements and
system-load samples while other applications are running:

```bash
python -m pip install psutil "faster-coco-eval==1.8.0"
python bench/make_dataset.py --images 5000 --cats 80 --gt-per-image 12 \
  --dt-per-image 40 --seed 0 --out bench/out/local-inputs
python bench/interleaved.py --gt bench/out/local-inputs/gt_5000.json \
  --pred bench/out/local-inputs/dt_5000.json --out bench/out/local-benchmark \
  --threads 1 2 --rounds 6 --input-mode files
```

Each scorer starts in a fresh process. One warmup per backend/thread setting
is saved but excluded from the summary. Six measured rounds use every ordering
of the three backends; the thread-setting order alternates between rounds.
Every round checks all four arrays for exact ultrafast parity and separately
checks faster-coco-eval's numerical tolerance. No timing outliers are removed.

`results.json` retains phase wall/CPU times, whole-process peak RSS, input and
array hashes, package versions, and whole-host CPU/memory samples. Its summary
reports the median, minimum, maximum, mean and standard deviation. Full arrays
and logs stay in the output directory. In file mode the scoring total includes
JSON loading, indexing, result loading, evaluate/accumulate/summarize; imports,
evaluator construction, input bookkeeping and output serialization are excluded.
CPU times cover the same phases and sum time consumed by process threads.

Thread environment variables limit supported pools; they do not enforce a
process-wide CPU quota or pin performance/efficiency cores. CPU time still
depends on clock frequency, cache effects and scheduling. Telemetry is sampled
about once per second over the entire child process lifetime, so it includes
benchmark load and work outside the scoring timer; it cannot attribute short
phase delays to a particular background process. Treat these as observations
under the recorded load, not idle-machine performance or CPU-normalized scores.

See the [Apple M2 desktop measurement](benchmark-apple-m2.md) and
[i5-10400 / RTX 3070 server comparison](benchmark-rtx3070.md) for examples
with identical inputs and all raw repetitions retained.
