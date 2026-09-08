# YOLO26n: real predictions, identical evaluator inputs

Ultrafast 0.1.2 evaluates saved YOLO26n predictions **19.1× faster than
pycocotools with 53.4% lower peak process memory** in this measured case.
Compared with faster-coco-eval, it is 3.7× faster with 54.8% lower peak memory.

| Scorer | Evaluation time | Peak RSS | AP@[0.50:0.95] | Comparison with pycocotools |
|---|---:|---:|---:|---|
| pycocotools 2.0.11 | 37.073 s | 1,599.2 MiB | 0.4033248437 | Reference |
| faster-coco-eval 1.8.0 | 7.257 s | 1,651.6 MiB | 0.4033248437 | Maximum precision difference 2.22e-16 |
| ultrafast-pycocotools 0.1.2 | **1.942 s** | **745.9 MiB** | 0.4033248437 | **All four arrays byte-identical** |

The complete `precision`, `recall`, `scores` and `stats` arrays match pycocotools
byte for byte. Faster-coco-eval passes a separate absolute 1e-12 tolerance check
(rtol=0); its precision array is not byte-identical even though the summaries
agree. [Raw measurements, versions and hashes](../bench/results/yolo26n_v012.json).

## Prediction generation

The [official YOLO26n weights](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)
were run on all **5,000 COCO val2017 images**, producing **596,202 predictions**.
This benchmark uses the nano member of the [YOLO26 family](https://docs.ultralytics.com/models/yolo26/);
it does not report results for the small, medium, large or extra-large models.

- Ultralytics 8.4.143, PyTorch 2.13.0+cu126, CPU FP32, no GPU inference.
- Four CPU threads, CPU affinity 16–19 on an AMD EPYC 9554 host.
- Image size 640, square letterboxing (`rect=False`), batch size 8.
- Confidence threshold 0.001, max 300 detections per image, no augmentation.
- Official model's default end-to-end prediction path.
- COCO category IDs mapped from model indices; box coordinates rounded to three
  decimal places; scores retain the predictor's float values.

The prediction loop and JSON output took **242.0 s**, approximately four minutes.
That timing excludes model loading/download and is recorded separately from
scoring. A 100-image pilot took about six seconds. These are measured preparation
times on this shared host, not a standalone detector latency benchmark.

The weight SHA-256 is
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
The result manifest records annotation and prediction hashes, settings, versions
and the ordered image-ID digest. Raw images, weights and predictions are not
committed to Git.

## Scoring controls

Measured on 2026-09-08 on a shared Linux host, Python 3.12.3 and NumPy 2.4.4.
Each backend ran once in a fresh process with the exact same saved prediction
file, CPU affinity 4–5, two Rayon/OpenMP threads and one OpenBLAS thread.
All scorers used their normal result-loading defaults.

Evaluation time includes GT indexing, detection-dictionary copies, result
loading, matching, accumulation and summarization. It excludes JSON parsing,
inference and serialization. Peak RSS is whole-process peak memory, including
parsed inputs and output serialization. Shared-host timing variation is not
represented by confidence intervals from this single run per backend.

The result is evaluator speed on YOLO26n outputs, not a 19.1× speedup of YOLO26n
inference or end-to-end validation. AP is from the recorded prediction settings;
it is not asserted to exactly reproduce every published YOLO26 accuracy recipe.

## Reproduce

Install the evaluator and reference packages. Use a separate prediction
environment if your training environment has different dependencies:

```bash
python -m pip install ".[test]" "numpy==2.4.4" "faster-coco-eval==1.8.0"
python -m pip install "ultralytics==8.4.143"

python bench/predict_coco.py --model yolo26n.pt \
  --images /path/to/coco/val2017 \
  --ann /path/to/coco/annotations/instances_val2017.json \
  --out bench/data/yolo26_predictions.json \
  --device cpu --threads 4 --batch 8 --imgsz 640 --square \
  --conf 0.001 --max-det 300
```

The predictor writes a companion `.metadata.json` with hashes and settings.
Add `--limit 100` and a different output path for a quick CPU pilot. The full
run checks that every annotated image exists. Record your installed PyTorch
version: different versions or hardware may produce different prediction bytes.
All scorer comparisons must use one shared saved file regardless of that variation.

```bash
python bench/compare_saved_predictions.py --backend pycocotools \
  --gt /path/to/coco/annotations/instances_val2017.json \
  --pred bench/data/yolo26_predictions.json --out bench/out/yolo26-reference
python bench/compare_saved_predictions.py --backend faster-coco-eval \
  --gt /path/to/coco/annotations/instances_val2017.json \
  --pred bench/data/yolo26_predictions.json --out bench/out/yolo26-faster
python bench/compare_saved_predictions.py --backend ultrafast \
  --gt /path/to/coco/annotations/instances_val2017.json \
  --pred bench/data/yolo26_predictions.json --out bench/out/yolo26-ultrafast
```

Each output directory must be new. For comparable resource limits, restrict
scoring to two CPU cores and set `RAYON_NUM_THREADS=2`, `OMP_NUM_THREADS=2` and
`OPENBLAS_NUM_THREADS=1`. Compare the input hashes and four `array_sha256` entries
in the reference and ultrafast `result.json` files. Array archives are retained
for numerical comparisons with the additional backend. See the
[general reproduction guide](reproducibility.md) for public COCO input sources.
