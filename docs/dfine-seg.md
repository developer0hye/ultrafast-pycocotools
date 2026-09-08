# D-FINE-seg integration

The local D-FINE-seg integration adds `train.coco_backend=ultrafast` for bbox
and instance-segmentation mAP. The default remains `faster_coco_eval`.
Both metric instances use ultrafast's COCO evaluator through TorchMetrics;
the application retains its prediction conversion, batching, reset and metric
reporting. No process-wide module aliases are installed.

The patch targets [ArgoHA/D-FINE-seg at
4c0a313](https://github.com/ArgoHA/D-FINE-seg/tree/4c0a313f664ef9712b43dd53865fd45636b52ed6).
The completed local checkout is `~/Documents/D-FINE-seg`.
[Download the integration patch](patches/dfine-seg-ultrafast.patch).
These changes are local and have not been submitted upstream.

## Measured application performance

The [COCO500 Validator benchmark](benchmark-dfine-seg.md) replays actual
D-FINE-seg-S predictions on the M2 and the i5-10400 / RTX 3070 server.
The published 0.1.6 wheel improves bbox validation but regresses full
segmentation validation. The [0.1.7 mask-encoding optimization](mask-encoding-optimization.md)
fixes that bottleneck: full bbox + segmentation validation is now
**17.8% / 23.4% faster** than faster-coco-eval, with all outputs unchanged.
The integration now requires the published 0.1.7 wheel, which includes this
optimization. The default backend remains faster-coco-eval. The reports also retain an existing cross-host F1
tie-order difference affecting one match.

## Use the modified checkout

Configure the dataset and model as described in D-FINE-seg's README, then run:

```bash
cd ~/Documents/D-FINE-seg
uv sync --no-dev --extra ultrafast
uv run --no-sync dfine train train.coco_backend=ultrafast
```

For a pip-based source installation, use `pip install -e '.[ultrafast]'`.
The extra installs `ultrafast-pycocotools>=0.1.7,<0.2`, including the optimized
mask encoder. The backend option is present in both the live configuration and
the packaged `dfine init` template. Older configurations default to the existing
backend, and training checks the selected backend before starting an epoch.

Direct callers can use:

```python
from dfine_seg.dl.validator import Validator

validator = Validator(gt, predictions, label_to_name, coco_backend="ultrafast")
metrics = validator.compute_metrics()
```

The option is forwarded through training validation, the shared benchmark
validator, and OpenVINO/TensorRT INT8 validation. F1, threshold sweeps, semantic
segmentation and inference retain their existing computations. In particular,
the `dfine bench` command currently sets `compute_maps=False`; selecting a COCO
backend does not accelerate its inference-latency or F1 measurements.

## Validation

The initial adapter was tested on Apple M2 / macOS 26.6.2 / Python 3.12.13 with both
the project's locked environment and a newer supported dependency combination:

| Package | Locked environment | Additional environment |
|---|---|---|
| PyTorch / torchvision | 2.13.0 / 0.28.0 | 2.14.0 / 0.29.0 |
| TorchMetrics | 1.9.0 | 1.9.0 |
| NumPy | 2.1.1 | 2.5.3 |
| faster-coco-eval | 1.6.5 | 1.8.0 |
| ultrafast-pycocotools | 0.1.6 (published wheel) | 0.1.6 (local source build) |
| pycocotools reference | 2.0.11 | 2.0.11 |

Validation covers complete precision/recall/scores arrays, aggregate and
per-class outputs, bbox and segmentation, crowd annotations, tied scores,
images with missing GT or predictions, boolean masks, dense/RLE input,
incremental updates, pickle/reset/reuse, and `compute_maps=False`.
Ultrafast's tested arrays match pycocotools byte for byte; comparisons with
faster-coco-eval allow absolute floating-point differences up to `1e-12`.
Validator F1/IoU/mAP outputs are also compared between both backends.

After the 0.1.7 release, the dependency floor and lock were upgraded without
changing other package versions. A fresh public-wheel installation passed
**11 backend/integration/pretrained-accuracy tests**; Ruff lint/format and
`uv lock --check` also pass.

The original locked environment passed **273 fast tests**. Ten tests were skipped
because optional Gradio, SAM3 and export runtimes were absent; slow/GPU tests
were excluded from that run. The D-FINE-S pretrained accuracy regression is
also run separately with both backends against the repository's two image
fixtures. Ruff lint/format checks and `uv lock --check` pass.

The adapter uses TorchMetrics 1.9's private `_coco_backend` slot, guarded by a
layout check. Its tests should be rerun when upgrading TorchMetrics. This is
an integration and correctness validation, not an end-to-end training speed
measurement. A full training run, distributed GPU execution and quantized
engine validation were not performed.

Local test evidence is retained under `bench/out/dfine-seg-integration/`.
The integration patch includes the new tests, configuration, dependency lock
and source changes. To apply it elsewhere, start from the upstream revision
linked above and run `git apply /path/to/dfine-seg-ultrafast.patch`.
