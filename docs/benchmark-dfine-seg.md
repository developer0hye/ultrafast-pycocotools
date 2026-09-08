# D-FINE-seg Validator benchmark

This benchmark measures the [D-FINE-seg integration](dfine-seg.md) on actual
COCO val2017 images and predictions from the public
[D-FINE-seg-S checkpoint](https://huggingface.co/ArgoSA/D-FINE-seg).
It compares `faster_coco_eval` with `ultrafast` inside the application's
`Validator`, including input conversion and the other validation metrics.

**Follow-up:** the [0.1.7 mask-encoder optimization](mask-encoding-optimization.md)
reverses the segmentation regression: full validation is 17.8% / 23.4% faster
than faster-coco-eval on M2 / the server. This page retains the original
published-0.1.6 measurements.

## Results — 2026-09-08

**Bbox validation improved; full segmentation validation regressed.** The
published ultrafast 0.1.6 wheel makes COCO scoring faster, but its mask encoding
cost outweighs the scoring savings in this application's segmentation path.

Wall-clock seconds, median of six runs; brackets show min–max:

| Host / evaluation | faster-coco-eval | ultrafast | Total change |
|---|---:|---:|---:|
| Apple M2 / bbox | 1.211 [1.156–1.230] | 0.768 [0.731–0.790] | **1.58× faster** |
| Apple M2 / bbox + segmentation | 10.881 [10.302–10.989] | 12.356 [12.048–12.517] | **13.5% slower** |
| 3070 server / bbox | 1.942 [1.930–2.004] | 1.242 [1.237–1.259] | **1.56× faster** |
| 3070 server / bbox + segmentation | 14.235 [14.207–14.287] | 14.991 [14.952–15.027] | **5.3% slower** |

Ratios above divide the two medians. The raw results also include paired-round
ratios. Every bbox pair favored ultrafast; every full-segmentation pair favored
faster-coco-eval. These are evaluator replay results, not training-epoch or
inference speedups. The one-time M2 input preparation took 232.749 seconds
including inference, loading and postprocessing; it is excluded from this table.

The M2 has 8 CPU cores and 16 GiB RAM. The 3070 server uses an Intel i5-10400
(6 cores / 12 threads), with 31.24 GiB RAM visible to Linux. Both use two
evaluation threads. Key versions are TorchMetrics 1.9.0, NumPy 2.1.1,
faster-coco-eval 1.6.5 and ultrafast-pycocotools 0.1.6.

Process CPU seconds and peak RSS (MiB), also medians of six runs:

| Host / evaluation | CPU: faster → ultra | Peak RSS: faster → ultra |
|---|---:|---:|
| M2 / bbox | 1.211 → 0.806 | 772.3 → 640.0 |
| M2 / bbox + segmentation | 11.116 → 12.634 | 1282.4 → 1321.1 |
| 3070 server / bbox | 1.970 → 1.327 | 819.6 → 689.4 |
| 3070 server / bbox + segmentation | 18.521 → 19.393 | 1223.8 → 1241.3 |

During measured worker lifetimes, M2 whole-host CPU utilization ranged from
10% to 80% (median 22.2%); available RAM was 4.04–4.92 GiB. Its five-second
pre-run CPU baseline had a 12.9% median. The server had a 0.1% baseline,
0–16.4% worker-lifetime CPU utilization (median 8.4%), and 28.36–29.45 GiB
available RAM. M2 system-wide swap-in increased by 774.6 MiB and swap-out by
4.6 MiB during the measured sequence; these counters cannot attribute activity
to this benchmark. The server had 88 KiB swap-in and no swap-out increase.
Background load was recorded, not subtracted from wall time.

### Where the segmentation time goes

Median phase wall times for **bbox + segmentation**, in seconds:

| Phase | M2 faster | M2 ultra | Server faster | Server ultra |
|---|---:|---:|---:|---:|
| Input preparation / metric update | 5.390 | 7.595 | 7.179 | 8.910 |
| Bbox mAP compute | 1.048 | 0.581 | 1.552 | 0.893 |
| Segmentation mAP compute | 0.296 | 0.050 | 0.456 | 0.089 |
| F1 / IoU | 4.047 | 4.043 | 4.912 | 4.956 |
| Cleanup | 0.089 | 0.089 | 0.129 | 0.134 |

Individual phase medians need not sum to the median total. Segmentation mAP
compute improves by 5.97× on M2 and 5.11× on the server, while the input/update
phase becomes slower. That phase decodes cached RLE into dense masks, binarizes
them, and lets TorchMetrics encode them back into RLE.

Separate M2 cProfile runs located the added cost in the native mask encoder:
the same **6,928 encode calls** took 1.375 seconds inside faster-coco-eval's
encoder and 3.676 seconds inside ultrafast's encoder. NumPy Fortran conversion
took 0.888 versus 0.921 seconds. These profiled timings are diagnostic only and
are excluded from the performance table. Dense-to-RLE encoding is the next
optimization target; these measurements do not justify enabling ultrafast as
the default segmentation backend.

### Correctness and a separate F1 portability finding

All paired runs pass the `1e-12` agreement gate. Bbox AP50:95 is
**0.50625509** and mask AP50:95 is **0.39766678** in both backends on both hosts.
For the complete COCO arrays, faster-coco-eval versus ultrafast precision
differs by at most **2.22e-16**; recall and scores are byte-identical. Comparing
the same backend across hosts, all three arrays are byte-identical for both
evaluation cases. This is the application/subset AP described below.

The application's separate greedy F1 matcher has a pre-existing cross-host
difference of **one true positive**, reproduced with both backends:

| Evaluation | M2 F1 / TP | Server F1 / TP |
|---|---:|---:|
| Bbox | 0.65502309 / 2269 | 0.65473441 / 2268 |
| Bbox + segmentation | 0.63770208 / 2209 | 0.63741339 / 2208 |

A bbox audit isolates image **9400**: two candidate predictions, labeled
keyboard and laptop, have identical IoU `0.9155844449996948` against the same
keyboard GT. The application's `torch.argsort(-iou_values)` does not request
stable sorting, and the equal-IoU candidates reverse order between the two
platforms. Its class-agnostic greedy matching therefore picks a different
label. The candidate IoUs and stable-sort orders agree across hosts. This
does not involve the COCO evaluator and does not change mAP. No F1 matching
behavior was changed during this benchmark; the discrepancy is retained in
the evidence instead of relaxing the backend comparison tolerance.

[Raw results, telemetry, hashes, curve parity, F1 audit and profiling evidence](../bench/results/dfine_seg_20260908.json)
are committed in the repository's results directory.
Per-process logs, curve archives, input cache and profiles remain under
`bench/out/dfine-validator-{m2,rtx3070}-20260908/` and
`bench/out/dfine-coco500-inputs/`. The server copy is
`/tmp/dfine-validator-20260908/`.

## Workload and timing boundary

- The first 500 COCO val2017 image IDs, sorted numerically; no image exclusions.
- Checkpoint `dfine_seg_s_coco.pt` at Hugging Face revision
  `93351f275ac9640b89ba89b3657b270e878279e7`.
- D-FINE-seg upstream `4c0a313f664ef9712b43dd53865fd45636b52ed6` plus the
  [local integration patch](patches/dfine-seg-ultrafast.patch).
- Unmodified training `Loader` and `Trainer.get_preds_and_gt`: CPU FP32,
  640 × 640, batch 1, `keep_ratio=False`, confidence threshold 0.5.
- Predictions are generated once on the M2 and saved as tensors and RLE masks.
  Both hosts replay the identical file. Bbox mAP uses all 300 predictions per
  image; segmentation mAP uses the predictions retained at confidence 0.5.
- **bbox** removes mask fields from those same segment-model predictions.
  **bbox + segmentation** preserves the full training-validation input.
  These are two evaluation views of one model, not two model comparisons.

The timed interval includes `Validator` construction and metric updates,
`compute_metrics(extended=True)`, bbox/mask mAP, F1/IoU, and metric cleanup.
It excludes imports, reading the saved input, model inference, dataset loading,
plotting, training, and serialization of benchmark output. Phase wrappers time
the existing methods without changing their computations. Mask batch size is
the project's default 150; confidence and matching IoU thresholds are 0.5.

This follows D-FINE-seg's annotation policy: its COCO loader skips crowd
annotations, rasterizes polygons, and does not pass original COCO area/crowd
fields to this Validator. Mask predictions are thresholded. Consequently,
the reported AP is a check of this application path on a subset, **not an
official full-val2017 checkpoint accuracy result**.

## Measurement method

Each host runs one excluded warmup for each case/backend, then six measured
rounds. Every run starts a fresh process. Backend order alternates between
rounds, as does case order. All measured rounds are retained. PyTorch and
Rayon/OpenMP are limited to two threads; interop and BLAS limits are one.
Cores are not pinned. Both hosts evaluate on CPU; the RTX 3070 is unused.

Wall time, process CPU time, process peak RSS, and one-second whole-host
CPU/memory/swap/load telemetry are saved. Peak RSS includes imports and input
loading; telemetry also includes worker startup and result serialization.
Other user workloads are left running. Medians and ranges therefore describe
these observed conditions, not isolated-machine performance guarantees.

Every paired round checks all returned Validator metrics and COCO summary
statistics at absolute tolerance `1e-12` (zero relative tolerance), and checks
repeatability across rounds. Separate diagnostic processes additionally
compare the full precision, recall and scores arrays. Their extended-summary
work is excluded from performance statistics.

## Reproduce

Use the patched D-FINE-seg checkout and its locked environment:

```bash
cd ~/Documents/D-FINE-seg
uv sync --no-dev --extra ultrafast
uv pip install --python .venv/bin/python psutil==7.2.2

cd ~/Documents/ultrafast-pycocotools
DFINE_PY=../D-FINE-seg/.venv/bin/python
OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 "$DFINE_PY" bench/dfine_validator.py prepare \
  --dfine ../D-FINE-seg \
  --annotations /path/to/instances_val2017.json \
  --revision 93351f275ac9640b89ba89b3657b270e878279e7 \
  --images 500 --threads 2 --out bench/out/dfine-coco500-inputs

"$DFINE_PY" bench/dfine_validator.py run \
  --inputs bench/out/dfine-coco500-inputs/inputs.pt \
  --threads 2 --rounds 6 --out bench/out/dfine-validator-new-run
```

Preparation downloads only the selected images from COCO's public S3 bucket,
and pins the model revision. Image, annotation, checkpoint, input, benchmark
script and application-source hashes are retained in the result provenance.
For another host, copy `inputs.pt` together with `inputs.json`, install the
same patched application and package versions, and run the same replay
command. The Linux environment uses the CPU builds of PyTorch/torchvision
(`2.13.0+cpu` / `0.28.0+cpu`); the M2 uses `2.13.0` / `0.28.0`.
