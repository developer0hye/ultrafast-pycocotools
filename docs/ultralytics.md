# Ultralytics evaluator replacement (0.1.6)

The [0.1.7 follow-up](ultralytics-pr26101-validation.md) adds complete real
detection, segmentation and pose validation, cached replay, full arrays and LVIS
mask evidence on the RTX 3070 server. This page retains the original 0.1.6 run.

This benchmark calls the actual `DetectionValidator.coco_evaluate` method
on saved YOLO26n predictions for all 5,000 COCO val2017 images (596,202
predictions). It compares upstream `4e6701f` against the replacement in
[developer0hye/ultralytics, branch perf/ultrafast-coco-eval](https://github.com/developer0hye/ultralytics/tree/perf/ultrafast-coco-eval).

| Backend | Evaluation time, median | Process peak RSS, median |
|---|---:|---:|
| faster-coco-eval 1.8.0 | 7.084 s | 2,423.1 MiB |
| ultrafast-pycocotools 0.1.6 | 0.668 s | 1,127.4 MiB |

The evaluation phase is **10.60x faster with 53.5% lower process peak RSS**.
These are three alternating fresh-process samples per backend on an AMD EPYC
9554, pinned to two CPUs. Both use the same saved predictions, input IDs,
maxDets and area ranges. All returned COCO metrics, including fitness, are
exactly equal in these runs. This is not an inference or whole-validation
speedup: prediction generation and initial `jdict` loading are outside the
clock. JSON loading inside the evaluator is included. Predictions remain in
`validator.jdict`, as in normal validation. Peak RSS covers the entire process,
including PyTorch, imports and the retained predictions. Filesystem caches are
warm; the cached COCO API object starts empty in each process.

[Raw samples and input hashes](../bench/results/ultralytics_v016.json) include
the complete returned metrics and source revisions. The installed Ultralytics
distribution metadata was 8.4.120, but `PYTHONPATH` selected the recorded 8.4.143
source checkouts. The evaluator was locally built from the exact 0.1.6 release
commit. Timings vary with hardware and concurrent system load.

An additional single-process replay of the official 100-image LVIS example
returned matching AP, AP50 and fitness. The largest difference in any returned
metric was 5.56e-17. Other metrics are compared with `rtol=0, atol=1e-12`, not
claimed to be bit-identical to faster-coco-eval for every input. This path uses
`lvis_protocol="coco"` to preserve the previous evaluator's COCO-style LVIS
maxDets and crowd behavior; it does not silently change to official LVIS's
global per-image cap. This small LVIS example is a compatibility check, not a
full-LVIS speed claim.

## Reproduce

Use Linux for the same RSS units and CPU affinity. Start with Python 3.12 and
install the two evaluators and Ultralytics into an isolated environment:

```bash
python -m pip install "ultralytics==8.4.143" "ultrafast-pycocotools==0.1.6" \
  "faster-coco-eval==1.8.0" "numpy==2.4.4"
git clone https://github.com/ultralytics/ultralytics.git /tmp/ultra-reference
git -C /tmp/ultra-reference checkout 4e6701f
git clone --branch perf/ultrafast-coco-eval \
  https://github.com/developer0hye/ultralytics.git /tmp/ultra-replacement
git -C /tmp/ultra-replacement checkout e600c6f2275acd4344127ea59942fccc179a7c2f
```

Generate shared predictions using the [YOLO26 reproduction commands](yolo26.md#reproduce)
or use an existing COCO prediction JSON. Do not regenerate predictions between
backends. Run from this repository with paths appropriate to your data:

```bash
git show 5d5da8c8dcc85ffd1be2f48541adbdc79adeb97f:bench/ultralytics_metric.py > /tmp/ultralytics-metric-v016.py
export CUDA_VISIBLE_DEVICES='' YOLO_AUTOINSTALL=false
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export YOLO_CONFIG_DIR=/tmp/ultra-benchmark-config
for sample in 1 2 3; do
  for backend in reference replacement; do
    PYTHONPATH="/tmp/ultra-$backend" taskset -c 8-9 \
      python /tmp/ultralytics-metric-v016.py \
        --gt /path/to/coco/annotations/instances_val2017.json \
        --pred /path/to/yolo26_predictions.json \
        --output "/tmp/$backend-$sample.json"
  done
done
```

Choose two available CPUs if 8-9 are unavailable. Do not overlap measured
processes. Compare every returned metric, and report the median of each
backend's three time and peak-RSS samples. Add `--lvis` with LVIS-format
annotations and predictions to replay the existing LVIS evaluation path.
