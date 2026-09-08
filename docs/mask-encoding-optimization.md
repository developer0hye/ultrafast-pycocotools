# Mask encoding optimization (0.1.7)

The published 0.1.7 release includes the dense-mask encoding optimization identified in
the [D-FINE-seg baseline](benchmark-dfine-seg.md). It is **not included in
the published 0.1.6 wheel**. The original benchmark wheels retained version
0.1.6, so source and native-library hashes identify those optimized builds.

## Results — 2026-09-08

The regression is reversed: **full bbox + segmentation validation is now
17.8% faster on M2 and 23.4% faster on the i5-10400 server than
faster-coco-eval**. Both hosts evaluate on CPU with two threads; the server's
RTX 3070 is unused. Values below are wall seconds, medians of six runs:

| Host | Published ultrafast, previous run | Optimized ultrafast | faster-coco-eval, current run | Reduction vs published / faster |
|---|---:|---:|---:|---:|
| Apple M2 / 16 GiB RAM | 12.356 | **8.944** | 10.880 | **27.6% / 17.8%** |
| i5-10400 / 31.24 GiB visible RAM | 14.991 | **10.899** | 14.236 | **27.3% / 23.4%** |

The published-ultrafast column comes from the preceding same-day benchmark;
it was not interleaved with the optimized run. The current comparison against
faster-coco-eval alternates backend order each round and retains all samples.
Every segmentation pair favors the optimized backend. The M2 ranges were
8.880–9.011 seconds (optimized) and 10.839–10.911 (faster); server ranges were
10.881–10.922 and 14.195–14.340 seconds, respectively.

Bbox-only timing remains close to the previous run: 0.775 seconds on M2 and
1.260 seconds on the server, versus current faster-coco-eval medians of
1.203 and 1.977 seconds. The optimization specifically targets mask encoding;
no bbox-only improvement is claimed.

### Attribution and resource use

In separate M2 cProfile runs over the same **6,928 real mask encodes**, time
inside the native ultrafast encoder fell from **3.676 to 0.116 seconds**
(**31.8×**). These are one-run diagnostic timings, excluded from the six-run
application medians. Full evaluation improves less because mask decoding,
TorchMetrics preparation, F1/IoU and COCO scoring also take time.

Median input-preparation/update time fell from 7.595 to **4.145 seconds** on
M2 and from 8.910 to **4.852 seconds** on the server. F1/IoU still costs
approximately 4.06 / 4.92 seconds in the optimized segmentation run.

| Host / backend | CPU seconds, median | Peak RSS MiB, median |
|---|---:|---:|
| M2 / faster | 11.112 | 1281.8 |
| M2 / optimized | 9.228 | 1319.5 |
| Server / faster | 18.528 | 1203.4 |
| Server / optimized | 15.278 | 1216.2 |

Peak RSS includes imports and input loading. During measured worker lifetimes,
M2 whole-host CPU was 10.3–37.0% (median 20.45%), with 4.24–4.94 GiB available
RAM; its pre-run CPU baseline median was 9.9%. Server CPU was 0–16.2%
(median 8.4%), with 28.28–29.42 GiB available RAM and a 0.1% baseline.
M2 system-wide swap-in/out counters increased by approximately 291.8 / 6.6 MiB;
the server counters did not increase. These counters include unrelated work.
No background processes were stopped and no samples were removed based on time.

### Output equivalence

**All returned metrics and complete COCO precision/recall/scores arrays are
identical before and after the optimization on each host**, including all
measured, warmup and diagnostic metric outputs. Current faster-coco-eval versus
ultrafast comparisons also pass the unchanged `1e-12` gate. The previously
documented one-TP cross-host difference in the application's existing F1
matcher is unchanged; this optimization does not alter its matching logic.

[Raw application runs, telemetry, source/wheel/native hashes, microbenchmarks,
profiling and before/after parity](../bench/results/mask_encode_20260908.json)
are saved with the [original baseline](../bench/results/dfine_seg_20260908.json).
Per-process logs, profiles and curve archives are under
`bench/out/dfine-validator-{m2,rtx3070}-mask-20260908/`. Native wheels are in
`bench/out/mask-encode-final-wheels/`; these are local builds, not releases.

## Implementation

- The Python binding directly borrows Fortran-contiguous mask storage.
  Previously it indexed every pixel through a dynamic-dimensional ndarray
  view and copied each mask into a temporary buffer before encoding.
- The RLE encoder skips 32 equal bytes at a time inside long runs. Mixed
  blocks and the tail retain the byte-by-byte transition logic. Comparing
  raw bytes preserves COCO's behavior for values other than 0 and 1.
- Arbitrarily strided and C-order inputs retain the existing fallback.
  No unsafe code, architecture-specific instructions or dependencies were added.

The change is limited to `rust/ufcoco-py/src/mask.rs` and
`rust/ufcoco-core/src/rle.rs`. Regression coverage adds contiguous/strided,
offset/reversed, read-only, singleton/empty and nonbinary inputs, plus run
transitions around the 32-byte block boundaries.

## Verification

- Rust core: **62 tests pass** in both debug and release builds.
- M2 Python API/parity suite: **262 passed**, followed by **3 additional
  full-COCO fixture checks** after linking the cached dataset. The optional
  RF-DETR integration module was unavailable; its collection skip remains.
- Linux / i5-10400: **105 mask parity tests pass**, using pycocotools 2.0.11
  as the byte-exact reference.
- D-FINE-seg: **11 integration and pretrained-accuracy tests pass** with the
  optimized M2 wheel.

## Reproduce the optimized application run

The integration now requires and locks the public 0.1.7 release, so
`uv sync --no-dev --extra ultrafast` installs the optimized encoder directly.
The historical benchmark builds can also be reproduced from source:

```bash
cd ~/Documents/ultrafast-pycocotools
maturin build --release --locked \
  --interpreter ../D-FINE-seg/.venv/bin/python --out bench/out/optimized-wheels
uv pip install --python ../D-FINE-seg/.venv/bin/python \
  --reinstall-package ultrafast-pycocotools bench/out/optimized-wheels/*.whl

../D-FINE-seg/.venv/bin/python bench/dfine_validator.py run \
  --inputs bench/out/dfine-coco500-inputs/inputs.pt \
  --threads 2 --rounds 6 --out bench/out/dfine-optimized-new-run
```

The input-generation procedure, fixed checkpoint revision and evaluation
semantics are in the [baseline report](benchmark-dfine-seg.md). Both hosts
replay exactly the same saved predictions; GPU inference, training and data
loading remain outside the evaluator timing boundary. The updated runs use
the same alternating-order, fresh-process procedure, with six retained
rounds per case/backend and separate excluded warmups/curve diagnostics.

`bench/mask_encode.py` additionally compares deterministic masks with the
pinned pycocotools reference. Its microbenchmark records seven samples of ten
calls each after one warmup. Input construction, hashing and reference
comparison are excluded. Separate processes select published/optimized
ultrafast or faster-coco-eval; these diagnostic microbenchmarks do not use the
balanced application-run schedule. Empty masks are too fast for robust
relative timing claims. The faster-coco-eval adapter converts strided/C-order
inputs to Fortran layout inside timing because its native encoder requires it.
