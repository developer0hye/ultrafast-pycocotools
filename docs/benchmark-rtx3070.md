# i5-10400 / RTX 3070 server benchmark — 2026-09-08

The RTX 3070 server completed the same synthetic 5,000-image bbox workload
in **0.193 s median with ultrafast 0.1.6**, using the two-thread pool setting.
The observed median speedups were **81.6× versus pycocotools** and
**9.9× versus faster-coco-eval**. Evaluation runs entirely on the CPU;
the RTX 3070 identifies this server and does not accelerate these measurements.

[All server results and telemetry](../bench/results/rtx3070_20260908.json) ·
[Apple M2 report](benchmark-apple-m2.md) ·
[Apple M2 raw results](../bench/results/apple_m2_20260908.json)

## CPU, RAM and execution environment

| Item | RTX 3070 server | Apple M2 desktop |
|---|---|---|
| CPU | Intel Core i5-10400, 6 cores / 12 threads | Apple M2, 4 performance + 4 efficiency cores |
| OS-reported RAM | 31.24 GiB | 16.00 GiB |
| OS | Ubuntu 24.04.4 LTS; Linux 7.0.0-28; x86_64 | macOS 26.6.2; arm64 |
| Python | 3.12.3 | 3.12.13 |
| Rust compiler | 1.97.1 | 1.98.0 |
| CPU use before warmup | 0.0–0.1% | 37.4–61.4% |

Both hosts used NumPy 2.4.4, pycocotools 2.0.11, faster-coco-eval 1.8.0,
ultrafast-pycocotools 0.1.6 and psutil 7.2.2. The library was built in release
mode from commit `5d5da8c8dcc85ffd1be2f48541adbdc79adeb97f` with two build jobs
and no custom RUSTFLAGS. The measurement scripts and input files have identical
SHA-256 hashes on both hosts. Source was transferred to a separate server
checkout with its own virtual environment; existing server work was left intact.

Server CPU affinity allowed logical CPUs 0–11; no core pinning was requested.
The existing CPU governor reported `powersave` and was left unchanged. Neither
core frequency nor thermal state was sampled. The measurement window, including
warmups, was **2026-09-08 21:27:33–21:32:37 KST**.

## Server results

The procedure matches the M2 run: one excluded warmup and six measured runs
per backend/thread setting, all in fresh processes. Each of the six backend
orders appears once, and the thread-setting order alternates between rounds.
All **42 executions (36 measured)** are retained, with no timing exclusions.

The requested Rayon/OpenMP thread settings are 1 and 2; OpenBLAS, Accelerate
and MKL thread environment limits are 1. These settings do not impose a hard
process CPU quota. Wall and process CPU totals cover JSON/GT loading, indexing,
result loading, evaluate, accumulate and summarize. Imports, evaluator
construction, bookkeeping and output serialization are excluded. RSS covers
the entire child process including serialization.

| Requested threads | Backend | Wall s: median (min–max) | CPU s: median | Peak RSS MiB: median (min–max) |
|---:|---|---:|---:|---:|
| 1 | pycocotools | 15.926 (15.657–16.121) | 15.925 | 850.9 (850.9–851.0) |
| 1 | faster-coco-eval | 1.918 (1.896–1.945) | 2.923 | 692.2 (691.9–692.4) |
| 1 | ultrafast | 0.274 (0.272–0.276) | 0.274 | 95.6 (95.6–95.6) |
| 2 | pycocotools | 15.725 (15.605–16.257) | 15.723 | 851.0 (850.7–851.0) |
| 2 | faster-coco-eval | 1.898 (1.894–1.924) | 2.910 | 692.5 (692.3–692.6) |
| 2 | ultrafast | 0.193 (0.192–0.196) | 0.283 | 96.3 (96.1–96.3) |

The raw JSON also retains per-phase timings, individual RSS values, means and
sample standard deviations. Ratios use medians from the same host and setting.

## Both hosts, same inputs

Two-thread-setting medians, including JSON loading:

| Backend | M2 wall s | Server wall s | M2 peak RSS MiB | Server peak RSS MiB |
|---|---:|---:|---:|---:|
| pycocotools | 11.560 | 15.725 | 827.7 | 851.0 |
| faster-coco-eval | 1.385 | 1.898 | 645.9 | 692.5 |
| ultrafast | 0.136 | 0.193 | 92.3 | 96.3 |

These are observations under each host's recorded conditions. The M2 desktop
had other active work and transient slowdowns; the server started with very
little CPU activity. OS, CPU architecture, Python patch version and compiler
also differ. The table is useful for deployment expectations under these
conditions; it does not isolate CPU hardware performance. Cross-platform RSS
accounting and allocator behavior also affect memory comparisons.

During measured server processes, whole-host CPU samples ranged from
**2.3% to 16.9%**
(median 8.4%), including benchmark work.
Available RAM ranged from **28.36 to
29.40 GiB**; swap usage was
**720.2–720.2 MiB**.
Telemetry spans the whole child process and is sampled about once per second;
it cannot resolve ultrafast's subsecond scoring phases individually.

## Correctness and provenance

- All server warmups and measured rounds matched pycocotools byte for byte for
  the complete `precision`, `recall`, `scores` and `stats` arrays.
- All three backends also produced **byte-identical arrays across the M2 and
  server hosts**, at both thread settings. Complete saved arrays were compared;
  all warmup and repetition hashes were stable for each backend.
- Faster-coco-eval's difference from pycocotools stayed within `atol=1e-12`,
  `rtol=0`; the largest absolute difference was `2.220446049250313e-16`.
- The published 64-image quick check passed its input and reference-array
  hashes. The 36 targeted reproduction/compact-file tests passed before timing.

The synthetic workload has 5,000 images, 80 categories, 30,812 annotations and
98,155 detections. Seed 0, GT limit 12 and detection limit 40 per image match
the M2 recipe. Original input JSON files were copied directly to the server.
This measures synthetic bbox evaluation; no images or model inference are used.

```text
GT SHA-256:   b23701d6733ce49a2344160bcbf30ea804eba99b80c0bc2a845e1a43f1bd13a5
Pred SHA-256: aa0c5b1a53ddc0d3cef46f6ad272af4980b38f68fcfbe88a33d34b4682f93df4
```

The result JSON contains the host metadata, all runs and telemetry, quick-check
results, and cross-host array comparisons. Full arrays and logs were copied
back to local `bench/out/rtx3070-20260908-interleaved/`. The isolated server
checkout and its original outputs remain under
`/tmp/ufcoco-bench-20260908-m2-comparison/repo/`.

## Reproduce

Follow the [reproduction guide](reproducibility.md#measurements-on-a-busy-desktop)
to install the recorded package versions and generate the input files. With
the environment and transferred inputs used for this run, the command was:

```bash
cd /tmp/ufcoco-bench-20260908-m2-comparison/repo
.benchmark-venv/bin/python bench/interleaved.py \
  --gt ../inputs/gt_5000.json --pred ../inputs/dt_5000.json \
  --out bench/out/rtx3070-repeat --threads 1 2 --rounds 6 --input-mode files
```

Choose a new output directory for each repetition; existing results are never
overwritten. The same input hashes are required for comparison with this report.
