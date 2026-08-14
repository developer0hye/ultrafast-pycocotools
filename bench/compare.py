"""Compare implementations in a way a busy machine cannot quietly distort.

Three things go wrong when you benchmark a parallel library against a
single-threaded one on a desktop:

1. **Background load steals cores.** pycocotools is single-threaded and barely
   notices; anything using rayon is fighting the antivirus for the same cores
   and loses wall-clock that says nothing about its own efficiency. This
   *understates* the parallel implementation.
2. **Drift.** Running all repetitions of A, then all of B, attributes any
   thermal or load drift to whichever ran second. Interleaving spreads it.
3. **One number hides the spread.** At tenths of a second the run-to-run
   variance is easily 15%, so a single measurement cannot support "faster".

So: repetitions are interleaved, the machine's load is sampled around every
run and reported, and both wall and CPU time come back. Wall is what a user
waits; CPU is what the work actually cost and is far less sensitive to
contention. When the two disagree, the machine was busy.

``--threads 1`` runs everything single-threaded. That is the comparison that
travels — it removes the core count and the background load from the question
and leaves the algorithmic difference.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

IMPLS = ["pycocotools", "faster", "hotcoco", "ufcoco"]


def run_once(impl: str, gt: str, dt: str, iou_type: str, threads: int | None) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_impl.py")),
        "--impl", impl,
        "--gt", gt,
        "--dt", dt,
        "--iou-type", iou_type,
    ]
    if threads:
        cmd += ["--threads", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise SystemExit(f"{impl} produced no result:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--iou-type", default="bbox")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--threads", type=int, help="pin every implementation to N threads")
    ap.add_argument("--impls", default=",".join(IMPLS))
    args = ap.parse_args()

    impls = [i for i in args.impls.split(",") if i]
    runs: dict[str, list[dict]] = {i: [] for i in impls}
    for rep in range(args.repeat):
        for impl in impls:  # interleaved, so drift hits everyone equally
            runs[impl].append(run_once(impl, args.gt, args.dt, args.iou_type, args.threads))
        print(f"  rep {rep + 1}/{args.repeat} done", file=sys.stderr, flush=True)

    loads = [r["cpu_load_before"] for v in runs.values() for r in v]
    loads = [x for x in loads if x == x]
    print(f"\n{args.iou_type}  threads={args.threads or 'default'}  "
          f"repeat={args.repeat}  machine load {min(loads):.0f}-{max(loads):.0f}% "
          f"(median {statistics.median(loads):.0f}%)")

    base = runs.get("pycocotools")
    base_wall = min(r["eval_total"] for r in base) if base else None
    base_cpu = min(r["eval_cpu"] for r in base) if base else None

    head = (f"{'impl':14} {'wall best':>10} {'median':>9} {'spread':>8} "
            f"{'cpu best':>9} {'wall x':>7} {'cpu x':>7} {'peak RSS':>10}  parity")
    print(head)
    print("-" * len(head))
    ref_digest = base[0]["digests"] if base else None
    for impl in impls:
        v = runs[impl]
        wall = sorted(r["eval_total"] for r in v)
        cpu = sorted(r["eval_cpu"] for r in v)
        spread = (wall[-1] - wall[0]) / wall[0] if wall[0] else float("nan")
        rss = min(r["peak_rss_mb"] for r in v)
        rss_s = f"{rss / 1000:.2f} GB" if rss >= 1000 else f"{rss:.0f} MB"
        wx = base_wall / wall[0] if base_wall else float("nan")
        cx = base_cpu / cpu[0] if base_cpu else float("nan")
        if impl == "pycocotools":
            parity = "(reference)"
        elif ref_digest and v[0]["digests"] == ref_digest:
            parity = "bit-identical"
        else:
            d = max(abs(x - y) for x, y in zip(base[0]["stats"], v[0]["stats"]))
            parity = f"max |diff| {d:.1e}"
        print(f"{impl:14} {wall[0]:10.3f} {statistics.median(wall):9.3f} {spread:7.0%} "
              f"{cpu[0]:9.3f} {wx:6.1f}x {cx:6.1f}x {rss_s:>10}  {parity}")

    if not args.threads:
        print("\nnote: wall-clock here includes whatever else the machine was doing.")
        print("      Re-run with --threads 1 for a comparison that does not depend on it.")


if __name__ == "__main__":
    main()
