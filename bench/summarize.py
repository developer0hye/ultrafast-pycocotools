"""Collect `run_impl.py --json-out` results into the comparison table.

Kept as a script rather than a shell one-liner because the numbers end up in
the README, and a table nobody can regenerate is a table nobody should trust.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--baseline", default="pycocotools")
    args = ap.parse_args()

    runs: dict[tuple[str, str], dict] = {}
    for f in args.files:
        r = json.loads(f.read_text())
        key = (r["iou_type"], r["impl"])
        # Keep the best run when a configuration was measured more than once.
        if key not in runs or r["eval_total"] < runs[key]["eval_total"]:
            runs[key] = r

    header = (
        f"{'iouType':8} {'impl':22} {'eval(s)':>9} {'speedup':>8} "
        f"{'wall(s)':>9} {'peak RSS':>10}  parity vs {args.baseline}"
    )
    print(header)
    print("-" * len(header))
    for iou_type in sorted({k[0] for k in runs}):
        base = runs.get((iou_type, args.baseline))
        for impl in ("pycocotools", "faster", "hotcoco", "ufcoco"):
            r = runs.get((iou_type, impl))
            if r is None:
                continue
            speed = base["eval_total"] / r["eval_total"] if base else float("nan")
            if base is None:
                parity = "?"
            elif impl == args.baseline:
                parity = "(reference)"
            else:
                d = max(abs(a - b) for a, b in zip(base["stats"], r["stats"]))
                parity = "bit-identical" if d == 0.0 else f"max |diff| {d:.1e}"
            rss = r["peak_rss_mb"]
            rss_s = f"{rss / 1000:.2f} GB" if rss >= 1000 else f"{rss:.0f} MB"
            print(
                f"{iou_type:8} {impl:22} {r['eval_total']:9.3f} {speed:7.1f}x "
                f"{r['wall_total']:9.3f} {rss_s:>10}  {parity}"
            )


if __name__ == "__main__":
    main()
