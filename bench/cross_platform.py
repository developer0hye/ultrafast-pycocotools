"""Compare the same evaluation run on two machines.

Bit-exactness so far has meant "matches pycocotools on this machine". That is
the claim that matters for a drop-in replacement, but it leaves a question
open: is the *number itself* stable across platforms, or does everyone get
their own AP?

Two places could genuinely differ:

* ``rleToString`` / ``rleFrString`` use C ``long``, which is 64-bit under
  gcc/clang and 32-bit under MSVC. pycocotools inherits that; this crate
  always uses ``i64``. On COCO-sized masks the counts never reach 2^31 so the
  two agree, but it is an assumption worth measuring rather than asserting.
* OKS calls ``exp``. numpy's vectorised ``exp`` and a platform libm are both
  accurate to well under an ulp, but they are different implementations and
  neither promises the other's exact bits.

Compares the digests of the whole precision/recall/scores arrays, not the
summary numbers — a summary can agree while the curve underneath does not.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(paths: list[Path]) -> dict:
    out: dict = {}
    for p in paths:
        r = json.loads(p.read_text())
        # bench/out/{win,linux}-<impl>-<preds>.json
        stem = p.stem
        host, rest = stem.split("-", 1)
        impl, preds = rest.split("-", 1)
        out[(preds, impl, host)] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    runs = load(args.files)
    hosts = sorted({k[2] for k in runs})
    if len(hosts) != 2:
        raise SystemExit(f"need exactly two hosts, found {hosts}")
    h1, h2 = hosts

    plat = {}
    for (_, _, host), r in runs.items():
        plat[host] = r.get("platform", "?")
    print(f"{h1}: {plat.get(h1)}    {h2}: {plat.get(h2)}")
    print()

    configs = sorted({k[0] for k in runs})
    impls = ["pycocotools", "faster", "hotcoco", "ufcoco"]
    header = f"{'predictions':22} {'impl':14} {'same across hosts':>18}  {'max |stat diff|':>16}"
    print(header)
    print("-" * len(header))
    same_host_parity: dict = defaultdict(dict)
    for cfg in configs:
        for impl in impls:
            a = runs.get((cfg, impl, h1))
            b = runs.get((cfg, impl, h2))
            if a is None or b is None:
                continue
            identical = a["digests"] == b["digests"]
            delta = max(abs(x - y) for x, y in zip(a["stats"], b["stats"]))
            mark = "identical" if identical else "DIFFERS"
            print(f"{cfg:22} {impl:14} {mark:>18}  {delta:16.3e}")
            for host, r in ((h1, a), (h2, b)):
                same_host_parity[(cfg, host)][impl] = r
        print()

    print("parity against pycocotools, computed separately on each host")
    print(f"{'predictions':22} {'host':8} {'impl':14} {'result':>22}")
    print("-" * 70)
    for (cfg, host), by_impl in sorted(same_host_parity.items()):
        ref = by_impl.get("pycocotools")
        if ref is None:
            continue
        for impl in impls:
            if impl == "pycocotools" or impl not in by_impl:
                continue
            r = by_impl[impl]
            identical = r["digests"] == ref["digests"]
            if identical:
                verdict = "bit-identical"
            else:
                d = max(abs(x - y) for x, y in zip(ref["stats"], r["stats"]))
                verdict = f"max |diff| {d:.1e}"
            print(f"{cfg:22} {host:8} {impl:14} {verdict:>22}")


if __name__ == "__main__":
    main()
