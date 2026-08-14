"""Split the JSON loader into read / parse / build.

The loader is the biggest single item before evaluation starts, and the obvious
next move is a SIMD tokenizer (simdjson does gigabytes a second). Whether that
is worth anything depends entirely on which half the time is in:

* If the **tokenizer** dominates, SIMD is the answer.
* If **building Python objects** dominates, SIMD buys almost nothing — a
  `PyDict` per annotation and a `PyFloat` per number is work no parser can
  skip, and it is work we are required to do, because `coco.anns[id]` must be
  a real dict for detectron2 and mmdetection to write to.

Three measurements, each a strict subset of the next: read the bytes; read and
run the parser discarding everything; read, parse, and build. Differences give
the split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultrafast_pycocotools import _ufcoco


def best(fn, repeat: int) -> float:
    t = float("inf")
    for _ in range(repeat):
        s = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - s)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    for path in args.files:
        p = str(path)
        mb = path.stat().st_size / 1e6
        t_read = best(lambda: _ufcoco.read_file_only(p), args.repeat)
        t_parse = best(lambda: _ufcoco.parse_json_only(p), args.repeat)
        t_build = best(lambda: _ufcoco.load_json(p), args.repeat)
        t_py = best(lambda: json.load(open(p, "rb")), args.repeat)

        print(f"{path.name}  ({mb:.0f} MB)")
        print(f"  read only                 {t_read:6.3f}s  {mb / t_read:7.0f} MB/s")
        print(f"  + parse, discard          {t_parse:6.3f}s  {mb / t_parse:7.0f} MB/s"
              f"   (parser {t_parse - t_read:.3f}s)")
        print(f"  + build Python objects    {t_build:6.3f}s  {mb / t_build:7.0f} MB/s"
              f"   (building {t_build - t_parse:.3f}s)")
        print(f"  stdlib json.load          {t_py:6.3f}s  {mb / t_py:7.0f} MB/s"
              f"   ({t_py / t_build:.2f}x ours)")
        share = (t_build - t_parse) / t_build
        print(f"  -> building Python objects is {share:.0%} of the loader; "
              f"the parser is {(t_parse - t_read) / t_build:.0%}, the read "
              f"{t_read / t_build:.0%}")
        print()


if __name__ == "__main__":
    main()
