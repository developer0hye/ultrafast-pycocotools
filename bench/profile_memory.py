"""Where the memory goes, phase by phase.

Peak RSS says the process got big. It does not say which phase made it big,
and on a 24 GB Objects365 run that is the only question worth asking. This
brackets each phase and reports three views of it, because they answer
different questions:

* **RSS delta** — what the operating system had to hand out. Includes Python
  objects, Rust data, allocator slack and fragmentation. This is the number a
  user hits when they run out of memory.
* **Python allocations** (``tracemalloc``) — the annotation dicts, the lists
  inside them, the floats inside those. Optional because tracing costs ~2x.
* **Rust allocations** — exact bytes and allocation counts from our own
  global allocator, available when the extension is built with::

      maturin develop --release --features alloc-stats

  Without that feature the Rust column reads "n/a" rather than zero: a silent
  zero would look like "we allocate nothing", which is the opposite of useful.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import ultrafast_pycocotools as ufc
from ultrafast_pycocotools import _ufcoco


def rss_bytes() -> int:
    """Current working set of this process."""
    if sys.platform == "darwin":
        import psutil
        return psutil.Process().memory_info().rss
    if sys.platform == "win32":
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        c = PMC()
        c.cb = ctypes.sizeof(c)
        fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None)
        if fn is None:
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        cur = ctypes.windll.kernel32.GetCurrentProcess
        cur.restype = wintypes.HANDLE
        fn(cur(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096


def mb(n: float) -> str:
    return f"{n / 1e6:9.1f} MB"


@dataclass
class Phase:
    name: str
    wall: float = 0.0
    rss_delta: int = 0
    py_delta: int = 0
    rust_live_delta: int = 0
    rust_peak: int = 0
    rust_allocs: int = 0


@dataclass
class Recorder:
    trace_python: bool = False
    phases: list[Phase] = field(default_factory=list)

    @contextmanager
    def phase(self, name: str):
        gc.collect()
        p = Phase(name)
        rust0 = _ufcoco.alloc_stats()
        _ufcoco.reset_alloc_peak()
        py0 = tracemalloc.get_traced_memory()[0] if self.trace_python else 0
        rss0 = rss_bytes()
        t0 = time.perf_counter()
        try:
            yield p
        finally:
            p.wall = time.perf_counter() - t0
            gc.collect()
            p.rss_delta = rss_bytes() - rss0
            if self.trace_python:
                p.py_delta = tracemalloc.get_traced_memory()[0] - py0
            rust1 = _ufcoco.alloc_stats()
            p.rust_live_delta = rust1["live_bytes"] - rust0["live_bytes"]
            p.rust_peak = rust1["peak_bytes"] - rust0["live_bytes"]
            p.rust_allocs = rust1["allocations"] - rust0["allocations"]
            self.phases.append(p)

    def report(self, rust_enabled: bool) -> None:
        head = f"{'phase':26} {'wall':>8} {'RSS delta':>12} {'Python':>12} {'Rust live':>12} {'Rust peak':>12} {'allocs':>12}"
        print(head)
        print("-" * len(head))
        for p in self.phases:
            py = mb(p.py_delta) if self.trace_python else "        n/a"
            rl = mb(p.rust_live_delta) if rust_enabled else "        n/a"
            rp = mb(p.rust_peak) if rust_enabled else "        n/a"
            al = f"{p.rust_allocs:12,}" if rust_enabled else "         n/a"
            print(f"{p.name:26} {p.wall:7.3f}s {mb(p.rss_delta)} {py} {rl} {rp} {al}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--dt", type=Path, required=True)
    ap.add_argument("--iou-type", default="segm")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--file-inputs", action="store_true",
                    help="Keep native snapshots; install psutil for macOS RSS measurements")
    ap.add_argument(
        "--no-derive-segmentation",
        action="store_true",
        help="skip the four-corner polygon loadRes stores for box-only results",
    )
    ap.add_argument(
        "--trace-python",
        action="store_true",
        help="also account Python allocations with tracemalloc (~2x slower)",
    )
    args = ap.parse_args()

    rust_enabled = bool(_ufcoco.alloc_stats()["enabled"])
    if not rust_enabled:
        print(
            "note: built without the alloc-stats feature, so the Rust columns "
            "are unavailable.\n      rebuild with: maturin develop --release "
            "--features alloc-stats\n"
        )
    if args.trace_python:
        tracemalloc.start()

    rec = Recorder(trace_python=args.trace_python)
    base_rss = rss_bytes()

    with rec.phase("COCO(gt) load"):
        gt = ufc.COCO(str(args.gt), verbose=False)
    with rec.phase("loadRes(dt)"):
        if args.file_inputs:
            dt = gt.loadRes(str(args.dt))
        else:
            with open(args.dt) as f:
                dets = json.load(f)
            dt = gt.loadRes(dets, derive_segmentation=not args.no_derive_segmentation)
            del dets

    ev = ufc.COCOeval(gt, dt, args.iou_type, print_function=lambda *_: None)
    p = ev.params
    p.imgIds = [int(i) for i in np.unique(p.imgIds)]
    p.catIds = [int(c) for c in np.unique(p.catIds)]
    p.maxDets = sorted(p.maxDets)

    with rec.phase("_prepare"):
        if args.file_inputs:
            gts = gt._compact if gt._compact is not None else gt._eval_annotations(p.imgIds, p.catIds)
            dts = dt._compact if dt._compact is not None else dt._eval_annotations(p.imgIds, p.catIds)
            img_sizes = ev._image_sizes()
        else:
            gts, dts, img_sizes = ev._collect()

    sigmas = getattr(p, "kpt_oks_sigmas", np.zeros(0))
    with rec.phase("engine build"):
        engine = _ufcoco.Evaluator(
            gts, dts, img_sizes, p.imgIds, p.catIds,
            [float(x) for x in p.iouThrs], [float(x) for x in p.recThrs],
            [int(m) for m in p.maxDets],
            [[float(a[0]), float(a[1])] for a in p.areaRng],
            True, p.iouType, [float(s) for s in np.asarray(sigmas).ravel()], True, 0.02,
        )
    with rec.phase("evaluate + accumulate"):
        arrays = engine.run(False)

    print()
    rec.report(rust_enabled)
    print()
    print(f"process RSS at start        : {mb(base_rss)}")
    print(f"process RSS at end          : {mb(rss_bytes())}")
    if rust_enabled:
        s = _ufcoco.alloc_stats()
        print(f"rust live at end            : {mb(s['live_bytes'])}")
        print(f"rust peak overall           : {mb(s['peak_bytes'])}")
        print(f"rust total allocated        : {mb(s['total_allocated_bytes'])}")
        print(f"rust allocations            : {s['allocations']:,}")
    print()
    counts = [x.annotation_count if isinstance(x, _ufcoco.CompactBbox) else len(x) for x in (gts, dts)]
    print(f"gt / dt annotations         : {counts[0]} / {counts[1]}")
    print(f"retained complete arrays    : {mb(sum(arrays[key].nbytes for key in ('precision', 'recall', 'scores')))}")
    print("note: RSS deltas include the allocator's own slack, so they exceed")
    print("      the Rust/Python figures and do not sum to the process total.")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "iou_type": args.iou_type,
            "file_inputs": args.file_inputs,
            "native_path": _ufcoco.__file__,
            "rust": _ufcoco.alloc_stats(),
            "phases": [asdict(phase) for phase in rec.phases],
            "rss_start": base_rss,
            "rss_end": rss_bytes(),
            "annotations": counts,
            "array_bytes": {key: arrays[key].nbytes for key in ("precision", "recall", "scores")},
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
