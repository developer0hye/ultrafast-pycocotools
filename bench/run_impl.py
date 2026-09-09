"""Run one COCO evaluation implementation and report stats, time and peak RSS.

Each implementation runs in its own process so the timings are not polluted by
another library's warm caches or allocator state, and so peak RSS means what it
says.

**Wall-clock alone is not a fair comparison on a busy machine.** pycocotools is
single-threaded and barely notices background load; anything using rayon is
competing for the same cores and loses wall-clock that has nothing to do with
its own efficiency. Two things here address that:

* ``--threads N`` configures the Rayon/OpenMP pools. It is not a process-wide
  thread cap or CPU affinity setting: streaming producers can overlap workers.
* CPU time is reported next to wall time. Under contention wall inflates and
  CPU does not, so a gap between them is the measurement telling you it was
  disturbed.

Usage:
    python bench/run_impl.py --impl pycocotools --gt GT.json --dt DT.json --iou-type bbox
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import sysconfig
import time
from pathlib import Path

import numpy as np

LOAD_BEFORE = float("nan")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    """Peak working set of this process, in MB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        # K32GetProcessMemoryInfo lives in kernel32 on Vista+; the psapi.dll
        # export exists but is a forwarder that ctypes.windll cannot always
        # resolve, and it fails silently (leaving the struct zeroed).
        fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None)
        if fn is None:
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
        # Without explicit argtypes ctypes truncates the HANDLE on win64 and
        # the call quietly returns zeroed counters.
        fn.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        fn.restype = wintypes.BOOL
        cur = ctypes.windll.kernel32.GetCurrentProcess
        cur.restype = wintypes.HANDLE
        fn(cur(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1e6
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS bytes.
    return ru / 1e3 if sys.platform.startswith("linux") else ru / 1e6


def cpu_load_percent() -> float:
    """Whole-machine CPU utilisation over a short sample."""
    if sys.platform == "win32":
        import subprocess

        try:
            out = subprocess.run(
                ["wmic", "cpu", "get", "loadpercentage"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            vals = [int(x) for x in out.split() if x.isdigit()]
            return float(sum(vals) / len(vals)) if vals else float("nan")
        except Exception:  # noqa: BLE001
            return float("nan")
    try:
        with open("/proc/stat") as f:
            a = [float(x) for x in f.readline().split()[1:]]
        time.sleep(0.2)
        with open("/proc/stat") as f:
            b = [float(x) for x in f.readline().split()[1:]]
        busy = (sum(b) - sum(a)) - (b[3] - a[3])
        total = sum(b) - sum(a)
        return 100.0 * busy / total if total else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def run(impl: str, gt_path: str, dt_path: str, iou_type: str, file_inputs: bool = False,
        lvis_protocol: str | None = None, arrays_out: Path | None = None) -> dict:
    timings: dict[str, float] = {}

    if impl == "pycocotools":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    elif impl == "faster":
        from faster_coco_eval import COCO
        from faster_coco_eval import COCOeval_faster as COCOeval
    elif impl == "hotcoco":
        from hotcoco import COCO, COCOeval
    elif impl == "ufcoco":
        from ultrafast_pycocotools import COCO, COCOeval
    else:
        raise SystemExit(f"unknown impl {impl}")

    cpu_start = time.process_time()
    t = time.perf_counter()
    gt = COCO(gt_path)
    timings["load_gt"] = time.perf_counter() - t

    t = time.perf_counter()
    if file_inputs:
        dt = gt.loadRes(dt_path)
    else:
        with open(dt_path) as f:
            dt_json = json.load(f)
        dt = gt.loadRes(dt_json)
    timings["load_dt"] = time.perf_counter() - t

    t = time.perf_counter()
    c = time.process_time()
    options = {}
    if lvis_protocol is not None:
        if impl != "ufcoco":
            raise ValueError("Explicit LVIS protocols in this runner require --impl ufcoco")
        options = dict(lvis_style=True, lvis_protocol=lvis_protocol)
    ev = COCOeval(gt, dt, iou_type, **options)
    ev.evaluate()
    timings["evaluate"] = time.perf_counter() - t
    cpu_eval = time.process_time() - c

    t = time.perf_counter()
    c = time.process_time()
    ev.accumulate()
    timings["accumulate"] = time.perf_counter() - t
    cpu_eval += time.process_time() - c

    t = time.perf_counter()
    ev.summarize()
    timings["summarize"] = time.perf_counter() - t

    # Array hashes can allocate a large contiguous byte buffer (notably LVIS).
    # Capture evaluation memory before any result-verification/provenance work.
    cpu_total = time.process_time() - cpu_start
    peak = peak_rss_mb()

    stats = [float(x) for x in ev.stats]
    # Digests of the full arrays, not just the twelve summary numbers, so two
    # machines can be compared without shipping ~8 MB of doubles around. A
    # summary can match while the curve underneath differs in a hundred places.
    import hashlib

    digests = {
        k: hashlib.sha256(
            np.ascontiguousarray(ev.eval[k], dtype=np.float64).tobytes()
        ).hexdigest()[:16]
        for k in ("precision", "recall", "scores")
    }
    if arrays_out is not None:
        np.savez_compressed(arrays_out, **{k: ev.eval[k] for k in ("precision", "recall", "scores")})
    runtime_paths = [Path(sys.executable).resolve()]
    library = sysconfig.get_config_var('LDLIBRARY')
    if library:
        candidates = [Path(sys.base_prefix) / 'lib' / library,
                      Path(sys.base_prefix) / library]
        libdir = sysconfig.get_config_var('LIBDIR')
        if libdir:
            candidates.append(Path(libdir) / library)
        runtime_paths.extend(path.resolve() for path in candidates if path.is_file())
    source_paths = [Path(__file__).resolve()]
    distribution = importlib.metadata.distribution({
        'ufcoco': 'ultrafast-pycocotools', 'faster': 'faster-coco-eval',
    }.get(impl, impl))
    # Native classes can report __module__='builtins' (hotcoco). Distribution
    # files still identify the exact installed binary for every backend.
    source_paths.extend(
        Path(distribution.locate_file(path)).resolve()
        for path in distribution.files or []
        if str(path).endswith(('.so', '.pyd', '.dylib'))
    )
    if impl == 'ufcoco':
        from ultrafast_pycocotools import _ufcoco
        source_paths.extend([Path(_ufcoco.__file__),
                             Path(sys.modules[COCO.__module__].__file__),
                             Path(sys.modules[COCOeval.__module__].__file__)])
        if lvis_protocol is not None:
            from ultrafast_pycocotools import _lvis
            source_paths.append(Path(_lvis.__file__))
    return {
        "impl": impl,
        "package_version": distribution.version,
        "params": {
            "maxDets": list(ev.params.maxDets),
            "useCats": ev.params.useCats,
            "iouThrs": np.asarray(ev.params.iouThrs).tolist(),
            "recThrs": np.asarray(ev.params.recThrs).tolist(),
            "areaRng": np.asarray(ev.params.areaRng).tolist(),
        },
        "file_inputs": file_inputs,
        "lvis_protocol": lvis_protocol,
        "diagnostic_only": arrays_out is not None,
        "iou_type": iou_type,
        "timings": timings,
        "eval_total": timings["evaluate"] + timings["accumulate"] + timings["summarize"],
        "eval_cpu": cpu_eval,
        "threads": os.environ.get("RAYON_NUM_THREADS", "default"),
        "cpu_load_before": LOAD_BEFORE,
        "cpu_load_after": cpu_load_percent(),
        "wall_total": sum(timings.values()),
        "cpu_total": cpu_total,
        "stats": stats,
        "digests": digests,
        "platform": f"{sys.platform}/{platform.machine()}",
        "peak_rss_mb": peak,
        "python": sys.version,
        "numpy": np.__version__,
        "runtime_file_sha256": {str(path): file_digest(path) for path in runtime_paths},
        "source_file_sha256": {str(path): file_digest(path) for path in source_paths},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument(
        "--threads",
        type=int,
        help="configure Rayon/OpenMP pool size (not a process-wide CPU/thread cap)",
    )
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--iou-type", default="bbox")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--file-inputs", action="store_true",
                    help="Pass the prediction filename directly to loadRes")
    ap.add_argument("--lvis-protocol", choices=["official", "coco"])
    ap.add_argument("--arrays-out", type=Path, help="Save full arrays after timing and peak-RSS capture")
    args = ap.parse_args()

    # Must happen before the extension is imported: rayon reads this when it
    # builds its global pool, which is on the first parallel call.
    if args.threads:
        os.environ["RAYON_NUM_THREADS"] = str(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    global LOAD_BEFORE
    LOAD_BEFORE = cpu_load_percent()
    res = run(args.impl, args.gt, args.dt, args.iou_type, args.file_inputs,
              args.lvis_protocol, args.arrays_out)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
