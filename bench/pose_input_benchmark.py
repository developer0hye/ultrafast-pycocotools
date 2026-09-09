"""Compare pose input implementations in alternating fresh processes."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import psutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline-python", required=True)
parser.add_argument("--candidate-python", required=True)
parser.add_argument("--gt", type=Path, required=True)
parser.add_argument("--dt", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--repeat", type=int, default=6)
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
interpreters = {"baseline": args.baseline_python, "candidate": args.candidate_python}
records = []
for mode in ("files", "list"):
    for iteration in range(args.repeat):
        for variant in (
            ("baseline", "candidate")
            if iteration % 2 == 0
            else ("candidate", "baseline")
        ):
            name = f"{mode}-{iteration}-{variant}"
            result = out / (name + ".json")
            cmd = [
                interpreters[variant],
                str(root / "bench/run_impl.py"),
                "--impl",
                "ufcoco",
                "--threads",
                "2",
                "--iou-type",
                "keypoints",
                "--gt",
                str(args.gt.resolve()),
                "--dt",
                str(args.dt.resolve()),
                "--json-out",
                str(result),
            ]
            if mode == "files":
                cmd.append("--file-inputs")
            print("START", name, flush=True)
            samples = []
            psutil.cpu_percent()
            start = time.perf_counter()
            with result.with_suffix(".log").open("w") as log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2"),
                    cwd=root,
                )
                while proc.poll() is None:
                    samples.append(
                        {
                            "seconds": time.perf_counter() - start,
                            "host_cpu_percent": psutil.cpu_percent(),
                            "available_ram": psutil.virtual_memory().available,
                        }
                    )
                    try:
                        proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
            assert proc.returncode == 0, name
            data = json.loads(result.read_text())
            if records:
                assert data["digests"] == records[0]["result"]["digests"], name
            records.append(dict(name=name, command=cmd, samples=samples, result=data))
            (out / "execution.json").write_text(json.dumps(records, indent=2))
            print("PASS", name, data["wall_total"], data["peak_rss_mb"], flush=True)
