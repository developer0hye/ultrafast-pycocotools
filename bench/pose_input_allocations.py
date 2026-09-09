"""Count native pose allocations separately from uninstrumented timing."""

import argparse
import gc
import hashlib
import json
import platform
import sys
from pathlib import Path

from ultrafast_pycocotools import COCO, COCOeval, _ufcoco

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["files", "list"], required=True)
parser.add_argument("--gt", type=Path, required=True)
parser.add_argument("--dt", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
mode, output = args.mode, args.out
gc.collect()
start = _ufcoco.alloc_stats()
assert start["enabled"]
_ufcoco.reset_alloc_peak()
gt = COCO(args.gt, verbose=False)
dt = gt.loadRes(args.dt if mode == "files" else json.loads(args.dt.read_text()))
loaded = _ufcoco.alloc_stats()
_ufcoco.reset_alloc_peak()
ev = COCOeval(gt, dt, "keypoints", print_function=lambda *_: None)
ev.run()
end = _ufcoco.alloc_stats()
arrays = {
    k: hashlib.sha256(ev.eval[k].tobytes()).hexdigest()
    for k in ("precision", "recall", "scores")
}
Path(output).write_text(
    json.dumps(
        dict(
            mode=mode,
            diagnostic_only=True,
            python=sys.version,
            platform=platform.platform(),
            native_sha256=hashlib.sha256(
                Path(_ufcoco.__file__).read_bytes()
            ).hexdigest(),
            start=start,
            loaded=loaded,
            evaluated=end,
            arrays=arrays,
        ),
        indent=2,
    )
)
