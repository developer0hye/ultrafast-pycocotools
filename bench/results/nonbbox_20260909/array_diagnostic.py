"""Untimed complete-array and native-storage diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import ultrafast_pycocotools as ufc
from ultrafast_pycocotools import _ufcoco

p = argparse.ArgumentParser()
p.add_argument('--gt', required=True)
p.add_argument('--dt', required=True)
p.add_argument('--iou-type', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
gt = ufc.COCO(a.gt, verbose=False)
dt = gt.loadRes(a.dt)
storage = {name: {'snapshot_bytes': handle._compact.snapshot_bytes,
                  'column_bytes': handle._compact.column_bytes,
                  'annotation_count': handle._compact.annotation_count}
           for name, handle in [('gt', gt), ('dt', dt)]}
ev = ufc.COCOeval(gt, dt, a.iou_type, print_function=lambda *_: None)
ev.run()
arrays = {key: ev.eval[key] for key in ('precision', 'recall', 'scores')}
np.savez_compressed(a.output.with_suffix('.npz'), **arrays)
record = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
          'iou_type': a.iou_type, 'storage': storage,
          'native_path': _ufcoco.__file__,
          'native_sha256': hashlib.sha256(Path(_ufcoco.__file__).read_bytes()).hexdigest(),
          'gt_sha256': hashlib.sha256(Path(a.gt).read_bytes()).hexdigest(),
          'dt_sha256': hashlib.sha256(Path(a.dt).read_bytes()).hexdigest(),
          'array_bytes': {k: v.nbytes for k,v in arrays.items()},
          'array_hashes': {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
          'compact_after': [handle._compact is not None for handle in (gt, dt)],
          'stats': ev.stats.tolist()}
a.output.write_text(json.dumps(record, indent=2))
