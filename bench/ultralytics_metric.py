"""Replay Ultralytics' COCO evaluation on saved predictions in a fresh process."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--gt', required=True)
p.add_argument('--pred', required=True)
p.add_argument('--output', required=True)
p.add_argument('--lvis', action='store_true')
a = p.parse_args()
import ultralytics
from ultralytics.models.yolo.detect import DetectionValidator
v = DetectionValidator(args={'save_json': True}, save_dir=Path(a.output).parent)
v.training, v.is_coco, v.is_lvis, v.gdict = False, not a.lvis, a.lvis, None
v.jdict = json.loads(Path(a.pred).read_text())
image_ids = [im['id'] for im in json.loads(Path(a.gt).read_text())['images']]
v.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=[f'{i}.jpg' for i in image_ids]))
start = time.perf_counter()
stats = v.coco_evaluate({}, a.pred, a.gt)
elapsed = time.perf_counter() - start
assert 'fitness' in stats, stats
record = {'seconds': elapsed, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == 'darwin' else 1024),
          'images': len(image_ids), 'predictions': len(v.jdict), 'metrics': stats,
          'scope': 'DetectionValidator.coco_evaluate; predictions retained in jdict; cold COCO object; warm filesystem; no inference',
          'ultralytics_source_version': ultralytics.__version__,
          'versions': {name: importlib.metadata.version(name) for name in ('ultralytics', 'faster-coco-eval', 'ultrafast-pycocotools', 'numpy', 'torch')}}
Path(a.output).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
