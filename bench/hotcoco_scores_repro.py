"""Reproduce an ignored-cell difference in the sampled COCO scores array.

Image 2 has no GT and one high-score detection outside the small-area range.
Its ignored score still occupies a rank in pycocotools accumulation.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    from hotcoco import COCO as HotCOCO, COCOeval as HotEval
    from pycocotools.coco import COCO as RefCOCO
    from pycocotools.cocoeval import COCOeval as RefEval
    from ultrafast_pycocotools import COCO as UltraCOCO, COCOeval as UltraEval
    gt_data = dict(images=[dict(id=1, width=200, height=200), dict(id=2, width=200, height=200)],
                   categories=[dict(id=1, name='object')],
                   annotations=[dict(id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10],
                                     area=100, iscrowd=0)])
    predictions = [dict(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.5),
                   dict(image_id=2, category_id=1, bbox=[0, 0, 100, 100], score=0.9)]
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        gt_path, dt_path = Path(directory)/'gt.json', Path(directory)/'dt.json'
        gt_path.write_text(json.dumps(gt_data))
        dt_path.write_text(json.dumps(predictions))
        for name, coco, evaluator in [('pycocotools', RefCOCO, RefEval),
                                      ('hotcoco', HotCOCO, HotEval),
                                      ('ultrafast', UltraCOCO, UltraEval)]:
            gt = coco(str(gt_path))
            ev = evaluator(gt, gt.loadRes(str(dt_path)), 'bbox')
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            arrays = ev.eval
            results[name] = dict(AP=float(ev.stats[0]),
                                 small_precision_recall0=float(np.asarray(arrays['precision'])[0, 0, 0, 1, 2]),
                                 small_score_recall0=float(np.asarray(arrays['scores'])[0, 0, 0, 1, 2]))
    report = dict(gt=gt_data, detections=predictions, results=results,
                  score_index=[0, 0, 0, 1, 2],
                  interpretation='IoU 0.5, recall 0, category 1, small area, maxDets 100')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
