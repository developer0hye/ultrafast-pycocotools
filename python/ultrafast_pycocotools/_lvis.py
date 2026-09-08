"""Federated LVIS protocol on the shared native COCO matching engine."""
from __future__ import annotations

import numpy as np


def collect(evaluator, ground_truth):
    p, gt, dt = evaluator.params, evaluator.cocoGt, evaluator.cocoDt
    if not p.useCats or p.iouType not in ('bbox', 'segm'):
        raise ValueError('LVIS supports category-aware bbox and segm evaluation')
    coco_protocol = evaluator.lvis_protocol == 'coco'
    if not coco_protocol and (len(p.maxDets) != 1 or p.maxDets[0] <= 0):
        raise ValueError('LVIS maxDets must contain one positive per-image limit, normally [300]')
    frequencies = {}
    for category_id in p.catIds:
        frequency = gt.cats[category_id].get('frequency')
        if frequency not in ('r', 'c', 'f'):
            raise ValueError(f'LVIS category {category_id} needs frequency r, c or f')
        frequencies[category_id] = frequency
    selected_categories = set(p.catIds)
    positives = {}
    for ann in ground_truth:
        positives.setdefault(ann['image_id'], set()).add(ann['category_id'])
    detections, not_exhaustive = [], []
    for image_id in p.imgIds:
        image = gt.imgs[image_id]
        if not coco_protocol and ('neg_category_ids' not in image or 'not_exhaustive_category_ids' not in image):
            raise ValueError(f'LVIS image {image_id} needs federated annotation metadata')
        verified = set(image.get('neg_category_ids', [])) | positives.get(image_id, set())
        not_exhaustive.extend((image_id, category) for category in image.get('not_exhaustive_category_ids', []))
        # The global per-image cap precedes both category selection and federated filtering.
        # Python's stable sort keeps original prediction order for tied scores.
        candidates = dt.imgToAnns.get(image_id, [])
        if not coco_protocol and len(candidates) > p.maxDets[0]:
            candidates = sorted(candidates, key=lambda ann: ann['score'], reverse=True)[:p.maxDets[0]]
        detections.extend(ann for ann in candidates if ann['category_id'] in selected_categories
                          and ann['category_id'] in verified)
    evaluator._lvis_not_exhaustive = not_exhaustive
    evaluator._lvis_freq_groups = {label: [i for i, cat in enumerate(p.catIds) if frequencies[cat] == label]
                                   for label in ('r', 'c', 'f')}
    return detections


def frequency_stats(evaluator):
    """Reduce COCO-style LVIS frequency AP at the largest per-category limit."""
    values = {}
    for frequency, categories in evaluator._lvis_freq_groups.items():
        precision = evaluator.eval['precision'][:, :, categories, 0, -1]
        valid = precision[precision > -1]
        values['AP' + frequency] = float(np.mean(valid)) if valid.size else -1.0
    return values


def names(max_dets):
    return ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'APr', 'APc', 'APf',
            f'AR@{max_dets}', f'ARs@{max_dets}', f'ARm@{max_dets}', f'ARl@{max_dets}']


def summarize(evaluator):
    if not evaluator.eval:
        raise RuntimeError('Please run accumulate() first')
    p = evaluator.params

    def mean(ap=True, threshold=None, area='all', frequency=None):
        # Native arrays keep the COCO maxDets axis; LVIS has exactly one limit.
        values = evaluator.eval['precision' if ap else 'recall'][..., 0]
        if threshold is not None:
            values = values[np.where(p.iouThrs == threshold)[0]]
        area_index = [i for i, label in enumerate(p.areaRngLbl) if label == area]
        if ap and frequency is not None:
            values = values[:, :, evaluator._lvis_freq_groups[frequency], area_index]
        elif ap:
            values = values[:, :, :, area_index]
        else:
            values = values[:, :, area_index]
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else -1.0

    values = [mean(), mean(threshold=.5), mean(threshold=.75),
              *[mean(area=area) for area in ('small', 'medium', 'large')],
              *[mean(frequency=frequency) for frequency in ('r', 'c', 'f')],
              mean(False), *[mean(False, area=area) for area in ('small', 'medium', 'large')]]
    evaluator.stats = np.asarray(values, dtype=np.float64)
    for key, value in zip(names(p.maxDets[0]), values):
        evaluator.print_function(f'{key}: {value:.3f}')
