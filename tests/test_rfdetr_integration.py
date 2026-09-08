"""Exercise RF-DETR's real metric lifecycle with a per-instance backend switch."""
import copy
import pickle

import pytest

pytest.importorskip('rfdetr', reason='optional RF-DETR integration dependencies')
pytest.importorskip('pytorch_lightning')
import torch
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

from ultrafast_pycocotools import COCO
from ultrafast_pycocotools.integrations.rfdetr import use_ultrafast


def inputs(segmentation=False):
    preds = [{'boxes': torch.tensor([[1., 2., 11., 12.], [1., 2., 11., 12.], [20., 20., 25., 25.]]),
              'labels': torch.tensor([1, 1, 3]), 'scores': torch.tensor([.8, .8, .4])},
             {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros(0, dtype=torch.int64),
              'scores': torch.zeros(0)}]
    targets = [{'boxes': torch.tensor([[1., 2., 11., 12.], [20., 20., 25., 25.]]),
                'labels': torch.tensor([1, 3]), 'iscrowd': torch.tensor([0, 1]),
                'area': torch.tensor([100., 25.])},
               {'boxes': torch.tensor([[0., 0., 4., 4.]]), 'labels': torch.tensor([1]),
                'iscrowd': torch.tensor([0]), 'area': torch.tensor([16.])}]
    if segmentation:
        for items in [preds, targets]:
            for item in items:
                masks = torch.zeros((len(item['boxes']), 32, 32), dtype=torch.bool)
                for i, (x0, y0, x1, y1) in enumerate(item['boxes'].int().tolist()):
                    masks[i, y0:y1, x0:x1] = True
                item['masks'] = masks
    return preds, targets


@pytest.mark.parametrize('iou_type', ['bbox', 'segm', ('bbox', 'segm')])
@pytest.mark.parametrize('max_dets', [100, 500])
def test_metric_update_compute_reset_and_pickle(iou_type, max_dets):
    preds, targets = inputs(iou_type != 'bbox')
    reference = OnePassCocoMeanAveragePrecision(iou_type=iou_type, class_metrics=True,
                                               max_detection_thresholds=[1, 10, max_dets])
    actual = use_ultrafast(copy.deepcopy(reference))
    assert actual._coco_backend.coco is COCO
    assert reference._coco_backend.coco is not COCO
    actual = pickle.loads(pickle.dumps(actual))
    for _ in range(2):
        for metric in [reference, actual]:
            for i in range(len(preds)):
                metric.update(preds[i:i+1], targets[i:i+1])
            metric.merge_distributed_state()
        expected, observed = reference.compute(), actual.compute()
        assert expected.keys() == observed.keys()
        for key in expected:
            assert expected[key].dtype == observed[key].dtype
            assert torch.equal(expected[key], observed[key]), key
        reference.reset()
        actual.reset()


def test_backend_switch_rejects_existing_state():
    metric = OnePassCocoMeanAveragePrecision()
    metric.update(*inputs())
    with pytest.raises(ValueError, match='before updating'):
        use_ultrafast(metric)


def test_backend_switch_rejects_unrelated_metric():
    with pytest.raises(TypeError, match='OnePass'):
        use_ultrafast(object())


def test_contract_failure_restores_original_backend(monkeypatch):
    metric = OnePassCocoMeanAveragePrecision()
    original = metric._coco_backend
    def fail():
        raise RuntimeError('changed contract')
    monkeypatch.setattr(metric, '_validate_private_contract', fail)
    with pytest.raises(RuntimeError, match='changed contract'):
        use_ultrafast(metric)
    assert metric._coco_backend is original


def distributed_worker(rank, rendezvous, output):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        predictions, targets = inputs(segmentation=True)
        metric = use_ultrafast(OnePassCocoMeanAveragePrecision(
            iou_type=('bbox', 'segm'), class_metrics=True, max_detection_thresholds=[1, 10, 500]))
        metric.update(predictions[rank:rank+1], targets[rank:rank+1])
        metric.merge_distributed_state()
        result = metric.compute()
        torch.save(result, output + str(rank) + '.pt')
    finally:
        dist.destroy_process_group()


def test_two_rank_cpu_merge_matches_single_process(tmp_path):
    rendezvous = (tmp_path/'rendezvous').as_uri()
    output = str(tmp_path/'rank')
    torch.multiprocessing.spawn(distributed_worker, args=(rendezvous, output), nprocs=2, join=True)
    reference = OnePassCocoMeanAveragePrecision(iou_type=('bbox', 'segm'), class_metrics=True,
                                               max_detection_thresholds=[1, 10, 500])
    reference.update(*inputs(segmentation=True))
    expected = reference.compute()
    for rank in range(2):
        actual = torch.load(output + str(rank) + '.pt', weights_only=True)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert torch.equal(actual[key], expected[key]), key
