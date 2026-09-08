"""The public benchmark verifier must reject wrong inputs and tiny metric drift."""
import json

import numpy as np
import pytest

from reproduce import compare


def pair(tmp_path):
    paths = [tmp_path / 'ref', tmp_path / 'fast']
    for p in paths:
        p.mkdir()
        (p / 'result.json').write_text(json.dumps({'gt_sha256': 'same_gt',
            'pred_sha256': 'same_predictions', 'images': 2, 'detections': 3}))
        np.savez(p / 'arrays.npz', **{k: np.array([.5, -1.], dtype=np.float64)
                                    for k in ('precision', 'recall', 'scores', 'stats')})
    return paths


def test_reproduce_rejects_one_ulp_precision_change(tmp_path):
    a, b = pair(tmp_path)
    assert all(compare(a, b)['byte_identical'].values())
    with np.load(b / 'arrays.npz') as z:
        values = {k: z[k].copy() for k in z.files}
    values['precision'][0] = np.nextafter(.5, 1.)
    np.savez(b / 'arrays.npz', **values)
    with pytest.raises(ValueError, match='Array parity failed'):
        compare(a, b)


def test_reproduce_rejects_different_predictions(tmp_path):
    a, b = pair(tmp_path)
    p = b / 'result.json'
    values = json.loads(p.read_text())
    values['pred_sha256'] = 'different_input'
    p.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='Inputs differ'):
        compare(a, b)
