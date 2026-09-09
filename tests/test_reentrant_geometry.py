"""Numeric conversion may re-enter Python and resize the coordinate list."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('geometry', ['bbox', 'polygon'])
def test_coordinate_list_resize_during_float_conversion(geometry):
    # Isolate a regression in native list access from the pytest process.
    script = r'''
import sys
from ultrafast_pycocotools._ufcoco import Evaluator

coordinates = []
class Coordinate:
    def __float__(self):
        coordinates.clear()
        return 0.0

geometry = sys.argv[1]
coordinates.extend([Coordinate(), 0., 1., 1.] if geometry == 'bbox'
                   else [Coordinate(), 0., 1., 0., 1., 1.])
ann = dict(id=1, image_id=1, category_id=1, area=1., iscrowd=0,
           bbox=coordinates if geometry == 'bbox' else [0., 0., 1., 1.])
if geometry == 'polygon':
    ann['segmentation'] = [coordinates]
try:
    Evaluator([ann], [], {1: (4, 4)}, [1], [1], [.5], [0., 1.],
              [1, 10, 100], [[0., 1e10]], True,
              'bbox' if geometry == 'bbox' else 'segm', [])
except IndexError:
    print('safe bounds error')
else:
    raise AssertionError('coordinate list resizing must be detected')
'''
    result = subprocess.run(
        [sys.executable, '-c', script, geometry],
        capture_output=True, text=True, timeout=30,
        env=dict(os.environ, RAYON_NUM_THREADS='2'),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'safe bounds error'
