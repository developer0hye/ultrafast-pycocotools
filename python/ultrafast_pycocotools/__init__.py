"""ultrafast-pycocotools — a bit-exact, fast drop-in replacement for pycocotools.

Three ways to use it:

Import it directly::

    from ultrafast_pycocotools import COCO, COCOeval

    gt = COCO("instances_val2017.json")
    dt = gt.loadRes("detections.json")
    ev = COCOeval(gt, dt, "bbox")
    ev.run()

Alias the module in code you already have::

    import ultrafast_pycocotools as pycocotools

Or take over the ``pycocotools`` import name for the whole process, which is
what you want when a third-party library imports it for you (torchvision,
detectron2, mmdetection)::

    from ultrafast_pycocotools import init_as_pycocotools
    init_as_pycocotools()

``init_as_pycocotools`` must run before whatever imports ``pycocotools``,
since Python caches modules on first import.
"""

from __future__ import annotations

import sys

from . import coco, cocoeval, mask
from .coco import COCO
from .cocoeval import COCOeval, Params

__version__ = "0.1.6"

__all__ = [
    "COCO",
    "COCOeval",
    "Params",
    "coco",
    "cocoeval",
    "mask",
    "init_as_pycocotools",
    "__version__",
]


def init_as_pycocotools() -> None:
    """Register this package under the ``pycocotools`` import name.

    Also registers ``pycocotools._mask``, because some code imports the
    private extension module directly rather than going through
    ``pycocotools.mask``.
    """
    import ultrafast_pycocotools

    sys.modules["pycocotools"] = ultrafast_pycocotools
    sys.modules["pycocotools.coco"] = coco
    sys.modules["pycocotools.cocoeval"] = cocoeval
    sys.modules["pycocotools.mask"] = mask
    sys.modules["pycocotools._mask"] = mask
