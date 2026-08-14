"""RLE mask utilities — the ``pycocotools.mask`` surface.

Same functions, same argument shapes, same return dtypes, same ``bytes``
``counts``. The implementation is a Rust port of ``maskApi.c`` that reproduces
the C arithmetic exactly, including the parts that look like bugs (see
``rust/ufcoco-core/src/rle.rs``).

Everything below is a thin dispatcher: ``pycocotools.mask`` itself is only a
few lines over ``pycocotools._mask``, and the list-vs-single-object rules it
implements are load-bearing for callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _ufcoco as _mask

__all__ = [
    "encode",
    "decode",
    "merge",
    "area",
    "toBbox",
    "iou",
    "frPyObjects",
    "frBbox",
    "frPoly",
    "frUncompressedRLE",
    "toBoundary",
]

iou = _mask.iou
merge = _mask.merge
frPyObjects = _mask.frPyObjects
frBbox = _mask.frBbox
frPoly = _mask.frPoly
frUncompressedRLE = _mask.frUncompressedRLE


def encode(bimask: np.ndarray) -> Any:
    """Encode a binary mask, or a stack of them, as RLE.

    A 2-D ``(h, w)`` input returns one RLE dict; a 3-D ``(h, w, n)`` input
    returns a list of ``n`` dicts. That asymmetry is pycocotools' and is
    preserved.
    """
    if bimask.ndim == 3:
        return _mask.encode(bimask)
    if bimask.ndim == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order="F"))[0]
    raise ValueError("encode() expects a 2-D or 3-D uint8 array")


def decode(rleObjs: Any) -> np.ndarray:
    """Decode RLE back to a binary mask, in Fortran order."""
    if isinstance(rleObjs, list):
        return _mask.decode(rleObjs)
    return _mask.decode([rleObjs])[:, :, 0]


def area(rleObjs: Any) -> Any:
    """Mask area(s), as ``uint32`` like the reference."""
    if isinstance(rleObjs, list):
        return _mask.area(rleObjs)
    return _mask.area([rleObjs])[0]


def toBbox(rleObjs: Any) -> np.ndarray:
    """Tight bounding box(es) ``[x, y, w, h]`` around the mask(s)."""
    if isinstance(rleObjs, list):
        return _mask.toBbox(rleObjs)
    return _mask.toBbox([rleObjs])[0]


def toBoundary(rleObjs: Any, dilation_ratio: float = 0.02) -> Any:
    """Boundary mask(s) for Boundary IoU (Cheng et al., CVPR 2021).

    Extension — pycocotools has no equivalent. The boundary is the mask minus
    its erosion by ``round(dilation_ratio * image_diagonal)`` pixels, computed
    on a zero-padded canvas so a mask touching the image border still has a
    boundary there.
    """
    if isinstance(rleObjs, list):
        return _mask.toBoundary(rleObjs, dilation_ratio)
    return _mask.toBoundary([rleObjs], dilation_ratio)[0]
