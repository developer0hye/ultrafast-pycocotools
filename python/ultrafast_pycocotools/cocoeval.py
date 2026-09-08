"""``COCOeval`` — API-compatible with ``pycocotools.cocoeval.COCOeval``.

The contract is stronger than "compatible": on the same input this produces
the *same bits*, not the same number to a tolerance. Two details carry most of
that weight and are easy to get wrong:

``Params.iouThrs`` / ``Params.recThrs``
    Built with ``np.linspace``, exactly as upstream, and handed to the engine
    rather than reconstructed there. ``np.linspace(0.5, 0.95, 10)`` is not
    ``[0.5 + 0.05 * i]`` — they differ by one ULP at two of the ten points,
    and ``np.linspace(0, 1, 101)`` differs at ten of its 101. Since
    ``searchsorted(..., side='left')`` is a strict comparison, that ULP picks
    a different precision value whenever recall lands exactly on a threshold,
    which is often. Reimplementations that rebuild the grid arithmetically
    land ~1e-6 away on COCO AP.

Stable sorting
    pycocotools sorts with ``kind='mergesort'``; ties in detection score
    resolve to annotation order. The greedy matcher is order-sensitive, so an
    unstable sort changes which ground truth a detection claims and moves AP
    and AR. Every sort in the engine is stable, and :meth:`_prepare` feeds it
    annotations in ``loadAnns(getAnnIds(...))`` order to make "annotation
    order" mean the same thing on both sides.

Deviations from pycocotools, all deliberate and none numeric:

* ``evaluate()`` does the matching and accumulation too, and ``accumulate()``
  publishes the result. Upstream materialises ``K*A*I`` per-image dicts
  between the two, which is where its memory goes; we keep the split in the
  API and not in the memory profile. Pass ``store_eval_imgs=True`` to get
  ``evalImgs`` populated anyway.
* ``ann['segmentation']`` is never rewritten to RLE. Upstream does that in
  place, editing the caller's data; :meth:`computeIoU` converts on the fly
  instead and produces the same masks. (``ignore`` / ``_ignore`` *are* set,
  exactly as upstream sets them, but only on the compatibility path below.)
* ``self._gts``, ``self._dts`` and ``self.ious`` are built on first access
  rather than during ``evaluate()``. The engine does not read them, and
  materialising every IoU matrix is a large part of what makes pycocotools'
  memory profile bad. Code that touches them still works and pays upstream's
  cost for it.

:meth:`computeIoU`, :meth:`computeOks` and :meth:`evaluateImg` are faithful
Python ports kept because they are public API — libraries subclass
``COCOeval`` to override ``evaluateImg`` — not because the fast path uses
them. It does the same work in Rust across all images at once.
"""

from __future__ import annotations

import copy
import datetime
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from . import _ufcoco
from . import mask as maskUtils
from .coco import COCO

__all__ = ["COCOeval", "Params"]

_DET_TYPES = frozenset({"segm", "bbox", "boundary"})


class Params:
    """Evaluation grid. Field names and defaults follow pycocotools."""

    def setDetParams(self) -> None:
        self.imgIds: list = []
        self.catIds: list = []
        # np.arange causes trouble: its data points are slightly larger than
        # the true value. linspace is what upstream uses and what the engine
        # is calibrated against — do not "simplify" these two lines.
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setKpParams(self) -> None:
        self.imgIds = []
        self.catIds = []
        self.iouThrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.recThrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "medium", "large"]
        self.useCats = 1
        self.kpt_oks_sigmas = (
            np.array([
                0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
                0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89,
            ])
            / 10.0
        )

    def __init__(self, iouType: str = "segm"):
        if iouType in _DET_TYPES:
            self.setDetParams()
        elif iouType.startswith("keypoints"):
            self.setKpParams()
        else:
            raise Exception("iouType not supported")
        self.iouType = iouType
        self.useSegm = None  # deprecated upstream; kept for compatibility


class COCOeval:
    def __init__(
        self,
        cocoGt: COCO | None = None,
        cocoDt: COCO | None = None,
        iouType: str = "segm",
        *,
        store_eval_imgs: bool = False,
        lvis_style: bool = False,
        lvis_protocol: str = "official",
        kpt_oks_sigmas: Any = None,
        use_area: bool = True,
        boundary_dilation_ratio: float = 0.02,
        print_function: Callable[[str], None] = print,
    ):
        """
        Args:
            cocoGt / cocoDt: ground truth and detection handles.
            iouType: ``"segm"``, ``"bbox"``, ``"keypoints"``, or the extension
                ``"boundary"``.
            store_eval_imgs: also build the per-image ``evalImgs`` records.
                Off by default because materialising them is what makes
                pycocotools' memory profile bad, and almost nothing reads them.
            lvis_protocol: ``"official"`` uses LVIS's global per-image cap and
                ignore flags. ``"coco"`` preserves faster-coco-eval's per-category
                caps and COCO crowd semantics with federated category filtering.
            kpt_oks_sigmas: override the 17 COCO keypoint sigmas.
            use_area: use the ground truth ``area`` for OKS. Set False for
                CrowdPose-style data with no usable area, which falls back to
                ``0.53 * w * h``.
            boundary_dilation_ratio: only for ``iouType="boundary"``.
            print_function: where ``summarize()`` writes. Extension.
        """
        if not iouType:
            print("iouType not specified. use default iouType segm")
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.evalImgs: list = []
        self.eval: dict = {}
        # Left as plain dicts until something asks for them; `_ensure_prepared`
        # swaps in the populated defaultdicts. `ious` is a property, so the
        # backing field is what gets initialised here — assigning `self.ious`
        # would go through the setter and defeat the laziness.
        self._gts: Any = {}
        self._dts: Any = {}
        self._ious: dict | None = None
        self.params = Params(iouType=iouType)
        self.lvis_style = bool(lvis_style)
        if lvis_protocol not in ("official", "coco"):
            raise ValueError('lvis_protocol must be official or coco')
        self.lvis_protocol = lvis_protocol
        if self.lvis_style and self.lvis_protocol == "official":
            self.params.maxDets = [300]
        self._paramsEval: Params | None = None
        self.stats: Any = []

        self.store_eval_imgs = store_eval_imgs
        self.use_area = use_area
        self.boundary_dilation_ratio = boundary_dilation_ratio
        self.print_function = print_function
        self._engine: Any = None
        self._raw: dict | None = None

        if kpt_oks_sigmas is not None:
            self.params.kpt_oks_sigmas = np.asarray(kpt_oks_sigmas, dtype=np.float64)

        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    # ------------------------------------------------------------------
    # core pipeline
    # ------------------------------------------------------------------

    def _collect(self) -> tuple[list, list, dict]:
        """Collect the annotations to evaluate, in pycocotools' order.

        The order matters: ``getAnnIds`` groups by image in ``imgIds`` order
        and keeps file order within an image, and that is what breaks ties in
        detection score. Do not sort, dedupe or filter further here.
        """
        p = self.params
        cat_filter = p.catIds if p.useCats else []
        def collect(handle):
            if type(handle) is COCO:
                return handle._eval_annotations(p.imgIds, cat_filter)
            return handle.loadAnns(handle.getAnnIds(imgIds=p.imgIds, catIds=cat_filter))

        gts = collect(self.cocoGt)
        dts = [] if self.lvis_style else collect(self.cocoDt)

        if self.lvis_style:
            from ._lvis import collect
            dts = collect(self, gts)

        return gts, dts, self._image_sizes()

    def _image_sizes(self) -> dict:
        img_sizes: dict = {}
        if self.params.iouType in ("segm", "boundary"):
            for src in (self.cocoGt, self.cocoDt):
                for img_id, img in src.imgs.items():
                    if img_id not in img_sizes:
                        img_sizes[int(img_id)] = (int(img["height"]), int(img["width"]))
        return img_sizes

    def _prepare(self) -> None:
        """Populate ``_gts`` / ``_dts``, keyed by ``(imgId, catId)``.

        The engine does not need this — it reads annotations straight into
        Rust — but ``computeIoU`` / ``evaluateImg`` do, and so does anything
        that subclasses ``COCOeval`` and reaches into ``self._gts``. Building
        it is therefore deferred until something asks.

        Unlike pycocotools this does **not** rewrite ``ann['segmentation']``
        to RLE in place. Upstream mutates the caller's annotations there;
        :meth:`computeIoU` converts on the fly instead, which produces the
        same masks without editing data we do not own.
        """
        gts, dts, _ = self._collect()
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        for source_gt in gts:
            gt = dict(source_gt) if self.lvis_style else source_gt
            gt.setdefault("ignore", 0)
            if self.lvis_style and self.lvis_protocol == "official":
                gt["iscrowd"] = 0
            else:
                gt["ignore"] = bool(gt.get("iscrowd", 0))
            if self.params.iouType.startswith("keypoints"):
                gt["ignore"] = (gt.get("num_keypoints", 0) == 0) or gt["ignore"]
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

    def _ensure_prepared(self) -> None:
        if not isinstance(self._gts, defaultdict):
            self._prepare()

    # ------------------------------------------------------------------
    # per-image entry points
    #
    # The engine never calls these; it does the same work in Rust over all
    # images at once. They exist because they are public API — code
    # subclasses ``COCOeval`` to override ``evaluateImg``, and calls
    # ``computeIoU`` directly — so they are kept as faithful ports that
    # produce the same values the engine does.
    # ------------------------------------------------------------------

    def computeIoU(self, imgId, catId):
        """IoU between every detection and ground truth of one (image, category)."""
        self._ensure_prepared()
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [g for cId in p.catIds for g in self._gts[imgId, cId]]
            dt = [d for cId in p.catIds for d in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType in ("segm", "boundary"):
            g = [self.cocoGt.annToRLE(x) for x in gt]
            d = [self.cocoDt.annToRLE(x) for x in dt]
        elif p.iouType == "bbox":
            g = [x["bbox"] for x in gt]
            d = [x["bbox"] for x in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        if p.iouType == "boundary" and len(ious) > 0:
            # Same rule the engine uses: min(mask, boundary), except against
            # crowd ground truth, which keeps the plain mask IoU.
            gb = maskUtils.toBoundary(g, self.boundary_dilation_ratio)
            db = maskUtils.toBoundary(d, self.boundary_dilation_ratio)
            boundary = np.asarray(maskUtils.iou(db, gb, iscrowd))
            ious = np.asarray(ious)
            keep = np.asarray(iscrowd) == 0
            ious[:, keep] = np.minimum(ious[:, keep], boundary[:, keep])
        return ious

    def computeOks(self, imgId, catId):
        """Object keypoint similarity for one (image, category)."""
        self._ensure_prepared()
        p = self.params
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0 : p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        variances = (sigmas * 2) ** 2
        k = len(sigmas)
        for j, gt in enumerate(gts):
            g = np.array(gt["keypoints"])
            xg, yg, vg = g[0::3], g[1::3], g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0, x1 = bb[0] - bb[2], bb[0] + bb[2] * 2
            y0, y1 = bb[1] - bb[3], bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                xd, yd = d[0::3], d[1::3]
                if k1 > 0:
                    dx = xd - xg
                    dy = yd - yg
                else:
                    z = np.zeros(k)
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                area = gt["area"] if self.use_area else bb[3] * bb[2] * 0.53
                e = (dx**2 + dy**2) / variances / (area + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """Greedy match for one (image, category, area range).

        Returns the same dict pycocotools returns, so code that overrides or
        post-processes it keeps working.
        """
        self._ensure_prepared()
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [g for cId in p.catIds for g in self._gts[imgId, cId]]
            dt = [d for cId in p.catIds for d in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            g["_ignore"] = 1 if (g["ignore"] or g["area"] < aRng[0] or g["area"] > aRng[1]) else 0

        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o.get("iscrowd", 0)) for o in gt]
        ious = self.ious[imgId, catId]
        ious = ious[:, gtind] if len(ious) > 0 else ious

        T, G, D = len(p.iouThrs), len(gt), len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))
        if len(ious) != 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, _g in enumerate(gt):
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        if ious[dind, gind] < iou:
                            continue
                        iou = ious[dind, gind]
                        m = gind
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
        if self.lvis_style and (imgId, catId) in set(self._lvis_not_exhaustive):
            a[:] = True
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
        }

    @property
    def ious(self) -> dict:
        """``{(imgId, catId): IoU matrix}``, computed on first access.

        pycocotools fills this during ``evaluate()``; we do not, because
        materialising every matrix is a large part of what makes its memory
        profile bad and nothing in the fast path reads it. Touching this
        attribute computes them, at pycocotools' cost — the point is that code
        which needs it keeps working, not that it is free.
        """
        if self._ious is None:
            p = self.params
            compute = (
                self.computeOks if p.iouType.startswith("keypoints") else self.computeIoU
            )
            cat_ids = p.catIds if p.useCats else [-1]
            self._ious = {
                (imgId, catId): compute(imgId, catId)
                for imgId in p.imgIds
                for catId in cat_ids
            }
        return self._ious

    @ious.setter
    def ious(self, value: dict) -> None:
        self._ious = value

    def evaluate(self) -> None:
        """Run per-image evaluation.

        Unlike pycocotools this also performs matching and accumulation; see
        the module docstring for why. ``accumulate()`` then publishes the
        result, so the usual three-call sequence keeps working unchanged.
        """
        tic = time.time()
        p = self.params
        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            self.print_function(
                f"useSegm (deprecated) is not None. Running {p.iouType} evaluation"
            )
        self.print_function(f"Evaluate annotation type *{p.iouType}*")

        p.imgIds = [int(i) for i in np.unique(p.imgIds)]
        if p.useCats:
            p.catIds = [int(c) for c in np.unique(p.catIds)]
        p.maxDets = sorted(p.maxDets)
        self.params = p

        if (type(self) is COCOeval and not self.lvis_style
                and type(self.cocoGt) is COCO and type(self.cocoDt) is COCO
                and (self.cocoGt._compact is not None or self.cocoDt._compact is not None)):
            categories = p.catIds if p.useCats else []
            gts = (self.cocoGt._compact if self.cocoGt._compact is not None else
                   self.cocoGt._eval_annotations(p.imgIds, categories))
            dts = (self.cocoDt._compact if self.cocoDt._compact is not None else
                   self.cocoDt._eval_annotations(p.imgIds, categories))
            img_sizes = self._image_sizes()
        else:
            gts, dts, img_sizes = self._collect()
        # A fresh run must not serve stale per-image views.
        self._gts, self._dts, self._ious = {}, {}, None

        sigmas = getattr(p, "kpt_oks_sigmas", np.zeros(0))
        self._engine = _ufcoco.Evaluator(
            gts,
            dts,
            img_sizes,
            p.imgIds,
            [int(c) for c in p.catIds],
            [float(t) for t in p.iouThrs],
            [float(t) for t in p.recThrs],
            [int(m) for m in p.maxDets],
            [[float(a[0]), float(a[1])] for a in p.areaRng],
            bool(p.useCats),
            p.iouType,
            [float(s) for s in np.asarray(sigmas).ravel()],
            bool(self.use_area),
            float(self.boundary_dilation_ratio),
        )
        if self.lvis_style:
            coco_protocol = self.lvis_protocol == "coco"
            self._engine.configure_lvis(
                [bool(gt.get("iscrowd" if coco_protocol else "ignore", 0)) for gt in gts],
                self._lvis_not_exhaustive, coco_protocol)
        # Native extraction owns the scalar/geometry data; release temporary
        # annotation-reference lists before allocating complete output tensors.
        del gts, dts
        self._raw = self._engine.run(self.store_eval_imgs)
        if self.store_eval_imgs:
            self.evalImgs = self._raw["evalImgs"]
            for item in self.evalImgs:
                if item is not None:
                    item["aRng"] = list(p.areaRng[item["aRng"]])
        self._paramsEval = copy.deepcopy(self.params)
        self.print_function(f"DONE (t={time.time() - tic:0.2f}s).")

    def accumulate(self, p: Params | None = None) -> None:
        """Publish the accumulated curves into ``self.eval``."""
        tic = time.time()
        if self._raw is None:
            self.print_function("Please run evaluate() first")
            return
        if p is not None and p is not self.params:
            raise NotImplementedError(
                "accumulate(p) with parameters other than self.params is not "
                "supported; set COCOeval.params before calling evaluate()"
            )
        p = self.params
        self.eval = {
            "params": p,
            "counts": list(self._raw["counts"]),
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": self._raw["precision"],
            "recall": self._raw["recall"],
            "scores": self._raw["scores"],
        }
        self.print_function(f"DONE (t={time.time() - tic:0.2f}s).")

    def _summarize(self, ap=1, iouThr=None, areaRng="all", maxDets=100) -> float:
        p = self.params
        iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
        titleStr = "Average Precision" if ap == 1 else "Average Recall"
        typeStr = "(AP)" if ap == 1 else "(AR)"
        iouStr = (
            f"{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}"
            if iouThr is None
            else f"{iouThr:0.2f}"
        )

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if self.lvis_style and len(aind) == len(mind) == 1:
            from ._lvis import valid_values
            array = self.eval["precision" if ap == 1 else "recall"]
            thresholds = (list(range(array.shape[0])) if iouThr is None
                          else np.where(iouThr == p.iouThrs)[0].tolist())
            valid = valid_values(array, thresholds, list(range(array.shape[-3])), aind[0], mind[0])
            if valid is not None:
                mean_s = float(np.mean(valid)) if valid.size else -1.0
                self.print_function(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                return mean_s
        if ap == 1:
            s = self.eval["precision"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            s = self.eval["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        mean_s = -1 if len(s[s > -1]) == 0 else np.mean(s[s > -1])
        self.print_function(
            iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        )
        return float(mean_s)

    def summarize(self) -> None:
        """Compute and print the 12 (detection) or 10 (keypoint) summary
        metrics, in pycocotools' order and format."""

        if self.lvis_style and self.lvis_protocol == "official":
            from ._lvis import summarize
            summarize(self)
            return

        def _summarizeDets():
            md = self.params.maxDets
            # Upstream indexes maxDets[2] unconditionally and raises on any
            # other length; fall back to the largest so custom maxDets work.
            m2 = md[2] if len(md) > 2 else md[-1]
            stats = np.zeros((12,))
            stats[0] = self._summarize(1)
            stats[1] = self._summarize(1, iouThr=0.5, maxDets=m2)
            stats[2] = self._summarize(1, iouThr=0.75, maxDets=m2)
            stats[3] = self._summarize(1, areaRng="small", maxDets=m2)
            stats[4] = self._summarize(1, areaRng="medium", maxDets=m2)
            stats[5] = self._summarize(1, areaRng="large", maxDets=m2)
            stats[6] = self._summarize(0, maxDets=md[0])
            stats[7] = self._summarize(0, maxDets=md[1] if len(md) > 1 else md[-1])
            stats[8] = self._summarize(0, maxDets=m2)
            stats[9] = self._summarize(0, areaRng="small", maxDets=m2)
            stats[10] = self._summarize(0, areaRng="medium", maxDets=m2)
            stats[11] = self._summarize(0, areaRng="large", maxDets=m2)
            return stats

        def _summarizeKps():
            md = self.params.maxDets[-1]
            stats = np.zeros((10,))
            stats[0] = self._summarize(1, maxDets=md)
            stats[1] = self._summarize(1, maxDets=md, iouThr=0.5)
            stats[2] = self._summarize(1, maxDets=md, iouThr=0.75)
            stats[3] = self._summarize(1, maxDets=md, areaRng="medium")
            stats[4] = self._summarize(1, maxDets=md, areaRng="large")
            stats[5] = self._summarize(0, maxDets=md)
            stats[6] = self._summarize(0, maxDets=md, iouThr=0.5)
            stats[7] = self._summarize(0, maxDets=md, iouThr=0.75)
            stats[8] = self._summarize(0, maxDets=md, areaRng="medium")
            stats[9] = self._summarize(0, maxDets=md, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType in _DET_TYPES:
            self.stats = _summarizeDets()
        elif iouType.startswith("keypoints"):
            self.stats = _summarizeKps()
        else:
            raise ValueError(f"unsupported iouType {iouType!r}")

        if self.lvis_style and self.lvis_protocol == "coco":
            from ._lvis import frequency_stats
            self.stats[0] = self._summarize(1, maxDets=self.params.maxDets[-1])
            self._lvis_frequency_stats = frequency_stats(self)

    def run(self) -> None:
        """``evaluate()`` + ``accumulate()`` + ``summarize()``."""
        self.evaluate()
        self.accumulate()
        self.summarize()

    def __str__(self) -> str:
        self.summarize()
        return ""

    # ------------------------------------------------------------------
    # extensions
    # ------------------------------------------------------------------

    @property
    def stats_as_dict(self) -> dict[str, float]:
        """The summary metrics keyed by name instead of by position."""
        if self.lvis_style and self.lvis_protocol == "official":
            from ._lvis import names
            labels = names(self.params.maxDets[0])
        elif self.params.iouType in _DET_TYPES:
            labels = [
                "AP", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
                f"AR_{self.params.maxDets[0]}",
                f"AR_{self.params.maxDets[1] if len(self.params.maxDets) > 1 else self.params.maxDets[-1]}",
                f"AR_{self.params.maxDets[2] if len(self.params.maxDets) > 2 else self.params.maxDets[-1]}",
                "AR_small", "AR_medium", "AR_large",
            ]
        else:
            labels = [
                "AP", "AP_50", "AP_75", "AP_medium", "AP_large",
                "AR", "AR_50", "AR_75", "AR_medium", "AR_large",
            ]
        values = {k: float(v) for k, v in zip(labels, self.stats)}
        if self.lvis_style and self.lvis_protocol == "coco":
            values.update(getattr(self, '_lvis_frequency_stats', {}))
        aliases = {'AP_all': 'AP', 'AP50': 'AP_50', 'AP75': 'AP_75',
                   'APs': 'AP_small', 'APm': 'AP_medium', 'APl': 'AP_large'}
        for canonical, legacy in aliases.items():
            if canonical in values:
                values.setdefault(legacy, values[canonical])
            elif legacy in values:
                values[canonical] = values[legacy]
        for key in list(values):
            if key.startswith('AR_') and key[3:].isdigit():
                values[f'AR@{key[3:]}'] = values[key]
        return values

    def per_category_stats(self, area: str = "all", max_dets: int | None = None) -> dict:
        """AP / AP50 / AP75 / AR per category.

        The usual reason to want this is to find the class that is dragging
        mAP down; the aggregate number cannot tell you.
        """
        if not self.eval:
            raise Exception("Please run accumulate() first")
        p = self.params
        aind = p.areaRngLbl.index(area)
        mdet = p.maxDets[-1] if max_dets is None else max_dets
        mind = p.maxDets.index(mdet)
        prec = self.eval["precision"]
        rec = self.eval["recall"]
        i50 = int(np.where(np.isclose(p.iouThrs, 0.50))[0][0])
        i75 = int(np.where(np.isclose(p.iouThrs, 0.75))[0][0])

        out = {}
        cat_ids = p.catIds if p.useCats else [-1]
        names = {}
        if p.useCats and self.cocoGt is not None:
            names = {c["id"]: c.get("name", str(c["id"])) for c in self.cocoGt.loadCats(cat_ids)}
        for k, cid in enumerate(cat_ids):
            s = prec[:, :, k, aind, mind]
            r = rec[:, k, aind, mind]

            def _mean(x):
                x = x[x > -1]
                return float(np.mean(x)) if x.size else float("nan")

            out[int(cid)] = {
                "name": names.get(int(cid), str(cid)),
                "AP": _mean(s),
                "AP_50": _mean(s[i50]),
                "AP_75": _mean(s[i75]),
                "AR": _mean(r),
            }
        return out

    def pr_curve(
        self,
        cat_id: int | None = None,
        iou_thr: float = 0.5,
        area: str = "all",
        max_dets: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Precision / recall / score arrays for one setting.

        ``recall`` is ``params.recThrs``; ``precision`` and ``score`` are the
        interpolated values at those recalls, which is exactly what AP
        averages. Averaged over categories when ``cat_id`` is None.
        """
        if not self.eval:
            raise Exception("Please run accumulate() first")
        p = self.params
        t = int(np.where(np.isclose(p.iouThrs, iou_thr))[0][0])
        aind = p.areaRngLbl.index(area)
        mdet = p.maxDets[-1] if max_dets is None else max_dets
        mind = p.maxDets.index(mdet)
        prec = self.eval["precision"][t, :, :, aind, mind]
        sc = self.eval["scores"][t, :, :, aind, mind]
        if cat_id is None:
            valid = prec > -1
            with np.errstate(invalid="ignore"):
                pr = np.where(valid.any(axis=1), (prec * valid).sum(1) / np.maximum(valid.sum(1), 1), -1.0)
                ss = np.where(valid.any(axis=1), (sc * valid).sum(1) / np.maximum(valid.sum(1), 1), -1.0)
        else:
            k = list(p.catIds).index(cat_id)
            pr = prec[:, k]
            ss = sc[:, k]
        return {"recall": np.asarray(p.recThrs), "precision": pr, "score": ss}

    def per_instance(
        self,
        iou_thr: float = 0.5,
        area: str = "all",
        max_dets: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Per-detection and per-ground-truth verdicts, column-wise.

        Returns ``(detections, ground_truths)``:

        * detections: ``image_id``, ``category_id``, ``dt_id``, ``score``,
          ``gt_id`` (-1 when unmatched), ``iou``, ``ignore``
        * ground_truths: ``image_id``, ``category_id``, ``gt_id``, ``ignore``,
          ``matched``

        ``ignore`` is the field that makes this trustworthy. A detection that
        matched a crowd region, or that falls outside the area range, is
        neither a true nor a false positive in COCO's arithmetic. Anything
        built on top of this must respect that or it will contradict the AP
        printed beside it.

        Detections past ``max_dets`` for an (image, category) do not appear at
        all, because the AP arithmetic never saw them either.
        """
        if self._engine is None:
            raise Exception("Please run evaluate() first")
        p = self.params
        t = int(np.where(np.isclose(p.iouThrs, iou_thr))[0][0])
        aind = p.areaRngLbl.index(area)
        mdet = p.maxDets[-1] if max_dets is None else max_dets
        return self._engine.per_instance(t, aind, int(mdet))

    def matches(
        self,
        iou_thr: float = 0.5,
        area: str = "all",
        max_dets: int | None = None,
    ) -> dict[str, np.ndarray]:
        """True positives: detections matched to a ground truth that counts.

        A filtered view of :meth:`per_instance` — matched *and* not ignored, so
        the row count is exactly the TP count behind AP at this threshold.
        """
        dets, _ = self.per_instance(iou_thr=iou_thr, area=area, max_dets=max_dets)
        keep = (dets["gt_id"] >= 0) & (~dets["ignore"])
        return {k: v[keep] for k, v in dets.items()}

    def confusion_matrix(
        self,
        iou_thr: float = 0.5,
        score_thr: float = 0.0,
        area: str = "all",
        max_dets: int | None = None,
    ) -> tuple[np.ndarray, list]:
        """Category counts with a background row and column.

        Cells, using COCO's own accounting so the numbers reconcile with AP:

        * ``cm[k, k]``   — detections of category k matched to a real ground
          truth (true positives)
        * ``cm[k, -1]``  — detections of category k matched to nothing (false
          positives). Detections that matched a crowd region or fell outside
          the area range are *not* counted, exactly as AP does not count them.
        * ``cm[-1, k]``  — ground truths of category k that nothing found
          (false negatives), ignoring crowd and out-of-range ground truth.

        Matching is within a category, so off-diagonal cells stay empty:
        genuine cross-category confusion requires a ``useCats=0`` evaluation,
        and this deliberately does not fake it.
        """
        dets, gts = self.per_instance(iou_thr=iou_thr, area=area, max_dets=max_dets)
        p = self.params
        cat_ids = list(p.catIds) if p.useCats else [-1]
        index = {int(c): i for i, c in enumerate(cat_ids)}
        n = len(cat_ids)
        cm = np.zeros((n + 1, n + 1), dtype=np.int64)

        live = (~dets["ignore"]) & (dets["score"] >= score_thr)
        for cid, gid in zip(dets["category_id"][live], dets["gt_id"][live]):
            k = index.get(int(cid))
            if k is None:
                continue
            cm[k, k if gid >= 0 else n] += 1

        # A ground truth counts as missed only if the detection that would
        # have matched it was not suppressed by the score threshold.
        found = set(
            int(g) for g in dets["gt_id"][live & (dets["gt_id"] >= 0)].tolist()
        )
        for cid, gid, ign in zip(gts["category_id"], gts["gt_id"], gts["ignore"]):
            if ign or int(gid) in found:
                continue
            k = index.get(int(cid))
            if k is not None:
                cm[n, k] += 1
        labels = [str(c) for c in cat_ids] + ["background"]
        return cm, labels

    def mean_iou(self, iou_thr: float = 0.5, area: str = "all") -> float:
        """Mean IoU over true positives — localisation quality, independent of
        how many objects were found."""
        m = self.matches(iou_thr=iou_thr, area=area)
        return float(np.mean(m["iou"])) if len(m["iou"]) else float("nan")
