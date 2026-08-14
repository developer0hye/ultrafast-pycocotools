"""``COCO`` — the dataset handle, API-compatible with ``pycocotools.coco.COCO``.

This class stays in Python and stays dict-based on purpose. Its public
attributes (``dataset``, ``anns``, ``imgs``, ``cats``, ``imgToAnns``,
``catToImgs``) are not merely readable in pycocotools, they are *the* interface
a lot of downstream code uses: detectron2, mmdetection and torchvision all
reach into ``coco.anns[id]`` and mutate the dicts they find there. Replacing
them with a Rust-backed view would be faster and would break all of that.

So the speed work happens where it actually pays and where nobody is looking:
the evaluator (see :mod:`ultrafast_pycocotools.cocoeval`) reads these dicts
once into a compact Rust representation and never touches Python again.

Ordering note: :meth:`getAnnIds` returns annotations grouped by image in the
order the image ids were given, and within an image in file order. The
evaluator relies on that order to break ties in detection score the same way
pycocotools does, so it is behaviour, not an accident.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

from . import _ufcoco
from . import mask as maskUtils

__all__ = ["COCO"]


def _is_array_like(obj: Any) -> bool:
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


def load_json(path: str | os.PathLike) -> Any:
    """Read a JSON file into Python objects.

    Uses the Rust reader, which builds the same objects ``json.load`` would but
    without an intermediate parse tree — worth about 2x on a large annotation
    file, where loading otherwise outweighs the evaluation itself.

    Float parsing agrees with CPython bit for bit (both are correctly
    rounded), which matters because an annotation's ``area`` decides its
    small/medium/large bucket. ``tests/test_json_loader.py`` checks that
    against a real file.

    Falls back to the standard library if anything goes wrong, so a malformed
    file still produces the error message people recognise.
    """
    try:
        return _ufcoco.load_json(str(path))
    except (ValueError, OSError):
        with open(path, "r") as f:
            return json.load(f)


class COCO:
    def __init__(self, annotation_file: str | os.PathLike | None = None, *, verbose: bool = True):
        """
        Args:
            annotation_file: path to a COCO-format JSON file. ``None`` builds
                an empty handle, which is how :meth:`loadRes` starts.
            verbose: print the progress lines pycocotools prints. Extension;
                pycocotools has no way to silence them.
        """
        self.dataset: dict = {}
        self.anns: dict = {}
        self.cats: dict = {}
        self.imgs: dict = {}
        self.imgToAnns: dict = defaultdict(list)
        self.catToImgs: dict = defaultdict(list)
        self.verbose = verbose
        if annotation_file is not None:
            self._log("loading annotations into memory...")
            tic = time.time()
            dataset = load_json(annotation_file)
            assert isinstance(dataset, dict), (
                f"annotation file format {type(dataset)} not supported"
            )
            self._log(f"Done (t={time.time() - tic:0.2f}s)")
            self.dataset = dataset
            self.createIndex()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def createIndex(self) -> None:
        self._log("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        self._log("index created!")
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self) -> None:
        for key, value in self.dataset["info"].items():
            print(f"{key}: {value}")

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None) -> list:
        """Annotation ids passing every supplied filter.

        Note the area filter is strict on both ends (``>`` and ``<``), unlike
        the evaluator's area ranges which are inclusive. That inconsistency is
        pycocotools' and is preserved.
        """
        imgIds = imgIds if _is_array_like(imgIds) else [imgIds]
        catIds = catIds if _is_array_like(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            if len(imgIds) != 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            if len(catIds) != 0:
                catIds = set(catIds)
                anns = [ann for ann in anns if ann["category_id"] in catIds]
            if len(areaRng) != 0:
                anns = [
                    ann for ann in anns if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
                ]
        if iscrowd is not None:
            return [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
        return [ann["id"] for ann in anns]

    def getCatIds(self, catNms=[], supNms=[], catIds=[]) -> list:
        catNms = catNms if _is_array_like(catNms) else [catNms]
        supNms = supNms if _is_array_like(supNms) else [supNms]
        catIds = catIds if _is_array_like(catIds) else [catIds]

        cats = self.dataset["categories"]
        if len(catNms) == len(supNms) == len(catIds) == 0:
            return [cat["id"] for cat in cats]
        if len(catNms) != 0:
            cats = [cat for cat in cats if cat["name"] in catNms]
        if len(supNms) != 0:
            cats = [cat for cat in cats if cat["supercategory"] in supNms]
        if len(catIds) != 0:
            cats = [cat for cat in cats if cat["id"] in catIds]
        return [cat["id"] for cat in cats]

    def getImgIds(self, imgIds=[], catIds=[]) -> list:
        imgIds = imgIds if _is_array_like(imgIds) else [imgIds]
        catIds = catIds if _is_array_like(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]) -> list:
        if _is_array_like(ids):
            return [self.anns[i] for i in ids]
        if isinstance(ids, int):
            return [self.anns[ids]]
        return []

    def loadCats(self, ids=[]) -> list:
        if _is_array_like(ids):
            return [self.cats[i] for i in ids]
        if isinstance(ids, int):
            return [self.cats[ids]]
        return []

    def loadImgs(self, ids=[]) -> list:
        if _is_array_like(ids):
            return [self.imgs[i] for i in ids]
        if isinstance(ids, int):
            return [self.imgs[ids]]
        return []

    def showAnns(self, anns: list, draw_bbox: bool = False):
        """Draw annotations on the current matplotlib axes."""
        if len(anns) == 0:
            return 0
        if "segmentation" in anns[0] or "keypoints" in anns[0]:
            datasetType = "instances"
        elif "caption" in anns[0]:
            datasetType = "captions"
        else:
            raise Exception("datasetType not supported")

        if datasetType == "captions":
            for ann in anns:
                print(ann["caption"])
            return None

        # matplotlib is an optional dependency; only visualisation needs it.
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons, color = [], []
        for ann in anns:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list):
                    for seg in ann["segmentation"]:
                        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                        polygons.append(Polygon(poly))
                        color.append(c)
                else:
                    t = self.imgs[ann["image_id"]]
                    if isinstance(ann["segmentation"]["counts"], list):
                        rle = maskUtils.frPyObjects(
                            [ann["segmentation"]], t["height"], t["width"]
                        )
                    else:
                        rle = [ann["segmentation"]]
                    m = maskUtils.decode(rle)
                    img = np.ones((m.shape[0], m.shape[1], 3))
                    if ann.get("iscrowd") == 1:
                        color_mask = np.array([2.0, 166.0, 101.0]) / 255
                    else:
                        color_mask = np.random.random((1, 3)).tolist()[0]
                    for i in range(3):
                        img[:, :, i] = color_mask[i]
                    ax.imshow(np.dstack((img, m * 0.5)))
            if "keypoints" in ann and isinstance(ann["keypoints"], list):
                sks = np.array(self.loadCats(ann["category_id"])[0]["skeleton"]) - 1
                kp = np.array(ann["keypoints"])
                x, y, v = kp[0::3], kp[1::3], kp[2::3]
                for sk in sks:
                    if np.all(v[sk] > 0):
                        plt.plot(x[sk], y[sk], linewidth=3, color=c)
                plt.plot(
                    x[v > 0], y[v > 0], "o", markersize=8,
                    markerfacecolor=c, markeredgecolor="k", markeredgewidth=2,
                )
                plt.plot(
                    x[v > 1], y[v > 1], "o", markersize=8,
                    markerfacecolor=c, markeredgecolor=c, markeredgewidth=2,
                )
            if draw_bbox:
                bx, by, bw, bh = ann["bbox"]
                np_poly = np.array(
                    [[bx, by], [bx, by + bh], [bx + bw, by + bh], [bx + bw, by]]
                ).reshape((4, 2))
                polygons.append(Polygon(np_poly))
                color.append(c)

        ax.add_collection(
            PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        )
        ax.add_collection(
            PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
        )
        return None

    def loadRes(self, resFile, *, derive_segmentation: bool = True) -> "COCO":
        """Build a result handle from detections.

        Accepts a path, a list of dicts, or an ``Nx7`` numpy array. The derived
        fields (``id``, ``area``, ``iscrowd``) are filled in exactly as
        pycocotools fills them, because ``area`` decides which area range a
        detection falls in and therefore changes AP_small/medium/large.

        Args:
            derive_segmentation: for box-only results, also store the
                equivalent four-corner polygon under ``segmentation``, as
                pycocotools does. It costs 376 bytes per detection — 440 MB on
                Objects365 — and is read by nothing here: the evaluator
                rasterises the box directly when the field is absent, so
                ``iouType="segm"`` gives identical results either way. Set
                False on large detection sets unless your own code reads
                ``ann["segmentation"]`` off a box-only result.
        """
        res = COCO(verbose=self.verbose)
        res.dataset["info"] = copy.deepcopy(self.dataset.get("info", {}))
        res.dataset["images"] = [img for img in self.dataset["images"]]

        self._log("Loading and preparing results...")
        tic = time.time()
        if isinstance(resFile, (str, os.PathLike)):
            anns = load_json(resFile)
        elif isinstance(resFile, np.ndarray):
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert isinstance(anns, list), "results in not an array of objects"
        if len(anns) > 0:
            annsImgIds = [ann["image_id"] for ann in anns]
            assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), (
                "Results do not correspond to current coco set"
            )

        if len(anns) == 0:
            res.dataset["categories"] = copy.deepcopy(self.dataset.get("categories", []))
        elif "caption" in anns[0]:
            imgIds = set(img["id"] for img in res.dataset["images"]) & set(
                ann["image_id"] for ann in anns
            )
            res.dataset["images"] = [
                img for img in res.dataset["images"] if img["id"] in imgIds
            ]
            for idx, ann in enumerate(anns):
                ann["id"] = idx + 1
        elif "bbox" in anns[0] and anns[0]["bbox"] != []:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for idx, ann in enumerate(anns):
                bb = ann["bbox"]
                if derive_segmentation and "segmentation" not in ann:
                    x1, x2, y1, y2 = bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = idx + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for idx, ann in enumerate(anns):
                ann["area"] = maskUtils.area(ann["segmentation"])
                if "bbox" not in ann:
                    ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
                ann["id"] = idx + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for idx, ann in enumerate(anns):
                s = ann["keypoints"]
                x, y = s[0::3], s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = idx + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        self._log(f"DONE (t={time.time() - tic:0.2f}s)")

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    # Snake-case aliases, for code written against faster-coco-eval.
    load_res = loadRes
    load_anns = loadAnns
    load_cats = loadCats
    load_imgs = loadImgs
    get_ann_ids = getAnnIds
    get_cat_ids = getCatIds
    get_img_ids = getImgIds

    def download(self, tarDir: str | None = None, imgIds=[]):
        """Fetch images referenced by the annotations."""
        from urllib.request import urlretrieve

        if tarDir is None:
            print("Please specify target directory")
            return -1
        imgs = self.imgs.values() if len(imgIds) == 0 else self.loadImgs(imgIds)
        n = len(imgs)
        os.makedirs(tarDir, exist_ok=True)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img["file_name"])
            if not os.path.exists(fname):
                urlretrieve(img["coco_url"], fname)
            print(f"downloaded {i}/{n} images (t={time.time() - tic:0.1f}s)")
        return None

    def loadNumpyAnnotations(self, data: np.ndarray) -> list:
        """Convert an ``Nx7`` array of ``[imageID, x1, y1, w, h, score, class]``."""
        print("Converting ndarray to lists...")
        assert isinstance(data, np.ndarray)
        assert data.shape[1] == 7
        n = data.shape[0]
        ann = []
        for i in range(n):
            if i % 1000000 == 0:
                print(f"{i}/{n}")
            ann.append({
                "image_id": int(data[i, 0]),
                "bbox": [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                "score": data[i, 5],
                "category_id": int(data[i, 6]),
            })
        return ann

    def annToRLE(self, ann: dict):
        """Segmentation of one annotation as a single RLE.

        Polygons are unioned; uncompressed RLE is compressed; compressed RLE is
        returned unchanged.
        """
        t = self.imgs[ann["image_id"]]
        h, w = t["height"], t["width"]
        segm = ann["segmentation"]
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, h, w)
            return maskUtils.merge(rles)
        if isinstance(segm["counts"], list):
            return maskUtils.frPyObjects(segm, h, w)
        return segm

    def annToMask(self, ann: dict) -> np.ndarray:
        """Binary mask for one annotation."""
        return maskUtils.decode(self.annToRLE(ann))


# pycocotools exposes this module-level constant; some code checks it.
PYTHON_VERSION = sys.version_info[0]
