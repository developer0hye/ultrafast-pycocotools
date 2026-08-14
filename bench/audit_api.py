"""Compare our public surface against pycocotools', attribute by attribute.

"Drop-in" is a claim about names and signatures, not about AP. This walks the
reference package and reports anything a caller could reach for and not find,
or find with a different signature.

Not every difference is a defect — extra keyword arguments with defaults are
additive, and a few upstream attributes are internal. The point is to see the
list rather than assume it is empty.
"""

from __future__ import annotations

import inspect

import pycocotools.coco as ref_coco
import pycocotools.cocoeval as ref_cocoeval
import pycocotools.mask as ref_mask

import ultrafast_pycocotools.coco as our_coco
import ultrafast_pycocotools.cocoeval as our_cocoeval
import ultrafast_pycocotools.mask as our_mask


def public(obj) -> list[str]:
    return sorted(n for n in dir(obj) if not n.startswith("__"))


def sig(fn) -> str:
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "<builtin>"


def compare_class(name: str, ref_cls, our_cls) -> int:
    problems = 0
    ref_names = [n for n in public(ref_cls) if callable(getattr(ref_cls, n, None))]
    for n in ref_names:
        if not hasattr(our_cls, n):
            print(f"  MISSING  {name}.{n}")
            problems += 1
            continue
        rs, os_ = sig(getattr(ref_cls, n)), sig(getattr(our_cls, n))
        if rs != os_ and rs != "<builtin>":
            # Extra keyword-only arguments with defaults are additive.
            r_params = inspect.signature(getattr(ref_cls, n)).parameters
            try:
                o_params = inspect.signature(getattr(our_cls, n)).parameters
            except (TypeError, ValueError):
                continue
            missing = [p for p in r_params if p not in o_params]
            added_required = [
                p
                for p, v in o_params.items()
                if p not in r_params
                and v.default is inspect.Parameter.empty
                and v.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            if missing or added_required:
                print(f"  SIGNATURE {name}.{n}\n      ref {rs}\n      our {os_}")
                problems += 1
            else:
                print(f"  ok(+kw)  {name}.{n}  our{os_}")
    return problems


def compare_module(name: str, ref_mod, our_mod) -> int:
    problems = 0
    for n in public(ref_mod):
        obj = getattr(ref_mod, n)
        if inspect.ismodule(obj) or n.startswith("_"):
            continue
        if not hasattr(our_mod, n):
            print(f"  MISSING  {name}.{n}  ({type(obj).__name__})")
            problems += 1
    return problems


def main() -> None:
    total = 0
    print("pycocotools.coco module")
    total += compare_module("coco", ref_coco, our_coco)
    print("pycocotools.coco.COCO")
    total += compare_class("COCO", ref_coco.COCO, our_coco.COCO)

    print("\npycocotools.cocoeval module")
    total += compare_module("cocoeval", ref_cocoeval, our_cocoeval)
    print("pycocotools.cocoeval.COCOeval")
    total += compare_class("COCOeval", ref_cocoeval.COCOeval, our_cocoeval.COCOeval)
    print("pycocotools.cocoeval.Params")
    total += compare_class("Params", ref_cocoeval.Params, our_cocoeval.Params)

    print("\npycocotools.mask module")
    total += compare_module("mask", ref_mask, our_mask)

    print("\nParams instance attributes")
    for iou_type in ("bbox", "segm", "keypoints"):
        r = ref_cocoeval.Params(iouType=iou_type)
        o = our_cocoeval.Params(iouType=iou_type)
        for n in public(r):
            if not hasattr(o, n):
                print(f"  MISSING  Params({iou_type}).{n}")
                total += 1

    print("\nCOCOeval instance attributes (after construction)")
    r = ref_cocoeval.COCOeval()
    o = our_cocoeval.COCOeval()
    for n in public(r):
        if not hasattr(o, n):
            print(f"  MISSING  COCOeval().{n}")
            total += 1

    print(f"\n{total} incompatibilities")


if __name__ == "__main__":
    main()
