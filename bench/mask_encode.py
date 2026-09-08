"""Deterministic mask-encoding microbenchmark, with byte-exact COCO checks.

Use separate processes/PYTHONPATH roots to compare installed wheel versions.
Reported samples time encoding only; setup, reference checks and hashing are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from pycocotools import mask as reference


def corpus():
    rng = np.random.default_rng(20260908)
    h, w = 640, 640
    y, x = np.ogrid[:h, :w]
    blob = ((y - 311) ** 2 / 140**2 + (x - 327) ** 2 / 180**2 < 1).astype(np.uint8)
    masks = {
        "empty": np.empty((0, 17, 2), dtype=np.uint8, order="F"),
        "zeros": np.zeros((h, w, 1), dtype=np.uint8, order="F"),
        "ones": np.ones((h, w, 1), dtype=np.uint8, order="F"),
        "blob": np.asfortranarray(blob[:, :, None]),
        "blob_stack": np.asfortranarray(
            np.stack([np.roll(blob, i * 13, axis=0) for i in range(8)], axis=2)
        ),
        "binary_noise": np.asfortranarray(
            rng.integers(0, 2, (h, w, 1), dtype=np.uint8)
        ),
        "byte_noise": np.asfortranarray(
            rng.integers(0, 256, (h, w, 1), dtype=np.uint8)
        ),
        "alternating": np.asfortranarray(
            np.broadcast_to((y % 2).astype(np.uint8), (h, w))[:, :, None]
        ),
    }
    masks["c_blob_stack"] = np.ascontiguousarray(masks["blob_stack"])
    masks["strided_blob_stack"] = masks["blob_stack"][::2, ::2, ::-1]
    return masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["ultrafast", "faster"], default="ultrafast")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iterations", type=int, default=10)
    a = p.parse_args()
    if a.rounds < 1 or a.iterations < 1:
        p.error("rounds and iterations must be positive")
    if a.backend == "ultrafast":
        from ultrafast_pycocotools import mask, _ufcoco

        encode = mask.encode
        native = Path(_ufcoco.__file__)
    else:
        from faster_coco_eval.core import mask
        import faster_coco_eval.mask_api_new_cpp as extension

        def encode(value):
            return mask.encode(np.asfortranarray(value))

        native = Path(extension.__file__)
    result = dict(
        backend=a.backend,
        platform=platform.platform(),
        native_path=str(native),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        numpy=importlib.metadata.version("numpy"),
        reference=importlib.metadata.version("pycocotools"),
        rounds=a.rounds,
        iterations=a.iterations,
        method="One excluded warmup per case; encoding-only wall/CPU seconds per call. "
        "faster-coco-eval adapter converts non-Fortran input inside timing.",
        cases={},
    )
    for name, value in corpus().items():
        expected = reference.encode(np.asfortranarray(value))
        assert encode(value) == expected, name
        samples = []
        for _ in range(a.rounds):
            start, cpu = time.perf_counter(), time.process_time()
            for _ in range(a.iterations):
                got = encode(value)
            samples.append(
                dict(
                    wall=(time.perf_counter() - start) / a.iterations,
                    cpu=(time.process_time() - cpu) / a.iterations,
                )
            )
            assert got == expected, name
        result["cases"][name] = dict(
            shape=list(value.shape),
            strides=list(value.strides),
            samples=samples,
            median_wall=statistics.median(s["wall"] for s in samples),
            counts_sha256=hashlib.sha256(
                b"".join(r["counts"] for r in expected)
            ).hexdigest(),
            reference_byte_identical=True,
        )
        print(name, result["cases"][name]["median_wall"], flush=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
