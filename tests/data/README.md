# Test fixtures

## `coco_subset_gt.json` / `coco_subset_dt.json`

The fixture contains 93 images with 785 annotations selected from COCO val2017,
and 780 detections derived from those annotations.

**Why the files are committed.** Synthetic data only approximates real
annotations. Polygons from `bench/make_dataset.py` are single rings with 3–8
vertices; real COCO annotations can have multiple rings and dozens of vertices.
Real crowds include arbitrary uncompressed RLE shapes, while synthetic crowds
are axis-aligned rectangles. Real data can expose `rleFrPoly` differences in
shapes that a synthetic generator does not produce.

Earlier tests used a 20 MB file under `bench/data/` that existed on only one
machine, silently skipping on fresh clones and CI. Committing this small fixture
makes those checks available on every checkout.

**Coverage**, enforced by `tests/test_fixture_coverage.py`:

| Feature | Count or coverage |
|---|---|
| Crowd annotations | 32, including uncompressed RLE |
| Multi-ring polygons | 145 |
| Longest polygon | More than 40 vertices |
| Area scales | 440 small / 130 large |
| Images without annotations | 3 |
| Categories without annotations | Included, exercising the `-1` sentinel path |

**Sensitivity.** The parity comparison detects even a one-ULP difference in a
single cell of the 969,600-cell precision array.

**Regeneration:**

```bash
python bench/make_real_fixture.py --gt path/to/instances_val2017.json
```

Selection is greedy coverage-per-byte selection, not random sampling. Crowd
images receive a separate quota because their annotation density makes them
expensive under a pure efficiency ranking. The initial selection without that
quota contained only two crowd annotations.

## Source and license

COCO annotations come from [cocodataset.org](https://cocodataset.org) under
**CC BY 4.0**. No image files are included; annotations alone are sufficient for
evaluation tests.
