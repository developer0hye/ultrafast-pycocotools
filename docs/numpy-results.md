# NumPy result arrays (unreleased)

`loadRes` accepts an `Nx7` NumPy array of
`[image_id, x, y, w, h, score, category_id]` besides result files and lists,
as pycocotools does. That route used to convert every row into a Python
dictionary. It now copies float64 and float32 arrays straight into the same
compact columns as result files, without a JSON snapshot.

## Results

COCO val2017 bbox, 733,070 YOLO26n detections, 2 threads, median of 5 fresh
processes (Intel Core i5-10400, 2026-09-24). The array is read from an `.npy`
file first; its 41 MB are included in peak RSS. AP is identical in every row.

| Build | Input | `loadRes` s | Total s | Peak RSS MB |
| --- | --- | ---: | ---: | ---: |
| main (0.1.11) | JSON file | 0.292 | 0.912 | 264 |
| main (0.1.11) | array | 1.782 | 2.554 | 695 |
| task-bottleneck branch (#16) | JSON file | 0.208 | 0.591 | 254 |
| this change | array | 0.033 | 0.417 | 194 |

A framework that already holds its detections as tensors can skip writing and
parsing JSON: evaluation takes 30% less time and 22% less memory than the
file route of the same build.

## Semantics

The array route reproduces `loadNumpyAnnotations` followed by `loadRes`:

- IDs are truncated with `int()`; non-finite or out-of-range IDs, foreign
  image IDs and other dtypes take the ordinary route and its errors.
- For float32 arrays the derived `area` and the box polygon used for `segm`
  are computed in float32, as pycocotools computes them on NumPy float32
  scalars.
- The array is copied at load time; later edits to it have no effect.
- Public views (`dataset`, `anns`, ...) are built on first access from the
  stored rows in their original dtype, so they equal the ordinary conversion.
- bbox and segm evaluation use the compact columns. Keypoint and LVIS
  evaluation read the results as dictionaries, as before.

The messages pycocotools prints for array input are printed unchanged.
[tests/test_numpy_results.py](../tests/test_numpy_results.py) compares complete
arrays with pycocotools for float64 and float32 input, views with the ordinary
conversion, and the error cases.

## Use from a framework

```python
rows = np.column_stack([image_ids, x, y, w, h, scores, category_ids])  # float64 or float32
dt = gt.loadRes(rows)
```

The values are used as given. Ultralytics, for example, rounds boxes to three
and scores to five decimals when it writes `predictions.json`; to reproduce the
file route's metrics exactly, round the arrays the same way first.
