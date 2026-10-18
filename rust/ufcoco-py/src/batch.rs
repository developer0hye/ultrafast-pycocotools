//! Per-image box batches for `metrics.DetectionMetrics.update`.
//!
//! Reads the per-image dicts, converts boxes, fills the engine's instances and
//! matches, all in one call: building annotation dicts, `COCO` handles and the
//! accumulated curves for a batch cost several times the matching itself.
//!
//! The instances are the ones the dictionary path builds from the same data
//! (`test_batch_metrics.py` compares the two bit for bit): ground truth is
//! ignored when crowded, its area is the given `area` or `w * h`, detection
//! areas are `w * h` as `loadRes` sets them, and the input order within each
//! image is the annotation order that breaks score ties.
//!
//! Values are read as `np.asarray(value, dtype=float64 | int64).ravel()`
//! would read them. NumPy arrays of the common dtypes are read in place;
//! anything else (lists, tensors, other dtypes) goes through the Python
//! `convert(value, kind)` the caller passes, so the conversion rules stay
//! NumPy's.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use ufcoco_core::eval::{EvalParams, Evaluator, GeomStore, Instances, IouType};
use ufcoco_core::IGNORED;

/// Python's `repr` of a shape tuple.
fn shape_repr(shape: &[usize]) -> String {
    match shape {
        [one] => format!("({one},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

// The NumPy fast paths read through `as_array` rather than `readonly()`.
// SAFETY: the GIL is held for as long as each view lives, no Python code runs
// meanwhile, and no Rust code holds a mutable view of any array. The crate's
// dynamic borrow tracking, which `readonly()` adds on top of that, only tracks
// Rust borrows: it never stopped Python code from writing to an array, so it
// adds no guarantee here, and it cost more than reading a batch's arrays.

/// Append `value` as float64 in C order; returns its shape.
fn read_f64(
    value: &Bound<'_, PyAny>,
    convert: &Bound<'_, PyAny>,
    out: &mut Vec<f64>,
) -> PyResult<Vec<usize>> {
    if let Ok(array) = value.cast::<PyArrayDyn<f64>>() {
        let view = unsafe { array.as_array() };
        match view.as_slice() {
            Some(values) => out.extend_from_slice(values),
            None => out.extend(view.iter().copied()),
        }
        return Ok(view.shape().to_vec());
    }
    macro_rules! typed {
        ($($t:ty),*) => {$(
            if let Ok(array) = value.cast::<PyArrayDyn<$t>>() {
                let view = unsafe { array.as_array() };
                out.extend(view.iter().map(|&x| x as f64));
                return Ok(view.shape().to_vec());
            }
        )*};
    }
    typed!(f32, i64, i32);
    let converted = convert.call1((value, intern!(value.py(), "f64")))?;
    let view = unsafe { converted.cast::<PyArrayDyn<f64>>()?.as_array() };
    out.extend(view.iter().copied());
    Ok(view.shape().to_vec())
}

/// Append `value` as int64 in C order; returns how many values were read.
fn read_i64(
    value: &Bound<'_, PyAny>,
    convert: &Bound<'_, PyAny>,
    out: &mut Vec<i64>,
) -> PyResult<usize> {
    let before = out.len();
    if let Ok(array) = value.cast::<PyArrayDyn<i64>>() {
        let view = unsafe { array.as_array() };
        match view.as_slice() {
            Some(values) => out.extend_from_slice(values),
            None => out.extend(view.iter().copied()),
        }
        return Ok(out.len() - before);
    }
    macro_rules! typed {
        ($($t:ty),*) => {$(
            if let Ok(array) = value.cast::<PyArrayDyn<$t>>() {
                out.extend(unsafe { array.as_array() }.iter().map(|&x| x as i64));
                return Ok(out.len() - before);
            }
        )*};
    }
    typed!(i32, i16, i8, u8, u16, u32, bool);
    let converted = convert.call1((value, intern!(value.py(), "i64")))?;
    out.extend(
        unsafe { converted.cast::<PyArrayDyn<i64>>()?.as_array() }
            .iter()
            .copied(),
    );
    Ok(out.len() - before)
}

fn lookup<'py>(
    item: &Bound<'py, PyAny>,
    key: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(dict) = item.cast::<PyDict>() {
        return dict.get_item(key);
    }
    if item.contains(key)? {
        Ok(Some(item.get_item(key)?))
    } else {
        Ok(None)
    }
}

/// One side of a batch, as engine columns.
#[derive(Default)]
struct Side {
    img: Vec<u32>,
    cat: Vec<i64>,
    boxes: Vec<[f64; 4]>,
    /// Ground truth only.
    area: Vec<f64>,
    crowd: Vec<bool>,
    score: Vec<f64>,
}

struct Reader<'a, 'py> {
    convert: &'a Bound<'py, PyAny>,
    xyxy: bool,
    flat: Vec<f64>,
    ints: Vec<i64>,
}

impl<'py> Reader<'_, 'py> {
    fn require(
        &self,
        item: &Bound<'py, PyAny>,
        key: &Bound<'py, PyString>,
        name: &str,
        i: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        lookup(item, key)?
            .ok_or_else(|| PyValueError::new_err(format!("{name}[{i}] has no '{key}'")))
    }

    /// Labels and boxes of one image, appended to `side`; returns the count.
    fn boxes(
        &mut self,
        [labels, boxes]: [&Bound<'py, PyAny>; 2],
        name: &str,
        i: usize,
        side: &mut Side,
    ) -> PyResult<usize> {
        let start = side.cat.len();
        let n = read_i64(labels, self.convert, &mut side.cat)?;
        self.flat.clear();
        let mut shape = read_f64(boxes, self.convert, &mut self.flat)?;
        if self.flat.is_empty() {
            shape = vec![0, 4];
        }
        if shape != [n, 4] {
            side.cat.truncate(start);
            return Err(PyValueError::new_err(format!(
                "{name}[{i}]['boxes'] must have shape ({n}, 4), not {}",
                shape_repr(&shape)
            )));
        }
        for b in self.flat.chunks_exact(4) {
            side.boxes.push(if self.xyxy {
                [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            } else {
                [b[0], b[1], b[2], b[3]]
            });
        }
        side.img.resize(side.img.len() + n, i as u32);
        Ok(n)
    }

    fn target(
        &mut self,
        item: &Bound<'py, PyAny>,
        columns: [&Bound<'py, PyAny>; 2],
        i: usize,
        gt: &mut Side,
    ) -> PyResult<()> {
        let py = item.py();
        let n = self.boxes(columns, "targets", i, gt)?;
        let first = gt.boxes.len() - n;
        match lookup(item, intern!(py, "iscrowd"))? {
            Some(value) => {
                self.ints.clear();
                let m = read_i64(&value, self.convert, &mut self.ints)?;
                if m != n {
                    return Err(PyValueError::new_err(format!(
                        "targets[{i}]['iscrowd'] has {m} entries, not {n}"
                    )));
                }
                gt.crowd.extend(self.ints.iter().map(|&v| v != 0));
            }
            None => gt.crowd.resize(gt.crowd.len() + n, false),
        }
        match lookup(item, intern!(py, "area"))? {
            Some(value) => {
                self.flat.clear();
                read_f64(&value, self.convert, &mut self.flat)?;
                if self.flat.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "targets[{i}]['area'] has {} entries, not {n}",
                        self.flat.len()
                    )));
                }
                gt.area.extend_from_slice(&self.flat);
            }
            // `np.where(given, area, w * h)` in the dictionary path.
            None => gt
                .area
                .extend(gt.boxes[first..].iter().map(|b| b[2] * b[3])),
        }
        Ok(())
    }

    fn pred(
        &mut self,
        [scores, labels, boxes]: [&Bound<'py, PyAny>; 3],
        i: usize,
        dt: &mut Side,
    ) -> PyResult<()> {
        let n = self.boxes([labels, boxes], "preds", i, dt)?;
        let before = dt.score.len();
        read_f64(scores, self.convert, &mut dt.score)?;
        let m = dt.score.len() - before;
        if m != n {
            return Err(PyValueError::new_err(format!(
                "preds[{i}] has {n} labels but {m} scores"
            )));
        }
        Ok(())
    }
}

fn image_id(item: &Bound<'_, PyAny>, convert: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    let py = item.py();
    let Some(value) = lookup(item, intern!(py, "image_id"))? else {
        return Ok(None);
    };
    if let Ok(v) = value.extract::<i64>() {
        return Ok(Some(v));
    }
    Ok(Some(
        convert
            .call1((value, intern!(py, "int")))?
            .extract::<i64>()?,
    ))
}

/// Read and match one batch of per-image box dicts.
///
/// Returns a dict: `image_ids` (the `image_id` each image gives, or None),
/// the counted detections' `det_cat` / `score` / `outcome` (`(D, T)` uint8;
/// detections ignored at every threshold are dropped), and `gt_cat` / `npig`
/// (non-ignored ground truth per category). Categories are the labels seen,
/// or the single id -1 without `use_cats`.
#[pyfunction]
#[pyo3(signature = (preds, targets, xyxy, iou_thrs, area_rng, max_det, use_cats, convert))]
#[allow(clippy::too_many_arguments)]
pub fn box_batch<'py>(
    py: Python<'py>,
    preds: &Bound<'py, PyList>,
    targets: &Bound<'py, PyList>,
    xyxy: bool,
    iou_thrs: Vec<f64>,
    area_rng: [f64; 2],
    max_det: usize,
    use_cats: bool,
    convert: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let n_img = preds.len();
    if targets.len() != n_img {
        return Err(PyValueError::new_err(format!(
            "preds has {n_img} images but targets has {}",
            targets.len()
        )));
    }
    let (mut gt, mut dt) = (Side::default(), Side::default());
    let mut reader = Reader {
        convert,
        xyxy,
        flat: Vec::new(),
        ints: Vec::new(),
    };
    let image_ids = PyList::empty(py);
    let (boxes, scores, labels) = (
        intern!(py, "boxes"),
        intern!(py, "scores"),
        intern!(py, "labels"),
    );
    for i in 0..n_img {
        let (pred, target) = (preds.get_item(i)?, targets.get_item(i)?);
        // Every required key is checked before any value is read, as the
        // dictionary path checks them.
        let (p_boxes, p_scores, p_labels) = (
            reader.require(&pred, boxes, "preds", i)?,
            reader.require(&pred, scores, "preds", i)?,
            reader.require(&pred, labels, "preds", i)?,
        );
        let (t_boxes, t_labels) = (
            reader.require(&target, boxes, "targets", i)?,
            reader.require(&target, labels, "targets", i)?,
        );
        reader.target(&target, [&t_labels, &t_boxes], i, &mut gt)?;
        reader.pred([&p_scores, &p_labels, &p_boxes], i, &mut dt)?;
        let given = match (image_id(&pred, convert)?, image_id(&target, convert)?) {
            (Some(a), Some(b)) if a != b => {
                return Err(PyValueError::new_err(format!(
                    "image {i}: preds and targets give different image_id [{}, {}]",
                    a.min(b),
                    a.max(b)
                )))
            }
            (a, b) => a.or(b),
        };
        image_ids.append(given)?;
    }

    let (cat_ids, gt_slot, dt_slot) = ufcoco_core::metrics::category_slots(&gt.cat, &dt.cat);
    let (g_n, d_n) = (gt.cat.len(), dt.cat.len());
    let gt_inst = Instances {
        ids: (1..=g_n as i64).collect(),
        scores: vec![0.0; g_n],
        areas: gt.area,
        ignore: gt.crowd.clone(),
        iscrowd: gt.crowd,
        lvis_mark: vec![false; g_n],
        bboxes: Vec::new(),
        img_slot: gt.img,
        cat_slot: gt_slot,
        geom: GeomStore::Bboxes(gt.boxes),
    };
    let dt_inst = Instances {
        ids: (1..=d_n as i64).collect(),
        // `prepare_bbox_results` / loadRes: `area = bb[2] * bb[3]`.
        areas: dt.boxes.iter().map(|b| b[2] * b[3]).collect(),
        scores: dt.score,
        iscrowd: vec![false; d_n],
        ignore: vec![false; d_n],
        lvis_mark: vec![false; d_n],
        bboxes: Vec::new(),
        img_slot: dt.img,
        cat_slot: dt_slot,
        geom: GeomStore::Bboxes(dt.boxes),
    };
    let t_n = iou_thrs.len();
    let params = EvalParams {
        img_ids: (0..n_img as i64).collect(),
        cat_ids: cat_ids.clone(),
        iou_thrs,
        // Only `run` reads the recall grid; outcomes do not.
        rec_thrs: vec![0.0],
        max_dets: vec![max_det],
        area_rng: vec![area_rng],
        use_cats,
        iou_type: IouType::Bbox,
        kpt_sigmas: Vec::new(),
        use_area: true,
    };
    let out = py.detach(|| Evaluator::new(params, gt_inst, dt_inst).outcomes(0, max_det));

    // A detection ignored at every threshold is in no count and no cut.
    let (mut det_cat, mut score, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
    for (d, row) in out
        .outcome
        .chunks_exact(t_n.max(1))
        .enumerate()
        .take(out.score.len())
    {
        if row.iter().any(|&o| o != IGNORED) {
            det_cat.push(if use_cats {
                cat_ids[out.cat_slot[d] as usize]
            } else {
                -1
            });
            score.push(out.score[d]);
            outcome.extend_from_slice(row);
        }
    }
    let gt_cat = if use_cats { cat_ids } else { vec![-1] };
    let npig: Vec<i64> = out.npig.iter().map(|&v| v as i64).collect();

    let kept = score.len();
    let d = PyDict::new(py);
    d.set_item("image_ids", image_ids)?;
    d.set_item("det_cat", Array1::from_vec(det_cat).into_pyarray(py))?;
    d.set_item("score", Array1::from_vec(score).into_pyarray(py))?;
    d.set_item(
        "outcome",
        Array2::from_shape_vec((kept, t_n), outcome)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py),
    )?;
    d.set_item("gt_cat", Array1::from_vec(gt_cat).into_pyarray(py))?;
    d.set_item("npig", Array1::from_vec(npig).into_pyarray(py))?;
    Ok(d)
}
