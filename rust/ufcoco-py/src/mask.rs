//! The `pycocotools.mask` / `pycocotools._mask` surface.
//!
//! Signatures, return dtypes and even the error messages follow the Cython
//! original, because callers do things like `rle["counts"].decode("ascii")`
//! and `mask_util.area(rle)[0]` and will break on a `str` where they expected
//! `bytes` or a `float64` where they expected `uint32`.

use numpy::ndarray::{Array1, Array2, Array3, ShapeBuilder};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use ufcoco_core::rle::{self, Rle};

/// Build the `{"size": [h, w], "counts": b"..."}` dict pycocotools returns.
///
/// `counts` must be `bytes`: downstream code decodes it, and handing back a
/// `str` is the single most common way a drop-in replacement breaks callers.
pub fn rle_to_dict<'py>(py: Python<'py>, r: &Rle) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("size", vec![r.h as u64, r.w as u64])?;
    d.set_item("counts", PyBytes::new(py, &r.to_string()))?;
    Ok(d)
}

/// Read one RLE out of a Python object.
///
/// Accepts both the compressed form (`counts` as `bytes`/`str`) and the
/// uncompressed form (`counts` as a list of ints). pycocotools only accepts
/// the former here and raises on the latter; accepting both is strictly more
/// permissive and cannot change a result that pycocotools would have produced.
pub fn rle_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Rle> {
    let dict = obj
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("expected an RLE dict with 'size' and 'counts'"))?;
    let size: Vec<u64> = dict
        .get_item("size")?
        .ok_or_else(|| PyValueError::new_err("RLE dict is missing 'size'"))?
        .extract()?;
    if size.len() != 2 {
        return Err(PyValueError::new_err("RLE 'size' must be [h, w]"));
    }
    let (h, w) = (size[0] as u32, size[1] as u32);
    let counts = dict
        .get_item("counts")?
        .ok_or_else(|| PyValueError::new_err("RLE dict is missing 'counts'"))?;

    if let Ok(b) = counts.cast::<PyBytes>() {
        return Ok(Rle::from_str(b.as_bytes(), h, w));
    }
    if let Ok(s) = counts.extract::<String>() {
        return Ok(Rle::from_str(s.as_bytes(), h, w));
    }
    let cnts: Vec<u32> = counts
        .extract()
        .map_err(|_| PyTypeError::new_err("RLE 'counts' must be bytes, str, or a list of ints"))?;
    Ok(Rle::new(h, w, cnts))
}

fn rles_from_seq(objs: &Bound<'_, PyAny>) -> PyResult<Vec<Rle>> {
    let list = objs.try_iter()?;
    let mut out = Vec::new();
    for item in list {
        out.push(rle_from_any(&item?)?);
    }
    Ok(out)
}

#[pyfunction]
pub fn encode<'py>(
    py: Python<'py>,
    mask: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<Bound<'py, PyList>> {
    let arr = mask.as_array();
    if arr.ndim() != 3 {
        return Err(PyValueError::new_err(
            "mask must be a 3-D (h, w, n) uint8 array",
        ));
    }
    let (h, w, n) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    let out = PyList::empty(py);
    let mut buf = vec![0u8; h * w];
    for i in 0..n {
        // Column-major serialisation, which is the order RLE counts run in.
        // Reading through ndarray indexing keeps this correct for C-order
        // input too, where pycocotools would simply refuse.
        for x in 0..w {
            for y in 0..h {
                buf[x * h + y] = arr[[y, x, i]];
            }
        }
        let r = Rle::encode(&buf, h as u32, w as u32);
        out.append(rle_to_dict(py, &r)?)?;
    }
    Ok(out)
}

#[pyfunction]
pub fn decode<'py>(
    py: Python<'py>,
    rle_objs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let rles = rles_from_seq(rle_objs)?;
    if rles.is_empty() {
        return Err(PyIndexError::new_err("decode() needs at least one RLE"));
    }
    let (h, w) = (rles[0].h as usize, rles[0].w as usize);
    let n = rles.len();
    let mut data = vec![0u8; h * w * n];
    for (i, r) in rles.iter().enumerate() {
        r.decode_into(&mut data[i * h * w..(i + 1) * h * w]);
    }
    // Fortran order: pycocotools reshapes with order='F' and callers index
    // [:, :, k].
    let arr = Array3::from_shape_vec((h, w, n).f(), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (rle_objs, intersect=0))]
pub fn merge<'py>(
    py: Python<'py>,
    rle_objs: &Bound<'py, PyAny>,
    intersect: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let rles = rles_from_seq(rle_objs)?;
    let m = rle::merge(&rles, intersect != 0);
    rle_to_dict(py, &m)
}

#[pyfunction]
pub fn area<'py>(py: Python<'py>, rle_objs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let rles = rles_from_seq(rle_objs)?;
    // uint32, matching the C `uint` the reference returns. Callers compare
    // these against annotation areas, and float64 would silently widen.
    let a: Array1<u32> = Array1::from_iter(rles.iter().map(|r| r.area()));
    Ok(a.into_pyarray(py).into_any())
}

#[pyfunction]
#[pyo3(name = "toBbox")]
pub fn to_bbox<'py>(
    py: Python<'py>,
    rle_objs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rles = rles_from_seq(rle_objs)?;
    let n = rles.len();
    let mut data = Vec::with_capacity(n * 4);
    for r in &rles {
        data.extend_from_slice(&r.to_bbox());
    }
    let arr =
        Array2::from_shape_vec((n, 4), data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Either a stack of boxes or a stack of masks — `iou` accepts both, and both
/// sides must be the same kind.
enum IouOperand {
    Boxes(Vec<[f64; 4]>),
    Masks(Vec<Rle>),
}

impl IouOperand {
    fn len(&self) -> usize {
        match self {
            IouOperand::Boxes(v) => v.len(),
            IouOperand::Masks(v) => v.len(),
        }
    }
}

fn preproc(obj: &Bound<'_, PyAny>) -> PyResult<IouOperand> {
    // An Nx4 float array (or anything that converts to one) is boxes.
    if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<f64>>() {
        let a = arr.as_array();
        if a.ndim() != 2 || a.shape()[1] != 4 {
            return Err(PyValueError::new_err(
                "numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension",
            ));
        }
        let mut v = Vec::with_capacity(a.shape()[0]);
        for r in 0..a.shape()[0] {
            v.push([a[[r, 0]], a[[r, 1]], a[[r, 2]], a[[r, 3]]]);
        }
        return Ok(IouOperand::Boxes(v));
    }
    let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Ok(IouOperand::Boxes(Vec::new()));
    }
    if items[0].cast::<PyDict>().is_ok() {
        let mut v = Vec::with_capacity(items.len());
        for it in &items {
            v.push(rle_from_any(it)?);
        }
        return Ok(IouOperand::Masks(v));
    }
    let mut v = Vec::with_capacity(items.len());
    for it in &items {
        let b: Vec<f64> = it.extract().map_err(|_| {
            PyValueError::new_err("list input can be bounding box (Nx4) or RLEs ([RLE])")
        })?;
        if b.len() != 4 {
            return Err(PyValueError::new_err(
                "list input can be bounding box (Nx4) or RLEs ([RLE])",
            ));
        }
        v.push([b[0], b[1], b[2], b[3]]);
    }
    Ok(IouOperand::Boxes(v))
}

#[pyfunction]
pub fn iou<'py>(
    py: Python<'py>,
    dt: &Bound<'py, PyAny>,
    gt: &Bound<'py, PyAny>,
    iscrowd: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let d = preproc(dt)?;
    let g = preproc(gt)?;
    let (m, n) = (d.len(), g.len());
    if m == 0 || n == 0 {
        // pycocotools returns a plain empty list here, and cocoeval checks
        // `len(ious) == 0`, so an empty ndarray would not be equivalent.
        return Ok(PyList::empty(py).into_any());
    }
    let crowd: Vec<u8> = iscrowd
        .try_iter()?
        .map(|x| x?.extract::<i64>().map(|v| v as u8))
        .collect::<PyResult<_>>()?;

    let mut out = vec![0.0f64; m * n];
    match (&d, &g) {
        (IouOperand::Boxes(dv), IouOperand::Boxes(gv)) => rle::bb_iou(dv, gv, &crowd, &mut out),
        (IouOperand::Masks(dv), IouOperand::Masks(gv)) => rle::rle_iou(dv, gv, &crowd, &mut out),
        _ => {
            return Err(PyTypeError::new_err(
                "The dt and gt should have the same data type, either RLEs, list or np.ndarray",
            ))
        }
    }
    let arr =
        Array2::from_shape_vec((m, n), out).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into_any())
}

#[pyfunction]
#[pyo3(name = "frBbox")]
pub fn fr_bbox<'py>(
    py: Python<'py>,
    bb: &Bound<'py, PyAny>,
    h: u32,
    w: u32,
) -> PyResult<Bound<'py, PyList>> {
    let boxes = match preproc(bb)? {
        IouOperand::Boxes(v) => v,
        IouOperand::Masks(_) => return Err(PyTypeError::new_err("expected bounding boxes")),
    };
    let out = PyList::empty(py);
    for b in &boxes {
        out.append(rle_to_dict(py, &rle::rle_fr_bbox(b, h, w))?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "frPoly")]
pub fn fr_poly<'py>(
    py: Python<'py>,
    poly: &Bound<'py, PyAny>,
    h: u32,
    w: u32,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for p in poly.try_iter()? {
        let xy: Vec<f64> = p?.extract()?;
        out.append(rle_to_dict(py, &rle::rle_fr_poly(&xy, h, w))?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "frUncompressedRLE")]
pub fn fr_uncompressed<'py>(
    py: Python<'py>,
    uc: &Bound<'py, PyAny>,
    _h: u32,
    _w: u32,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for item in uc.try_iter()? {
        let item = item?;
        let dict = item
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("expected uncompressed RLE dicts"))?;
        let size: Vec<u64> = dict
            .get_item("size")?
            .ok_or_else(|| PyValueError::new_err("RLE dict is missing 'size'"))?
            .extract()?;
        let cnts: Vec<u32> = dict
            .get_item("counts")?
            .ok_or_else(|| PyValueError::new_err("RLE dict is missing 'counts'"))?
            .extract()?;
        // Note the reference uses the *annotation's* size, not the h/w
        // arguments; passing a mismatched h/w is silently ignored upstream
        // and we keep that.
        let r = rle::rle_fr_uncompressed(&cnts, size[0] as u32, size[1] as u32);
        out.append(rle_to_dict(py, &r)?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "frPyObjects")]
pub fn fr_py_objects<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    h: u32,
    w: u32,
) -> PyResult<Bound<'py, PyAny>> {
    // The dispatch below mirrors the Cython original branch for branch,
    // including the "a bare list of 4 numbers is one box, a bare list of more
    // is one polygon" rule that makes a 2-point polygon impossible to express.
    if obj.extract::<PyReadonlyArrayDyn<f64>>().is_ok() {
        return Ok(fr_bbox(py, obj, h, w)?.into_any());
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        if dict.contains("counts")? && dict.contains("size")? {
            let l = PyList::empty(py);
            l.append(dict)?;
            let objs = fr_uncompressed(py, l.as_any(), h, w)?;
            return objs.get_item(0);
        }
        return Err(PyTypeError::new_err("input type is not supported."));
    }
    let items: Vec<Bound<'py, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Ok(PyList::empty(py).into_any());
    }
    let first = &items[0];
    if let Ok(d) = first.cast::<PyDict>() {
        if d.contains("counts")? && d.contains("size")? {
            return Ok(fr_uncompressed(py, obj, h, w)?.into_any());
        }
    }
    if first.extract::<f64>().is_ok() {
        // A flat list of numbers: 4 means one box, more means one polygon.
        let flat: Vec<f64> = obj.extract()?;
        return if flat.len() == 4 {
            let r = rle::rle_fr_bbox(&[flat[0], flat[1], flat[2], flat[3]], h, w);
            Ok(rle_to_dict(py, &r)?.into_any())
        } else {
            let r = rle::rle_fr_poly(&flat, h, w);
            Ok(rle_to_dict(py, &r)?.into_any())
        };
    }
    let inner_len = first.len()?;
    if inner_len == 4 {
        Ok(fr_bbox(py, obj, h, w)?.into_any())
    } else {
        Ok(fr_poly(py, obj, h, w)?.into_any())
    }
}

/// Extension: boundary mask of an RLE, for Boundary IoU.
#[pyfunction]
#[pyo3(name = "toBoundary", signature = (rle_objs, dilation_ratio=0.02))]
pub fn to_boundary<'py>(
    py: Python<'py>,
    rle_objs: &Bound<'py, PyAny>,
    dilation_ratio: f64,
) -> PyResult<Bound<'py, PyList>> {
    let rles = rles_from_seq(rle_objs)?;
    let out = PyList::empty(py);
    for r in &rles {
        out.append(rle_to_dict(py, &rle::rle_to_boundary(r, dilation_ratio))?)?;
    }
    Ok(out)
}
