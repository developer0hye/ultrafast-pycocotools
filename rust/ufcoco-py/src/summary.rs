//! Gather valid summary values without dense advanced-index and mask copies.
//! NumPy still performs the mean on this same C-order sequence, preserving its
//! floating-point reduction order rather than introducing a different sum.
use numpy::ndarray::{Ix4, Ix5};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub(crate) fn summary_values<'py>(
    py: Python<'py>,
    array: PyReadonlyArrayDyn<'py, f64>,
    thresholds: Vec<usize>,
    categories: Vec<usize>,
    area: usize,
    max_det: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = array.as_array();
    let shape = a.shape();
    if !(shape.len() == 4 || shape.len() == 5) {
        return Err(PyValueError::new_err(
            "summary array must have four or five axes",
        ));
    }
    let n = shape.len();
    if thresholds.iter().any(|&t| t >= shape[0])
        || categories.iter().any(|&k| k >= shape[n - 3])
        || area >= shape[n - 2]
        || max_det >= shape[n - 1]
    {
        return Err(PyValueError::new_err(
            "summary selection is outside the array",
        ));
    }
    let mut values = Vec::new();
    if n == 5 {
        let a = a.into_dimensionality::<Ix5>().unwrap();
        for t in thresholds {
            for r in 0..a.shape()[1] {
                for &k in &categories {
                    let v = a[[t, r, k, area, max_det]];
                    if v > -1.0 {
                        values.push(v);
                    }
                }
            }
        }
    } else {
        let a = a.into_dimensionality::<Ix4>().unwrap();
        for t in thresholds {
            for &k in &categories {
                let v = a[[t, k, area, max_det]];
                if v > -1.0 {
                    values.push(v);
                }
            }
        }
    }
    Ok(values.into_pyarray(py))
}
