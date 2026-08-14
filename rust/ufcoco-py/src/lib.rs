//! PyO3 bindings for `ufcoco-core`.
//!
//! Two responsibilities live here and nowhere else:
//!
//! * translating Python annotation dicts into the crate's struct-of-arrays
//!   form, once, up front — after which no Python object is touched again and
//!   the GIL can be released for the whole evaluation;
//! * reproducing the `pycocotools.mask` surface exactly (see `mask.rs`).
//!
//! Segmentation conversion is chunked: raw polygons/RLE strings are read under
//! the GIL a chunk at a time and rasterised without it, so peak memory is
//! "one chunk of raw geometry + the finished RLEs" rather than "every polygon
//! in the dataset + every RLE".

mod json;
mod mask;

use numpy::ndarray::{Array1, Array2, ArrayD, IxDyn};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rayon::prelude::*;
use std::collections::HashMap;
use ufcoco_core::eval::{
    EvalParams, Evaluator as CoreEvaluator, GeomStore, ImgEval, Instances, IouType,
};
use ufcoco_core::rle::{self, Rle};

/// How many annotations are read out of Python before their geometry is
/// rasterised. Big enough to amortise the GIL round-trip, small enough that
/// the raw polygon buffer stays bounded on million-annotation datasets.
const CHUNK: usize = 16384;

fn iou_type_from_str(s: &str) -> PyResult<IouType> {
    match s {
        "segm" => Ok(IouType::Segm),
        "bbox" => Ok(IouType::Bbox),
        "keypoints" | "keypoints_crowd" => Ok(IouType::Keypoints),
        "boundary" => Ok(IouType::Boundary),
        other => Err(PyValueError::new_err(format!(
            "iouType must be one of segm, bbox, keypoints, boundary; got {other:?}"
        ))),
    }
}

/// Raw, not-yet-rasterised segmentation as it appears in the JSON.
enum RawSegm {
    /// One or more polygon rings, unioned into a single mask.
    Poly(Vec<Vec<f64>>),
    /// `counts` as a run list.
    Uncompressed(Vec<u32>),
    /// `counts` in the LEB128-ish string form.
    Compressed(Vec<u8>),
    Missing,
}

fn get_f64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    Ok(match d.get_item(key)? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    })
}

fn get_i64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i64>> {
    Ok(match d.get_item(key)? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    })
}

fn read_segm(d: &Bound<'_, PyDict>) -> PyResult<RawSegm> {
    let Some(seg) = d.get_item("segmentation")? else {
        return Ok(RawSegm::Missing);
    };
    if seg.is_none() {
        return Ok(RawSegm::Missing);
    }
    if let Ok(sd) = seg.cast::<PyDict>() {
        let counts = sd
            .get_item("counts")?
            .ok_or_else(|| PyValueError::new_err("RLE segmentation is missing 'counts'"))?;
        if let Ok(b) = counts.cast::<PyBytes>() {
            return Ok(RawSegm::Compressed(b.as_bytes().to_vec()));
        }
        if let Ok(s) = counts.extract::<String>() {
            return Ok(RawSegm::Compressed(s.into_bytes()));
        }
        return Ok(RawSegm::Uncompressed(counts.extract()?));
    }
    // A list of rings, each a flat [x0, y0, x1, y1, ...].
    let rings: Vec<Vec<f64>> = seg.extract()?;
    Ok(RawSegm::Poly(rings))
}

/// pycocotools' `annToRLE`: polygons are unioned, uncompressed RLE is taken
/// as-is, compressed RLE is decoded.
fn raw_to_rle(raw: &RawSegm, h: u32, w: u32, scratch: &mut rle::PolyScratch) -> Rle {
    match raw {
        RawSegm::Poly(rings) => rle::rle_fr_polys_into(rings, h, w, scratch),
        RawSegm::Uncompressed(c) => rle::rle_fr_uncompressed(c, h, w),
        RawSegm::Compressed(s) => Rle::from_str(s, h, w),
        RawSegm::Missing => Rle::new(h, w, vec![(h as u64 * w as u64) as u32]),
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_instances(
    py: Python<'_>,
    anns: &Bound<'_, PyList>,
    is_gt: bool,
    iou_type: IouType,
    img_sizes: &HashMap<i64, (u32, u32)>,
    img_slot: &HashMap<i64, u32>,
    cat_slot: &HashMap<i64, u32>,
    boundary_dilation: f64,
) -> PyResult<Instances> {
    let n = anns.len();
    let mut inst = Instances {
        ids: Vec::with_capacity(n),
        scores: Vec::with_capacity(n),
        areas: Vec::with_capacity(n),
        iscrowd: Vec::with_capacity(n),
        ignore: Vec::with_capacity(n),
        lvis_mark: Vec::with_capacity(n),
        bboxes: Vec::new(),
        img_slot: Vec::with_capacity(n),
        cat_slot: Vec::with_capacity(n),
        geom: match iou_type {
            IouType::Bbox => GeomStore::Bboxes(Vec::with_capacity(n)),
            IouType::Segm => GeomStore::Masks(Vec::with_capacity(n)),
            IouType::Boundary => GeomStore::Boundaries {
                masks: Vec::with_capacity(n),
                boundaries: Vec::with_capacity(n),
            },
            IouType::Keypoints => GeomStore::Keypoints {
                data: Vec::new(),
                k: 0,
            },
        },
    };

    let needs_mask = matches!(iou_type, IouType::Segm | IouType::Boundary);
    let mut raw_chunk: Vec<(RawSegm, u32, u32)> = Vec::with_capacity(CHUNK.min(n.max(1)));

    for (i, item) in anns.iter().enumerate() {
        let d = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("annotations must be dicts"))?;
        let image_id = get_i64(d, "image_id")?
            .ok_or_else(|| PyKeyError::new_err("annotation is missing 'image_id'"))?;
        let category_id = get_i64(d, "category_id")?.unwrap_or(-1);
        inst.ids.push(get_i64(d, "id")?.unwrap_or(i as i64 + 1));
        inst.img_slot
            .push(img_slot.get(&image_id).copied().unwrap_or(u32::MAX));
        inst.cat_slot
            .push(cat_slot.get(&category_id).copied().unwrap_or(u32::MAX));
        inst.scores.push(get_f64(d, "score")?.unwrap_or(0.0));

        let bbox: [f64; 4] = match d.get_item("bbox")? {
            Some(v) if !v.is_none() => {
                let b: Vec<f64> = v.extract()?;
                if b.len() != 4 {
                    return Err(PyValueError::new_err("'bbox' must have 4 entries"));
                }
                [b[0], b[1], b[2], b[3]]
            }
            _ => [0.0; 4],
        };
        inst.areas
            .push(get_f64(d, "area")?.unwrap_or(bbox[2] * bbox[3]));
        let iscrowd = get_i64(d, "iscrowd")?.unwrap_or(0) != 0;
        inst.iscrowd.push(iscrowd);

        if is_gt {
            // pycocotools' `_prepare` writes
            //     gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            //     gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            // The second line unconditionally overwrites the first, so the
            // annotation's own `ignore` field has no effect upstream — a
            // surprise for CrowdHuman-style data that sets `ignore: 1`.
            // Reproducing the bug is the whole point of this crate, so we do,
            // and the README says so out loud rather than quietly fixing it.
            let mut ig = iscrowd;
            if iou_type == IouType::Keypoints {
                ig = get_i64(d, "num_keypoints")?.unwrap_or(0) == 0 || ig;
            }
            inst.ignore.push(ig);
        } else {
            inst.ignore.push(false);
        }
        inst.lvis_mark
            .push(get_i64(d, "lvis_mark")?.unwrap_or(0) != 0);

        match &mut inst.geom {
            GeomStore::Bboxes(v) => v.push(bbox),
            GeomStore::Keypoints { data, k } => {
                inst.bboxes.push(bbox);
                let kp: Vec<f64> = match d.get_item("keypoints")? {
                    Some(v) if !v.is_none() => v.extract()?,
                    _ => Vec::new(),
                };
                if *k == 0 && !kp.is_empty() {
                    *k = kp.len() / 3;
                }
                data.extend_from_slice(&kp);
            }
            _ => {
                let (h, w) = *img_sizes.get(&image_id).ok_or_else(|| {
                    PyKeyError::new_err(format!("no image entry for image_id {image_id}"))
                })?;
                raw_chunk.push((read_segm(d)?, h, w));
            }
        }

        if needs_mask && raw_chunk.len() >= CHUNK {
            flush_masks(py, &mut raw_chunk, &mut inst.geom, boundary_dilation);
        }
    }
    if needs_mask && !raw_chunk.is_empty() {
        flush_masks(py, &mut raw_chunk, &mut inst.geom, boundary_dilation);
    }
    Ok(inst)
}

/// Rasterise a chunk of raw segmentations with the GIL released.
fn flush_masks(
    py: Python<'_>,
    raw: &mut Vec<(RawSegm, u32, u32)>,
    geom: &mut GeomStore,
    boundary_dilation: f64,
) {
    let want_boundary = matches!(geom, GeomStore::Boundaries { .. });
    let built: Vec<(Rle, Option<Rle>)> = py.detach(|| {
        // One scratch per worker thread: polygon rasterisation is otherwise
        // dominated by allocating and freeing the same seven vectors.
        raw.par_iter()
            .map_init(rle::PolyScratch::default, |scratch, (r, h, w)| {
                let m = raw_to_rle(r, *h, *w, scratch);
                let b = if want_boundary {
                    Some(rle::rle_to_boundary(&m, boundary_dilation))
                } else {
                    None
                };
                (m, b)
            })
            .collect()
    });
    match geom {
        GeomStore::Masks(v) => v.extend(built.into_iter().map(|(m, _)| m)),
        GeomStore::Boundaries { masks, boundaries } => {
            for (m, b) in built {
                masks.push(m);
                boundaries.push(b.unwrap());
            }
        }
        _ => unreachable!("flush_masks called for a non-mask geometry"),
    }
    raw.clear();
}

/// The evaluation engine, constructed once from prepared annotations.
#[pyclass(module = "ultrafast_pycocotools._ufcoco")]
pub struct Evaluator {
    inner: CoreEvaluator,
}

#[pymethods]
impl Evaluator {
    #[new]
    #[pyo3(signature = (
        gt_anns, dt_anns, img_sizes, img_ids, cat_ids, iou_thrs, rec_thrs,
        max_dets, area_rng, use_cats, iou_type, kpt_sigmas, use_area=true,
        boundary_dilation=0.02,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        gt_anns: &Bound<'_, PyList>,
        dt_anns: &Bound<'_, PyList>,
        img_sizes: HashMap<i64, (u32, u32)>,
        img_ids: Vec<i64>,
        cat_ids: Vec<i64>,
        iou_thrs: Vec<f64>,
        rec_thrs: Vec<f64>,
        max_dets: Vec<usize>,
        area_rng: Vec<[f64; 2]>,
        use_cats: bool,
        iou_type: &str,
        kpt_sigmas: Vec<f64>,
        use_area: bool,
        boundary_dilation: f64,
    ) -> PyResult<Self> {
        let it = iou_type_from_str(iou_type)?;
        let img_map: HashMap<i64, u32> = img_ids
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();
        let cat_map: HashMap<i64, u32> = cat_ids
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect();

        let gt = extract_instances(
            py,
            gt_anns,
            true,
            it,
            &img_sizes,
            &img_map,
            &cat_map,
            boundary_dilation,
        )?;
        let dt = extract_instances(
            py,
            dt_anns,
            false,
            it,
            &img_sizes,
            &img_map,
            &cat_map,
            boundary_dilation,
        )?;

        let params = EvalParams {
            img_ids,
            cat_ids,
            iou_thrs,
            rec_thrs,
            max_dets,
            area_rng,
            use_cats,
            iou_type: it,
            kpt_sigmas,
            use_area,
        };
        Ok(Evaluator {
            inner: CoreEvaluator::new(params, gt, dt),
        })
    }

    /// Run evaluation and accumulation, returning the arrays pycocotools
    /// stores in `COCOeval.eval`.
    #[pyo3(signature = (collect_eval_imgs=false))]
    fn run<'py>(&self, py: Python<'py>, collect_eval_imgs: bool) -> PyResult<Bound<'py, PyDict>> {
        let (res, eval_imgs) = py.detach(|| self.inner.run(collect_eval_imgs));
        let [t, r, k, a, m] = res.counts;

        let out = PyDict::new(py);
        out.set_item("counts", vec![t, r, k, a, m])?;
        let prec = ArrayD::from_shape_vec(IxDyn(&[t, r, k, a, m]), res.precision)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rec = ArrayD::from_shape_vec(IxDyn(&[t, k, a, m]), res.recall)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let sc = ArrayD::from_shape_vec(IxDyn(&[t, r, k, a, m]), res.scores)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        out.set_item("precision", prec.into_pyarray(py))?;
        out.set_item("recall", rec.into_pyarray(py))?;
        out.set_item("scores", sc.into_pyarray(py))?;
        if collect_eval_imgs {
            out.set_item("evalImgs", eval_imgs_to_py(py, &eval_imgs, t)?)?;
        }
        Ok(out)
    }

    /// Per-instance verdicts at one setting, column-wise.
    ///
    /// Returns `(detections, ground_truths)`, each a dict of numpy arrays. The
    /// column form is not a stylistic choice: a COCO-scale run has hundreds of
    /// thousands of instances and building dicts for them costs more than the
    /// evaluation did.
    #[pyo3(signature = (t_idx=0, a_idx=0, max_det=100))]
    fn per_instance<'py>(
        &self,
        py: Python<'py>,
        t_idx: usize,
        a_idx: usize,
        max_det: usize,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        let (dets, gts) = py.detach(|| self.inner.per_instance(t_idx, a_idx, max_det));

        let d = PyDict::new(py);
        let col = |f: fn(&ufcoco_core::DetRecord) -> i64| -> Array1<i64> {
            Array1::from_iter(dets.iter().map(f))
        };
        d.set_item("image_id", col(|r| r.img_id).into_pyarray(py))?;
        d.set_item("category_id", col(|r| r.cat_id).into_pyarray(py))?;
        d.set_item("dt_id", col(|r| r.dt_id).into_pyarray(py))?;
        d.set_item("gt_id", col(|r| r.gt_id).into_pyarray(py))?;
        d.set_item(
            "score",
            Array1::from_iter(dets.iter().map(|r| r.score)).into_pyarray(py),
        )?;
        d.set_item(
            "iou",
            Array1::from_iter(dets.iter().map(|r| r.iou)).into_pyarray(py),
        )?;
        d.set_item(
            "ignore",
            Array1::from_iter(dets.iter().map(|r| r.ignore)).into_pyarray(py),
        )?;

        let g = PyDict::new(py);
        let gcol = |f: fn(&ufcoco_core::GtRecord) -> i64| -> Array1<i64> {
            Array1::from_iter(gts.iter().map(f))
        };
        g.set_item("image_id", gcol(|r| r.img_id).into_pyarray(py))?;
        g.set_item("category_id", gcol(|r| r.cat_id).into_pyarray(py))?;
        g.set_item("gt_id", gcol(|r| r.gt_id).into_pyarray(py))?;
        g.set_item(
            "ignore",
            Array1::from_iter(gts.iter().map(|r| r.ignore)).into_pyarray(py),
        )?;
        g.set_item(
            "matched",
            Array1::from_iter(gts.iter().map(|r| r.matched)).into_pyarray(py),
        )?;
        Ok((d, g))
    }
}

fn eval_imgs_to_py<'py>(
    py: Python<'py>,
    imgs: &[Option<ImgEval>],
    t_n: usize,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for e in imgs {
        match e {
            None => out.append(py.None())?,
            Some(e) => {
                let d = PyDict::new(py);
                d.set_item("image_id", e.img_id)?;
                d.set_item("category_id", e.cat_id)?;
                d.set_item("aRng", e.area_idx)?;
                d.set_item("maxDet", e.max_det)?;
                d.set_item("dtIds", e.dt_ids.clone())?;
                d.set_item("gtIds", e.gt_ids.clone())?;
                d.set_item("dtScores", e.dt_scores.clone())?;
                let d_n = e.dt_scores.len();
                let g_n = e.gt_ids.len();
                // pycocotools stores these as float64 matrices of ids.
                let f = |v: &[i64]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
                d.set_item(
                    "dtMatches",
                    Array2::from_shape_vec((t_n, d_n), f(&e.dt_matches))
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .into_pyarray(py),
                )?;
                d.set_item(
                    "gtMatches",
                    Array2::from_shape_vec((t_n, g_n), f(&e.gt_matches))
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                        .into_pyarray(py),
                )?;
                d.set_item(
                    "gtIgnore",
                    Array1::from_iter(e.gt_ignore.iter().map(|&b| b as u8 as f64)).into_pyarray(py),
                )?;
                d.set_item(
                    "dtIgnore",
                    Array2::from_shape_vec(
                        (t_n, d_n),
                        e.dt_ignore.iter().map(|&b| b as u8 as f64).collect(),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
                )?;
                out.append(d)?;
            }
        }
    }
    Ok(out)
}

#[pymodule]
fn _ufcoco(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Evaluator>()?;
    m.add_function(wrap_pyfunction!(mask::encode, m)?)?;
    m.add_function(wrap_pyfunction!(mask::decode, m)?)?;
    m.add_function(wrap_pyfunction!(mask::merge, m)?)?;
    m.add_function(wrap_pyfunction!(mask::area, m)?)?;
    m.add_function(wrap_pyfunction!(mask::to_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(mask::iou, m)?)?;
    m.add_function(wrap_pyfunction!(mask::fr_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(mask::fr_poly, m)?)?;
    m.add_function(wrap_pyfunction!(mask::fr_uncompressed, m)?)?;
    m.add_function(wrap_pyfunction!(mask::fr_py_objects, m)?)?;
    m.add_function(wrap_pyfunction!(mask::to_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(json::load_json, m)?)?;
    Ok(())
}
