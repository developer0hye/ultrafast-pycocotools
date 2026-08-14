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

mod alloc;
mod json;
mod mask;

use numpy::ndarray::{Array1, Array2, ArrayD, IxDyn};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyList, PyString};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use ufcoco_core::eval::{
    EvalParams, Evaluator as CoreEvaluator, GeomStore, ImgEval, Instances, IouType,
};
use ufcoco_core::rle::{self, Rle};

/// How many annotations are read out of Python before their geometry is
/// rasterised. Big enough to amortise the GIL round-trip, small enough that
/// the raw polygon buffer stays bounded on million-annotation datasets.
const CHUNK: usize = 4096;

#[cfg(feature = "alloc-stats")]
#[global_allocator]
static GLOBAL: alloc::Counting = alloc::Counting;

/// Rust-side allocation counters; see `alloc.rs`.
///
/// `enabled` is false unless the extension was built with the `alloc-stats`
/// feature, in which case every other field is zero and should be ignored
/// rather than reported as "no allocations".
#[pyfunction]
fn alloc_stats(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("enabled", cfg!(feature = "alloc-stats"))?;
    let (live, peak, total, allocs, frees) = alloc::snapshot();
    d.set_item("live_bytes", live)?;
    d.set_item("peak_bytes", peak)?;
    d.set_item("total_allocated_bytes", total)?;
    d.set_item("allocations", allocs)?;
    d.set_item("frees", frees)?;
    Ok(d)
}

/// Drop the allocation high-water mark to the current live figure, so the
/// next phase can be measured on its own.
#[pyfunction]
fn reset_alloc_peak() {
    alloc::reset_peak();
}

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
    /// One or more polygon rings, unioned into a single mask. Stored flat
    /// (`ends[i]` is the exclusive end of ring `i`) so a multi-ring polygon
    /// costs one allocation instead of one per ring.
    Poly { coords: Vec<f64>, ends: Vec<u32> },
    /// `counts` as a run list.
    Uncompressed(Vec<u32>),
    /// `counts` in the LEB128-ish string form.
    Compressed(Vec<u8>),
    /// No `segmentation` field: rasterise the bounding box instead.
    ///
    /// This is exactly what `loadRes` stores for a box-only detection
    /// (`[[x1, y1, x1, y2, x2, y2, x2, y1]]`), so deriving it here rather than
    /// materialising it in Python is invisible to the result and saves 376
    /// bytes per detection — 440 MB on Objects365.
    FromBbox([f64; 4]),
}

fn get_f64(d: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> PyResult<Option<f64>> {
    Ok(match d.get_item(key)? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    })
}

fn get_i64(d: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> PyResult<Option<i64>> {
    Ok(match d.get_item(key)? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    })
}

/// Read a flat `[x0, y0, x1, y1, ...]` ring into `out`.
///
/// `Vec<f64>::extract` would do this too, but it allocates a fresh `Vec` per
/// ring and goes through the generic sequence path. COCO ground truth has
/// ~900k polygon vertices, so appending into one reused buffer and taking the
/// `PyFloat` fast path is worth the explicit loop.
fn read_ring(obj: &Bound<'_, PyAny>, out: &mut Vec<f64>) -> PyResult<()> {
    if let Ok(list) = obj.cast::<PyList>() {
        let n = list.len();
        out.reserve(n);
        for i in 0..n {
            // SAFETY: `i < n`, and `list` is a list for as long as we hold
            // the GIL here.
            let item = unsafe { list.get_item_unchecked(i) };
            out.push(read_float(&item)?);
        }
        return Ok(());
    }
    out.extend(obj.extract::<Vec<f64>>()?);
    Ok(())
}

/// Read one number, taking the `PyFloat` fast path.
///
/// COCO coordinates are floats, but ints appear (a polygon vertex on a whole
/// pixel serialises as `12`, not `12.0`), so the slow path has to stay.
#[inline]
fn read_float(item: &Bound<'_, PyAny>) -> PyResult<f64> {
    match item.cast::<PyFloat>() {
        Ok(f) => Ok(f.value()),
        Err(_) => item.extract::<f64>(),
    }
}

/// Read `[x, y, w, h]` without allocating.
///
/// `Vec<f64>::extract` would heap-allocate four doubles per annotation; on
/// Objects365 that is 2.4 million allocations for data that fits in a
/// register pair.
fn read_bbox(v: &Bound<'_, PyAny>) -> PyResult<[f64; 4]> {
    let mut out = [0.0f64; 4];
    if let Ok(list) = v.cast::<PyList>() {
        if list.len() == 4 {
            for (i, slot) in out.iter_mut().enumerate() {
                // SAFETY: length checked immediately above.
                let item = unsafe { list.get_item_unchecked(i) };
                *slot = read_float(&item)?;
            }
            return Ok(out);
        }
    }
    let b: Vec<f64> = v.extract()?;
    if b.len() != 4 {
        return Err(PyValueError::new_err("'bbox' must have 4 entries"));
    }
    out.copy_from_slice(&b);
    Ok(out)
}

fn read_segm(d: &Bound<'_, PyDict>, keys: &Keys<'_>, bbox: [f64; 4]) -> PyResult<RawSegm> {
    let Some(seg) = d.get_item(keys.segmentation)? else {
        return Ok(RawSegm::FromBbox(bbox));
    };
    if seg.is_none() {
        return Ok(RawSegm::FromBbox(bbox));
    }
    if let Ok(sd) = seg.cast::<PyDict>() {
        let counts = sd
            .get_item(keys.counts)?
            .ok_or_else(|| PyValueError::new_err("RLE segmentation is missing 'counts'"))?;
        if let Ok(b) = counts.cast::<PyBytes>() {
            return Ok(RawSegm::Compressed(b.as_bytes().to_vec()));
        }
        if let Ok(s) = counts.extract::<String>() {
            return Ok(RawSegm::Compressed(s.into_bytes()));
        }
        return Ok(RawSegm::Uncompressed(counts.extract()?));
    }
    // A list of rings, each a flat [x0, y0, x1, y1, ...]. Stored flat with
    // offsets so a polygon with several rings costs one allocation, not one
    // per ring.
    let mut coords: Vec<f64> = Vec::new();
    let mut ends: Vec<u32> = Vec::new();
    for ring in seg.try_iter()? {
        read_ring(&ring?, &mut coords)?;
        ends.push(coords.len() as u32);
    }
    Ok(RawSegm::Poly { coords, ends })
}

/// Interned annotation keys.
///
/// `dict.get_item("image_id")` builds a fresh Python string for the lookup
/// every time. At six lookups per annotation that is 14 million string
/// objects on Objects365 — more work than the evaluation. `intern!` resolves
/// each to a cached `PyString` once per interpreter.
struct Keys<'py> {
    id: &'py Bound<'py, PyString>,
    image_id: &'py Bound<'py, PyString>,
    category_id: &'py Bound<'py, PyString>,
    score: &'py Bound<'py, PyString>,
    bbox: &'py Bound<'py, PyString>,
    area: &'py Bound<'py, PyString>,
    iscrowd: &'py Bound<'py, PyString>,
    num_keypoints: &'py Bound<'py, PyString>,
    keypoints: &'py Bound<'py, PyString>,
    lvis_mark: &'py Bound<'py, PyString>,
    segmentation: &'py Bound<'py, PyString>,
    counts: &'py Bound<'py, PyString>,
}

impl<'py> Keys<'py> {
    fn new(py: Python<'py>) -> Keys<'py> {
        Keys {
            id: intern!(py, "id"),
            image_id: intern!(py, "image_id"),
            category_id: intern!(py, "category_id"),
            score: intern!(py, "score"),
            bbox: intern!(py, "bbox"),
            area: intern!(py, "area"),
            iscrowd: intern!(py, "iscrowd"),
            num_keypoints: intern!(py, "num_keypoints"),
            keypoints: intern!(py, "keypoints"),
            lvis_mark: intern!(py, "lvis_mark"),
            segmentation: intern!(py, "segmentation"),
            counts: intern!(py, "counts"),
        }
    }
}

/// pycocotools' `annToRLE`: polygons are unioned, uncompressed RLE is taken
/// as-is, compressed RLE is decoded.
fn raw_to_rle(raw: &RawSegm, h: u32, w: u32, scratch: &mut rle::PolyScratch) -> Rle {
    match raw {
        RawSegm::Poly { coords, ends } => rle::rle_fr_polys_flat_into(coords, ends, h, w, scratch),
        RawSegm::Uncompressed(c) => rle::rle_fr_uncompressed(c, h, w),
        RawSegm::Compressed(s) => Rle::from_str(s, h, w),
        RawSegm::FromBbox(b) => rle::rle_fr_bbox(b, h, w),
    }
}

#[allow(clippy::too_many_arguments)]
/// An empty `Instances` sized for `n` annotations of `iou_type`.
fn new_instances(n: usize, iou_type: IouType) -> Instances {
    Instances {
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
    }
}

fn attach_masks(geom: &mut GeomStore, masks: Vec<(Rle, Option<Rle>)>) {
    match geom {
        GeomStore::Masks(v) => v.extend(masks.into_iter().map(|(m, _)| m)),
        GeomStore::Boundaries { masks: ms, boundaries } => {
            for (m, b) in masks {
                ms.push(m);
                boundaries.push(b.expect("boundary requested but not built"));
            }
        }
        _ => debug_assert!(masks.is_empty()),
    }
}

/// Read both annotation sets, rasterising behind a single worker.
///
/// Reading annotations out of Python needs the GIL; rasterising them does not.
/// So one worker thread takes finished chunks and rasterises them across the
/// rayon pool while this thread keeps reading, and total time becomes
/// max(read, rasterise) instead of their sum.
///
/// The two sides share that worker rather than getting one each. With separate
/// pipelines the detection read could not start until the ground-truth
/// rasteriser had drained, and on COCO segmentation that barrier cost 0.02s of
/// a 0.13s extraction — the reader sat idle while the worker finished, then
/// the worker sat idle while the reader restarted. Ground truth is sent first
/// and the channel preserves order, so the finished masks split back apart at
/// a known offset.
#[allow(clippy::too_many_arguments)]
fn extract_both(
    py: Python<'_>,
    gt_anns: &Bound<'_, PyList>,
    dt_anns: &Bound<'_, PyList>,
    iou_type: IouType,
    img_sizes: &HashMap<i64, (u32, u32)>,
    img_slot: &HashMap<i64, u32>,
    cat_slot: &HashMap<i64, u32>,
    boundary_dilation: f64,
) -> PyResult<(Instances, Instances, ExtractTimings)> {
    let keys = Keys::new(py);
    let timings: Arc<ExtractTimings> = Arc::default();
    let needs_mask = matches!(iou_type, IouType::Segm | IouType::Boundary);
    let want_boundary = iou_type == IouType::Boundary;
    let mut gt = new_instances(gt_anns.len(), iou_type);
    let mut dt = new_instances(dt_anns.len(), iou_type);
    // Every annotation contributes exactly one raw segmentation when masks are
    // in play, so this is where the worker's output splits.
    let gt_mask_count = if needs_mask { gt_anns.len() } else { 0 };

    let (gt_masks, dt_masks) = std::thread::scope(
        |scope| -> PyResult<(Vec<(Rle, Option<Rle>)>, Vec<(Rle, Option<Rle>)>)> {
            let (tx_raw, rx_raw) =
                std::sync::mpsc::sync_channel::<Vec<(RawSegm, u32, u32)>>(2);
            let worker_timings = Arc::clone(&timings);
            let worker = scope.spawn(move || {
                let mut out: Vec<(Rle, Option<Rle>)> = Vec::new();
                while let Ok(chunk) = rx_raw.recv() {
                    let t = Instant::now();
                    let built: Vec<(Rle, Option<Rle>)> = chunk
                        .par_iter()
                        // One scratch per worker: polygon rasterisation
                        // otherwise spends its time allocating and freeing the
                        // same buffers.
                        .map_init(rle::PolyScratch::default, |scratch, (r, h, w)| {
                            let m = raw_to_rle(r, *h, *w, scratch);
                            let b =
                                want_boundary.then(|| rle::rle_to_boundary(&m, boundary_dilation));
                            (m, b)
                        })
                        .collect();
                    worker_timings
                        .rasterise_ns
                        .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    out.extend(built);
                }
                out
            });

            let mut raw_chunk: Vec<(RawSegm, u32, u32)> = Vec::with_capacity(CHUNK);
            let outcome = read_annotations(
                gt_anns, true, iou_type, img_sizes, img_slot, cat_slot, &keys, needs_mask,
                &mut gt, &mut raw_chunk, &tx_raw, &timings, true,
            )
            .and_then(|()| {
                read_annotations(
                    dt_anns, false, iou_type, img_sizes, img_slot, cat_slot, &keys, needs_mask,
                    &mut dt, &mut raw_chunk, &tx_raw, &timings, false,
                )
            });
            if outcome.is_ok() && !raw_chunk.is_empty() {
                let _ = tx_raw.send(raw_chunk);
            }
            // Closing the channel is what lets the worker finish; it must
            // happen on the error path too.
            drop(tx_raw);
            let mut built = worker.join().expect("mask worker panicked");
            outcome?;
            let dt_masks = built.split_off(gt_mask_count.min(built.len()));
            Ok((built, dt_masks))
        },
    )?;

    attach_masks(&mut gt.geom, gt_masks);
    attach_masks(&mut dt.geom, dt_masks);
    let timings = Arc::try_unwrap(timings).unwrap_or_default();
    Ok((gt, dt, timings))
}

/// Wall-clock breakdown of building an `Evaluator`.
///
/// `read` and `rasterise` overlap by design, so they do not sum to the total.
/// `read_blocked` is the part of `read` spent waiting for the rasteriser to
/// catch up: if it is large the rasteriser is the bottleneck, if it is ~0 the
/// GIL-bound reading is.
#[derive(Default)]
struct ExtractTimings {
    gt_read_ns: AtomicU64,
    dt_read_ns: AtomicU64,
    read_blocked_ns: AtomicU64,
    rasterise_ns: AtomicU64,
}

/// The GIL-bound half of extraction: scalar fields straight into `inst`, raw
/// geometry batched out to the rasteriser.
#[allow(clippy::too_many_arguments)]
fn read_annotations(
    anns: &Bound<'_, PyList>,
    is_gt: bool,
    iou_type: IouType,
    img_sizes: &HashMap<i64, (u32, u32)>,
    img_slot: &HashMap<i64, u32>,
    cat_slot: &HashMap<i64, u32>,
    keys: &Keys<'_>,
    needs_mask: bool,
    inst: &mut Instances,
    raw_chunk: &mut Vec<(RawSegm, u32, u32)>,
    tx_raw: &std::sync::mpsc::SyncSender<Vec<(RawSegm, u32, u32)>>,
    timings: &ExtractTimings,
    is_gt_side: bool,
) -> PyResult<()> {
    let t_read = Instant::now();
    for (i, item) in anns.iter().enumerate() {
        let d = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("annotations must be dicts"))?;
        let image_id = get_i64(d, keys.image_id)?
            .ok_or_else(|| PyKeyError::new_err("annotation is missing 'image_id'"))?;
        let category_id = get_i64(d, keys.category_id)?.unwrap_or(-1);
        inst.ids.push(get_i64(d, keys.id)?.unwrap_or(i as i64 + 1));
        inst.img_slot
            .push(img_slot.get(&image_id).copied().unwrap_or(u32::MAX));
        inst.cat_slot
            .push(cat_slot.get(&category_id).copied().unwrap_or(u32::MAX));
        inst.scores.push(get_f64(d, keys.score)?.unwrap_or(0.0));

        let bbox: [f64; 4] = match d.get_item(keys.bbox)? {
            Some(v) if !v.is_none() => read_bbox(&v)?,
            _ => [0.0; 4],
        };
        inst.areas
            .push(get_f64(d, keys.area)?.unwrap_or(bbox[2] * bbox[3]));
        let iscrowd = get_i64(d, keys.iscrowd)?.unwrap_or(0) != 0;
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
                ig = get_i64(d, keys.num_keypoints)?.unwrap_or(0) == 0 || ig;
            }
            inst.ignore.push(ig);
        } else {
            inst.ignore.push(false);
        }
        inst.lvis_mark
            .push(get_i64(d, keys.lvis_mark)?.unwrap_or(0) != 0);

        match &mut inst.geom {
            GeomStore::Bboxes(v) => v.push(bbox),
            GeomStore::Keypoints { data, k } => {
                inst.bboxes.push(bbox);
                let kp: Vec<f64> = match d.get_item(keys.keypoints)? {
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
                raw_chunk.push((read_segm(d, keys, bbox)?, h, w));
            }
        }

        if needs_mask && raw_chunk.len() >= CHUNK {
            // Blocks once the worker is two chunks behind, which is the
            // backpressure that bounds raw-geometry memory.
            let chunk = std::mem::replace(raw_chunk, Vec::with_capacity(CHUNK));
            let t_block = Instant::now();
            let sent = tx_raw.send(chunk);
            timings
                .read_blocked_ns
                .fetch_add(t_block.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if sent.is_err() {
                return Err(PyValueError::new_err("mask rasteriser stopped"));
            }
        }
    }
    let counter = if is_gt_side {
        &timings.gt_read_ns
    } else {
        &timings.dt_read_ns
    };
    counter.fetch_add(t_read.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

/// The evaluation engine, constructed once from prepared annotations.
#[pyclass(module = "ultrafast_pycocotools._ufcoco")]
pub struct Evaluator {
    inner: CoreEvaluator,
    extract: ExtractTimings,
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

        let (gt, dt, extract) = extract_both(
            py,
            gt_anns,
            dt_anns,
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
            extract,
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

    /// Per-phase timings, in seconds.
    ///
    /// Extraction and evaluation are measured differently on purpose.
    /// `gt_read` / `gt_rasterise` are wall-clock on two threads that overlap,
    /// so they do not sum; `gt_read_blocked` is how much of the reading was
    /// spent waiting on the rasteriser, which is what says who the bottleneck
    /// is. The evaluation phases are summed across rayon workers, so compare
    /// them to the caller's wall-clock to see whether they parallelised.
    fn timings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let ns = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e9;
        d.set_item("gt_read", ns(&self.extract.gt_read_ns))?;
        d.set_item("dt_read", ns(&self.extract.dt_read_ns))?;
        d.set_item("read_blocked", ns(&self.extract.read_blocked_ns))?;
        d.set_item("rasterise", ns(&self.extract.rasterise_ns))?;
        let [group, iou, matching, accumulate] = self.inner.timings().as_secs();
        d.set_item("group_index", group)?;
        d.set_item("iou_cpu", iou)?;
        d.set_item("match_cpu", matching)?;
        d.set_item("accumulate_cpu", accumulate)?;
        Ok(d)
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
    m.add_function(wrap_pyfunction!(alloc_stats, m)?)?;
    m.add_function(wrap_pyfunction!(reset_alloc_peak, m)?)?;
    Ok(())
}
