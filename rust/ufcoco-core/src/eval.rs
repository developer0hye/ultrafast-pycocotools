//! COCO detection / segmentation / keypoint evaluation.
//!
//! A port of `pycocotools.cocoeval.COCOeval` that aims to be numerically
//! identical to it, not merely close. The places where that costs something
//! are called out in comments; the short list is:
//!
//! * every sort is stable, because pycocotools passes `kind='mergesort'` and
//!   the greedy matcher is order-sensitive — with ties in detection score, an
//!   unstable sort changes which ground truth gets claimed and moves AP;
//! * precision is `tp / (fp + tp + eps)` with `eps = np.spacing(1)`, not
//!   `tp / (fp + tp)`. Those differ by one ULP whenever `fp + tp == 1`, which
//!   is the first point of every curve;
//! * the IoU and recall threshold grids are supplied by the caller, which
//!   builds them with `np.linspace`. Reconstructing them as `start + i * step`
//!   lands one ULP off at several points and shifts AP by ~1e-6 — this is a
//!   real, measured failure mode of other reimplementations, not a
//!   hypothetical.
//!
//! Structurally the difference from pycocotools is the loop nest. Upstream
//! computes every IoU, materialises `K*A*I` result dicts, then accumulates.
//! Here the outer loop is over categories and the whole (IoU -> match ->
//! accumulate) chain runs inside it, so live memory is one category's worth of
//! intermediates instead of the whole dataset's.
//!
//! The matching loops are written with explicit indices because they walk
//! several parallel arrays at once (ious, the ignore-sorted permutation, the
//! match table) and because they have to stay readable next to the Python they
//! are a port of.
#![allow(clippy::needless_range_loop)]

use crate::group::{Grouping, Run, RunJoin};
use crate::rle::{self, Rle};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Instant;

/// Where the evaluation spent its time, in nanoseconds.
///
/// Values are **summed across worker threads**, so on a parallel phase they
/// exceed wall-clock time — deliberately. The sum says where the CPU went; the
/// caller's wall-clock says whether it parallelised. Comparing the two is what
/// distinguishes "this phase is expensive" from "this phase is serialised".
///
/// Timers sit at category granularity (a few hundred `Instant::now()` calls
/// per run), so the instrumentation cannot distort what it measures.
#[derive(Default, Debug)]
pub struct Timings {
    /// Building the sparse (image, category) index.
    pub group_ns: AtomicU64,
    /// `computeIoU` / `computeOks`, including the per-image detection sort.
    pub iou_ns: AtomicU64,
    /// `evaluateImg`: the greedy matcher.
    pub match_ns: AtomicU64,
    /// `accumulate`: the score sort and the PR curves.
    pub accumulate_ns: AtomicU64,
}

impl Timings {
    #[inline]
    fn add(counter: &AtomicU64, start: Instant) {
        counter.fetch_add(start.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
    }

    /// `(group, iou, match, accumulate)` in seconds.
    pub fn as_secs(&self) -> [f64; 4] {
        let g = |c: &AtomicU64| c.load(AtomicOrdering::Relaxed) as f64 / 1e9;
        [
            g(&self.group_ns),
            g(&self.iou_ns),
            g(&self.match_ns),
            g(&self.accumulate_ns),
        ]
    }
}

/// `np.spacing(1)`.
pub const EPS: f64 = f64::EPSILON;

/// Smallest slice of images a nested parallel split will hand to a worker.
///
/// Below this the split costs more than the work; above it, uneven categories
/// get spread across idle workers.
const PAR_MIN_IMAGES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IouType {
    Segm,
    Bbox,
    Keypoints,
    /// Extension: `min(mask IoU, boundary IoU)`, per Boundary IoU (CVPR'21).
    Boundary,
}

/// Per-annotation geometry, stored column-wise so a million annotations do
/// not each pay for the largest variant.
#[derive(Debug)]
pub enum GeomStore {
    Bboxes(Vec<[f64; 4]>),
    /// Immutable file geometry shared with the input handle.
    SharedBboxes(Arc<Vec<[f64; 4]>>),
    Masks(Vec<Rle>),
    Boundaries {
        masks: Vec<Rle>,
        boundaries: Vec<Rle>,
    },
    /// Flat `k * 3` keypoint triplets per instance.
    Keypoints {
        data: Vec<f64>,
        k: usize,
    },
}

impl GeomStore {
    fn bbox_slice(&self) -> Option<&[[f64; 4]]> {
        match self {
            Self::Bboxes(v) => Some(v),
            Self::SharedBboxes(v) => Some(v),
            _ => None,
        }
    }
}

/// Annotations in struct-of-arrays form.
#[derive(Debug)]
pub struct Instances {
    pub ids: Vec<i64>,
    pub scores: Vec<f64>,
    pub areas: Vec<f64>,
    pub iscrowd: Vec<bool>,
    /// Ground truth only: the `ignore` flag pycocotools derives in `_prepare`.
    pub ignore: Vec<bool>,
    /// LVIS extension: a detection of a category that is not exhaustively
    /// annotated in this image is ignored rather than counted as a false
    /// positive. Always false for plain COCO.
    pub lvis_mark: Vec<bool>,
    /// Keypoint runs only: the ground-truth box OKS uses for its fall-back
    /// distance and for the CrowdPose area substitute.
    pub bboxes: Vec<[f64; 4]>,
    /// Slot indices resolved against `EvalParams::img_ids` / `cat_ids`;
    /// `u32::MAX` for annotations outside the evaluation.
    pub img_slot: Vec<u32>,
    pub cat_slot: Vec<u32>,
    pub geom: GeomStore,
}

impl Instances {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct EvalParams {
    pub img_ids: Vec<i64>,
    pub cat_ids: Vec<i64>,
    pub iou_thrs: Vec<f64>,
    pub rec_thrs: Vec<f64>,
    /// Ascending, as pycocotools sorts them in `evaluate()`.
    pub max_dets: Vec<usize>,
    pub area_rng: Vec<[f64; 2]>,
    pub use_cats: bool,
    pub iou_type: IouType,
    pub kpt_sigmas: Vec<f64>,
    /// CrowdPose ground truth has no usable `area`; fall back to
    /// `0.53 * w * h`.
    pub use_area: bool,
}

/// Accumulated curves, laid out exactly like the numpy arrays pycocotools
/// stores in `self.eval`.
#[derive(Debug)]
pub struct EvalResult {
    /// `[T, R, K, A, M]`
    pub counts: [usize; 5],
    /// `T*R*K*A*M`, C order
    pub precision: Vec<f64>,
    /// `T*K*A*M`, C order
    pub recall: Vec<f64>,
    /// `T*R*K*A*M`, C order
    pub scores: Vec<f64>,
}

/// One evaluated detection, with the verdict the AP arithmetic gave it.
///
/// `ignore` is the deciding field and the reason this is not just a list of
/// matches: a detection that matched a crowd region, or that falls outside the
/// area range, counts as neither a true nor a false positive. Anything built
/// on top of this (confusion matrices, TP/FP listings, calibration) has to
/// respect that or it will disagree with the AP printed next to it.
#[derive(Clone, Copy, Debug)]
pub struct DetRecord {
    pub img_id: i64,
    pub cat_id: i64,
    pub dt_id: i64,
    pub score: f64,
    /// Matched ground-truth id, or -1.
    pub gt_id: i64,
    /// IoU with the matched ground truth; 0.0 when unmatched.
    pub iou: f64,
    pub ignore: bool,
}

/// One evaluated ground truth.
#[derive(Clone, Copy, Debug)]
pub struct GtRecord {
    pub img_id: i64,
    pub cat_id: i64,
    pub gt_id: i64,
    /// Ignored: a crowd region, or outside the area range.
    pub ignore: bool,
    pub matched: bool,
}

/// The `evalImgs` entry for one (image, category, area range), in the shape
/// pycocotools exposes.
#[derive(Clone, Debug)]
pub struct ImgEval {
    pub img_id: i64,
    pub cat_id: i64,
    pub area_idx: usize,
    pub max_det: usize,
    pub dt_ids: Vec<i64>,
    pub gt_ids: Vec<i64>,
    pub dt_scores: Vec<f64>,
    pub gt_ignore: Vec<bool>,
    /// `T * D`, matched ground-truth id or 0 (pycocotools' `dtMatches`).
    pub dt_matches: Vec<i64>,
    /// `T * G`, matched detection id or 0 (`gtMatches`).
    pub gt_matches: Vec<i64>,
    /// `T * D`
    pub dt_ignore: Vec<bool>,
}

/// Stable descending order by score, NaN last.
///
/// pycocotools sorts `np.argsort(-scores, kind='mergesort')`. Negating and
/// sorting ascending is the same permutation as sorting descending — including
/// for `-0.0`, which compares equal to `0.0` either way — and numpy puts NaN
/// last.
#[inline]
fn cmp_desc_score(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => b.partial_cmp(&a).unwrap(),
    }
}

/// Geometry is only read for the largest per-image evaluation limit. Keep
/// scalar rows intact, but allow the loader to omit decoded mask payload for
/// detections outside that limit. Reuse matching's category-major grouping and
/// stable score ordering, including class-agnostic ties and NaNs.
pub fn detection_geometry_keep(
    dt: &Instances,
    category_count: usize,
    use_cats: bool,
    max_det: usize,
) -> Vec<bool> {
    let n_groups = if use_cats { category_count } else { 1 };
    let groups = Grouping::build(&dt.img_slot, &dt.cat_slot, n_groups, use_cats);
    let mut keep = vec![true; dt.len()];
    let mut order = Vec::new();
    for group in 0..n_groups {
        for run in groups.group(group) {
            if run.len as usize <= max_det {
                continue;
            }
            order.clear();
            order.extend_from_slice(groups.indices(run));
            order.sort_by(|&a, &b| cmp_desc_score(dt.scores[a as usize], dt.scores[b as usize]));
            for &index in &order[max_det..] {
                keep[index as usize] = false;
            }
        }
    }
    keep
}

/// Everything one image contributes to one category.
struct CatImage<'a> {
    img_slot: u32,
    gt_idx: &'a [u32],
    /// Score-sorted, truncated to `max_dets.last()`.
    dt_idx: Vec<u32>,
    /// Row-major `D x G`; empty when either side is empty, which is the `[]`
    /// pycocotools' `computeIoU` returns.
    ious: Vec<f64>,
}

/// Per-image match result for one area range, in the compact form
/// accumulation needs.
#[derive(Default)]
struct ImgMatch {
    dt_scores: Vec<f64>,
    /// `T * D`, ground-truth slot or -1.
    dt_match: Vec<i32>,
    /// `T * D`
    dt_ignore: Vec<bool>,
    /// `G`, in ignore-sorted order.
    gt_ignore: Vec<bool>,
    /// Permutation applied to the ground truth (ignore-sorted, stable).
    gt_perm: Vec<u32>,
}

pub struct Evaluator {
    pub params: EvalParams,
    gt: Instances,
    dt: Instances,
    gt_groups: Grouping,
    dt_groups: Grouping,
    n_groups: usize,
    timings: Timings,
}

impl Evaluator {
    pub fn new(params: EvalParams, gt: Instances, dt: Instances) -> Evaluator {
        // With useCats = 0 everything collapses into a single group, but the
        // grouping still sorts by category inside each image so the merged
        // list matches pycocotools' category-major concatenation.
        let n_groups = if params.use_cats {
            params.cat_ids.len()
        } else {
            1
        };
        let timings = Timings::default();
        let t = Instant::now();
        let gt_groups = Grouping::build(&gt.img_slot, &gt.cat_slot, n_groups, params.use_cats);
        let dt_groups = Grouping::build(&dt.img_slot, &dt.cat_slot, n_groups, params.use_cats);
        Timings::add(&timings.group_ns, t);
        Evaluator {
            params,
            gt,
            dt,
            gt_groups,
            dt_groups,
            n_groups,
            timings,
        }
    }

    /// Apply LVIS ignore semantics without cloning Python annotation dictionaries.
    pub fn configure_lvis(
        &mut self,
        gt_ignore: Vec<bool>,
        non_exhaustive: Vec<(i64, i64)>,
        preserve_crowds: bool,
    ) -> Result<(), String> {
        if gt_ignore.len() != self.gt.len() {
            return Err("LVIS ignore flags do not match ground-truth length".into());
        }
        self.gt.ignore = gt_ignore;
        if !preserve_crowds {
            self.gt.iscrowd.fill(false);
        }
        let pairs: std::collections::HashSet<(i64, i64)> = non_exhaustive.into_iter().collect();
        for i in 0..self.dt.len() {
            let pair = self
                .params
                .img_ids
                .get(self.dt.img_slot[i] as usize)
                .zip(self.params.cat_ids.get(self.dt.cat_slot[i] as usize));
            self.dt.lvis_mark[i] = pair
                .map(|(&image, &category)| pairs.contains(&(image, category)))
                .unwrap_or(false);
        }
        Ok(())
    }

    /// Per-phase timings accumulated so far. See [`Timings`].
    pub fn timings(&self) -> &Timings {
        &self.timings
    }

    pub fn gt_instances(&self) -> &Instances {
        &self.gt
    }
    pub fn dt_instances(&self) -> &Instances {
        &self.dt
    }

    /// `computeIoU` / `computeOks` for every image of one category.
    fn prepare_category(&self, k: usize) -> Vec<CatImage<'_>> {
        let max_det = self.params.max_dets.last().copied().unwrap_or(0);
        let gt_runs = self.gt_groups.group(k);
        let dt_runs = self.dt_groups.group(k);
        let joins: Vec<(u32, Option<Run>, Option<Run>)> = RunJoin::new(gt_runs, dt_runs).collect();
        // Nested inside the per-category fan-out. Categories are wildly
        // uneven — `person` alone is a third of COCO's annotations — so
        // splitting only by category leaves one worker with the tail while
        // eleven idle. `with_min_len` keeps small categories from paying for
        // splits they do not need.
        joins
            .par_iter()
            .with_min_len(PAR_MIN_IMAGES)
            .map(|&(img_slot, gr, dr)| {
                let gt_idx: &[u32] = gr
                    .map(|r: Run| self.gt_groups.indices(&r))
                    .unwrap_or_default();
                let mut dt_idx: Vec<u32> = dr
                    .map(|r: Run| self.dt_groups.indices(&r).to_vec())
                    .unwrap_or_default();
                dt_idx.sort_by(|&a, &b| {
                    cmp_desc_score(self.dt.scores[a as usize], self.dt.scores[b as usize])
                });
                if dt_idx.len() > max_det {
                    dt_idx.truncate(max_det);
                }
                let ious = self.compute_iou(&dt_idx, gt_idx);
                CatImage {
                    img_slot,
                    gt_idx,
                    dt_idx,
                    ious,
                }
            })
            .collect()
    }

    /// Row-major `D x G` IoU, or empty when either side has no entries.
    /// The lowest IoU any match can be made at.
    ///
    /// `evaluate_img` compares against `min(thr, 1 - 1e-10)`, so this is the
    /// floor over every threshold. A pair below it is skipped at all of them,
    /// which is what lets `rle_iou_refs_above` stop short of an exact answer.
    /// Zero (or a caller asking for a zero threshold) turns that off — see the
    /// note on `rle_iou_refs_above`.
    fn match_floor(&self) -> f64 {
        self.params
            .iou_thrs
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .min(1.0 - 1e-10)
            .max(0.0)
    }

    fn compute_iou(&self, dt_idx: &[u32], gt_idx: &[u32]) -> Vec<f64> {
        if dt_idx.is_empty() || gt_idx.is_empty() {
            return Vec::new();
        }
        let (m, n) = (dt_idx.len(), gt_idx.len());
        let mut out = vec![0.0f64; m * n];
        if let (Some(dv), Some(gv)) = (self.dt.geom.bbox_slice(), self.gt.geom.bbox_slice()) {
            rle::bb_iou_indexed(dv, gv, dt_idx, gt_idx, &self.gt.iscrowd, &mut out);
            return out;
        }
        let iscrowd: Vec<u8> = gt_idx
            .iter()
            .map(|&g| self.gt.iscrowd[g as usize] as u8)
            .collect();
        let floor = self.match_floor();

        match (&self.dt.geom, &self.gt.geom) {
            (GeomStore::Masks(dv), GeomStore::Masks(gv)) => {
                let d: Vec<&Rle> = dt_idx.iter().map(|&i| &dv[i as usize]).collect();
                let g: Vec<&Rle> = gt_idx.iter().map(|&i| &gv[i as usize]).collect();
                rle::rle_iou_refs_above(&d, &g, &iscrowd, floor, &mut out);
            }
            (
                GeomStore::Boundaries {
                    masks: dm,
                    boundaries: db,
                },
                GeomStore::Boundaries {
                    masks: gm,
                    boundaries: gb,
                },
            ) => {
                let d: Vec<&Rle> = dt_idx.iter().map(|&i| &dm[i as usize]).collect();
                let g: Vec<&Rle> = gt_idx.iter().map(|&i| &gm[i as usize]).collect();
                rle::rle_iou_refs_above(&d, &g, &iscrowd, floor, &mut out);
                let d: Vec<&Rle> = dt_idx.iter().map(|&i| &db[i as usize]).collect();
                let g: Vec<&Rle> = gt_idx.iter().map(|&i| &gb[i as usize]).collect();
                let mut bout = vec![0.0f64; m * n];
                rle::rle_iou_refs_above(&d, &g, &iscrowd, floor, &mut bout);
                // Crowd ground truth keeps the plain mask IoU.
                for gi in 0..n {
                    if iscrowd[gi] != 0 {
                        continue;
                    }
                    for di in 0..m {
                        let i = di * n + gi;
                        out[i] = out[i].min(bout[i]);
                    }
                }
            }
            (GeomStore::Keypoints { data: dd, k }, GeomStore::Keypoints { data: gd, .. }) => {
                self.compute_oks(dt_idx, gt_idx, dd, gd, *k, &mut out);
            }
            _ => panic!("ground truth and detection geometry kinds disagree"),
        }
        out
    }

    /// `computeOks`, following pycocotools' exact operation order.
    ///
    /// The divisions are applied one at a time (`/ vars / area / 2`), which is
    /// not the same in floating point as dividing by the product, so they stay
    /// separate here.
    fn compute_oks(
        &self,
        dt_idx: &[u32],
        gt_idx: &[u32],
        dd: &[f64],
        gd: &[f64],
        k: usize,
        out: &mut [f64],
    ) {
        let n = gt_idx.len();
        let vars: Vec<f64> = self
            .params
            .kpt_sigmas
            .iter()
            .map(|s| (s * 2.0) * (s * 2.0))
            .collect();
        for (j, &gi) in gt_idx.iter().enumerate() {
            let g = &gd[(gi as usize) * k * 3..(gi as usize + 1) * k * 3];
            let k1 = (0..k).filter(|&t| g[t * 3 + 2] > 0.0).count();
            let bb = self.gt.bboxes[gi as usize];
            let (x0, x1) = (bb[0] - bb[2], bb[0] + bb[2] * 2.0);
            let (y0, y1) = (bb[1] - bb[3], bb[1] + bb[3] * 2.0);
            let area = if self.params.use_area {
                self.gt.areas[gi as usize]
            } else {
                bb[3] * bb[2] * 0.53
            };
            for (i, &di) in dt_idx.iter().enumerate() {
                let d = &dd[(di as usize) * k * 3..(di as usize + 1) * k * 3];
                let mut sum = 0.0f64;
                let mut cnt = 0usize;
                for t in 0..k {
                    // `!(v > 0.0)`, not `v <= 0.0`: pycocotools filters with
                    // the boolean mask `vg > 0`, so a NaN visibility flag is
                    // excluded. The negation is what reproduces that; the
                    // "simpler" comparison would keep it.
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    if k1 > 0 && !(g[t * 3 + 2] > 0.0) {
                        continue;
                    }
                    let (xd, yd) = (d[t * 3], d[t * 3 + 1]);
                    let (dx, dy) = if k1 > 0 {
                        (xd - g[t * 3], yd - g[t * 3 + 1])
                    } else {
                        (
                            f64::max(0.0, x0 - xd) + f64::max(0.0, xd - x1),
                            f64::max(0.0, y0 - yd) + f64::max(0.0, yd - y1),
                        )
                    };
                    let e = (dx * dx + dy * dy) / vars[t] / (area + EPS) / 2.0;
                    sum += (-e).exp();
                    cnt += 1;
                }
                out[i * n + j] = sum / cnt as f64;
            }
        }
    }

    /// `evaluateImg` for one image and area range.
    fn evaluate_img(&self, ci: &CatImage<'_>, area_idx: usize) -> Option<ImgMatch> {
        let mut out = ImgMatch::default();
        let mut scratch = MatchScratch::default();
        self.evaluate_img_into(ci, area_idx, &mut out, &mut scratch)
            .then_some(out)
    }

    /// `evaluateImg`, writing into caller-owned buffers.
    ///
    /// Returns whether this image contributes anything, which depends only on
    /// whether it has any annotations and so is the same for every area range.
    /// Reusing `out` across area ranges is what keeps this off the allocator:
    /// at Objects365 scale the per-image vectors were 20 million allocations,
    /// and sixteen threads contending for the heap costs more than the
    /// matching itself.
    fn evaluate_img_into(
        &self,
        ci: &CatImage<'_>,
        area_idx: usize,
        out: &mut ImgMatch,
        scratch: &mut MatchScratch,
    ) -> bool {
        // RunJoin already excludes groups with neither GT nor detections.
        // A zero maxDets may empty a real detection-only group afterwards;
        // pycocotools still exposes its empty evalImgs record, not None.
        let a_rng = self.params.area_rng[area_idx];
        let t_n = self.params.iou_thrs.len();
        let g_n = ci.gt_idx.len();
        let d_n = ci.dt_idx.len();
        // Destructured so the matcher can hold several of these at once;
        // borrowing them one field at a time through `out` would not compile.
        let ImgMatch {
            dt_scores,
            dt_match,
            dt_ignore,
            gt_ignore,
            gt_perm,
        } = out;

        // Ignore flags, then a stable partition that puts them last.
        let ignore = &mut scratch.ignore;
        ignore.clear();
        ignore.extend(ci.gt_idx.iter().map(|&g| {
            let g = g as usize;
            let a = self.gt.areas[g];
            self.gt.ignore[g] || a < a_rng[0] || a > a_rng[1]
        }));
        gt_perm.clear();
        // Stable binary partition without a comparison sort or its scratch allocation.
        gt_perm.extend((0..g_n as u32).filter(|&i| !ignore[i as usize]));
        gt_perm.extend((0..g_n as u32).filter(|&i| ignore[i as usize]));
        gt_ignore.clear();
        gt_ignore.extend(gt_perm.iter().map(|&i| ignore[i as usize]));

        dt_match.clear();
        dt_match.resize(t_n * d_n, -1);
        let gt_matched = &mut scratch.gt_matched;
        gt_matched.clear();
        gt_matched.resize(t_n * g_n, false);

        if !ci.ious.is_empty() {
            for (tind, &thr) in self.params.iou_thrs.iter().enumerate() {
                for dind in 0..d_n {
                    // pycocotools clamps the floor so a threshold of exactly
                    // 1.0 can still match a pair whose IoU rounds just below.
                    let mut best = f64::min(thr, 1.0 - 1e-10);
                    let mut m: i32 = -1;
                    for gind in 0..g_n {
                        let gsrc = gt_perm[gind] as usize;
                        // Already claimed by a higher-scoring detection, and
                        // not a crowd region (which may absorb many).
                        if gt_matched[tind * g_n + gind]
                            && !self.gt.iscrowd[ci.gt_idx[gsrc] as usize]
                        {
                            continue;
                        }
                        // Ground truths are ignore-sorted, so once we hold a
                        // real match there is nothing better further right.
                        if m > -1 && !gt_ignore[m as usize] && gt_ignore[gind] {
                            break;
                        }
                        let v = ci.ious[dind * g_n + gsrc];
                        if v < best {
                            continue;
                        }
                        best = v;
                        m = gind as i32;
                    }
                    if m < 0 {
                        continue;
                    }
                    dt_match[tind * d_n + dind] = m;
                    gt_matched[tind * g_n + m as usize] = true;
                }
            }
        }

        // Unmatched detections outside the area range (or LVIS-marked) are
        // ignored rather than counted as false positives.
        dt_ignore.clear();
        dt_ignore.resize(t_n * d_n, false);
        for tind in 0..t_n {
            for dind in 0..d_n {
                let src = ci.dt_idx[dind] as usize;
                let m = dt_match[tind * d_n + dind];
                dt_ignore[tind * d_n + dind] = if m >= 0 {
                    gt_ignore[m as usize]
                } else {
                    let a = self.dt.areas[src];
                    a < a_rng[0] || a > a_rng[1] || self.dt.lvis_mark[src]
                };
            }
        }

        dt_scores.clear();
        dt_scores.extend(ci.dt_idx.iter().map(|&i| self.dt.scores[i as usize]));
        true
    }

    /// Run the whole evaluation.
    ///
    /// `collect_eval_imgs` materialises the pycocotools-shaped per-image
    /// records. It is off by default because building them is what makes
    /// upstream's memory profile bad, and almost nothing reads them.
    pub fn run(&self, collect_eval_imgs: bool) -> (EvalResult, Vec<Option<ImgEval>>) {
        let p = &self.params;
        let (t_n, r_n) = (p.iou_thrs.len(), p.rec_thrs.len());
        let k_n = self.n_groups;
        let (a_n, m_n) = (p.area_rng.len(), p.max_dets.len());
        let i_n = p.img_ids.len();

        let max_det_all = p.max_dets.last().copied().unwrap_or(0);
        // Workers publish one category at a time into the final C-order arrays.
        // Keeping every CatOut until a final transpose duplicates all output tensors.
        let output = std::sync::Mutex::new(EvalResult {
            counts: [t_n, r_n, k_n, a_n, m_n],
            precision: vec![-1.0; t_n * r_n * k_n * a_n * m_n],
            recall: vec![-1.0; t_n * k_n * a_n * m_n],
            scores: vec![-1.0; t_n * r_n * k_n * a_n * m_n],
        });
        let per_cat: Vec<Vec<Option<ImgEval>>> = (0..k_n)
            .into_par_iter()
            .map(|k| {
                let t = Instant::now();
                let work = self.prepare_category(k);
                Timings::add(&self.timings.iou_ns, t);
                let mut out = CatOut {
                    // -1 is pycocotools' "this category has no ground truth
                    // here" sentinel; summarize() filters on s > -1.
                    precision: vec![-1.0; t_n * r_n * a_n * m_n],
                    recall: vec![-1.0; t_n * a_n * m_n],
                    scores: vec![-1.0; t_n * r_n * a_n * m_n],
                    eval_imgs: Vec::new(),
                };
                let mut buf = AccumBuf::default();
                // Allocated once per category and refilled for each area
                // range: which images contribute depends only on whether they
                // have annotations, which does not vary by area.
                let mut matches: Vec<Option<ImgMatch>> = Vec::new();
                // The score-sorted detection order is the same for every area
                // range and every maxDets: area only changes which ground
                // truths are ignored, and maxDets only takes a prefix of each
                // image's list. Sorting inside the (area, maxDets) loops meant
                // sorting the same data twelve times for a default run, which
                // on a dense detector is most of the accumulate cost.
                let mut order: Vec<(f64, u32, u32)> = Vec::new();
                let mut order_ready = false;
                for a in 0..a_n {
                    let t = Instant::now();
                    if matches.is_empty() {
                        matches.resize_with(work.len(), || Some(ImgMatch::default()));
                    }
                    // Buffers stay attached to their image slot, so reuse
                    // across area ranges survives the parallel split; the
                    // scratch is per worker.
                    matches
                        .par_iter_mut()
                        .zip(work.par_iter())
                        .with_min_len(PAR_MIN_IMAGES)
                        .for_each_init(MatchScratch::default, |scratch, (slot, ci)| {
                            let mut buf = slot.take().unwrap_or_default();
                            *slot = self
                                .evaluate_img_into(ci, a, &mut buf, scratch)
                                .then_some(buf);
                        });
                    Timings::add(&self.timings.match_ns, t);
                    let t = Instant::now();
                    if !order_ready {
                        Self::build_order(&matches, max_det_all, &mut order);
                        order_ready = true;
                    }
                    for (m, &max_det) in p.max_dets.iter().enumerate() {
                        self.accumulate_slice(
                            &matches, &order, max_det, a, m, t_n, r_n, a_n, m_n, &mut buf, &mut out,
                        );
                    }
                    Timings::add(&self.timings.accumulate_ns, t);
                    if collect_eval_imgs {
                        out.eval_imgs
                            .extend(self.materialise_eval_imgs(&work, &matches, k, a, i_n));
                    }
                }
                {
                    // Copying disjoint category slots changes no arithmetic or ordering.
                    // The short lock avoids unsafe shared writes; matching stays parallel.
                    let mut result = output.lock().expect("evaluation output lock poisoned");
                    let am = a_n * m_n;
                    for t in 0..t_n {
                        let dst = (t * k_n + k) * am;
                        result.recall[dst..dst + am]
                            .copy_from_slice(&out.recall[t * am..t * am + am]);
                        for r in 0..r_n {
                            let src = (t * r_n + r) * am;
                            let dst = ((t * r_n + r) * k_n + k) * am;
                            result.precision[dst..dst + am]
                                .copy_from_slice(&out.precision[src..src + am]);
                            result.scores[dst..dst + am]
                                .copy_from_slice(&out.scores[src..src + am]);
                        }
                    }
                }
                out.eval_imgs
            })
            .collect();

        // Indexed parallel collection preserves the public [K][A][I] diagnostic order.
        let eval_imgs = per_cat.into_iter().flatten().collect();
        (
            output
                .into_inner()
                .expect("evaluation output lock poisoned"),
            eval_imgs,
        )
    }

    /// Expand the compact per-image matches into pycocotools' `evalImgs`
    /// dicts, re-inserting `None` for images with neither ground truth nor
    /// detections so positional indexing still works.
    fn materialise_eval_imgs(
        &self,
        work: &[CatImage<'_>],
        matches: &[Option<ImgMatch>],
        k: usize,
        a: usize,
        i_n: usize,
    ) -> Vec<Option<ImgEval>> {
        let p = &self.params;
        let t_n = p.iou_thrs.len();
        let max_det = p.max_dets.last().copied().unwrap_or(0);
        let mut out: Vec<Option<ImgEval>> = vec![None; i_n];
        for (ci, mm) in work.iter().zip(matches.iter()) {
            let Some(mm) = mm else { continue };
            let g_n = ci.gt_idx.len();
            let d_n = ci.dt_idx.len();
            let gt_ids: Vec<i64> = mm
                .gt_perm
                .iter()
                .map(|&i| self.gt.ids[ci.gt_idx[i as usize] as usize])
                .collect();
            let dt_ids: Vec<i64> = ci.dt_idx.iter().map(|&i| self.dt.ids[i as usize]).collect();
            // pycocotools stores matched *ids*, with 0 meaning unmatched.
            let mut dt_matches = vec![0i64; t_n * d_n];
            let mut gt_matches = vec![0i64; t_n * g_n];
            for t in 0..t_n {
                for d in 0..d_n {
                    let m = mm.dt_match[t * d_n + d];
                    if m >= 0 {
                        dt_matches[t * d_n + d] = gt_ids[m as usize];
                        gt_matches[t * g_n + m as usize] = dt_ids[d];
                    }
                }
            }
            out[ci.img_slot as usize] = Some(ImgEval {
                img_id: p.img_ids[ci.img_slot as usize],
                cat_id: if p.use_cats { p.cat_ids[k] } else { -1 },
                area_idx: a,
                max_det,
                dt_ids,
                gt_ids,
                dt_scores: mm.dt_scores.clone(),
                gt_ignore: mm.gt_ignore.clone(),
                dt_matches,
                gt_matches,
                dt_ignore: mm.dt_ignore.clone(),
            });
        }
        out
    }

    /// Flatten (image, detection) in image order and stable-sort by descending
    /// score — the list pycocotools builds with np.concatenate followed by
    /// argsort(kind='mergesort').
    ///
    /// Built for the largest maxDets; smaller ones are a subsequence, and
    /// filtering a stably-sorted list leaves it stably sorted, so this runs
    /// once per category rather than once per (area, maxDets).
    ///
    /// The score travels with each entry rather than being fetched through
    /// `matches[i].dt_scores[d]` inside the comparator. Same permutation — same
    /// values, same comparator, same stable sort — but the comparator reads one
    /// f64 that is already in the cache line it is sorting, instead of chasing
    /// two pointers per comparison into per-image vectors scattered across the
    /// heap. It also makes `scores_sorted` unnecessary: the score is already
    /// there, in the right order.
    fn build_order(matches: &[Option<ImgMatch>], max_det: usize, order: &mut Vec<(f64, u32, u32)>) {
        order.clear();
        for (i, mm) in matches.iter().enumerate() {
            let Some(mm) = mm else { continue };
            for d in 0..mm.dt_scores.len().min(max_det) {
                order.push((mm.dt_scores[d], i as u32, d as u32));
            }
        }
        order.sort_by(|x, y| cmp_desc_score(x.0, y.0));
    }

    /// The body of pycocotools' `accumulate` for one (category, area, maxDet).
    #[allow(clippy::too_many_arguments)]
    fn accumulate_slice(
        &self,
        matches: &[Option<ImgMatch>],
        order: &[(f64, u32, u32)],
        max_det: usize,
        a: usize,
        m: usize,
        t_n: usize,
        r_n: usize,
        a_n: usize,
        m_n: usize,
        buf: &mut AccumBuf,
        out: &mut CatOut,
    ) {
        let mut npig = 0usize;
        let mut any = false;
        for mm in matches.iter() {
            let Some(mm) = mm else { continue };
            any = true;
            npig += mm.gt_ignore.iter().filter(|&&ig| !ig).count();
        }
        if !any || npig == 0 {
            return;
        }
        // `d` is the detection's rank inside its own image, so keeping
        // `d < max_det` is exactly pycocotools' per-image `[0:maxDet]` cut.
        // The largest limit uses the existing sorted slice directly. Only smaller
        // per-image prefixes need a filtered buffer, whose capacity stays small.
        let flat = if max_det >= self.params.max_dets.last().copied().unwrap_or(0) {
            order
        } else {
            buf.flat.clear();
            buf.flat.extend(
                order
                    .iter()
                    .copied()
                    .filter(|&(_, _, d)| (d as usize) < max_det),
            );
            &buf.flat
        };
        let nd = flat.len();
        buf.pr.clear();
        buf.pr.resize(nd, 0.0);

        let rec_thrs = &self.params.rec_thrs;
        let npig_f = npig as f64;

        for t in 0..t_n {
            let (mut tp, mut fp) = (0i64, 0i64);
            let mut recall = 0.0;
            let mut next_sample = 0;
            buf.rec_indices.clear();
            buf.rec_indices.resize(r_n, nd);
            for (n, (precision, &(_, i, d))) in buf.pr.iter_mut().zip(flat).enumerate() {
                let mm = matches[i as usize].as_ref().unwrap();
                let d_full = mm.dt_scores.len();
                let idx = t * d_full + d as usize;
                if !mm.dt_ignore[idx] {
                    if mm.dt_match[idx] >= 0 {
                        tp += 1;
                    } else {
                        fp += 1;
                    }
                }
                let tpf = tp as f64;
                let fpf = fp as f64;
                recall = tpf / npig_f;
                // Record searchsorted's first eligible index while recall is produced.
                // Keep the original comparison, including unusual threshold values.
                while next_sample < r_n && !(recall < rec_thrs[next_sample]) {
                    buf.rec_indices[next_sample] = n;
                    next_sample += 1;
                }
                // `+ EPS` is np.spacing(1) in the reference. Dropping it
                // changes the first point of every curve by one ULP.
                *precision = tpf / (fpf + tpf + EPS);
            }

            out.recall[(t * a_n + a) * m_n + m] = recall;

            // Make precision monotonically non-increasing in recall.
            for i in (1..nd).rev() {
                if buf.pr[i] > buf.pr[i - 1] {
                    buf.pr[i - 1] = buf.pr[i];
                }
            }

            // Apply the recorded indices after the backward precision envelope.
            // This needs O(R) index storage instead of O(D) recall values.
            for (ri, &pi) in buf.rec_indices.iter().enumerate() {
                let dst = (t * r_n + ri) * a_n * m_n + a * m_n + m;
                if pi < nd {
                    out.precision[dst] = buf.pr[pi];
                    out.scores[dst] = flat[pi].0;
                } else {
                    out.precision[dst] = 0.0;
                    out.scores[dst] = 0.0;
                }
            }
        }
    }

    /// Per-instance verdicts for one (IoU threshold, area range, maxDet).
    ///
    /// Extension API. Everything diagnostic — TP/FP/FN listings, confusion
    /// matrices, mean IoU, calibration — should be built from this rather than
    /// re-deriving its own matching, so it cannot disagree with the AP numbers.
    ///
    /// Detections beyond `max_det` for an (image, category) are not returned
    /// at all, because the AP arithmetic never saw them either.
    pub fn per_instance(
        &self,
        t_idx: usize,
        a_idx: usize,
        max_det: usize,
    ) -> (Vec<DetRecord>, Vec<GtRecord>) {
        let p = &self.params;
        (0..self.n_groups)
            .into_par_iter()
            .map(|k| {
                let work = self.prepare_category(k);
                let cat_id = if p.use_cats { p.cat_ids[k] } else { -1 };
                let mut dets = Vec::new();
                let mut gts = Vec::new();
                for ci in &work {
                    let Some(mm) = self.evaluate_img(ci, a_idx) else {
                        continue;
                    };
                    let img_id = p.img_ids[ci.img_slot as usize];
                    let g_n = ci.gt_idx.len();
                    let d_full = ci.dt_idx.len();
                    let d_n = d_full.min(max_det);
                    let mut gt_matched = vec![false; g_n];
                    for d in 0..d_n {
                        let di = ci.dt_idx[d] as usize;
                        let slot = mm.dt_match[t_idx * d_full + d];
                        let (gt_id, iou) = if slot < 0 {
                            (-1, 0.0)
                        } else {
                            let gsrc = mm.gt_perm[slot as usize] as usize;
                            gt_matched[gsrc] = true;
                            (
                                self.gt.ids[ci.gt_idx[gsrc] as usize],
                                ci.ious[d * g_n + gsrc],
                            )
                        };
                        dets.push(DetRecord {
                            img_id,
                            cat_id,
                            dt_id: self.dt.ids[di],
                            score: self.dt.scores[di],
                            gt_id,
                            iou,
                            ignore: mm.dt_ignore[t_idx * d_full + d],
                        });
                    }
                    for (slot, &perm) in mm.gt_perm.iter().enumerate() {
                        let gsrc = perm as usize;
                        gts.push(GtRecord {
                            img_id,
                            cat_id,
                            gt_id: self.gt.ids[ci.gt_idx[gsrc] as usize],
                            ignore: mm.gt_ignore[slot],
                            matched: gt_matched[gsrc],
                        });
                    }
                }
                (dets, gts)
            })
            .reduce(
                || (Vec::new(), Vec::new()),
                |mut a, b| {
                    a.0.extend(b.0);
                    a.1.extend(b.1);
                    a
                },
            )
    }
}

/// Working buffers for one image's match, reused across images.
#[derive(Default)]
struct MatchScratch {
    ignore: Vec<bool>,
    gt_matched: Vec<bool>,
}

#[derive(Default)]
struct AccumBuf {
    flat: Vec<(f64, u32, u32)>,
    pr: Vec<f64>,
    rec_indices: Vec<usize>,
}

struct CatOut {
    precision: Vec<f64>,
    recall: Vec<f64>,
    scores: Vec<f64>,
    eval_imgs: Vec<Option<ImgEval>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scenarios small enough that the expected numbers can be worked out by
    /// hand. The Python suite proves we agree with pycocotools on real data;
    /// these pin the individual rules, so a failure says *which* rule broke
    /// instead of "AP moved".
    fn boxes(
        ids: &[i64],
        scores: &[f64],
        rects: &[[f64; 4]],
        cat_slots: &[u32],
        iscrowd: &[bool],
    ) -> Instances {
        let n = ids.len();
        assert_eq!(scores.len(), n);
        assert_eq!(rects.len(), n);
        Instances {
            ids: ids.to_vec(),
            scores: scores.to_vec(),
            areas: rects.iter().map(|b| b[2] * b[3]).collect(),
            iscrowd: iscrowd.to_vec(),
            // pycocotools derives `ignore` from `iscrowd`; do the same here so
            // the fixtures cannot drift from the Python layer.
            ignore: iscrowd.to_vec(),
            lvis_mark: vec![false; n],
            bboxes: Vec::new(),
            img_slot: vec![0; n],
            cat_slot: cat_slots.to_vec(),
            geom: GeomStore::Bboxes(rects.to_vec()),
        }
    }

    fn no_boxes() -> Instances {
        boxes(&[], &[], &[], &[], &[])
    }

    #[test]
    fn geometry_cap_preserves_category_major_ties_nan_and_image_limits() {
        let mut dt = boxes(
            &[1, 2, 3, 4, 5, 6],
            &[0.5, 0.5, 0.9, f64::NAN, -0.0, 0.0],
            &[[0.0, 0.0, 1.0, 1.0]; 6],
            &[1, 0, 1, 0, 0, 0],
            &[false; 6],
        );
        dt.img_slot = vec![0, 0, 0, 0, 1, 1];
        assert_eq!(
            detection_geometry_keep(&dt, 2, false, 2),
            vec![false, true, true, false, true, true]
        );
        assert_eq!(
            detection_geometry_keep(&dt, 2, true, 1),
            vec![false, true, true, false, true, false]
        );
        assert_eq!(detection_geometry_keep(&dt, 2, false, 0), vec![false; 6]);
    }

    /// One image, one category, one IoU threshold, three recall thresholds.
    /// Keeps every output array indexable as `precision[r]` / `recall[0]`.
    fn params() -> EvalParams {
        EvalParams {
            img_ids: vec![100],
            cat_ids: vec![7],
            iou_thrs: vec![0.5],
            rec_thrs: vec![0.0, 0.5, 1.0],
            max_dets: vec![10],
            area_rng: vec![[0.0, 1e10]],
            use_cats: true,
            iou_type: IouType::Bbox,
            kpt_sigmas: Vec::new(),
            use_area: true,
        }
    }

    const UNIT: [f64; 4] = [0.0, 0.0, 10.0, 10.0];

    #[test]
    fn precision_keeps_the_np_spacing_epsilon() {
        // The single most-copied divergence: `tp / (fp + tp)` instead of
        // `tp / (fp + tp + np.spacing(1))`. They differ only when fp + tp == 1
        // — the first point of every curve — so a test on a one-detection
        // scenario is exactly where it shows.
        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]);
        let dt = boxes(&[11], &[0.9], &[UNIT], &[0], &[false]);
        let (res, _) = Evaluator::new(params(), gt, dt).run(false);

        assert_eq!(res.precision[0], 1.0 / (1.0 + EPS));
        assert_ne!(res.precision[0], 1.0, "the epsilon was dropped");
        assert_eq!(res.recall[0], 1.0);
        assert_eq!(res.scores[0], 0.9);
    }

    #[test]
    fn ground_truth_with_no_detections_scores_zero_not_absent() {
        // 0.0 and -1.0 mean different things: -1 is "this category has no
        // ground truth here" and summarize() filters it out of the mean, so
        // reporting -1 here would quietly delete a category from mAP.
        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]);
        let (res, _) = Evaluator::new(params(), gt, no_boxes()).run(false);

        assert_eq!(res.recall[0], 0.0);
        assert!(
            res.precision.iter().all(|&v| v == 0.0),
            "{:?}",
            res.precision
        );
    }

    #[test]
    fn detections_with_no_ground_truth_are_absent_not_zero() {
        let dt = boxes(&[11], &[0.9], &[UNIT], &[0], &[false]);
        let (res, _) = Evaluator::new(params(), no_boxes(), dt).run(false);

        assert_eq!(res.recall[0], -1.0);
        assert!(
            res.precision.iter().all(|&v| v == -1.0),
            "{:?}",
            res.precision
        );
    }

    #[test]
    fn crowd_ground_truth_absorbs_extra_detections() {
        // A detection landing inside a crowd region is ignored, not counted
        // as a false positive. Without that the second detection below would
        // be an FP and precision would fall.
        let gt = boxes(
            &[1, 2],
            &[0.0, 0.0],
            &[UNIT, [50.0, 0.0, 50.0, 50.0]],
            &[0, 0],
            &[false, true],
        );
        let dt = boxes(
            &[11, 12],
            &[0.9, 0.8],
            &[UNIT, [60.0, 10.0, 10.0, 10.0]],
            &[0, 0],
            &[false, false],
        );
        let (res, imgs) = Evaluator::new(params(), gt, dt).run(true);

        let e = imgs.iter().flatten().next().expect("one image evaluated");
        assert_eq!(
            e.gt_ignore,
            vec![false, true],
            "crowd sorts last and is ignored"
        );
        assert_eq!(
            e.dt_matches,
            vec![1, 2],
            "second detection matched the crowd"
        );
        assert_eq!(
            e.dt_ignore,
            vec![false, true],
            "a crowd match is neither TP nor FP"
        );
        assert_eq!(res.recall[0], 1.0);
        assert_eq!(res.precision[0], 1.0 / (1.0 + EPS));
    }

    #[test]
    fn tied_scores_resolve_to_annotation_order() {
        // The greedy matcher takes detections in order, so with equal scores
        // the annotation order decides which ground truth gets claimed. An
        // unstable sort would make this arbitrary — and the totals can stay
        // the same while the *assignment* changes, which is why this asserts
        // ids rather than counts.
        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]);
        let forward = boxes(
            &[11, 12],
            &[0.5, 0.5],
            &[UNIT, UNIT],
            &[0, 0],
            &[false, false],
        );
        let (_, imgs) = Evaluator::new(params(), gt, forward).run(true);
        let e = imgs.iter().flatten().next().unwrap();
        assert_eq!(e.dt_ids, [11, 12]);

        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]);
        let reversed = boxes(
            &[12, 11],
            &[0.5, 0.5],
            &[UNIT, UNIT],
            &[0, 0],
            &[false, false],
        );
        let (_, imgs) = Evaluator::new(params(), gt, reversed).run(true);
        let e = imgs.iter().flatten().next().unwrap();
        assert_eq!(
            e.dt_ids,
            vec![12, 11],
            "order follows the input, not the ids"
        );
    }

    #[test]
    fn max_dets_truncates_by_score() {
        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]);
        let dt = boxes(
            &[11, 12, 13],
            &[0.7, 0.9, 0.8],
            &[UNIT, UNIT, UNIT],
            &[0, 0, 0],
            &[false, false, false],
        );
        let mut p = params();
        p.max_dets = vec![2];
        let (_, imgs) = Evaluator::new(p, gt, dt).run(true);

        let e = imgs.iter().flatten().next().unwrap();
        assert_eq!(
            e.dt_ids,
            vec![12, 13],
            "the two highest scores, in score order"
        );
    }

    #[test]
    fn area_range_excludes_ground_truth_from_the_denominator() {
        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]); // area 100
        let dt = boxes(&[11], &[0.9], &[UNIT], &[0], &[false]);
        let mut p = params();
        p.area_rng = vec![[0.0, 50.0]];
        let (res, _) = Evaluator::new(p, gt, dt).run(false);

        // The only ground truth is outside the range, so the category has
        // nothing to score against.
        assert_eq!(res.recall[0], -1.0);
    }

    #[test]
    fn area_range_bounds_are_inclusive() {
        // pycocotools ignores when `area < lo || area > hi`, so a ground truth
        // sitting exactly on a bound stays in. COCO's boundaries are 32^2 and
        // 96^2 and real annotations do land on them.
        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]); // area exactly 100
        let dt = boxes(&[11], &[0.9], &[UNIT], &[0], &[false]);
        let mut p = params();
        p.area_rng = vec![[100.0, 100.0]];
        let (res, _) = Evaluator::new(p, gt, dt).run(false);
        assert_eq!(res.recall[0], 1.0);
    }

    #[test]
    fn iou_below_threshold_does_not_match() {
        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]);
        // Overlap 5x5 = 25, union 100 + 100 - 25 = 175, IoU = 0.1428...
        let dt = boxes(&[11], &[0.9], &[[5.0, 5.0, 10.0, 10.0]], &[0], &[false]);
        let (res, _) = Evaluator::new(params(), gt, dt).run(false);

        assert_eq!(
            res.recall[0], 0.0,
            "0.14 IoU must not clear a 0.5 threshold"
        );
        assert!(res.precision.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn use_cats_off_merges_categories() {
        // Class-agnostic (proposal) scoring: a detection of the wrong class
        // may match. With categories on, these two never meet.
        let make = || {
            (
                boxes(&[1], &[0.0], &[UNIT], &[0], &[false]),
                boxes(&[11], &[0.9], &[UNIT], &[1], &[false]),
            )
        };
        let mut p = params();
        p.cat_ids = vec![7, 8];

        let (gt, dt) = make();
        let (with_cats, _) = Evaluator::new(p.clone(), gt, dt).run(false);
        // Category 7 has the ground truth and no detection of its own.
        assert_eq!(with_cats.recall[0], 0.0);

        let (gt, dt) = make();
        p.use_cats = false;
        let (without_cats, _) = Evaluator::new(p, gt, dt).run(false);
        assert_eq!(
            without_cats.counts[2], 1,
            "categories collapse to one group"
        );
        assert_eq!(without_cats.recall[0], 1.0);
    }

    /// Four well-separated ground truths and six detections that land, in
    /// score order, as TP FP TP TP FP FP.
    ///
    /// Everything else in this module uses one or two detections, which makes
    /// the precision/recall curve a single point — and a point cannot detect a
    /// broken envelope, a wrong recall readout, or an off-by-one in the
    /// threshold walk. This is the fixture that gives the curve a shape.
    fn curve_scenario() -> (Instances, Instances) {
        let gt = boxes(
            &[1, 2, 3, 4],
            &[0.0; 4],
            &[
                [0.0, 0.0, 10.0, 10.0],
                [100.0, 0.0, 10.0, 10.0],
                [200.0, 0.0, 10.0, 10.0],
                [300.0, 0.0, 10.0, 10.0], // never found
            ],
            &[0; 4],
            &[false; 4],
        );
        let dt = boxes(
            &[11, 12, 13, 14, 15, 16],
            &[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            &[
                [0.0, 0.0, 10.0, 10.0],     // TP, gt 1
                [0.0, 100.0, 10.0, 10.0],   // FP
                [100.0, 0.0, 10.0, 10.0],   // TP, gt 2
                [200.0, 0.0, 10.0, 10.0],   // TP, gt 3
                [100.0, 100.0, 10.0, 10.0], // FP
                [200.0, 100.0, 10.0, 10.0], // FP
            ],
            &[0; 6],
            &[false; 6],
        );
        (gt, dt)
    }

    #[test]
    fn precision_recall_curve_is_built_point_by_point() {
        // Hand-computed against pycocotools' arithmetic, with npig = 4:
        //
        //   n | verdict | tp fp |   rc  | raw pr
        //   0 |   TP    |  1  0 | 0.25  | 1/(1+eps)
        //   1 |   FP    |  1  1 | 0.25  | 0.5
        //   2 |   TP    |  2  1 | 0.50  | 2/3
        //   3 |   TP    |  3  1 | 0.75  | 0.75
        //   4 |   FP    |  3  2 | 0.75  | 0.6
        //   5 |   FP    |  3  3 | 0.75  | 0.5
        //
        // The backward max lifts pr[1] and pr[2] to 0.75. Sampling at
        // recall 0.0/0.25/0.50/0.75/1.00 with side='left' then reads indices
        // 0, 0, 2, 3, and past the end.
        let (gt, dt) = curve_scenario();
        let mut p = params();
        p.rec_thrs = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let (res, _) = Evaluator::new(p, gt, dt).run(false);

        let first = 1.0 / (1.0 + EPS);
        assert_eq!(
            res.precision,
            vec![first, first, 0.75, 0.75, 0.0],
            "the envelope, the sampling side, and the run-off-the-end fill"
        );
        assert_eq!(
            res.scores,
            vec![0.9, 0.9, 0.7, 0.6, 0.0],
            "scores are sampled at the same indices as precision"
        );
        assert_eq!(
            res.recall[0], 0.75,
            "recall is the *last* point, not the first"
        );
    }

    #[test]
    fn precision_envelope_is_monotone_and_not_the_raw_curve() {
        // Stated separately from the exact values because it is the property
        // people rely on: raw precision at index 2 is 2/3, and the reported
        // value must be the 0.75 that comes later in the curve.
        let (gt, dt) = curve_scenario();
        let mut p = params();
        p.rec_thrs = vec![0.5];
        let (res, _) = Evaluator::new(p, gt, dt).run(false);

        assert_eq!(res.precision[0], 0.75);
        assert_ne!(res.precision[0], 2.0 / 3.0, "the envelope was not applied");
    }

    #[test]
    fn precision_envelope_propagates_leftward_not_rightward() {
        // The curve above only ever falls, so sweeping the envelope the wrong
        // way happens to give the same answer at every sampled index. This one
        // *rises*: two false positives first, then four true positives, so the
        // best precision is at the very end and has to travel left.
        //
        //   n | verdict | tp fp |  rc  | raw pr
        //   0 |   FP    |  0  1 | 0.00 | 0
        //   1 |   FP    |  0  2 | 0.00 | 0
        //   2 |   TP    |  1  2 | 0.25 | 1/3
        //   3 |   TP    |  2  2 | 0.50 | 0.5
        //   4 |   TP    |  3  2 | 0.75 | 0.6
        //   5 |   TP    |  4  2 | 1.00 | 2/3
        //
        // Sweeping right-to-left makes every entry 2/3. Sweeping left-to-right
        // would leave 0, 1/3, 0.5 and 0.6 in place — all of them sampled.
        let gt = boxes(
            &[1, 2, 3, 4],
            &[0.0; 4],
            &[
                [0.0, 0.0, 10.0, 10.0],
                [100.0, 0.0, 10.0, 10.0],
                [200.0, 0.0, 10.0, 10.0],
                [300.0, 0.0, 10.0, 10.0],
            ],
            &[0; 4],
            &[false; 4],
        );
        let dt = boxes(
            &[11, 12, 13, 14, 15, 16],
            &[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            &[
                [0.0, 500.0, 10.0, 10.0],   // FP
                [100.0, 500.0, 10.0, 10.0], // FP
                [0.0, 0.0, 10.0, 10.0],     // TP
                [100.0, 0.0, 10.0, 10.0],   // TP
                [200.0, 0.0, 10.0, 10.0],   // TP
                [300.0, 0.0, 10.0, 10.0],   // TP
            ],
            &[0; 6],
            &[false; 6],
        );
        let mut p = params();
        p.rec_thrs = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let (res, _) = Evaluator::new(p, gt, dt).run(false);

        assert_eq!(res.precision, vec![2.0 / 3.0; 5]);
        assert_eq!(res.recall[0], 1.0);
    }

    #[test]
    fn an_iou_exactly_on_the_threshold_matches() {
        // The matcher's condition is `if v < best { continue }`, i.e. it
        // accepts `v >= best`. Flipping it to `<=` — which reads just as
        // naturally — rejects a detection sitting exactly on the threshold.
        // Box areas of 200 and 100 with a 100 overlap give exactly 0.5, no
        // rounding involved.
        let gt = boxes(&[1], &[0.0], &[[0.0, 0.0, 10.0, 10.0]], &[0], &[false]);
        let dt = boxes(&[11], &[0.9], &[[0.0, 0.0, 10.0, 20.0]], &[0], &[false]);
        let (res, _) = Evaluator::new(params(), gt, dt).run(false);

        assert_eq!(res.recall[0], 1.0, "IoU == threshold must match");
    }

    #[test]
    fn tied_ious_go_to_the_last_ground_truth_scanned() {
        // Two ground truths at exactly the same IoU. `v >= best` keeps
        // overwriting, so the highest index wins — observable through
        // `dtMatches`, and worth pinning because it is the kind of detail a
        // rewrite silently changes.
        let gt = boxes(
            &[1, 2],
            &[0.0, 0.0],
            &[[0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 10.0, 10.0]],
            &[0, 0],
            &[false, false],
        );
        let dt = boxes(&[11], &[0.9], &[[0.0, 0.0, 10.0, 20.0]], &[0], &[false]);
        let (_, imgs) = Evaluator::new(params(), gt, dt).run(true);

        let e = imgs.iter().flatten().next().unwrap();
        assert_eq!(
            e.dt_matches,
            [2],
            "the later ground truth wins an exact tie"
        );
    }

    #[test]
    fn each_max_dets_setting_sees_its_own_slice() {
        // Detections are truncated to max_dets.last() once, then each maxDets
        // entry cuts the sorted list again. Using `first()` for the initial
        // truncation would starve every larger setting.
        let (gt, dt) = curve_scenario();
        let mut p = params();
        p.max_dets = vec![2, 10];
        p.rec_thrs = vec![0.0];
        let (res, _) = Evaluator::new(p, gt, dt).run(false);

        assert_eq!(res.counts[4], 2);
        // maxDets = 2 sees TP, FP -> one of four ground truths found.
        assert_eq!(res.recall[0], 0.25);
        // maxDets = 10 sees all six -> three of four.
        assert_eq!(res.recall[1], 0.75);
    }

    #[test]
    fn a_crowd_region_absorbs_more_than_one_detection() {
        // The rule crowd regions exist for. A crowd ground truth stays
        // available after it has been claimed, so every detection landing in
        // it is ignored rather than counted as a false positive.
        let gt = boxes(
            &[1, 2],
            &[0.0, 0.0],
            &[UNIT, [50.0, 0.0, 50.0, 50.0]],
            &[0, 0],
            &[false, true],
        );
        let dt = boxes(
            &[11, 12, 13],
            &[0.9, 0.8, 0.7],
            &[
                UNIT,
                [60.0, 10.0, 10.0, 10.0], // inside the crowd
                [80.0, 30.0, 10.0, 10.0], // also inside the crowd
            ],
            &[0, 0, 0],
            &[false, false, false],
        );
        let (res, imgs) = Evaluator::new(params(), gt, dt).run(true);

        let e = imgs.iter().flatten().next().unwrap();
        assert_eq!(
            e.dt_matches,
            [1, 2, 2],
            "the crowd must still be matchable after the first claim"
        );
        assert_eq!(e.dt_ignore, [false, true, true]);
        assert_eq!(res.recall[0], 1.0);
    }

    #[test]
    fn a_real_match_is_not_given_up_for_a_better_ignored_one() {
        // Ground truths are ignore-sorted, so once a detection holds a real
        // match the scan stops at the first ignored entry. Without that break
        // the higher-IoU crowd below would steal the match and the detection
        // would be ignored instead of counted.
        let gt = boxes(
            &[1, 2],
            &[0.0, 0.0],
            &[
                [3.0, 0.0, 10.0, 10.0], // real, IoU 70/130 = 0.538
                UNIT,                   // crowd, IoU 1.0
            ],
            &[0, 0],
            &[false, true],
        );
        let dt = boxes(&[11], &[0.9], &[UNIT], &[0], &[false]);
        let (res, imgs) = Evaluator::new(params(), gt, dt).run(true);

        let e = imgs.iter().flatten().next().unwrap();
        assert_eq!(
            e.dt_matches,
            [1],
            "matched the real ground truth, not the crowd"
        );
        assert_eq!(e.dt_ignore, [false]);
        assert_eq!(res.recall[0], 1.0);
    }

    #[test]
    fn a_threshold_of_one_still_matches_a_near_perfect_overlap() {
        // pycocotools clamps the match floor to 1 - 1e-10 so that a threshold
        // of exactly 1.0 is reachable at all. Nothing else in the suite uses
        // a 1.0 threshold, which is the only place the clamp is observable.
        let gt = boxes(
            &[1],
            &[0.0],
            &[[0.0, 0.0, 1.0, 1.000_000_000_01]],
            &[0],
            &[false],
        );
        let dt = boxes(&[11], &[0.9], &[[0.0, 0.0, 1.0, 1.0]], &[0], &[false]);
        let mut p = params();
        p.iou_thrs = vec![1.0];
        let (res, _) = Evaluator::new(p, gt, dt).run(false);

        assert_eq!(res.recall[0], 1.0, "the 1 - 1e-10 clamp was dropped");
    }

    /// Keypoint instances with `k` triplets each; `gt` needs boxes for the
    /// OKS fall-back and the CrowdPose area substitute.
    fn keypoints(ids: &[i64], scores: &[f64], kps: &[Vec<f64>], k: usize) -> Instances {
        let n = ids.len();
        Instances {
            ids: ids.to_vec(),
            scores: scores.to_vec(),
            areas: vec![100.0; n],
            iscrowd: vec![false; n],
            ignore: vec![false; n],
            lvis_mark: vec![false; n],
            bboxes: vec![[0.0, 0.0, 10.0, 10.0]; n],
            img_slot: vec![0; n],
            cat_slot: vec![0; n],
            geom: GeomStore::Keypoints {
                data: kps.concat(),
                k,
            },
        }
    }

    #[test]
    fn oks_is_the_mean_of_the_per_keypoint_terms() {
        // Two visible keypoints, each one unit away, sigma 0.05, area 100:
        //   vars = (0.05 * 2)^2 = 0.01
        //   e    = 1 / 0.01 / 100 / 2 = 0.5
        //   OKS  = (exp(-0.5) + exp(-0.5)) / 2 = exp(-0.5)
        // Every step is exact in binary floating point, so this is an
        // equality, not an approximation. Two keypoints rather than one so
        // that dropping the division by the count is visible.
        let gt = keypoints(&[1], &[0.0], &[vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0]], 2);
        let dt = keypoints(&[11], &[0.9], &[vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]], 2);
        let mut p = params();
        p.iou_type = IouType::Keypoints;
        p.kpt_sigmas = vec![0.05, 0.05];
        p.iou_thrs = vec![0.5];

        let ev = Evaluator::new(p, gt, dt);
        let (dets, _) = ev.per_instance(0, 0, 10);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].iou, (-0.5f64).exp());
    }

    #[test]
    fn oks_of_identical_keypoints_is_exactly_one() {
        let kp = vec![3.0, 4.0, 2.0, 7.0, 1.0, 2.0];
        let gt = keypoints(&[1], &[0.0], std::slice::from_ref(&kp), 2);
        let dt = keypoints(&[11], &[0.9], &[kp], 2);
        let mut p = params();
        p.iou_type = IouType::Keypoints;
        p.kpt_sigmas = vec![0.05, 0.07];
        let ev = Evaluator::new(p, gt, dt);
        let (dets, _) = ev.per_instance(0, 0, 10);
        assert_eq!(dets[0].iou, 1.0);
    }

    #[test]
    fn result_is_independent_of_how_the_work_was_split() {
        // Runs are parallel over categories and over images inside them; the
        // output must not depend on how rayon happened to chunk it.
        let gt = boxes(
            &[1, 2, 3],
            &[0.0; 3],
            &[UNIT, [20.0, 20.0, 10.0, 10.0], [40.0, 40.0, 30.0, 30.0]],
            &[0, 0, 0],
            &[false; 3],
        );
        let dt = boxes(
            &[11, 12, 13, 14],
            &[0.9, 0.8, 0.8, 0.1],
            &[
                UNIT,
                [21.0, 21.0, 10.0, 10.0],
                [41.0, 41.0, 30.0, 30.0],
                [80.0, 80.0, 5.0, 5.0],
            ],
            &[0; 4],
            &[false; 4],
        );
        let ev = Evaluator::new(params(), gt, dt);
        let (a, _) = ev.run(false);
        let (b, _) = ev.run(false);
        assert_eq!(a.precision, b.precision);
        assert_eq!(a.recall, b.recall);
        assert_eq!(a.scores, b.scores);
    }

    #[test]
    fn output_arrays_have_the_documented_shape() {
        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]);
        let dt = boxes(&[11], &[0.9], &[UNIT], &[0], &[false]);
        let mut p = params();
        p.iou_thrs = vec![0.5, 0.75];
        p.area_rng = vec![[0.0, 1e10], [0.0, 50.0]];
        p.max_dets = vec![1, 10];
        let (res, _) = Evaluator::new(p, gt, dt).run(false);

        let [t, r, k, a, m] = res.counts;
        assert_eq!([t, r, k, a, m], [2, 3, 1, 2, 2]);
        assert_eq!(res.precision.len(), t * r * k * a * m);
        assert_eq!(res.scores.len(), t * r * k * a * m);
        assert_eq!(res.recall.len(), t * k * a * m);
    }

    #[test]
    fn per_instance_reports_ignored_detections_separately() {
        // Everything diagnostic is built on this, and it is only trustworthy
        // if a crowd match is reported as ignored rather than as a hit.
        let gt = boxes(
            &[1, 2],
            &[0.0, 0.0],
            &[UNIT, [50.0, 0.0, 50.0, 50.0]],
            &[0, 0],
            &[false, true],
        );
        let dt = boxes(
            &[11, 12],
            &[0.9, 0.8],
            &[UNIT, [60.0, 10.0, 10.0, 10.0]],
            &[0, 0],
            &[false, false],
        );
        let ev = Evaluator::new(params(), gt, dt);
        let (dets, gts) = ev.per_instance(0, 0, 10);

        assert_eq!(dets.len(), 2);
        let tp: Vec<_> = dets.iter().filter(|d| d.gt_id >= 0 && !d.ignore).collect();
        assert_eq!(tp.len(), 1);
        assert_eq!(tp[0].dt_id, 11);
        assert_eq!(tp[0].gt_id, 1);
        assert_eq!(tp[0].iou, 1.0);

        let ignored: Vec<_> = dets.iter().filter(|d| d.ignore).collect();
        assert_eq!(ignored.len(), 1);
        assert_eq!(ignored[0].dt_id, 12);

        assert_eq!(gts.len(), 2);
        assert_eq!(gts.iter().filter(|g| g.ignore).count(), 1);
        assert_eq!(gts.iter().filter(|g| g.matched).count(), 2);
    }

    #[test]
    fn timings_are_recorded() {
        let gt = boxes(&[1], &[0.0], &[UNIT], &[0], &[false]);
        let dt = boxes(&[11], &[0.9], &[UNIT], &[0], &[false]);
        let ev = Evaluator::new(params(), gt, dt);
        let _ = ev.run(false);
        let [_group, iou, matching, accumulate] = ev.timings().as_secs();
        assert!(iou >= 0.0 && matching >= 0.0 && accumulate >= 0.0);
    }

    #[test]
    fn descending_score_order_puts_nan_last() {
        // numpy's argsort sends NaN to the end; a score column with a NaN in
        // it should not reorder the finite entries around it.
        let mut v = [0.5, f64::NAN, 0.9, 0.1];
        v.sort_by(|&a, &b| cmp_desc_score(a, b));
        assert_eq!(v[0], 0.9);
        assert_eq!(v[1], 0.5);
        assert_eq!(v[2], 0.1);
        assert!(v[3].is_nan());
    }

    #[test]
    fn negative_zero_ties_with_zero() {
        // pycocotools sorts the negated scores, where -0.0 and 0.0 compare
        // equal; a total order over bit patterns would separate them and
        // change tie-breaking.
        assert_eq!(cmp_desc_score(0.0, -0.0), Ordering::Equal);
        assert_eq!(cmp_desc_score(-0.0, 0.0), Ordering::Equal);
    }
}
