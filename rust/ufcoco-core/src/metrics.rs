//! Precision, recall and F1 from per-detection outcomes.
//!
//! The input is what [`Evaluator::outcomes`](crate::eval::Evaluator::outcomes)
//! reports: one outcome byte per detection and IoU threshold, the same bytes
//! accumulation reads, plus the non-ignored ground truth per category. Counting
//! them, rather than matching again, keeps these numbers consistent with AP.
//!
//! A score threshold applies *after* matching, which equals matching only the
//! detections that pass it: matching visits an (image, category)'s detections
//! in stable descending score order and each match depends only on earlier
//! detections; `score >= s` keeps a prefix of that order (ties at `s` are all
//! kept and adjacent, NaN sorts last and fails), and a prefix matches the same
//! without the rest. `maxDets` truncation is a prefix too.
//!
//! The arithmetic is fixed so that every caller gets the same bits:
//! `tp / (tp + fp)` and `tp / (tp + fn)` divide integer counts converted once,
//! `f1 = (2 * p * r) / (p + r)`, zero denominators give 0, and the macro mean
//! sums categories one at a time in ascending category id, then divides.

use crate::eval::{desc_key, FALSE_POSITIVE, IGNORED, TRUE_POSITIVE};
use rayon::prelude::*;

/// Which detections count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cut {
    /// Every detection, NaN scores included.
    All,
    /// `score >= threshold`.
    AtLeast(f64),
    /// Per IoU threshold, the score cut that maximizes F1 (`micro`, or macro).
    BestF1 { micro: bool },
}

/// Per-category, micro and macro values. `T x R` arrays are threshold-major.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    /// `R` reported category ids, ascending.
    pub category_ids: Vec<i64>,
    /// `T`; `-inf` for [`Cut::All`], `+inf` when a sweep has nothing to keep.
    pub score_thr: Vec<f64>,
    pub tp: Vec<i64>,
    pub fp: Vec<i64>,
    pub fn_: Vec<i64>,
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub f1: Vec<f64>,
    /// `T` each.
    pub micro_tp: Vec<i64>,
    pub micro_fp: Vec<i64>,
    pub micro_fn: Vec<i64>,
    pub micro_precision: Vec<f64>,
    pub micro_recall: Vec<f64>,
    pub micro_f1: Vec<f64>,
    pub macro_precision: Vec<f64>,
    pub macro_recall: Vec<f64>,
    pub macro_f1: Vec<f64>,
}

#[inline]
fn divide(num: f64, den: f64) -> f64 {
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Precision, recall and F1 of one set of counts.
#[inline]
pub fn ratios(tp: i64, fp: i64, fn_: i64) -> (f64, f64, f64) {
    let precision = divide(tp as f64, (tp + fp) as f64);
    let recall = divide(tp as f64, (tp + fn_) as f64);
    let f1 = divide(2.0 * precision * recall, precision + recall);
    (precision, recall, f1)
}

/// F1 as `2 tp / (2 tp + fp + fn)`: one rounding of exact integers, so within
/// eps/2 of the real F1, while [`ratios`] rounds five times and lands within
/// 3 eps of it. Only the macro sweep's running sum uses it; the gap is part of
/// [`sweep_tolerance`].
#[inline]
fn quick_f1(tp: i64, fp: i64, fn_: i64) -> f64 {
    divide((2 * tp) as f64, (2 * tp + fp + fn_) as f64)
}

/// The distinct ids of `a` and `b`, ascending, and each entry's position
/// among them. Category ids are small integers in practice, so a table over
/// their range replaces sorting one id per entry.
pub fn category_slots(a: &[i64], b: &[i64]) -> (Vec<i64>, Vec<u32>, Vec<u32>) {
    let all = || a.iter().chain(b);
    let (lo, hi) = all().fold((i64::MAX, i64::MIN), |(lo, hi), &c| (lo.min(c), hi.max(c)));
    // The table is worth it only when it is not much larger than the input,
    // the same rule `Grouping::build` uses for its counting sort.
    let limit = 4 * (a.len() + b.len()) as i64 + 1024;
    if lo <= hi && hi.checked_sub(lo).is_some_and(|span| span < limit) {
        let mut slot = vec![u32::MAX; (hi - lo + 1) as usize];
        for &c in all() {
            slot[(c - lo) as usize] = 0;
        }
        let mut ids = Vec::new();
        for (offset, entry) in slot.iter_mut().enumerate() {
            if *entry == 0 {
                *entry = ids.len() as u32;
                ids.push(lo + offset as i64);
            }
        }
        let map = |v: &[i64]| v.iter().map(|&c| slot[(c - lo) as usize]).collect();
        return (ids, map(a), map(b));
    }
    let mut ids: Vec<i64> = all().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let map = |v: &[i64]| {
        v.iter()
            .map(|c| ids.binary_search(c).expect("collected above") as u32)
            .collect()
    };
    let (sa, sb) = (map(a), map(b));
    (ids, sa, sb)
}

/// Left-to-right mean, 0 when empty.
fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut n = 0usize;
    for v in values {
        total += v;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        total / n as f64
    }
}

/// Summarize detections given as category ids, scores and `D x T` outcomes,
/// with `(category id, non-ignored ground truth)` pairs.
///
/// Categories are reported when they have non-ignored ground truth or a
/// detection that is not ignored at some threshold; macro values average the
/// reported categories with ground truth. `exhaustive` evaluates every cut of
/// a macro sweep exactly instead of only the candidates (for tests).
#[allow(clippy::too_many_arguments)]
pub fn summarize(
    det_cat: &[i64],
    score: &[f64],
    outcome: &[u8],
    t_n: usize,
    gt_cat: &[i64],
    npig_in: &[i64],
    cut: Cut,
    exhaustive: bool,
) -> Summary {
    let d_n = score.len();
    assert_eq!(det_cat.len(), d_n);
    assert_eq!(outcome.len(), d_n * t_n);
    assert_eq!(gt_cat.len(), npig_in.len());

    // Every category id seen, ascending, and each detection's index into it.
    let (ids, cat, gt_slot) = category_slots(det_cat, gt_cat);
    let k_n = ids.len();
    let mut npig = vec![0i64; k_n];
    for (&k, &v) in gt_slot.iter().zip(npig_in) {
        npig[k as usize] += v;
    }
    let mut counted = vec![false; k_n];
    for d in 0..d_n {
        if outcome[d * t_n..(d + 1) * t_n]
            .iter()
            .any(|&o| o != IGNORED)
        {
            counted[cat[d] as usize] = true;
        }
    }
    let reported: Vec<usize> = (0..k_n).filter(|&k| npig[k] > 0 || counted[k]).collect();

    let score_thr: Vec<f64> = match cut {
        Cut::All => vec![f64::NEG_INFINITY; t_n],
        Cut::AtLeast(v) => vec![v; t_n],
        Cut::BestF1 { micro } => {
            let sequential = crate::eval::run_on_caller(score.len());
            best_thresholds(
                &cat, score, outcome, t_n, &npig, micro, exhaustive, sequential,
            )
        }
    };

    let r_n = reported.len();
    let mut out = Summary {
        category_ids: reported.iter().map(|&k| ids[k]).collect(),
        score_thr,
        ..Summary::default()
    };
    // `T x K` counts, filled detection by detection so `outcome` is read in
    // order.
    let mut tp_all = vec![0i64; t_n * k_n];
    let mut fp_all = vec![0i64; t_n * k_n];
    for d in 0..d_n {
        let k = cat[d] as usize;
        for (t, &o) in outcome[d * t_n..(d + 1) * t_n].iter().enumerate() {
            // `!(s >= thr)` rather than `s < thr`: a NaN score fails a threshold.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if cut != Cut::All && !(score[d] >= out.score_thr[t]) {
                continue;
            }
            match o {
                TRUE_POSITIVE => tp_all[t * k_n + k] += 1,
                FALSE_POSITIVE => fp_all[t * k_n + k] += 1,
                _ => {}
            }
        }
    }
    for t in 0..t_n {
        let tp = &tp_all[t * k_n..(t + 1) * k_n];
        let fp = &fp_all[t * k_n..(t + 1) * k_n];
        let (mut mtp, mut mfp, mut mfn) = (0i64, 0i64, 0i64);
        let row = out.precision.len();
        for &k in &reported {
            let f = npig[k] - tp[k];
            let (p, r, f1) = ratios(tp[k], fp[k], f);
            out.tp.push(tp[k]);
            out.fp.push(fp[k]);
            out.fn_.push(f);
            out.precision.push(p);
            out.recall.push(r);
            out.f1.push(f1);
            mtp += tp[k];
            mfp += fp[k];
            mfn += f;
        }
        let (p, r, f1) = ratios(mtp, mfp, mfn);
        out.micro_tp.push(mtp);
        out.micro_fp.push(mfp);
        out.micro_fn.push(mfn);
        out.micro_precision.push(p);
        out.micro_recall.push(r);
        out.micro_f1.push(f1);
        let with_gt = |values: &[f64]| {
            mean(
                reported
                    .iter()
                    .enumerate()
                    .filter(|&(_, &k)| npig[k] > 0)
                    .map(|(j, _)| values[row + j]),
            )
        };
        let mp = with_gt(&out.precision);
        let mr = with_gt(&out.recall);
        let mf = with_gt(&out.f1);
        out.macro_precision.push(mp);
        out.macro_recall.push(mr);
        out.macro_f1.push(mf);
    }
    debug_assert_eq!(out.tp.len(), t_n * r_n);
    out
}

/// The score cut that maximizes F1 at each IoU threshold.
///
/// Candidates are the distinct scores of detections that count (not ignored,
/// not NaN), so a cut never splits tied scores. Ties in F1 go to the highest
/// threshold; with nothing to keep the threshold is `+inf`.
#[allow(clippy::too_many_arguments)]
fn best_thresholds(
    cat: &[u32],
    score: &[f64],
    outcome: &[u8],
    t_n: usize,
    npig: &[i64],
    micro: bool,
    exhaustive: bool,
    sequential: bool,
) -> Vec<f64> {
    // Sorted once: a stable sort keeps the relative order of any subset, so
    // filtering the sorted detections equals sorting each threshold's live ones.
    let mut keyed: Vec<(u64, u32)> = score
        .iter()
        .enumerate()
        .map(|(d, &s)| (desc_key(s), d as u32))
        .collect();
    keyed.sort_by_key(|&(key, _)| key);
    // Gathered once in sorted order, so every pass below reads sequentially.
    let s_score: Vec<f64> = keyed.iter().map(|&(_, d)| score[d as usize]).collect();
    let s_cat: Vec<u32> = keyed.iter().map(|&(_, d)| cat[d as usize]).collect();
    let mut s_out: Vec<u8> = Vec::with_capacity(outcome.len());
    for &(_, d) in &keyed {
        s_out.extend_from_slice(&outcome[d as usize * t_n..(d as usize + 1) * t_n]);
    }
    drop(keyed);
    let npig_total: i64 = npig.iter().sum();
    // Thresholds are independent; each computes only its own value. Small
    // inputs stay on the calling thread, as `outcomes()` does.
    let per_threshold = |t: usize| {
        let (mut scores, mut cats, mut hits) = (Vec::new(), Vec::new(), Vec::new());
        for (i, &s) in s_score.iter().enumerate() {
            let o = s_out[i * t_n + t];
            if o != IGNORED && !s.is_nan() {
                scores.push(s);
                cats.push(s_cat[i]);
                hits.push(o == TRUE_POSITIVE);
            }
        }
        if scores.is_empty() {
            return f64::INFINITY;
        }
        let cuts: Vec<usize> = (0..scores.len())
            .filter(|&i| i + 1 == scores.len() || scores[i] != scores[i + 1])
            .collect();
        let pick = if micro {
            let mut tp = 0i64;
            let mut next = 0usize;
            let (mut at, mut top) = (0usize, f64::NEG_INFINITY);
            for (j, &c) in cuts.iter().enumerate() {
                while next <= c {
                    tp += i64::from(hits[next]);
                    next += 1;
                }
                let fp = (c + 1) as i64 - tp;
                let (_, _, f1) = ratios(tp, fp, npig_total - tp);
                if f1 > top {
                    top = f1;
                    at = j;
                }
            }
            at
        } else {
            best_macro_cut(&cats, &hits, &cuts, npig, exhaustive)
        };
        scores[cuts[pick]]
    };
    if sequential {
        (0..t_n).map(per_threshold).collect()
    } else {
        (0..t_n).into_par_iter().map(per_threshold).collect()
    }
}

/// Bound on |running macro-F1 sum - left-to-right sum|, with a factor 2 to
/// spare.
///
/// The running sum adds `n_dets` differences of [`quick_f1`] values in [0, 1];
/// each difference rounds by at most eps/2 and each addition by at most eps/2
/// times a partial sum, which stays within `n_gt + 1`: in all at most
/// `n_dets * (n_gt + 1) * eps / 2`. The differences telescope, so what is left
/// of the F1 values themselves is the current one per category, each within
/// 3.5 eps of the [`ratios`] value: `3.5 * n_gt * eps`. The left-to-right sum
/// of `n_gt` values rounds by at most `n_gt * n_gt * eps / 2`. The sum of the
/// three is below `((n_dets + n_gt) * (n_gt + 1) + 4 * n_gt) * eps / 2`; this
/// returns twice that. The caller widens the margin to `2 * tol` because both
/// the best cut and the running maximum carry the error.
pub fn sweep_tolerance(n_dets: usize, n_gt: usize) -> f64 {
    ((n_dets + n_gt) as f64 * (n_gt + 1) as f64 + 4.0 * n_gt as f64) * f64::EPSILON
}

/// Index into `cuts` of the highest cut with the largest macro F1.
///
/// Exact without evaluating every cut. A running sum, updated only by the
/// category each detection belongs to, gives every cut's macro-F1 sum within
/// `tol` of the left-to-right sum a fixed threshold computes. The best cut's
/// running sum is then at least `max - 2 * tol`, so only cuts within that
/// margin are evaluated with the fixed-threshold arithmetic, and the largest of
/// those, highest threshold first, is the answer. A cut that adds no detection
/// of a category with ground truth has the same counts as the cut before it and
/// a lower threshold, so it never wins and is skipped.
fn best_macro_cut(
    cats: &[u32],
    hits: &[bool],
    cuts: &[usize],
    npig: &[i64],
    exhaustive: bool,
) -> usize {
    // Categories with ground truth, in ascending id: the macro mean's order.
    let mut column = vec![u32::MAX; npig.len()];
    let mut n_gt = 0usize;
    for (k, &v) in npig.iter().enumerate() {
        if v > 0 {
            column[k] = n_gt as u32;
            n_gt += 1;
        }
    }
    let cols: Vec<u32> = cats.iter().map(|&k| column[k as usize]).collect();
    if n_gt == 0 || cols.iter().all(|&k| k == u32::MAX) {
        // Every cut scores 0; the highest threshold wins.
        return 0;
    }
    let npig_gt: Vec<i64> = npig.iter().copied().filter(|&v| v > 0).collect();

    let mut candidates: Vec<usize> = if exhaustive {
        (0..cuts.len()).collect()
    } else {
        let mut tp = vec![0i64; n_gt];
        let mut fp = vec![0i64; n_gt];
        let mut f1 = vec![0.0f64; n_gt];
        let mut running = Vec::with_capacity(cuts.len());
        let mut moved = Vec::with_capacity(cuts.len());
        let (mut total, mut next, mut counted, mut last) = (0.0f64, 0usize, 0usize, usize::MAX);
        for &c in cuts {
            while next <= c {
                let k = cols[next];
                if k != u32::MAX {
                    let k = k as usize;
                    if hits[next] {
                        tp[k] += 1;
                    } else {
                        fp[k] += 1;
                    }
                    let after = quick_f1(tp[k], fp[k], npig_gt[k] - tp[k]);
                    total += after - f1[k];
                    f1[k] = after;
                    counted += 1;
                }
                next += 1;
            }
            running.push(total);
            moved.push(counted != last);
            last = counted;
        }
        let top = running.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let margin = 2.0 * sweep_tolerance(cats.len(), n_gt);
        (0..cuts.len())
            .filter(|&j| moved[j] && running[j] >= top - margin)
            .collect()
    };
    candidates.sort_unstable();

    // Exact macro F1 at the candidates, walking the detections once.
    let mut tp = vec![0i64; n_gt];
    let mut fp = vec![0i64; n_gt];
    let mut next = 0usize;
    let (mut at, mut top) = (candidates[0], f64::NEG_INFINITY);
    for &j in &candidates {
        while next <= cuts[j] {
            let k = cols[next];
            if k != u32::MAX {
                if hits[next] {
                    tp[k as usize] += 1;
                } else {
                    fp[k as usize] += 1;
                }
            }
            next += 1;
        }
        let value = mean((0..n_gt).map(|k| ratios(tp[k], fp[k], npig_gt[k] - tp[k]).2));
        if value > top {
            top = value;
            at = j;
        }
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    const TP: u8 = TRUE_POSITIVE;
    const FP: u8 = FALSE_POSITIVE;
    const IG: u8 = IGNORED;

    #[test]
    fn hand_computed_values_and_zero_denominators() {
        // Category 1 found once of two, category 2 missed, category 3 only a
        // false positive.
        let s = summarize(
            &[1, 1, 3],
            &[0.9, 0.8, 0.7],
            &[TP, FP, FP],
            1,
            &[1, 2, 3],
            &[2, 1, 0],
            Cut::All,
            false,
        );
        assert_eq!(s.category_ids, vec![1, 2, 3]);
        assert_eq!(
            (s.tp, s.fp, s.fn_),
            (vec![1, 0, 0], vec![1, 0, 1], vec![1, 1, 0])
        );
        assert_eq!(s.precision, vec![0.5, 0.0, 0.0]);
        assert_eq!(s.f1, vec![0.5, 0.0, 0.0]);
        assert_eq!((s.micro_tp[0], s.micro_fp[0], s.micro_fn[0]), (1, 2, 2));
        assert_eq!(s.micro_precision[0], 1.0 / 3.0);
        assert_eq!(s.macro_f1[0], 0.25);
        assert_eq!(s.score_thr, vec![f64::NEG_INFINITY]);
    }

    #[test]
    fn nan_scores_count_only_without_a_threshold() {
        let args = (&[1i64, 1][..], &[f64::NAN, 0.5][..], &[TP, TP][..]);
        let all = summarize(args.0, args.1, args.2, 1, &[1], &[2], Cut::All, false);
        let cut = summarize(
            args.0,
            args.1,
            args.2,
            1,
            &[1],
            &[2],
            Cut::AtLeast(0.0),
            false,
        );
        assert_eq!((all.tp[0], cut.tp[0]), (2, 1));
        let best = summarize(
            args.0,
            args.1,
            args.2,
            1,
            &[1],
            &[2],
            Cut::BestF1 { micro: false },
            false,
        );
        assert_eq!(best.score_thr, vec![0.5]);
    }

    #[test]
    fn best_f1_breaks_ties_toward_the_highest_threshold() {
        // Cutting at 0.9 or at 0.5 both give F1 2/3 for the single category;
        // the second false positive at 0.4 lowers it.
        let s = summarize(
            &[1, 2, 1, 1],
            &[0.9, 0.7, 0.5, 0.4],
            &[TP, FP, TP, FP],
            1,
            &[1],
            &[3],
            Cut::BestF1 { micro: false },
            false,
        );
        assert_eq!(s.score_thr, vec![0.5]);
        let first = summarize(
            &[1, 1],
            &[0.9, 0.5],
            &[TP, FP],
            1,
            &[1],
            &[1],
            Cut::BestF1 { micro: true },
            false,
        );
        assert_eq!(first.score_thr, vec![0.9]);
    }

    #[test]
    fn macro_sweep_without_ground_truth_detections_keeps_the_top_cut() {
        let s = summarize(
            &[2, 2, 1],
            &[0.9, 0.4, 0.6],
            &[FP, FP, FP, FP, IG, TP],
            2,
            &[1, 2],
            &[1, 0],
            Cut::BestF1 { micro: false },
            false,
        );
        assert_eq!(s.score_thr, vec![0.9, 0.6]);
        assert_eq!(s.macro_f1, vec![0.0, 1.0]);
    }

    #[test]
    fn category_slots_with_dense_and_sparse_ids() {
        assert_eq!(
            category_slots(&[5, 3, 5, -2], &[9, 3]),
            (vec![-2, 3, 5, 9], vec![2, 1, 2, 0], vec![3, 1])
        );
        assert_eq!(
            category_slots(&[i64::MAX, 0], &[i64::MIN]),
            (vec![i64::MIN, 0, i64::MAX], vec![2, 1], vec![0])
        );
        assert_eq!(category_slots(&[], &[]), (vec![], vec![], vec![]));
        // Sparse ids take the sorted path.
        assert_eq!(
            category_slots(&[1 << 40, 3], &[3]),
            (vec![3, 1 << 40], vec![1, 0], vec![0])
        );
    }

    #[test]
    fn sort_key_orders_like_cmp_desc_score() {
        let values = [
            0.5,
            -0.0,
            0.0,
            f64::NAN,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e-300,
            -1e-300,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE / 2.0,
        ];
        for &a in &values {
            for &b in &values {
                assert_eq!(
                    desc_key(a).cmp(&desc_key(b)),
                    crate::eval::cmp_desc_score(a, b),
                    "{a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn sweep_does_not_depend_on_the_thread_layout() {
        let mut seed = 9u64;
        let mut next = |m: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % m
        };
        let n = 5000;
        let cat: Vec<u32> = (0..n).map(|_| next(7) as u32).collect();
        let score: Vec<f64> = (0..n).map(|_| next(200) as f64 / 200.0).collect();
        let outcome: Vec<u8> = (0..n * 4).map(|_| [IG, TP, FP][next(3) as usize]).collect();
        let npig: Vec<i64> = (0..7).map(|k| if k == 0 { 0 } else { 300 + k }).collect();
        for micro in [false, true] {
            let seq = best_thresholds(&cat, &score, &outcome, 4, &npig, micro, false, true);
            let par = best_thresholds(&cat, &score, &outcome, 4, &npig, micro, false, false);
            assert!(seq
                .iter()
                .zip(&par)
                .all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }

    #[test]
    fn candidate_search_equals_exhaustive_search() {
        // Coarse scores (many ties), ignored detections and two categories
        // without ground truth.
        let mut seed = 7u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..20 {
            let n = 3000;
            let (mut cats, mut scores, mut outcome) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..n {
                let c = (next() * 12.0) as i64;
                cats.push(c);
                scores.push((next() * 100.0).round() / 100.0);
                for _ in 0..2 {
                    let u = next();
                    outcome.push(if c < 2 || u > 0.55 {
                        FP
                    } else if u < 0.1 {
                        IG
                    } else {
                        TP
                    });
                }
            }
            let ids: Vec<i64> = (0..12).collect();
            let npig: Vec<i64> = ids
                .iter()
                .map(|&c| if c < 2 { 0 } else { 150 + c })
                .collect();
            for micro in [false, true] {
                let fast = summarize(
                    &cats,
                    &scores,
                    &outcome,
                    2,
                    &ids,
                    &npig,
                    Cut::BestF1 { micro },
                    false,
                );
                let slow = summarize(
                    &cats,
                    &scores,
                    &outcome,
                    2,
                    &ids,
                    &npig,
                    Cut::BestF1 { micro },
                    true,
                );
                assert_eq!(fast, slow);
            }
        }
    }
}
