//! Run-length-encoded binary masks.
//!
//! This is a line-by-line port of `cocoapi/common/maskApi.c`. Every routine
//! here is expected to be **bit-identical** to the C original, because the
//! numbers it produces (mask areas, IoU values) feed straight into AP and a
//! single-ULP difference can flip a greedy match and move AP in the 4th
//! decimal. Where the C code relies on a specific integer width, an
//! implementation-defined cast, or an off-by-one that looks like a bug, the
//! port reproduces it deliberately and says so in a comment.
//!
//! The one intentional deviation from the C source: `rleToString` /
//! `rleFrString` use `long`, which is 64-bit on Linux/macOS but 32-bit under
//! MSVC. We always use `i64` (the Linux behaviour), because that is what the
//! reference wheels most users compare against are built with, and because
//! 32-bit `long` silently corrupts counts above 2^31 (an image with more than
//! two billion pixels). See `to_string` for details.
//!
//! The index-based loops below mirror the C line for line on purpose. Turning
//! them into iterator chains reads better and buys nothing, while making a
//! future diff against `maskApi.c` harder to check by eye — which is the only
//! review that can catch a divergence here.
#![allow(clippy::needless_range_loop)]

/// C's `(int)` cast on a `double`, as x86-64 actually implements it.
///
/// This matters: `rleFrPoly` divides by an edge length that is zero for a
/// repeated polygon vertex, producing NaN, and then casts it. C leaves that
/// UB, but every real build lowers it to `cvttsd2si`, which yields
/// `INT_MIN` for NaN and for out-of-range values. Rust's `as i32` saturates
/// instead (NaN -> 0), which would silently disagree with pycocotools on
/// annotations that contain duplicate consecutive vertices — and those do
/// occur in the wild.
///
/// It also sits in the innermost polygon-tracing loop — roughly five calls per
/// boundary pixel — so it is one range test plus a raw convert. `x as i32`
/// would repeat that range test internally to guarantee saturation, and NaN
/// fails both comparisons here, which is exactly the case we want sent to
/// `INT_MIN`.
#[inline]
fn c_i32(x: f64) -> i32 {
    if x > -2147483649.0 && x < 2147483648.0 {
        // SAFETY: the comparison above excludes NaN and everything outside
        // i32's range after truncation, which is `to_int_unchecked`'s
        // precondition.
        unsafe { x.to_int_unchecked::<i32>() }
    } else {
        i32::MIN
    }
}

/// Hand back a run array with no capacity slack.
///
/// The counts vectors are built by pushing, so they carry up to 2x the bytes
/// they need. That is invisible on one mask and 30 MB across a COCO
/// segmentation run, where the RLEs are the bulk of live memory — and they
/// are built once and read many times, so paying one realloc to shed it is
/// the right trade.
#[inline]
fn tight(mut cnts: Vec<u32>) -> Vec<u32> {
    cnts.shrink_to_fit();
    cnts
}

/// A run-length-encoded binary mask, column-major (Fortran order) like COCO.
///
/// `cnts` alternates run lengths of 0s and 1s starting with 0s, so the mask
/// area is the sum of the odd-indexed entries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rle {
    pub h: u32,
    pub w: u32,
    pub cnts: Vec<u32>,
}

impl Rle {
    pub fn new(h: u32, w: u32, cnts: Vec<u32>) -> Self {
        Rle { h, w, cnts }
    }

    /// Number of pixels, as u64 so 2^32-pixel images do not wrap.
    #[inline]
    pub fn npix(&self) -> u64 {
        self.h as u64 * self.w as u64
    }

    /// `rleArea`: sum of the odd-indexed runs, wrapping like the C `uint`.
    pub fn area(&self) -> u32 {
        let mut a: u32 = 0;
        let mut j = 1;
        while j < self.cnts.len() {
            a = a.wrapping_add(self.cnts[j]);
            j += 2;
        }
        a
    }

    /// `rleToBbox`: tight [x, y, w, h] box around the set pixels.
    ///
    /// Reproduces the C exactly, including the truncation of `m` to an even
    /// count (a trailing run of 1s that is not closed by a run of 0s is
    /// dropped) and the `xp` bookkeeping that widens the y-extent to the full
    /// column height as soon as a run spans a column boundary.
    pub fn to_bbox(&self) -> [f64; 4] {
        let h = self.h;
        let w = self.w;
        // Only complete (start, length) pairs participate, exactly as in C.
        let m = (self.cnts.len() / 2) * 2;
        if m == 0 {
            return [0.0, 0.0, 0.0, 0.0];
        }
        let (mut xs, mut ys) = (w, h);
        let (mut xe, mut ye) = (0u32, 0u32);
        let mut cc: u32 = 0;
        let mut xp: u32 = 0;
        for j in 0..m {
            cc = cc.wrapping_add(self.cnts[j]);
            let t = cc.wrapping_sub((j % 2) as u32);
            // Column-major linear index -> (x, y). h == 0 cannot happen for a
            // real mask; guard so we do not divide by zero on malformed input.
            let (x, y) = if h == 0 { (0, 0) } else { (t / h, t % h) };
            if j % 2 == 0 {
                xp = x;
            } else if xp < x {
                // The run crossed into a new column, so it covers every row.
                ys = 0;
                ye = h.saturating_sub(1);
            }
            xs = xs.min(x);
            xe = xe.max(x);
            ys = ys.min(y);
            ye = ye.max(y);
        }
        [
            xs as f64,
            ys as f64,
            (xe.wrapping_sub(xs).wrapping_add(1)) as f64,
            (ye.wrapping_sub(ys).wrapping_add(1)) as f64,
        ]
    }

    /// `rleDecode` for a single mask; writes `h*w` bytes in column-major order.
    pub fn decode_into(&self, out: &mut [u8]) {
        let mut v: u8 = 0;
        let mut p = 0usize;
        for &c in &self.cnts {
            let c = c as usize;
            let end = (p + c).min(out.len());
            out[p..end].fill(v);
            p = end;
            v ^= 1;
        }
    }

    pub fn decode(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.npix() as usize];
        self.decode_into(&mut out);
        out
    }

    /// `rleEncode` for a single mask given in column-major order.
    ///
    /// Note the comparison is on raw byte values, not truthiness: a mask
    /// containing 2s produces transitions between 1 and 2 just like the C.
    pub fn encode(mask: &[u8], h: u32, w: u32) -> Rle {
        let a = mask.len();
        let mut cnts: Vec<u32> = Vec::with_capacity(16);
        let mut p: u8 = 0;
        let mut c: u32 = 0;
        for &t in mask.iter().take(a) {
            if t != p {
                cnts.push(c);
                c = 0;
                p = t;
            }
            c += 1;
        }
        cnts.push(c);
        Rle {
            h,
            w,
            cnts: tight(cnts),
        }
    }

    /// `rleToString`: LEB128-like, 6 bits per char, ASCII 48..111.
    ///
    /// The delta step (`x -= cnts[i-2]` for `i > 2`) is copied verbatim,
    /// including the fact that it starts at index 3 rather than 2 — that
    /// off-by-one is part of the on-disk format and every COCO annotation
    /// file in the wild is encoded with it.
    pub fn to_string(&self) -> Vec<u8> {
        let m = self.cnts.len();
        let mut s: Vec<u8> = Vec::with_capacity(m * 2);
        for i in 0..m {
            // i64 rather than C `long`: see the module docs.
            let mut x = self.cnts[i] as i64;
            if i > 2 {
                x -= self.cnts[i - 2] as i64;
            }
            let mut more = true;
            while more {
                let mut c = (x & 0x1f) as u8;
                x >>= 5; // arithmetic shift, matching signed `long` in C
                more = if c & 0x10 != 0 { x != -1 } else { x != 0 };
                if more {
                    c |= 0x20;
                }
                s.push(c + 48);
            }
        }
        s
    }

    /// `rleFrString`: inverse of [`Rle::to_string`].
    pub fn from_str(s: &[u8], h: u32, w: u32) -> Rle {
        let mut cnts: Vec<u32> = Vec::with_capacity(s.len() / 2 + 1);
        let mut p = 0usize;
        while p < s.len() {
            let mut x: i64 = 0;
            let mut k = 0u32;
            let mut more = true;
            while more && p < s.len() {
                let c = s[p].wrapping_sub(48);
                let shift = 5 * k;
                if shift < 64 {
                    x |= ((c & 0x1f) as i64) << shift;
                }
                more = (c & 0x20) != 0;
                p += 1;
                k += 1;
                if !more && (c & 0x10) != 0 {
                    // Sign-extend. C would invoke UB once the shift reaches the
                    // width of `long`; we simply stop, which keeps the value
                    // already accumulated.
                    let shift = 5 * k;
                    if shift < 64 {
                        x |= -1i64 << shift;
                    }
                }
            }
            let m = cnts.len();
            if m > 2 {
                x += cnts[m - 2] as i64;
            }
            cnts.push(x as u32);
        }
        Rle {
            h,
            w,
            cnts: tight(cnts),
        }
    }
}

/// `rleMerge`: union (`intersect == false`) or intersection of `n` masks.
///
/// Mirrors the C control flow, including the early-out for n == 0 / n == 1 and
/// the "shape mismatch collapses the result to an empty 0x0 RLE" behaviour.
pub fn merge(rles: &[Rle], intersect: bool) -> Rle {
    if rles.is_empty() {
        return Rle::default();
    }
    if rles.len() == 1 {
        return rles[0].clone();
    }
    let mut h = rles[0].h;
    let mut w = rles[0].w;
    let mut cnts: Vec<u32> = rles[0].cnts.clone();

    for b_rle in &rles[1..] {
        if b_rle.h != h || b_rle.w != w {
            // C sets h = w = m = 0 and breaks out, yielding an empty RLE.
            return Rle::default();
        }
        let a_cnts = std::mem::take(&mut cnts);
        if a_cnts.is_empty() || b_rle.cnts.is_empty() {
            // C would read cnts[0] out of bounds here; an empty operand can
            // only come from malformed input, and an empty result is the
            // only sane answer for both union and intersection.
            cnts = Vec::new();
            continue;
        }
        let (mut ca, mut cb) = (a_cnts[0], b_rle.cnts[0]);
        let (mut v, mut va, mut vb) = (false, false, false);
        let (mut a, mut b) = (1usize, 1usize);
        let mut cc: u32 = 0;
        let mut ct: u32 = 1;
        let mut out: Vec<u32> = Vec::with_capacity(a_cnts.len() + b_rle.cnts.len());

        while ct > 0 {
            let c = ca.min(cb);
            cc = cc.wrapping_add(c);
            ct = 0;
            ca -= c;
            if ca == 0 && a < a_cnts.len() {
                ca = a_cnts[a];
                a += 1;
                va = !va;
            }
            ct = ct.wrapping_add(ca);
            cb -= c;
            if cb == 0 && b < b_rle.cnts.len() {
                cb = b_rle.cnts[b];
                b += 1;
                vb = !vb;
            }
            ct = ct.wrapping_add(cb);
            let vp = v;
            v = if intersect { va && vb } else { va || vb };
            if v != vp || ct == 0 {
                out.push(cc);
                cc = 0;
            }
        }
        cnts = out;
        h = rles[0].h;
        w = rles[0].w;
    }
    Rle {
        h,
        w,
        cnts: tight(cnts),
    }
}

/// `bbIou`, writing a row-major `m x n` matrix (`out[d * n + g]`).
///
/// The C writes column-major into `o[g*m+d]` and Cython reshapes with
/// `order='F'`, so the *logical* layout is `[dt, gt]` either way; we store it
/// row-major so the numpy view is C-contiguous.
pub fn bb_iou(dt: &[[f64; 4]], gt: &[[f64; 4]], iscrowd: &[u8], out: &mut [f64]) {
    let m = dt.len();
    let n = gt.len();
    for (g, gbox) in gt.iter().enumerate() {
        let ga = gbox[2] * gbox[3];
        let crowd = iscrowd.get(g).is_some_and(|&c| c != 0);
        for (d, dbox) in dt.iter().enumerate() {
            let da = dbox[2] * dbox[3];
            let mut o = 0.0f64;
            let w = f64::min(dbox[2] + dbox[0], gbox[2] + gbox[0]) - f64::max(dbox[0], gbox[0]);
            if w > 0.0 {
                let h = f64::min(dbox[3] + dbox[1], gbox[3] + gbox[1]) - f64::max(dbox[1], gbox[1]);
                if h > 0.0 {
                    let i = w * h;
                    let u = if crowd { da } else { da + ga - i };
                    o = i / u;
                }
            }
            out[d * n + g] = o;
        }
    }
    debug_assert_eq!(out.len(), m * n);
}

/// `rleIou`, writing a row-major `m x n` matrix (`out[d * n + g]`).
///
/// Keeps the C's two-stage structure: a bbox-IoU pre-pass, then the exact
/// run-length intersection *only* where the bbox IoU is strictly positive.
/// That gate is not just an optimisation — pairs whose boxes miss are reported
/// as exactly 0.0 without ever looking at the masks, and dropping the gate
/// would change results for degenerate (zero-area box) masks.
pub fn rle_iou(dt: &[Rle], gt: &[Rle], iscrowd: &[u8], out: &mut [f64]) {
    let d: Vec<&Rle> = dt.iter().collect();
    let g: Vec<&Rle> = gt.iter().collect();
    rle_iou_refs(&d, &g, iscrowd, out)
}

/// [`rle_iou`] over borrowed masks, so callers that gather a subset out of a
/// larger store do not have to clone the run arrays first.
pub fn rle_iou_refs(dt: &[&Rle], gt: &[&Rle], iscrowd: &[u8], out: &mut [f64]) {
    let n = gt.len();
    let db: Vec<[f64; 4]> = dt.iter().map(|r| r.to_bbox()).collect();
    let gb: Vec<[f64; 4]> = gt.iter().map(|r| r.to_bbox()).collect();
    bb_iou(&db, &gb, iscrowd, out);

    for g in 0..n {
        let crowd = iscrowd.get(g).is_some_and(|&c| c != 0);
        for (d, d_rle) in dt.iter().enumerate() {
            let idx = d * n + g;
            if out[idx] <= 0.0 {
                continue;
            }
            let g_rle = &gt[g];
            if d_rle.h != g_rle.h || d_rle.w != g_rle.w {
                out[idx] = -1.0;
                continue;
            }
            if d_rle.cnts.is_empty() || g_rle.cnts.is_empty() {
                out[idx] = 0.0;
                continue;
            }
            let (mut ca, mut cb) = (d_rle.cnts[0], g_rle.cnts[0]);
            let (ka, kb) = (d_rle.cnts.len(), g_rle.cnts.len());
            let (mut va, mut vb) = (false, false);
            let (mut a, mut b) = (1usize, 1usize);
            let (mut i, mut u): (u32, u32) = (0, 0);
            let mut ct: u32 = 1;
            while ct > 0 {
                let c = ca.min(cb);
                if va || vb {
                    u = u.wrapping_add(c);
                    if va && vb {
                        i = i.wrapping_add(c);
                    }
                }
                ct = 0;
                ca -= c;
                if ca == 0 && a < ka {
                    ca = d_rle.cnts[a];
                    a += 1;
                    va = !va;
                }
                ct = ct.wrapping_add(ca);
                cb -= c;
                if cb == 0 && b < kb {
                    cb = g_rle.cnts[b];
                    b += 1;
                    vb = !vb;
                }
                ct = ct.wrapping_add(cb);
            }
            // Crowd ground truth: score against the detection's own area so a
            // detection may match any sub-region of the crowd.
            if i == 0 {
                u = 1;
            } else if crowd {
                u = d_rle.area();
            }
            out[idx] = i as f64 / u as f64;
        }
    }
}

/// `rleFrPoly`: rasterise one polygon (flat `[x0, y0, x1, y1, ...]`).
///
/// The upsample-by-5 / trace / downsample dance is reproduced verbatim; the
/// `+ .5` truncating casts are C `(int)` casts (round toward zero), which is
/// also what Rust's `as i32` does.
pub fn rle_fr_poly(xy: &[f64], h: u32, w: u32) -> Rle {
    rle_fr_poly_into(xy, h, w, &mut PolyScratch::default())
}

/// Reusable working buffers for [`rle_fr_poly_into`].
///
/// Rasterising a polygon needs seven temporary vectors. A COCO-scale
/// segmentation run does that ~70k times, and the allocator traffic was 80% of
/// the segm setup cost before these were hoisted out; hand one of these per
/// worker thread (`rayon`'s `map_init`) and the allocations disappear.
#[derive(Default)]
pub struct PolyScratch {
    x: Vec<i32>,
    y: Vec<i32>,
    u: Vec<i32>,
    v: Vec<i32>,
    px: Vec<i32>,
    py: Vec<i32>,
    a: Vec<u32>,
    parts: Vec<Rle>,
}

/// `rleFrPoly`, reusing caller-owned scratch buffers.
pub fn rle_fr_poly_into(xy: &[f64], h: u32, w: u32, s: &mut PolyScratch) -> Rle {
    let k = xy.len() / 2;
    if k == 0 {
        return Rle::new(h, w, vec![(h as u64 * w as u64) as u32]);
    }
    let scale = 5.0f64;

    // Upsample and close the polygon.
    let x = &mut s.x;
    let y = &mut s.y;
    x.clear();
    y.clear();
    for j in 0..k {
        x.push(c_i32(scale * xy[j * 2] + 0.5));
    }
    x.push(x[0]);
    for j in 0..k {
        y.push(c_i32(scale * xy[j * 2 + 1] + 0.5));
    }
    y.push(y[0]);

    // Walk each edge with a Bresenham-ish trace. `s` is deliberately left to
    // divide by zero on a degenerate edge (NaN/inf), because c_i32 then
    // reproduces the C result; see c_i32.
    let u = &mut s.u;
    let v = &mut s.v;
    u.clear();
    v.clear();
    for j in 0..k {
        let (mut xs, mut xe) = (x[j], x[j + 1]);
        let (mut ys, mut ye) = (y[j], y[j + 1]);
        let dx = xe.wrapping_sub(xs).wrapping_abs();
        let dy = ys.wrapping_sub(ye).wrapping_abs();
        let flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if flip {
            std::mem::swap(&mut xs, &mut xe);
            std::mem::swap(&mut ys, &mut ye);
        }
        if dx >= dy {
            let s = (ye - ys) as f64 / dx as f64;
            for d in 0..=dx {
                let t = if flip { dx - d } else { d };
                u.push(t.wrapping_add(xs));
                v.push(c_i32(ys as f64 + s * t as f64 + 0.5));
            }
        } else {
            let s = (xe - xs) as f64 / dy as f64;
            for d in 0..=dy {
                let t = if flip { dy - d } else { d };
                v.push(t.wrapping_add(ys));
                u.push(c_i32(xs as f64 + s * t as f64 + 0.5));
            }
        }
    }

    // Keep the points where the trace crosses a column boundary, downsampled
    // back to pixel coordinates.
    let kk = u.len();
    let px = &mut s.px;
    let py = &mut s.py;
    px.clear();
    py.clear();
    for j in 1..kk {
        if u[j] == u[j - 1] {
            continue;
        }
        let mut xd = (if u[j] < u[j - 1] {
            u[j]
        } else {
            u[j].wrapping_sub(1)
        }) as f64;
        xd = (xd + 0.5) / scale - 0.5;
        // `w - 1` on C's unsigned `siz`: for w == 0 it wraps to a huge value
        // rather than -1, which changes the comparison. Reproduce it.
        if xd.floor() != xd || xd < 0.0 || xd > (w as u64).wrapping_sub(1) as f64 {
            continue;
        }
        let mut yd = (if v[j] < v[j - 1] { v[j] } else { v[j - 1] }) as f64;
        yd = (yd + 0.5) / scale - 0.5;
        if yd < 0.0 {
            yd = 0.0;
        } else if yd > h as f64 {
            yd = h as f64;
        }
        yd = yd.ceil();
        px.push(c_i32(xd));
        py.push(c_i32(yd));
    }

    // Column-major linear indices of the crossings, plus a sentinel at h*w.
    let m = px.len();
    let a = &mut s.a;
    a.clear();
    for j in 0..m {
        // C does the multiply in `int` and then casts to `uint`; wrapping
        // keeps us identical on the (pathological) overflow path.
        a.push((px[j].wrapping_mul(h as i32).wrapping_add(py[j])) as u32);
    }
    a.push((h as u64 * w as u64) as u32);
    a.sort_unstable();

    // Delta-encode, then collapse zero-length runs.
    let mut p: u32 = 0;
    for v in a.iter_mut() {
        let t = *v;
        *v = v.wrapping_sub(p);
        p = t;
    }
    let k2 = a.len();
    let mut b: Vec<u32> = Vec::with_capacity(k2);
    let mut j = 0usize;
    b.push(a[j]);
    j += 1;
    while j < k2 {
        if a[j] > 0 {
            b.push(a[j]);
            j += 1;
        } else {
            j += 1;
            if j < k2 {
                let last = b.len() - 1;
                b[last] = b[last].wrapping_add(a[j]);
                j += 1;
            }
        }
    }
    Rle {
        h,
        w,
        cnts: tight(b),
    }
}

/// `rleFrBbox`: a box is just its four-corner polygon.
pub fn rle_fr_bbox(bb: &[f64; 4], h: u32, w: u32) -> Rle {
    let (xs, ys) = (bb[0], bb[1]);
    let (xe, ye) = (xs + bb[2], ys + bb[3]);
    let xy = [xs, ys, xs, ye, xe, ye, xe, ys];
    rle_fr_poly(&xy, h, w)
}

/// Union of the polygon parts of one annotation, the `annToRLE` polygon path.
pub fn rle_fr_polys(polys: &[Vec<f64>], h: u32, w: u32) -> Rle {
    rle_fr_polys_into(polys, h, w, &mut PolyScratch::default())
}

/// [`rle_fr_polys`] with caller-owned scratch.
pub fn rle_fr_polys_into(polys: &[Vec<f64>], h: u32, w: u32, s: &mut PolyScratch) -> Rle {
    if polys.len() == 1 {
        // The overwhelmingly common case. `merge` of a single RLE is a clone
        // upstream; skipping it saves copying every run array.
        return rle_fr_poly_into(&polys[0], h, w, s);
    }
    let mut parts = std::mem::take(&mut s.parts);
    parts.clear();
    for p in polys {
        parts.push(rle_fr_poly_into(p, h, w, s));
    }
    let out = merge(&parts, false);
    s.parts = parts;
    out
}

/// [`rle_fr_polys`] over rings stored flat, with `ends[i]` the exclusive end
/// of ring `i` in `coords`.
///
/// The flat form exists so that reading a multi-ring polygon out of Python
/// costs one allocation instead of one per ring; the rasterisation is
/// identical.
pub fn rle_fr_polys_flat_into(
    coords: &[f64],
    ends: &[u32],
    h: u32,
    w: u32,
    s: &mut PolyScratch,
) -> Rle {
    if ends.len() == 1 {
        return rle_fr_poly_into(coords, h, w, s);
    }
    let mut parts = std::mem::take(&mut s.parts);
    parts.clear();
    let mut start = 0usize;
    for &end in ends {
        let end = end as usize;
        parts.push(rle_fr_poly_into(&coords[start..end], h, w, s));
        start = end;
    }
    let out = merge(&parts, false);
    s.parts = parts;
    out
}

/// Uncompressed-RLE annotation (`{"counts": [...], "size": [h, w]}`) to [`Rle`].
pub fn rle_fr_uncompressed(counts: &[u32], h: u32, w: u32) -> Rle {
    Rle {
        h,
        w,
        cnts: counts.to_vec(),
    }
}

/// Boundary mask of an RLE: the mask minus its erosion by `dilation` pixels.
///
/// This is the Boundary-IoU construction (Cheng et al., CVPR 2021) as used by
/// faster-coco-eval, and it is an *extension* — pycocotools has no equivalent.
/// The erosion uses a 3x3 structuring element applied `dilation` times on a
/// 1-pixel zero-padded canvas, so mask pixels touching the image border count
/// as boundary.
pub fn rle_to_boundary(rle: &Rle, dilation_ratio: f64) -> Rle {
    let (h, w) = (rle.h as usize, rle.w as usize);
    if h == 0 || w == 0 {
        return rle.clone();
    }
    let img_diag = ((h * h + w * w) as f64).sqrt();
    let mut dilation = (dilation_ratio * img_diag).round() as i64;
    if dilation < 1 {
        dilation = 1;
    }
    let mask = rle.decode(); // column-major, h*w

    // Pad by one so border-touching pixels erode away.
    let (ph, pw) = (h + 2, w + 2);
    let mut cur = vec![0u8; ph * pw];
    for xx in 0..w {
        for yy in 0..h {
            cur[(xx + 1) * ph + (yy + 1)] = mask[xx * h + yy];
        }
    }
    let mut next = vec![0u8; ph * pw];
    for _ in 0..dilation {
        // 3x3 min filter; the padded ring stays 0 and keeps eroding inward.
        for xx in 1..pw - 1 {
            for yy in 1..ph - 1 {
                let mut v = 1u8;
                'k: for dx in 0..3 {
                    for dy in 0..3 {
                        if cur[(xx + dx - 1) * ph + (yy + dy - 1)] == 0 {
                            v = 0;
                            break 'k;
                        }
                    }
                }
                next[xx * ph + yy] = v;
            }
        }
        std::mem::swap(&mut cur, &mut next);
        next.iter_mut().for_each(|p| *p = 0);
    }

    let mut boundary = vec![0u8; h * w];
    for xx in 0..w {
        for yy in 0..h {
            let m = mask[xx * h + yy];
            let e = cur[(xx + 1) * ph + (yy + 1)];
            boundary[xx * h + yy] = m.saturating_sub(e);
        }
    }
    Rle::encode(&boundary, rle.h, rle.w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let h = 6u32;
        let w = 5u32;
        let mut mask = vec![0u8; (h * w) as usize];
        // Column-major: set a 2x2 block at rows 1..3, cols 2..4.
        for x in 2..4u32 {
            for y in 1..3u32 {
                mask[(x * h + y) as usize] = 1;
            }
        }
        let rle = Rle::encode(&mask, h, w);
        assert_eq!(rle.area(), 4);
        assert_eq!(rle.decode(), mask);
        let s = rle.to_string();
        assert_eq!(Rle::from_str(&s, h, w), rle);
        assert_eq!(rle.to_bbox(), [2.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn merge_union_and_intersect() {
        let h = 4u32;
        let w = 4u32;
        let mut a = vec![0u8; 16];
        let mut b = vec![0u8; 16];
        for i in 0..8 {
            a[i] = 1;
        }
        for i in 4..12 {
            b[i] = 1;
        }
        let ra = Rle::encode(&a, h, w);
        let rb = Rle::encode(&b, h, w);
        assert_eq!(merge(&[ra.clone(), rb.clone()], false).area(), 12);
        assert_eq!(merge(&[ra, rb], true).area(), 4);
    }

    #[test]
    fn bbox_iou_matches_hand_computation() {
        let dt = [[0.0, 0.0, 10.0, 10.0]];
        let gt = [[5.0, 5.0, 10.0, 10.0]];
        let mut out = vec![0.0; 1];
        bb_iou(&dt, &gt, &[0], &mut out);
        // intersection 25, union 175
        assert!((out[0] - 25.0 / 175.0).abs() < 1e-15);
    }

    #[test]
    fn crowd_iou_uses_detection_area() {
        let dt = [[0.0, 0.0, 10.0, 10.0]];
        let gt = [[0.0, 0.0, 100.0, 100.0]];
        let mut out = vec![0.0; 1];
        bb_iou(&dt, &gt, &[1], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn poly_square_area() {
        // A 10x10 axis-aligned square starting at (0, 0).
        let rle = rle_fr_poly(&[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0], 20, 20);
        assert_eq!(rle.area(), 100);
        assert_eq!(rle.to_bbox(), [0.0, 0.0, 10.0, 10.0]);
    }

    #[test]
    fn c_int_cast_follows_the_hardware_not_rust() {
        // Rust's `as i32` saturates and sends NaN to 0; the C this ports from
        // lowers to `cvttsd2si`, which sends both NaN and out-of-range to
        // INT_MIN. `rleFrPoly` reaches the NaN case on a repeated vertex, so
        // the difference is observable on real annotations.
        assert_eq!(c_i32(3.9), 3);
        assert_eq!(c_i32(-3.9), -3, "truncation is toward zero, not down");
        assert_eq!(c_i32(0.0), 0);
        assert_eq!(c_i32(f64::NAN), i32::MIN);
        assert_eq!(c_i32(f64::INFINITY), i32::MIN);
        assert_eq!(c_i32(f64::NEG_INFINITY), i32::MIN);
        assert_eq!(c_i32(2147483647.5), 2147483647);
        assert_eq!(c_i32(2147483648.0), i32::MIN);
        assert_eq!(c_i32(-2147483649.0), i32::MIN);
    }

    #[test]
    fn polygon_with_a_repeated_vertex_stays_a_valid_rle() {
        // A zero-length edge makes rleFrPoly divide by zero. The result is
        // whatever the C produces; what must hold is that we do not panic and
        // that the run lengths still tile the image.
        let poly = [5.0, 5.0, 5.0, 5.0, 5.0, 20.0, 20.0, 20.0, 20.0, 5.0];
        let r = rle_fr_poly(&poly, 32, 32);
        assert_eq!((r.h, r.w), (32, 32));
        assert_eq!(r.cnts.iter().map(|&c| c as u64).sum::<u64>(), 32 * 32);
    }

    #[test]
    fn polygon_outside_the_image_is_clipped() {
        let r = rle_fr_poly(&[-50.0, -50.0, -50.0, 5.0, 5.0, 5.0, 5.0, -50.0], 20, 20);
        assert_eq!(r.cnts.iter().map(|&c| c as u64).sum::<u64>(), 400);
        assert!(r.area() <= 400);
    }

    #[test]
    fn string_round_trip_survives_the_delta_rule() {
        // `rleToString` subtracts cnts[i-2] from index 3 onward, so most
        // encoded values are signed deltas. Anything that mishandles the sign
        // extension breaks here rather than on some rare mask.
        for cnts in [
            vec![0u32],
            vec![0, 1],
            vec![5, 3, 5, 3, 5],
            vec![1_000_000, 1, 999_999, 2],
            vec![0, 1, 0, 1, 0, 1],
            vec![u32::MAX / 4, 7, 3, 100_000],
        ] {
            let r = Rle::new(64, 64, cnts.clone());
            assert_eq!(Rle::from_str(&r.to_string(), 64, 64).cnts, cnts, "{cnts:?}");
        }
    }

    #[test]
    fn encode_compares_bytes_not_truthiness() {
        // A mask holding a 2 produces a transition between 1 and 2, exactly as
        // the C does. Treating it as "non-zero" would merge the runs.
        let r = Rle::encode(&[0u8, 0, 2, 2, 1, 1], 6, 1);
        assert_eq!(r.cnts, vec![2, 2, 2]);
    }

    #[test]
    fn area_sums_the_odd_runs() {
        assert_eq!(Rle::new(4, 4, vec![2, 3, 4, 5, 2]).area(), 8);
        assert_eq!(Rle::new(4, 4, vec![16]).area(), 0);
    }

    #[test]
    fn bbox_of_degenerate_rles() {
        // An odd count means the trailing run of ones is not closed; the C
        // drops it, and so do we.
        assert_eq!(Rle::new(10, 10, vec![]).to_bbox(), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(Rle::new(10, 10, vec![100]).to_bbox(), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn merge_of_one_is_a_copy_and_of_none_is_empty() {
        let a = Rle::new(4, 4, vec![2, 14]);
        assert_eq!(merge(std::slice::from_ref(&a), false), a);
        assert_eq!(merge(&[], false), Rle::default());
    }

    #[test]
    fn merge_of_mismatched_sizes_collapses() {
        let a = Rle::new(4, 4, vec![16]);
        let b = Rle::new(5, 5, vec![25]);
        assert_eq!(merge(&[a, b], false), Rle::default());
    }

    #[test]
    fn iou_of_mismatched_sizes_is_negative_one() {
        // The bounding boxes overlap, so the pair survives the bbox pre-pass
        // and reaches the size check — which is the branch under test.
        let a = Rle::encode(&[1u8; 100], 10, 10);
        let b = Rle::encode(&[1u8; 144], 12, 12);
        let mut out = vec![0.0f64; 1];
        rle_iou(&[a], &[b], &[0], &mut out);
        assert_eq!(out[0], -1.0);
    }

    #[test]
    fn disjoint_boxes_are_exactly_zero() {
        let dt = [[0.0, 0.0, 5.0, 5.0]];
        let gt = [[100.0, 100.0, 5.0, 5.0]];
        let mut out = vec![9.0; 1];
        bb_iou(&dt, &gt, &[0], &mut out);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn iou_matrix_is_row_major_over_detections() {
        // out[d * n + g]: callers index [detection, ground truth], and a
        // transposed matrix would silently pair the wrong objects.
        let dt = [[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 10.0, 10.0]];
        let gt = [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 10.0, 10.0]];
        let mut out = vec![0.0; 4];
        bb_iou(&dt, &gt, &[0, 0], &mut out);
        assert_eq!(out[0], 1.0, "dt0 vs gt0");
        assert_eq!(out[1], 0.0, "dt0 vs gt1");
        assert_eq!(out[2], 0.0, "dt1 vs gt0");
        assert_eq!(out[3], 0.0, "dt1 vs gt1");
    }

    #[test]
    fn boundary_is_the_rim_of_the_mask() {
        let (h, w) = (20u32, 20u32);
        let mut mask = vec![0u8; (h * w) as usize];
        for x in 5..15u32 {
            for y in 5..15u32 {
                mask[(x * h + y) as usize] = 1;
            }
        }
        let solid = Rle::encode(&mask, h, w);
        assert_eq!(solid.area(), 100);

        // dilation = round(0.02 * sqrt(20^2 + 20^2)) = 1, so a 10x10 block
        // erodes to 8x8 and the rim is 100 - 64.
        let boundary = rle_to_boundary(&solid, 0.02);
        assert_eq!(boundary.area(), 36);
    }

    #[test]
    fn run_arrays_carry_no_capacity_slack() {
        // The RLEs are the bulk of live memory on a segmentation run, and they
        // are built by pushing, so trimming is what keeps them from costing
        // twice what they need.
        let r = rle_fr_poly(&[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0], 64, 64);
        assert_eq!(r.cnts.len(), r.cnts.capacity());
        let s = Rle::from_str(&r.to_string(), 64, 64);
        assert_eq!(s.cnts.len(), s.cnts.capacity());
    }

    #[test]
    fn scratch_reuse_does_not_change_results() {
        // PolyScratch is shared across polygons for speed; a stale buffer
        // would leak one polygon's trace into the next.
        let polys: [&[f64]; 3] = [
            &[0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0],
            &[2.0, 3.0, 2.0, 19.0, 17.0, 19.0, 17.0, 3.0],
            &[1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
        ];
        let fresh: Vec<Rle> = polys.iter().map(|p| rle_fr_poly(p, 32, 32)).collect();
        let mut scratch = PolyScratch::default();
        let reused: Vec<Rle> = polys
            .iter()
            .map(|p| rle_fr_poly_into(p, 32, 32, &mut scratch))
            .collect();
        assert_eq!(fresh, reused);
    }
}
