//! `ufcoco-core` — COCO mask RLE and detection evaluation, in pure Rust.
//!
//! The contract this crate signs up to is *bit-exactness with pycocotools*,
//! not "close enough". Everything downstream of it — AP, AR, per-category
//! breakdowns — is expected to agree with `pycocotools` to the last bit on the
//! same input, and the test suite asserts that rather than a tolerance.
//!
//! That constraint drives three rules the code follows everywhere:
//!
//! 1. **No floating-point reassociation.** No fast-math flags, no
//!    `mul_add`/FMA contraction, no horizontal SIMD reductions over
//!    floats. Vectorisation is allowed only where it performs the same
//!    operations in the same order per element (or where the operation is
//!    exact, like integer arithmetic and `max`).
//! 2. **Every sort is stable.** pycocotools uses `kind='mergesort'` and the
//!    greedy matcher is order-sensitive, so ties in detection score must
//!    resolve to annotation order.
//! 3. **Threshold grids come from the caller.** They are built with
//!    `np.linspace` on the Python side; regenerating them here as
//!    `start + i * step` lands one ULP away at several points and shifts AP
//!    by ~1e-6.
//!
//! Speed and memory come from structure rather than from cutting corners on
//! the arithmetic: a sparse (image, category) index instead of a dense table,
//! a fused evaluate/accumulate loop that never materialises per-image result
//! dicts, and rayon across categories.

pub mod eval;
pub mod group;
pub mod rle;

pub use eval::{
    DetRecord, EvalParams, EvalResult, Evaluator, GeomStore, GtRecord, ImgEval, Instances, IouType,
    Timings, EPS,
};
pub use rle::Rle;
