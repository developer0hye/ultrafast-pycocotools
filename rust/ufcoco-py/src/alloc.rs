//! Allocation accounting, for the memory harness.
//!
//! Peak RSS tells you the process got big; it does not tell you *which* phase
//! made it big, and on a 24 GB run that is the only question that matters.
//! This wraps the system allocator to keep live bytes, a high-water mark, and
//! allocation counts, so `bench/profile_memory.py` can bracket a phase and
//! read off exactly what it cost on the Rust side.
//!
//! Gated behind the `alloc-stats` feature and off in shipped wheels: the
//! counters are two relaxed atomics per allocation, which is cheap but not
//! free, and a benchmark should not be able to slow down the thing it
//! measures by default. The numbers it reports (byte counts, allocation
//! counts) do not depend on the counting, so measuring with the feature on and
//! shipping with it off is sound.

#[cfg(feature = "alloc-stats")]
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);

// The wrapper itself only exists when it is installed; the counters and
// readers below stay unconditional so `alloc_stats()` compiles either way
// and reports zeros with `enabled: false`.
#[cfg(feature = "alloc-stats")]
pub struct Counting;

#[cfg(feature = "alloc-stats")]
#[inline]
fn on_alloc(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
    TOTAL_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    ALLOCS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(feature = "alloc-stats")]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            on_alloc(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        FREES.fetch_add(1, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Forwarding to System::realloc keeps the growth in place when it can,
        // which is exactly what `Vec::push` depends on; the default
        // alloc-copy-free implementation would make every growing vector look
        // like a fresh allocation and inflate the counts.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            let old = layout.size();
            if new_size >= old {
                on_alloc(new_size - old);
            } else {
                LIVE.fetch_sub(old - new_size, Ordering::Relaxed);
            }
        }
        p
    }
}

/// `(live, peak, total_allocated, allocs, frees)` in bytes / counts.
pub fn snapshot() -> (usize, usize, u64, u64, u64) {
    (
        LIVE.load(Ordering::Relaxed),
        PEAK.load(Ordering::Relaxed),
        TOTAL_BYTES.load(Ordering::Relaxed),
        ALLOCS.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed),
    )
}

/// Drop the high-water mark to the current live figure.
///
/// This is what makes per-phase attribution possible: reset, run one phase,
/// read the peak.
pub fn reset_peak() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}
