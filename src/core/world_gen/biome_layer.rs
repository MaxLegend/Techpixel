//! # Biome Layer System — Core
//!
//! This module provides the foundational types and traits for QubePixel's
//! layer-based biome generation pipeline.
//!
//! The system works as a stack of composable [`BiomeLayer`]s, each of which
//! transforms or refines a 2-D grid of integer biome/terrain IDs.  Layers are
//! wired together in a pipeline: a root layer produces a coarse grid, and each
//! successive layer doubles the resolution or applies biome-specific logic.
//!
//! ## Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`GenContext`] | Holds the world seed and a pool of reusable `Vec<u32>` buffers |
//! | [`BiomeLayer`] | Trait implemented by every layer in the pipeline |
//! | [`GenMode`] | Distinguishes the two-pass (Pre / Post) generation modes |
//!
//! ## Determinism
//!
//! **All** randomness must go through [`coord_hash`] / [`coord_hash_to_f64`].
//! Never use `thread_rng`, `RandomState`, or any other non-deterministic source.
//! This guarantees that the same `(seed, x, y)` always produces the same output.

use crate::{debug_log, ext_debug_log};
// ---------------------------------------------------------------------------
// GenContext — buffer pooling
// ---------------------------------------------------------------------------

/// Pools reusable `Vec<u32>` buffers so that hot generation paths avoid repeated
/// heap allocations.
///
/// Each call to [`GenContext::acquire_buffer`] returns a buffer whose *contents*
/// are undefined (the buffer is reused as-is).  Callers must overwrite every
/// element before reading.
///
/// # Buffer lifecycle
///
/// ```text
///   acquire_buffer  ──►  (caller writes into buffer)  ──►  release_buffer
///                                                                      │
///                                                                      ▼
///   acquire_buffer  ◄────────────────────────────────────────────  (returned to pool)
/// ```
///
/// The pool is capped at 64 buffers; excess buffers returned via
/// [`release_buffer`](GenContext::release_buffer) are dropped.
pub struct GenContext {
    /// World seed, threaded through every layer for deterministic hashing.
    pub seed: u64,

    /// Pool of reusable scratch buffers.
    buffers: Vec<Vec<u32>>,
}

impl GenContext {
    /// Create a new context with the given world seed and an empty buffer pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use qubepixel_terrain::GenContext;
    /// let ctx = GenContext::new(42);
    /// ```
    pub fn new(seed: u64) -> Self {
        debug_log!("GenContext", "new", "seed={}", seed);
        Self {
            seed,
            buffers: Vec::with_capacity(16),
        }
    }

    /// Acquire a buffer of at least `min_size` elements.
    ///
    /// If a buffer of sufficient capacity exists in the pool it is returned
    /// directly (its contents are **undefined**).  Otherwise a new `Vec` is
    /// allocated.
    ///
    /// This method is intentionally cheap — it performs at most one `Vec`
    /// allocation in the rare case that the pool is empty or all pooled
    /// buffers are too small.
    pub fn acquire_buffer(&mut self, min_size: usize) -> Vec<u32> {
        // Try to find a buffer with enough capacity.
        if let Some(idx) = self
            .buffers
            .iter()
            .rposition(|b| b.capacity() >= min_size)
        {
            let mut buf = self.buffers.swap_remove(idx);
            unsafe {
                buf.set_len(min_size);
            }
            ext_debug_log!(
                "GenContext",
                "acquire_buffer",
                "reused buffer cap={} requested={}",
                buf.capacity(),
                min_size
            );
            return buf;
        }

        // No suitable buffer — allocate a fresh one.
        let buf = vec![0u32; min_size];
        ext_debug_log!(
            "GenContext",
            "acquire_buffer",
            "allocated new buffer size={}",
            min_size
        );
        buf
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// The pool is trimmed to a maximum of **64** buffers; any surplus is
    /// silently dropped.
    pub fn release_buffer(&mut self, mut buf: Vec<u32>) {
        const MAX_POOL_SIZE: usize = 64;
        buf.clear();
        if self.buffers.len() < MAX_POOL_SIZE {
            self.buffers.push(buf);
            ext_debug_log!(
                "GenContext",
                "release_buffer",
                "returned buffer, pool size={}",
                self.buffers.len()
            );
        } else {
            ext_debug_log!(
                "GenContext",
                "release_buffer",
                "pool full ({}), dropping buffer",
                self.buffers.len()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// BiomeLayer trait
// ---------------------------------------------------------------------------

/// A single stage in the biome generation pipeline.
///
/// Every layer is **Send + Sync** so it can be shared across rayon worker
/// threads.  Layers are composed by wrapping a parent in a `Box<dyn BiomeLayer>`.
///
/// # Contract
///
/// * `output.len()` *must* equal `width * height`.  If it does not, the
///   implementation should return early without writing (graceful handling).
/// * The implementation must be deterministic: calling `generate` twice with
///   the same `(x, y, width, height, seed)` must yield identical results.
/// * Implementations should prefer [`GenContext::acquire_buffer`] for any
///   temporary allocations rather than creating fresh `Vec`s on every call.
pub trait BiomeLayer: Send + Sync {
    /// Fill `output` with biome IDs for the region starting at world
    /// coordinates `(x, y)` and extending `width` cells right and `height`
    /// cells down.
    ///
    /// # Arguments
    ///
    /// * `x`, `y` — world-space origin of the requested chunk (in cell units
    ///   at *this* layer's resolution).
    /// * `width`, `height` — dimensions of the requested area.
    /// * `ctx` — generation context providing the seed and buffer pool.
    /// * `output` — destination slice; **must** contain exactly
    ///   `width * height` elements.
    fn generate(
        &self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        ctx: &mut GenContext,
        output: &mut [u32],
    );
}

// ---------------------------------------------------------------------------
// Hash utilities
// ---------------------------------------------------------------------------

/// Deterministic hash for a `(seed, x, y)` coordinate triple.
///
/// Uses a simple but fast bit-mixing approach inspired by wyhash /
/// splitmix64.  The result is a full `u64` suitable for use as a
/// probabilistic discriminator.
///
/// # Properties
///
/// * **Deterministic** — same inputs always yield the same output.
/// * **Well-distributed** — small changes in input cause large changes in
///   output (avalanche effect).
///
/// # Examples
///
/// ```
/// let h = coord_hash(12345, 100, -200);
/// assert_eq!(h, coord_hash(12345, 100, -200)); // idempotent
/// ```
#[inline]
pub fn coord_hash(seed: u64, x: i32, y: i32) -> u64 {
    // Mix x and y into seed using a splitmix64-style finaliser.
    let mut h = seed;
    h = h.wrapping_add(x as u64);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = h.wrapping_add(y as u64);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h = h ^ (h >> 31);
    h
}

/// Convert a deterministic coordinate hash into a float in `[0.0, 1.0)`.
///
/// Equivalent to `coord_hash(seed, x, y) as f64 / (u64::MAX as f64 + 1.0)`.
///
/// # Examples
///
/// ```
/// let f = coord_hash_to_f64(42, 0, 0);
/// assert!(f >= 0.0 && f < 1.0);
/// ```
#[inline]
pub fn coord_hash_to_f64(seed: u64, x: i32, y: i32) -> f64 {
    let h = coord_hash(seed, x, y);
    // Use the upper 53 bits for a clean f64 mantissa.
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Deterministic hash for a single `u64` value (convenience wrapper).
///
/// Useful when the "key" is already a combined integer rather than a 2-D
/// coordinate.
#[inline]
pub fn single_hash(seed: u64, value: u64) -> u64 {
    let mut h = seed.wrapping_add(value);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    h ^ (h >> 31)
}

/// Convert a [`single_hash`] result into `[0.0, 1.0)`.
#[inline]
pub fn single_hash_to_f64(seed: u64, value: u64) -> f64 {
    let h = single_hash(seed, value);
    (h >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// GenMode
// ---------------------------------------------------------------------------

/// Indicates which pass of a two-pass generation scheme is being executed.
///
/// Some layers need to run twice:
///
/// 1. **Pre** — place "marker" values into the grid that will be resolved
///    later (e.g. tentative biome placements that depend on neighboring
///    regions).
/// 2. **Post** — resolve markers into final biome IDs now that the full
///    neighborhood is known.
///
/// Layers that don't need two passes simply ignore the mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GenMode {
    /// First pass — place markers / tentative values.
    Pre,
    /// Second pass — resolve markers into final values.
    Post,
}

impl std::fmt::Display for GenMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenMode::Pre => write!(f, "Pre"),
            GenMode::Post => write!(f, "Post"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: safe output length check
// ---------------------------------------------------------------------------

/// Returns `true` if `output.len() == width * height`, logging a warning
/// otherwise.  Layers should call this at the top of `generate()` and return
/// early if it returns `false`.
#[inline]
pub fn check_output_size(x: i32, y: i32, width: usize, height: usize, output: &[u32]) -> bool {
    let expected = width * height;
    if output.len() != expected {
        debug_log!(
            "BiomeLayer",
            "check_output_size",
            "output.len()={} != width*height={} at ({}, {}) size={}x{}",
            output.len(),
            expected,
            x,
            y,
            width,
            height
        );
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_hash_deterministic() {
        let a = coord_hash(42, 10, 20);
        let b = coord_hash(42, 10, 20);
        assert_eq!(a, b);
    }

    #[test]
    fn test_coord_hash_spread() {
        let a = coord_hash(42, 0, 0);
        let b = coord_hash(42, 1, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn test_coord_hash_to_f64_range() {
        for seed in [0u64, 1, 42, u64::MAX] {
            for x in -10i32..=10 {
                for y in -10i32..=10 {
                    let f = coord_hash_to_f64(seed, x, y);
                    assert!(
                        f >= 0.0 && f < 1.0,
                        "f={} out of range for seed={} ({},{})",
                        f,
                        seed,
                        x,
                        y
                    );
                }
            }
        }
    }

    #[test]
    fn test_gen_context_buffer_pool() {
        let mut ctx = GenContext::new(123);
        let buf = ctx.acquire_buffer(100);
        assert!(buf.len() >= 100);
        ctx.release_buffer(buf);
        // Should reuse the same buffer.
        let buf2 = ctx.acquire_buffer(100);
        assert!(buf2.len() >= 100);
        ctx.release_buffer(buf2);
    }

    #[test]
    fn test_gen_context_pool_trim() {
        let mut ctx = GenContext::new(0);
        // Push 70 buffers — only 64 should be kept.
        for _ in 0..70 {
            let buf = ctx.acquire_buffer(1);
            ctx.release_buffer(buf);
        }
        assert!(ctx.buffers.len() <= 64);
    }
}
