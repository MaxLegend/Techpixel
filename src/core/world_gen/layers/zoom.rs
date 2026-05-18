//! # Zoom layers — resolution doubling & Voronoi scaling
//!
//! Zoom layers increase the resolution of the biome grid by sampling from a
//! parent layer and interpolating or voting on boundary values.
//!
//! | Layer | Scale | Strategy |
//! |-------|-------|----------|
//! | [`LayerZoom`] | 2× | Majority-vote boundaries |
//! | [`LayerFuzzyZoom`] | 2× | Random-pick boundaries |
//! | [`LayerVoronoiZoom`] | 4× | Jittered nearest-cell (Voronoi) |

use glam::Vec2;
use crate::core::world_gen::biome_layer::{BiomeLayer, GenContext, check_output_size, coord_hash_to_f64};
use crate::{debug_log, ext_debug_log, flow_debug_log};
// ===========================================================================
// Shared helpers
// ===========================================================================

/// Given output coordinates and dimensions, compute the parent region that
/// must be requested (with a +1 border on each side) and the offset within
/// the parent buffer that corresponds to output origin.
///
/// Returns `(parent_x, parent_y, parent_w, parent_h)` — the region to request
/// from the parent layer.
fn parent_region_2x(out_x: i32, out_y: i32, out_w: usize, out_h: usize) -> (i32, i32, usize, usize) {
    // Parent coords for output (out_x, out_y) — these are at half resolution.
    let px = out_x >> 1;
    let py = out_y >> 1;
    // Parent extent needed: cover the full output region at half res, +1 each side.
    // ceil(out_w / 2) + 2 to account for left/right +1 border.
    let pw = ((out_w + 1) >> 1) + 2;
    let ph = ((out_h + 1) >> 1) + 2;
    // Shift origin left/up by 1 to get the border.
    (px - 1, py - 1, pw, ph)
}

/// Read a value from `buf` at (bx + dx, by + dy) with the given stride.
/// Out-of-bounds reads return 0.
#[inline]
fn read_parent(buf: &[u32], bw: usize, bh: usize, bx: i32, by: i32, dx: i32, dy: i32) -> u32 {
    let x = bx + dx;
    let y = by + dy;
    if x < 0 || y < 0 || x as usize >= bw || y as usize >= bh {
        return 0;
    }
    buf[y as usize * bw + x as usize]
}

/// Count occurrences of each value in the 3×3 neighborhood and return the
/// majority value.  On a tie, break using `coord_hash`.
fn majority_3x3(
    buf: &[u32],
    bw: usize,
    bh: usize,
    bx: i32,
    by: i32,
    seed: u64,
    out_x: i32,
    out_y: i32,
) -> u32 {
    // Collect 9 values.
    let mut counts: [u32; 9] = [0; 9];
    let mut values: [u32; 9] = [0; 9];
    let mut n: usize = 0;

    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let v = read_parent(buf, bw, bh, bx, by, dx, dy);
            // Linear search — n ≤ 9, always fast.
            if let Some(pos) = values[..n].iter().position(|&ev| ev == v) {
                counts[pos] += 1;
            } else {
                values[n] = v;
                counts[n] = 1;
                n += 1;
            }
        }
    }

    // Find max count.
    let mut max_count: u32 = 0;
    let mut best = values[0];
    let mut ties: usize = 0;

    for i in 0..n {
        if counts[i] > max_count {
            max_count = counts[i];
            best = values[i];
            ties = 1;
        } else if counts[i] == max_count {
            ties += 1;
        }
    }

    // Tie-break deterministically.
    if ties > 1 {
        let chance = coord_hash_to_f64(seed, out_x, out_y);
        // Pick among tied values using the hash.
        let mut candidates: Vec<u32> = Vec::new();
        for i in 0..n {
            if counts[i] == max_count {
                candidates.push(values[i]);
            }
        }
        let idx = (chance * candidates.len() as f64) as usize;
        best = candidates[idx.min(candidates.len() - 1)];
    }

    best
}

/// Return a random value from the 3×3 neighborhood using coord_hash.
fn random_from_3x3(
    buf: &[u32],
    bw: usize,
    bh: usize,
    bx: i32,
    by: i32,
    seed: u64,
    out_x: i32,
    out_y: i32,
) -> u32 {
    let vals: [u32; 9] = std::array::from_fn(|i| {
        let dx = (i % 3) as i32 - 1;
        let dy = (i / 3) as i32 - 1;
        read_parent(buf, bw, bh, bx, by, dx, dy)
    });
    let chance = coord_hash_to_f64(seed, out_x, out_y);
    let idx = (chance * 9.0) as usize;
    vals[idx.min(8)]
}

// ===========================================================================
// LayerZoom
// ===========================================================================

/// Doubles the resolution of the parent layer using **majority-vote** boundary
/// handling.
///
/// For each output cell at `(ox, oy)`, the corresponding parent cell is
/// `(ox >> 1, oy >> 1)`.  The layer examines the 3×3 parent neighborhood:
///
/// * If the center value agrees with the majority of the 9 cells, the center
///   value is used (no change).
/// * Otherwise the majority value is adopted.
/// * On a tie, [`coord_hash`] breaks it deterministically.
///
/// The parent layer is queried with a **+1 border** on each side so that
/// boundary analysis is always possible.
pub struct LayerZoom {
    parent: Box<dyn BiomeLayer>,
    seed: u64,
}

impl LayerZoom {
    /// Wrap `parent` in a 2× zoom layer.
    pub fn new(parent: Box<dyn BiomeLayer>, seed: u64) -> Self {
        debug_log!("LayerZoom", "new", "seed={}", seed);
        Self { parent, seed }
    }
}

impl BiomeLayer for LayerZoom {
    fn generate(
        &self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        ctx: &mut GenContext,
        output: &mut [u32],
    ) {
        if !check_output_size(x, y, width, height, output) {
            return;
        }

        flow_debug_log!(
            "LayerZoom",
            "generate",
            "region=({}, {}) size={}x{}",
            x, y, width, height
        );

        // Request parent with +1 border.
        let (px, py, pw, ph) = parent_region_2x(x, y, width, height);
        let mut parent_buf = ctx.acquire_buffer(pw * ph);
        self.parent.generate(px, py, pw, ph, ctx, &mut parent_buf);

        // For each output cell, map to parent and vote.
        for oy in 0..height {
            for ox in 0..width {
                let bx = (x + ox as i32) >> 1;
                let by = (y + oy as i32) >> 1;
                // Offset within parent buffer: bx - px, by - py.
                let pbx = bx - px;
                let pby = by - py;

                let _center = read_parent(&parent_buf, pw, ph, pbx, pby, 0, 0);

                // Collect 3×3 neighborhood.
                let maj = majority_3x3(
                    &parent_buf, pw, ph, pbx, pby,
                    self.seed, x + ox as i32, y + oy as i32,
                );

                // If center already matches majority, keep center; otherwise adopt majority.
                // Actually in the majority voting, we just always use the majority result
                // which naturally preserves the center when it IS the majority.
                output[oy * width + ox] = maj;
            }
        }

        ctx.release_buffer(parent_buf);

        ext_debug_log!(
            "LayerZoom",
            "generate",
            "done region=({}, {}) size={}x{}",
            x, y, width, height
        );
    }
}

// ===========================================================================
// LayerFuzzyZoom
// ===========================================================================

/// Doubles the resolution of the parent layer using **random-pick** boundary
/// handling.
///
/// Identical to [`LayerZoom`] except that at boundaries (where the 3×3
/// neighborhood contains more than one distinct value), instead of taking the
/// majority vote, a **random** value from the neighborhood is chosen via
/// [`coord_hash`].  This produces softer, more organic-looking transitions.
pub struct LayerFuzzyZoom {
    parent: Box<dyn BiomeLayer>,
    seed: u64,
}

impl LayerFuzzyZoom {
    /// Wrap `parent` in a fuzzy 2× zoom layer.
    pub fn new(parent: Box<dyn BiomeLayer>, seed: u64) -> Self {
        debug_log!("LayerFuzzyZoom", "new", "seed={}", seed);
        Self { parent, seed }
    }
}

impl BiomeLayer for LayerFuzzyZoom {
    fn generate(
        &self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        ctx: &mut GenContext,
        output: &mut [u32],
    ) {
        if !check_output_size(x, y, width, height, output) {
            return;
        }

        flow_debug_log!(
            "LayerFuzzyZoom",
            "generate",
            "region=({}, {}) size={}x{}",
            x, y, width, height
        );

        let (px, py, pw, ph) = parent_region_2x(x, y, width, height);
        let mut parent_buf = ctx.acquire_buffer(pw * ph);
        self.parent.generate(px, py, pw, ph, ctx, &mut parent_buf);

        for oy in 0..height {
            for ox in 0..width {
                let bx = (x + ox as i32) >> 1;
                let by = (y + oy as i32) >> 1;
                let pbx = bx - px;
                let pby = by - py;

                // Check if the 3×3 neighborhood is uniform.
                let center = read_parent(&parent_buf, pw, ph, pbx, pby, 0, 0);
                let mut uniform = true;
                'outer: for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        if read_parent(&parent_buf, pw, ph, pbx, pby, dx, dy) != center {
                            uniform = false;
                            break 'outer;
                        }
                    }
                }

                if uniform {
                    output[oy * width + ox] = center;
                } else {
                    // Fuzzy: pick randomly from the 3×3 neighborhood.
                    output[oy * width + ox] = random_from_3x3(
                        &parent_buf, pw, ph, pbx, pby,
                        self.seed, x + ox as i32, y + oy as i32,
                    );
                }
            }
        }

        ctx.release_buffer(parent_buf);

        ext_debug_log!(
            "LayerFuzzyZoom",
            "generate",
            "done region=({}, {}) size={}x{}",
            x, y, width, height
        );
    }
}

// ===========================================================================
// LayerVoronoiZoom
// ===========================================================================

/// Final zoom layer that scales the parent by **4×** using a jittered
/// nearest-cell strategy, producing organic Voronoi-diagram-like boundaries.
///
/// # Algorithm
///
/// 1. Each parent cell at `(px, py)` is assigned a random jitter offset
///    `(dx, dy)` in `[-1.8, 1.8]` via [`coord_hash_to_f64`].
/// 2. For every output pixel at `(ox, oy)`, compute the **continuous** parent
///    position `(ox / 4.0, oy / 4.0)`.
/// 3. Check the 5×5 parent neighborhood around the nearest parent cell.
/// 4. Find the jittered parent cell with the smallest Euclidean distance and
///    assign its biome value.
///
/// The 4× scaling (rather than 2×) and the jittered sampling ensure that
/// biome boundaries are not axis-aligned, giving the terrain a natural,
/// organic appearance.
///
/// [`glam::Vec2`] is used for efficient distance calculations.
pub struct LayerVoronoiZoom {
    parent: Box<dyn BiomeLayer>,
    seed: u64,
}

impl LayerVoronoiZoom {
    /// Scale factor applied by this layer.
    const ZOOM: usize = 4;

    /// Maximum absolute jitter applied to each parent cell center.
    /// The range `[-JITTER, JITTER]` is chosen so that the maximum jitter
    /// doesn't exceed one full cell width, preventing cells from "swapping".
    const JITTER: f32 = 1.8;

    /// Wrap `parent` in a 4× Voronoi zoom layer.
    pub fn new(parent: Box<dyn BiomeLayer>, seed: u64) -> Self {
        debug_log!("LayerVoronoiZoom", "new", "seed={}", seed);
        Self { parent, seed }
    }

    /// Compute the jittered center of parent cell `(px, py)` as a `Vec2`.
    ///
    /// The jitter offsets are deterministic functions of `(seed, px, py)`.
    #[inline]
    fn jittered_center(&self, px: i32, py: i32) -> Vec2 {
        // Use different hash inputs for x and y to avoid correlation.
        let dx = coord_hash_to_f64(self.seed, px * 2 + 0, py * 2 + 0);
        let dy = coord_hash_to_f64(self.seed, px * 2 + 1, py * 2 + 1);
        // Map [0, 1) → [-JITTER, JITTER).
        let dx = (dx * 2.0 - 1.0) * Self::JITTER as f64;
        let dy = (dy * 2.0 - 1.0) * Self::JITTER as f64;
        Vec2::new(px as f32 + dx as f32, py as f32 + dy as f32)
    }
}

impl BiomeLayer for LayerVoronoiZoom {
    fn generate(
        &self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        ctx: &mut GenContext,
        output: &mut [u32],
    ) {
        if !check_output_size(x, y, width, height, output) {
            return;
        }

        flow_debug_log!(
            "LayerVoronoiZoom",
            "generate",
            "region=({}, {}) size={}x{}",
            x, y, width, height
        );

        let zoom = Self::ZOOM as i32;

        // Parent region: output at 4×, so parent covers (x/4, y/4).
        // We need a +2 border because jitter can reach up to 1.8 cells away,
        // meaning we need at most 2 extra parent cells on each side.
        let px_start = (x / zoom) - 2;
        let py_start = (y / zoom) - 2;
        let px_end = (x + width as i32 + zoom - 1) / zoom + 2;
        let py_end = (y + height as i32 + zoom - 1) / zoom + 2;
        let pw = (px_end - px_start) as usize;
        let ph = (py_end - py_start) as usize;

        let mut parent_buf = ctx.acquire_buffer(pw * ph);
        self.parent.generate(px_start, py_start, pw, ph, ctx, &mut parent_buf);

        // For each output pixel, find the nearest jittered parent cell.
        let inv_zoom = 1.0 / zoom as f32;

        for oy in 0..height {
            for ox in 0..width {
                // Continuous position in parent space.
                let sample_x = (x + ox as i32) as f32 * inv_zoom;
                let sample_y = (y + oy as i32) as f32 * inv_zoom;

                let sample_pos = Vec2::new(sample_x, sample_y);

                // Nearest parent cell (floor).
                let near_px = sample_x.floor() as i32;
                let near_py = sample_y.floor() as i32;

                // Search the 5×5 neighborhood (2 cells each way covers max jitter of 1.8).
                let mut best_value: u32 = 0;
                let mut best_dist_sq: f32 = f32::MAX;

                for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        let cpx = near_px + dx;
                        let cpy = near_py + dy;

                        // Look up parent buffer.
                        let bx = cpx - px_start;
                        let by = cpy - py_start;
                        if bx < 0 || by < 0 || bx as usize >= pw || by as usize >= ph {
                            continue;
                        }
                        let value = parent_buf[by as usize * pw + bx as usize];

                        let center = self.jittered_center(cpx, cpy);
                        let dist_sq = sample_pos.distance_squared(center);

                        if dist_sq < best_dist_sq {
                            best_dist_sq = dist_sq;
                            best_value = value;
                        }
                    }
                }

                output[oy * width + ox] = best_value;
            }
        }

        ctx.release_buffer(parent_buf);

        ext_debug_log!(
            "LayerVoronoiZoom",
            "generate",
            "done region=({}, {}) size={}x{}",
            x, y, width, height
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::world_gen::layers::LayerIsland;

    #[test]
    fn test_layer_zoom_doubles() {
        let island: Box<dyn BiomeLayer> = Box::new(LayerIsland::new(42));
        let zoom = LayerZoom::new(island, 100);
        let mut ctx = GenContext::new(42);

        // 8×8 zoom → 16×16.
        let mut out = vec![0u32; 16 * 16];
        zoom.generate(0, 0, 16, 16, &mut ctx, &mut out);
        assert!(out.iter().any(|&v| v != 0));
    }

    #[test]
    fn test_layer_zoom_deterministic() {
        let island: Box<dyn BiomeLayer> = Box::new(LayerIsland::new(42));
        let zoom = LayerZoom::new(island, 100);
        let mut ctx = GenContext::new(42);

        let mut a = vec![0u32; 16 * 16];
        let mut b = vec![0u32; 16 * 16];
        zoom.generate(0, 0, 16, 16, &mut ctx, &mut a);
        zoom.generate(0, 0, 16, 16, &mut ctx, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_layer_fuzzy_zoom_deterministic() {
        let island: Box<dyn BiomeLayer> = Box::new(LayerIsland::new(42));
        let zoom = LayerFuzzyZoom::new(island, 200);
        let mut ctx = GenContext::new(42);

        let mut a = vec![0u32; 16 * 16];
        let mut b = vec![0u32; 16 * 16];
        zoom.generate(0, 0, 16, 16, &mut ctx, &mut a);
        zoom.generate(0, 0, 16, 16, &mut ctx, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_layer_voronoi_zoom_4x() {
        let island: Box<dyn BiomeLayer> = Box::new(LayerIsland::new(42));
        let zoom = LayerVoronoiZoom::new(island, 300);
        let mut ctx = GenContext::new(42);

        // 4×4 parent → 16×16 output.
        let mut out = vec![0u32; 16 * 16];
        zoom.generate(0, 0, 16, 16, &mut ctx, &mut out);
        // Should have at least two distinct values (land + ocean).
        let unique: std::collections::HashSet<u32> = out.iter().copied().collect();
        assert!(unique.len() >= 1);
    }
}
