// =============================================================================
// QubePixel — Fluid Simulation System
// =============================================================================
//
// Generalized fluid simulation supporting multiple fluid types (water, lava,
// etc.) with per-fluid flow rates, source/flowing distinction, and fluid
// interactions (e.g. water + lava → stone).
//
// Key concepts:
//   - Source blocks: level == 1.0, never deplete (infinite supply)
//   - Flowing blocks: level < 1.0, created by flow from sources/other flowing
//   - Spread distance: limits how far a fluid can flow horizontally from source
//   - Interactions: when two different fluids meet, they may produce a solid
//
// Cross-chunk flow:
//   Same snapshot-based approach as the original water system:
//   snapshot current chunk + neighbor boundary slices (immutable phase),
//   then apply all changes (mutable phase).
// =============================================================================

use std::collections::{HashMap, HashSet};
use crate::core::config;
use crate::core::gameobjects::chunk::Chunk;
use crate::core::gameobjects::block::BlockRegistry;
use crate::debug_log;

// ---------------------------------------------------------------------------
// Simulation constants
// ---------------------------------------------------------------------------

/// Minimum fluid level — below this the fluid cell is removed.
const MIN_LEVEL: f32 = 0.005;
/// Minimum level difference to trigger horizontal flow.
const FLOW_THRESHOLD: f32 = 0.01;
/// Maximum simulation sub-steps per frame.
const MAX_SUB_STEPS: usize = 8;

const NEIGHBOR_OFFSETS: [(i32, i32, i32); 6] = [
    ( 1,  0,  0),
    (-1,  0,  0),
    ( 0,  1,  0),
    ( 0, -1,  0),
    ( 0,  0,  1),
    ( 0,  0, -1),
];

// ---------------------------------------------------------------------------
// Per-fluid simulation parameters (extracted from BlockRegistry at init)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FluidInfo {
    flow_rate: f32,
    gravity_rate: f32,
    spread_distance: u8,
}

// ---------------------------------------------------------------------------
// Pending cross-chunk change
// ---------------------------------------------------------------------------

struct CrossChange {
    chunk_key: (i32, i32, i32),
    idx:       usize,
    delta:     f32,
    /// Block ID to set. 0 = don't change block type.
    /// For normal flow into air: set to the fluid block ID.
    /// For interactions: set to the product block ID.
    set_block: u8,
}

// ---------------------------------------------------------------------------
// Fluid interaction rule
// ---------------------------------------------------------------------------

struct FluidInteraction {
    /// The fluid that is flowing into the other.
    fluid_a: u8,
    /// The fluid that is being flowed into.
    fluid_b: u8,
    /// The solid block produced by the interaction.
    product: u8,
}

// ---------------------------------------------------------------------------
// FluidSimulator
// ---------------------------------------------------------------------------

/// Generalized fluid simulator supporting multiple fluid types.
///
/// Created from a [`BlockRegistry`] which provides per-fluid properties
/// (flow rate, gravity rate, spread distance) via `FluidProperties`.
pub struct FluidSimulator {
    /// All registered fluid block IDs.
    fluid_ids:   Vec<u8>,
    /// Per-fluid simulation parameters, keyed by block ID.
    fluid_info:  HashMap<u8, FluidInfo>,
    /// Interaction rules between different fluid types.
    interactions: Vec<FluidInteraction>,
    /// Chunks that need simulation next frame.
    dirty_chunks: HashSet<(i32, i32, i32)>,
}

impl FluidSimulator {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Build a fluid simulator from the block registry.
    ///
    /// Discovers all fluid blocks (those with `fluid: Some(..)` or legacy
    /// `is_water: true`) and sets up default interaction rules
    /// (water + lava → stone).
    pub fn new(registry: &BlockRegistry) -> Self {
        let fluid_blocks = registry.fluid_blocks();
        let mut fluid_ids = Vec::new();
        let mut fluid_info = HashMap::new();

        for (id, props) in &fluid_blocks {
            fluid_ids.push(*id);
            fluid_info.insert(*id, FluidInfo {
                flow_rate:       props.flow_rate,
                gravity_rate:    props.gravity_rate,
                spread_distance: props.spread_distance,
            });
        }

        // Default interactions: water + lava → stone (andesite as fallback)
        let mut interactions = Vec::new();
        let water_id = registry.id_for("water").unwrap_or(0);
        let lava_id  = registry.id_for("lava").unwrap_or(0);
        let stone_id = registry.id_for("rocks/andesite").unwrap_or(0);

        if water_id != 0 && lava_id != 0 && stone_id != 0 {
            // Water flowing into lava → stone
            interactions.push(FluidInteraction {
                fluid_a: water_id,
                fluid_b: lava_id,
                product: stone_id,
            });
            // Lava flowing into water → stone
            interactions.push(FluidInteraction {
                fluid_a: lava_id,
                fluid_b: water_id,
                product: stone_id,
            });
        }

        debug_log!(
            "FluidSimulator", "new",
            "Created fluid simulator: {} fluid types {:?}, {} interactions",
            fluid_ids.len(), fluid_ids, interactions.len()
        );

        Self {
            fluid_ids,
            fluid_info,
            interactions,
            dirty_chunks: HashSet::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Check if a block ID is a registered fluid.
    #[inline]
    pub fn is_fluid(&self, block_id: u8) -> bool {
        block_id != 0 && self.fluid_info.contains_key(&block_id)
    }

    /// Get all registered fluid IDs as a slice (for mesh building).
    pub fn fluid_ids(&self) -> &[u8] {
        &self.fluid_ids
    }

    /// Mark a chunk (and its 6 neighbors) as dirty for simulation.
    pub fn mark_dirty(&mut self, key: (i32, i32, i32)) {
        self.dirty_chunks.insert(key);
        for &(dx, dy, dz) in &NEIGHBOR_OFFSETS {
            self.dirty_chunks.insert((key.0 + dx, key.1 + dy, key.2 + dz));
        }
    }

    /// Mark only this chunk as dirty (no neighbor propagation).
    pub fn mark_dirty_local(&mut self, key: (i32, i32, i32)) {
        self.dirty_chunks.insert(key);
    }

    /// Whether any chunks need simulation.
    pub fn has_dirty(&self) -> bool {
        !self.dirty_chunks.is_empty()
    }

    // -----------------------------------------------------------------------
    // Interaction lookup
    // -----------------------------------------------------------------------

    /// Check if fluid_a flowing into fluid_b produces a solid.
    /// Returns the product block ID, or `None` if no interaction.
    fn check_interaction(&self, fluid_a: u8, fluid_b: u8) -> Option<u8> {
        for ix in &self.interactions {
            if ix.fluid_a == fluid_a && ix.fluid_b == fluid_b {
                return Some(ix.product);
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Main simulation entry point
    // -----------------------------------------------------------------------

    /// Run up to `MAX_SUB_STEPS` sub-steps of fluid simulation.
    pub fn simulate(&mut self, chunks: &mut HashMap<(i32, i32, i32), Chunk>) {
        for _step in 0..MAX_SUB_STEPS {
            if self.dirty_chunks.is_empty() { return; }

            let dirty: Vec<(i32, i32, i32)> = self.dirty_chunks.drain().collect();
            let mut any_changed = false;
            let mut all_cross: Vec<CrossChange> = Vec::new();

            // --- Compute phase (immutable reads + in-chunk writes) ---
            for key in &dirty {
                let (changed, cross) = self.simulate_chunk(key, chunks);
                if changed { any_changed = true; }
                all_cross.extend(cross);
            }

            // --- Apply cross-chunk changes ---
            for cc in all_cross {
                if let Some(nc) = chunks.get_mut(&cc.chunk_key) {
                    let old = nc.fluid_levels[cc.idx];
                    let new_val = (old + cc.delta).clamp(0.0, 1.0);
                    nc.fluid_levels[cc.idx] = new_val;
                    if cc.set_block != 0 {
                        let current = nc.blocks[cc.idx];
                        // Only overwrite air or fluid blocks, never solids
                        if current == 0 || self.is_fluid(current) {
                            nc.blocks[cc.idx] = cc.set_block;
                        }
                    }
                    nc.mesh_dirty = true;
                    any_changed = true;
                    self.dirty_chunks.insert(cc.chunk_key);
                    for &(dx, dy, dz) in &NEIGHBOR_OFFSETS {
                        let nk = (cc.chunk_key.0 + dx, cc.chunk_key.1 + dy, cc.chunk_key.2 + dz);
                        if chunks.contains_key(&nk) {
                            self.dirty_chunks.insert(nk);
                        }
                    }
                }
            }

            if !any_changed { return; }
        }
    }

    // -----------------------------------------------------------------------
    // Single-chunk simulation
    // Returns (changed, cross_chunk_changes)
    // -----------------------------------------------------------------------

    fn simulate_chunk(
        &mut self,
        key: &(i32, i32, i32),
        chunks: &mut HashMap<(i32, i32, i32), Chunk>,
    ) -> (bool, Vec<CrossChange>) {
        let sx = config::chunk_size_x();
        let sy = config::chunk_size_y();
        let sz = config::chunk_size_z();

        // --- Snapshot current chunk (immutable borrow released after block) ---
        let (blocks_snap, levels_snap, _cx, _cy, _cz) = {
            let chunk = match chunks.get(key) {
                Some(c) => c,
                None    => return (false, vec![]),
            };
            (chunk.blocks.clone(), chunk.fluid_levels.clone(),
             chunk.cx, chunk.cy, chunk.cz)
        };

        // --- Snapshot boundary slices of direct face-neighbors ---
        let below_key = (key.0, key.1 - 1, key.2);
        let above_key = (key.0, key.1 + 1, key.2);
        let xn_key    = (key.0 - 1, key.1, key.2);
        let xp_key    = (key.0 + 1, key.1, key.2);
        let zn_key    = (key.0, key.1, key.2 - 1);
        let zp_key    = (key.0, key.1, key.2 + 1);

        let below_slice = Self::snap_y_slice(chunks, &below_key, sy - 1, sx, sy, sz);
        let _above_slice = Self::snap_y_slice(chunks, &above_key, 0,      sx, sy, sz);
        let xn_slice    = Self::snap_x_slice(chunks, &xn_key,    sx - 1, sx, sy, sz);
        let xp_slice    = Self::snap_x_slice(chunks, &xp_key,    0,      sx, sy, sz);
        let zn_slice    = Self::snap_z_slice(chunks, &zn_key,    sz - 1, sx, sy, sz);
        let zp_slice    = Self::snap_z_slice(chunks, &zp_key,    0,      sx, sy, sz);

        // --- Working copies ---
        let mut new_levels  = levels_snap.clone();
        let mut new_blocks  = blocks_snap.clone();
        let mut cross_out: Vec<CrossChange> = Vec::new();
        let mut changed = false;
        // In-chunk interaction changes: (idx, product_block_id)
        let mut interaction_changes: Vec<(usize, u8)> = Vec::new();

        for x in 0..sx {
            for y in 0..sy {
                for z in 0..sz {
                    let idx = x * sy * sz + y * sz + z;
                    let block_id = new_blocks[idx];

                    // Skip non-fluid blocks
                    if !self.is_fluid(block_id) { continue; }

                    let level = new_levels[idx];
                    if level < MIN_LEVEL { continue; }

                    // Get per-fluid simulation parameters
                    let info = match self.fluid_info.get(&block_id) {
                        Some(i) => i,
                        None => continue,
                    };

                    // Source blocks (level == 1.0) never deplete
                    let is_source = levels_snap[idx] >= 1.0 - FLOW_THRESHOLD;

                    let mut remaining = level;

                    // ====================================================
                    // Priority 1: FLOW DOWN (gravity)
                    // ====================================================
                    if remaining >= MIN_LEVEL {
                        if y > 0 {
                            let bi = x * sy * sz + (y - 1) * sz + z;
                            let bb = new_blocks[bi];

                            if bb == 0 {
                                // Air below — flow down
                                let space = 1.0 - new_levels[bi];
                                let flow = remaining.min(space) * info.gravity_rate;
                                if flow > MIN_LEVEL {
                                    new_levels[bi] += flow;
                                    new_blocks[bi]  = block_id;
                                    if !is_source { new_levels[idx] -= flow; }
                                    remaining = if is_source { remaining } else { remaining - flow };
                                    changed = true;
                                }
                            } else if bb == block_id {
                                // Same fluid below — top up
                                let space = 1.0 - new_levels[bi];
                                if space > FLOW_THRESHOLD {
                                    let flow = remaining.min(space) * info.gravity_rate;
                                    new_levels[bi] += flow;
                                    if !is_source { new_levels[idx] -= flow; }
                                    remaining = if is_source { remaining } else { remaining - flow };
                                    changed = true;
                                }
                            } else if self.is_fluid(bb) && bb != block_id {
                                // Different fluid below — check interaction
                                if let Some(product) = self.check_interaction(block_id, bb) {
                                    interaction_changes.push((bi, product));
                                    changed = true;
                                }
                            }
                            // else: solid below — can't flow
                        } else if let Some(ref sl) = below_slice {
                            // Cross-chunk downward flow
                            let si = x * sz + z;
                            let (bb, bl) = sl[si];
                            if bb == 0 {
                                let space = 1.0 - bl;
                                let flow = remaining.min(space) * info.gravity_rate;
                                if flow > MIN_LEVEL {
                                    if !is_source { new_levels[idx] -= flow; }
                                    remaining = if is_source { remaining } else { remaining - flow };
                                    changed = true;
                                    cross_out.push(CrossChange {
                                        chunk_key: below_key,
                                        idx: x * sy * sz + (sy - 1) * sz + z,
                                        delta: flow,
                                        set_block: block_id,
                                    });
                                }
                            } else if bb == block_id {
                                let space = 1.0 - bl;
                                if space > FLOW_THRESHOLD {
                                    let flow = remaining.min(space) * info.gravity_rate;
                                    if !is_source { new_levels[idx] -= flow; }
                                    remaining = if is_source { remaining } else { remaining - flow };
                                    changed = true;
                                    cross_out.push(CrossChange {
                                        chunk_key: below_key,
                                        idx: x * sy * sz + (sy - 1) * sz + z,
                                        delta: flow,
                                        set_block: 0,
                                    });
                                }
                            } else if self.is_fluid(bb) && bb != block_id {
                                if let Some(product) = self.check_interaction(block_id, bb) {
                                    cross_out.push(CrossChange {
                                        chunk_key: below_key,
                                        idx: x * sy * sz + (sy - 1) * sz + z,
                                        delta: -bl,
                                        set_block: product,
                                    });
                                    changed = true;
                                }
                            }
                        }
                    }

                    if remaining < MIN_LEVEL && !is_source {
                        new_levels[idx] = 0.0;
                        continue;
                    }

                    // Check that the block directly below is full / solid
                    // (horizontal flow only when can't flow down)
                    let blocked_below = {
                        if y > 0 {
                            let bi = x * sy * sz + (y - 1) * sz + z;
                            let bb = new_blocks[bi];
                            (bb != 0 && !self.is_fluid(bb))
                                || (bb == block_id && new_levels[bi] >= 1.0 - FLOW_THRESHOLD)
                        } else {
                            match &below_slice {
                                None => true,
                                Some(sl) => {
                                    let (bb, bl) = sl[x * sz + z];
                                    (bb != 0 && !self.is_fluid(bb))
                                        || (bb == block_id && bl >= 1.0 - FLOW_THRESHOLD)
                                }
                            }
                        }
                    };

                    if !blocked_below { continue; }

                    // ====================================================
                    // Priority 2: FLOW HORIZONTALLY (equalization)
                    // ====================================================

                    struct HNeighbor {
                        is_cross:       bool,
                        in_chunk_idx:   usize,
                        cross_chunk:    (i32, i32, i32),
                        cross_idx:      usize,
                        block:          u8,
                        level:          f32,
                    }

                    let mut hneighbors: Vec<HNeighbor> = Vec::with_capacity(4);

                    // +X
                    if x + 1 < sx {
                        let ni = (x + 1) * sy * sz + y * sz + z;
                        hneighbors.push(HNeighbor {
                            is_cross: false, in_chunk_idx: ni,
                            cross_chunk: xp_key, cross_idx: 0,
                            block: new_blocks[ni], level: new_levels[ni],
                        });
                    } else if let Some(ref sl) = xp_slice {
                        let si = y * sz + z;
                        let (nb, nl) = sl[si];
                        hneighbors.push(HNeighbor {
                            is_cross: true, in_chunk_idx: 0,
                            cross_chunk: xp_key,
                            cross_idx: 0 * sy * sz + y * sz + z,
                            block: nb, level: nl,
                        });
                    }
                    // -X
                    if x > 0 {
                        let ni = (x - 1) * sy * sz + y * sz + z;
                        hneighbors.push(HNeighbor {
                            is_cross: false, in_chunk_idx: ni,
                            cross_chunk: xn_key, cross_idx: 0,
                            block: new_blocks[ni], level: new_levels[ni],
                        });
                    } else if let Some(ref sl) = xn_slice {
                        let si = y * sz + z;
                        let (nb, nl) = sl[si];
                        hneighbors.push(HNeighbor {
                            is_cross: true, in_chunk_idx: 0,
                            cross_chunk: xn_key,
                            cross_idx: (sx - 1) * sy * sz + y * sz + z,
                            block: nb, level: nl,
                        });
                    }
                    // +Z
                    if z + 1 < sz {
                        let ni = x * sy * sz + y * sz + (z + 1);
                        hneighbors.push(HNeighbor {
                            is_cross: false, in_chunk_idx: ni,
                            cross_chunk: zp_key, cross_idx: 0,
                            block: new_blocks[ni], level: new_levels[ni],
                        });
                    } else if let Some(ref sl) = zp_slice {
                        let si = x * sy + y;
                        let (nb, nl) = sl[si];
                        hneighbors.push(HNeighbor {
                            is_cross: true, in_chunk_idx: 0,
                            cross_chunk: zp_key,
                            cross_idx: x * sy * sz + y * sz + 0,
                            block: nb, level: nl,
                        });
                    }
                    // -Z
                    if z > 0 {
                        let ni = x * sy * sz + y * sz + (z - 1);
                        hneighbors.push(HNeighbor {
                            is_cross: false, in_chunk_idx: ni,
                            cross_chunk: zn_key, cross_idx: 0,
                            block: new_blocks[ni], level: new_levels[ni],
                        });
                    } else if let Some(ref sl) = zn_slice {
                        let si = x * sy + y;
                        let (nb, nl) = sl[si];
                        hneighbors.push(HNeighbor {
                            is_cross: true, in_chunk_idx: 0,
                            cross_chunk: zn_key,
                            cross_idx: x * sy * sz + y * sz + (sz - 1),
                            block: nb, level: nl,
                        });
                    }

                    for hn in &hneighbors {
                        if remaining < MIN_LEVEL { break; }

                        let n_block = hn.block;
                        let n_level = hn.level;

                        // --- Fluid interaction check ---
                        if n_block != 0 && n_block != block_id && self.is_fluid(n_block) {
                            if let Some(product) = self.check_interaction(block_id, n_block) {
                                if hn.is_cross {
                                    cross_out.push(CrossChange {
                                        chunk_key: hn.cross_chunk,
                                        idx: hn.cross_idx,
                                        delta: -n_level,
                                        set_block: product,
                                    });
                                } else {
                                    interaction_changes.push((hn.in_chunk_idx, product));
                                }
                                changed = true;
                                continue; // Don't flow into it, just interact
                            }
                            // Different fluid, no interaction rule — treat as solid
                            continue;
                        }

                        // --- Normal flow computation ---
                        let flow = if n_block == 0 {
                            // Air — spread into it
                            let diff = remaining - n_level;
                            if diff > FLOW_THRESHOLD {
                                let raw_flow = diff * info.flow_rate;
                                // Apply spread distance cap: the level can't drop below
                                // (remaining - 1.0/spread_distance), which limits horizontal reach
                                let min_level_after = (remaining - 1.0 / info.spread_distance.max(1) as f32).max(0.0);
                                let max_flow = (remaining - min_level_after).max(0.0);
                                raw_flow.min(max_flow).min(remaining)
                            } else { 0.0 }
                        } else if n_block == block_id {
                            // Same fluid — equalize
                            let diff = remaining - n_level;
                            if diff > FLOW_THRESHOLD {
                                (diff * info.flow_rate * 0.5).min(remaining)
                            } else { 0.0 }
                        } else {
                            0.0 // Solid block
                        };

                        if flow <= FLOW_THRESHOLD { continue; }

                        if !is_source { new_levels[idx] -= flow; }
                        remaining = if is_source { remaining } else { remaining - flow };
                        changed = true;

                        if hn.is_cross {
                            cross_out.push(CrossChange {
                                chunk_key: hn.cross_chunk,
                                idx: hn.cross_idx,
                                delta: flow,
                                set_block: if n_block == 0 { block_id } else { 0 },
                            });
                        } else {
                            new_levels[hn.in_chunk_idx] += flow;
                            if n_block == 0 {
                                new_blocks[hn.in_chunk_idx] = block_id;
                            }
                        }
                    }

                    // Restore source level (sources never deplete)
                    if is_source {
                        new_levels[idx] = 1.0;
                    } else if remaining < MIN_LEVEL {
                        new_levels[idx] = 0.0;
                    }
                }
            }
        }

        // --- Apply in-chunk interaction changes ---
        for (idx, product) in &interaction_changes {
            if new_blocks[*idx] != *product {
                new_blocks[*idx] = *product;
                new_levels[*idx] = 0.0;
                changed = true;
            }
        }

        // --- Apply changes to own chunk ---
        if changed {
            let chunk = chunks.get_mut(key).unwrap();

            for i in 0..new_levels.len() {
                let nl = new_levels[i];
                let nb = new_blocks[i];
                let ol = chunk.fluid_levels[i];
                let ob = chunk.blocks[i];

                if (nl - ol).abs() > 1e-7 {
                    chunk.fluid_levels[i] = nl;
                    chunk.mesh_dirty = true;
                }
                if nb != ob {
                    chunk.blocks[i] = nb;
                    chunk.mesh_dirty = true;
                }
            }

            // Remove dead fluid cells
            for i in 0..chunk.fluid_levels.len() {
                if chunk.fluid_levels[i] < MIN_LEVEL && self.is_fluid(chunk.blocks[i]) {
                    chunk.blocks[i] = 0;
                    chunk.fluid_levels[i] = 0.0;
                    chunk.mesh_dirty = true;
                }
            }

            // Re-mark if still active
            let still_active = chunk.fluid_levels.iter()
                .zip(chunk.blocks.iter())
                .any(|(&l, &b)| self.is_fluid(b) && l > MIN_LEVEL && l < 1.0 - FLOW_THRESHOLD);
            if still_active {
                self.dirty_chunks.insert(*key);
            }
        }

        // Mark neighbors dirty for cross-chunk spillover
        if !cross_out.is_empty() {
            for &(dx, dy, dz) in &NEIGHBOR_OFFSETS {
                let nk = (key.0 + dx, key.1 + dy, key.2 + dz);
                if chunks.contains_key(&nk) {
                    self.dirty_chunks.insert(nk);
                }
            }
        }

        (changed, cross_out)
    }

    // -----------------------------------------------------------------------
    // Slice snapshot helpers
    // -----------------------------------------------------------------------

    /// Snapshot one horizontal slice (constant y) of a chunk.
    /// Returns Vec<(block, level)> indexed by [x * sz + z].
    fn snap_y_slice(
        chunks: &HashMap<(i32, i32, i32), Chunk>,
        key: &(i32, i32, i32),
        y: usize,
        sx: usize, sy: usize, sz: usize,
    ) -> Option<Vec<(u8, f32)>> {
        let chunk = chunks.get(key)?;
        let mut out = Vec::with_capacity(sx * sz);
        for x in 0..sx {
            for z in 0..sz {
                let idx = x * sy * sz + y * sz + z;
                out.push((chunk.blocks[idx], chunk.fluid_levels[idx]));
            }
        }
        Some(out)
    }

    /// Snapshot one YZ plane (constant x) of a chunk.
    /// Returns Vec<(block, level)> indexed by [y * sz + z].
    fn snap_x_slice(
        chunks: &HashMap<(i32, i32, i32), Chunk>,
        key: &(i32, i32, i32),
        x: usize,
        _sx: usize, sy: usize, sz: usize,
    ) -> Option<Vec<(u8, f32)>> {
        let chunk = chunks.get(key)?;
        let mut out = Vec::with_capacity(sy * sz);
        for y in 0..sy {
            for z in 0..sz {
                let idx = x * sy * sz + y * sz + z;
                out.push((chunk.blocks[idx], chunk.fluid_levels[idx]));
            }
        }
        Some(out)
    }

    /// Snapshot one XY plane (constant z) of a chunk.
    /// Returns Vec<(block, level)> indexed by [x * sy + y].
    fn snap_z_slice(
        chunks: &HashMap<(i32, i32, i32), Chunk>,
        key: &(i32, i32, i32),
        z: usize,
        sx: usize, sy: usize, sz: usize,
    ) -> Option<Vec<(u8, f32)>> {
        let chunk = chunks.get(key)?;
        let mut out = Vec::with_capacity(sx * sy);
        for x in 0..sx {
            for y in 0..sy {
                let idx = x * sy * sz + y * sz + z;
                out.push((chunk.blocks[idx], chunk.fluid_levels[idx]));
            }
        }
        Some(out)
    }
}
