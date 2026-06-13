// =============================================================================
// QubePixel — Fluid Simulation System (flowing_fluids approach)
// =============================================================================
//
// Discrete level system (0–8) inspired by the flowing_fluids Minecraft mod:
//
//   Level 0   = no fluid / air
//   Level 1–7 = flowing fluid at various fill heights
//   Level 8   = source block (permanent, never depletes)
//
// Algorithm per tick:
//   1. GRAVITY FIRST — flow down into air or top-up same-fluid below.
//   2. HORIZONTAL LEVELING — mechanical averaging with neighbors
//      (only when blocked below AND a downward slope exists within
//       slope_find_distance blocks).
//   3. Source blocks (level 8) regenerate to 8 every tick.
//
// Cross-chunk flow uses the same snapshot-then-apply pattern as before.
// =============================================================================

use std::collections::{HashMap, HashSet};
use crate::core::config;
use crate::core::gameobjects::chunk::Chunk;
use crate::core::gameobjects::block::BlockRegistry;
use crate::debug_log;
use crate::flow_debug_log;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Discrete fluid level of a source block (infinite supply).
pub const FLUID_SOURCE_LEVEL: u8 = 8;
/// Minimum level a flowing block can have before being removed.
const MIN_LEVEL: u8 = 1;

const NEIGHBOR_OFFSETS: [(i32, i32, i32); 6] = [
    ( 1,  0,  0), (-1,  0,  0),
    ( 0,  1,  0), ( 0, -1,  0),
    ( 0,  0,  1), ( 0,  0, -1),
];

// ---------------------------------------------------------------------------
// Per-fluid simulation info (extracted from FluidProperties at init)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FluidInfo {
    /// Levels lost per horizontal spread step (water:1, lava:2).
    level_decrease: u8,
    /// Frames between simulation ticks for this fluid.
    tick_delay: u32,
    /// How far horizontally to search for a downward slope before spreading.
    slope_find_distance: u8,
}

// ---------------------------------------------------------------------------
// Pending cross-chunk changes
// ---------------------------------------------------------------------------

struct CrossChange {
    chunk_key: (i32, i32, i32),
    idx:       usize,
    /// New level to write (0 = remove).  i16 so we can represent "remove".
    new_level: u8,
    /// Block ID to set (0 = don't change block type).
    set_block: u8,
}

// ---------------------------------------------------------------------------
// Fluid interaction rule
// ---------------------------------------------------------------------------

struct FluidInteraction {
    fluid_a: u8,
    fluid_b: u8,
    product: u8,
}

// ---------------------------------------------------------------------------
// FluidSimulator
// ---------------------------------------------------------------------------

pub struct FluidSimulator {
    fluid_ids:    Vec<u8>,
    fluid_info:   HashMap<u8, FluidInfo>,
    interactions: Vec<FluidInteraction>,
    dirty_chunks: HashSet<(i32, i32, i32)>,
    /// Per-chunk tick countdown.  Chunk simulates when counter reaches 0.
    tick_counters: HashMap<(i32, i32, i32), u32>,
    /// Global frame counter for tick scheduling.
    frame: u64,
}

impl FluidSimulator {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    pub fn new(registry: &BlockRegistry) -> Self {
        let fluid_blocks = registry.fluid_blocks();
        let mut fluid_ids = Vec::new();
        let mut fluid_info = HashMap::new();

        for (id, props) in &fluid_blocks {
            fluid_ids.push(*id);
            fluid_info.insert(*id, FluidInfo {
                level_decrease:      props.level_decrease.max(1),
                tick_delay:          props.tick_delay.max(1),
                slope_find_distance: props.slope_find_distance,
            });
        }

        let mut interactions = Vec::new();
        let water_id = registry.id_for("water").unwrap_or(0);
        let lava_id  = registry.id_for("lava").unwrap_or(0);
        let stone_id = registry.id_for("rocks/andesite").unwrap_or(0);
        if water_id != 0 && lava_id != 0 && stone_id != 0 {
            interactions.push(FluidInteraction { fluid_a: water_id, fluid_b: lava_id,  product: stone_id });
            interactions.push(FluidInteraction { fluid_a: lava_id,  fluid_b: water_id, product: stone_id });
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
            tick_counters: HashMap::new(),
            frame: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    #[inline]
    pub fn is_fluid(&self, block_id: u8) -> bool {
        block_id != 0 && self.fluid_info.contains_key(&block_id)
    }

    pub fn fluid_ids(&self) -> &[u8] { &self.fluid_ids }

    pub fn mark_dirty(&mut self, key: (i32, i32, i32)) {
        self.dirty_chunks.insert(key);
        for &(dx, dy, dz) in &NEIGHBOR_OFFSETS {
            self.dirty_chunks.insert((key.0 + dx, key.1 + dy, key.2 + dz));
        }
    }

    pub fn mark_dirty_local(&mut self, key: (i32, i32, i32)) {
        self.dirty_chunks.insert(key);
    }

    pub fn has_dirty(&self) -> bool { !self.dirty_chunks.is_empty() }

    // -----------------------------------------------------------------------
    // Interaction lookup
    // -----------------------------------------------------------------------

    fn check_interaction(&self, fluid_a: u8, fluid_b: u8) -> Option<u8> {
        self.interactions.iter()
            .find(|ix| ix.fluid_a == fluid_a && ix.fluid_b == fluid_b)
            .map(|ix| ix.product)
    }

    // -----------------------------------------------------------------------
    // Main entry point
    // -----------------------------------------------------------------------

    pub fn simulate(&mut self, chunks: &mut HashMap<(i32, i32, i32), Chunk>) {
        self.frame = self.frame.wrapping_add(1);

        if self.dirty_chunks.is_empty() { return; }

        let dirty: Vec<(i32, i32, i32)> = self.dirty_chunks.drain().collect();
        let mut all_cross: Vec<CrossChange> = Vec::new();

        for key in &dirty {
            // --- Tick-rate gating: each chunk ticks at its fluid's rate ---
            let delay = self.chunk_tick_delay(key, chunks);
            let counter = self.tick_counters.entry(*key).or_insert(0);
            if *counter > 0 {
                *counter -= 1;
                // Re-queue so we check again next frame
                self.dirty_chunks.insert(*key);
                continue;
            }
            *counter = delay.saturating_sub(1);

            let (changed, cross) = self.simulate_chunk(key, chunks);
            if changed { all_cross.extend(cross); }
        }

        // --- Apply cross-chunk changes ---
        for cc in all_cross {
            if let Some(nc) = chunks.get_mut(&cc.chunk_key) {
                let cur_block = nc.blocks[cc.idx];
                let cur_level = nc.fluid_levels[cc.idx];

                // Only write to air or same-fluid cells, never overwrite solid
                let can_write = cur_block == 0 || self.is_fluid(cur_block);
                if !can_write { continue; }

                if cc.new_level == 0 {
                    nc.blocks[cc.idx]       = 0;
                    nc.fluid_levels[cc.idx] = 0;
                } else {
                    // Take the higher of the two levels (prevent overwriting a
                    // fuller cell with a shallower one from another tick)
                    if cc.set_block != 0 { nc.blocks[cc.idx] = cc.set_block; }
                    nc.fluid_levels[cc.idx] = nc.fluid_levels[cc.idx].max(cc.new_level);
                }

                if nc.fluid_levels[cc.idx] != cur_level || nc.blocks[cc.idx] != cur_block {
                    nc.mesh_dirty = true;
                    self.dirty_chunks.insert(cc.chunk_key);
                    for &(dx, dy, dz) in &NEIGHBOR_OFFSETS {
                        let nk = (cc.chunk_key.0 + dx, cc.chunk_key.1 + dy, cc.chunk_key.2 + dz);
                        if chunks.contains_key(&nk) { self.dirty_chunks.insert(nk); }
                    }
                }
            }
        }
    }

    /// Returns the tick delay for the dominant fluid in the chunk.
    fn chunk_tick_delay(&self, key: &(i32, i32, i32), chunks: &HashMap<(i32, i32, i32), Chunk>) -> u32 {
        if let Some(chunk) = chunks.get(key) {
            for (i, &b) in chunk.blocks.iter().enumerate() {
                if let Some(info) = self.fluid_info.get(&b) {
                    if chunk.fluid_levels[i] >= MIN_LEVEL {
                        return info.tick_delay;
                    }
                }
            }
        }
        5
    }

    // -----------------------------------------------------------------------
    // Single-chunk simulation
    // -----------------------------------------------------------------------

    fn simulate_chunk(
        &mut self,
        key: &(i32, i32, i32),
        chunks: &mut HashMap<(i32, i32, i32), Chunk>,
    ) -> (bool, Vec<CrossChange>) {
        let sx = config::chunk_size_x();
        let sy = config::chunk_size_y();
        let sz = config::chunk_size_z();

        // --- Snapshot (immutable) ---
        let (blocks_snap, levels_snap) = {
            let chunk = match chunks.get(key) {
                Some(c) => c,
                None    => return (false, vec![]),
            };
            (chunk.blocks.clone(), chunk.fluid_levels.clone())
        };

        // --- Boundary slice snapshots of face-neighbours ---
        let below_key = (key.0, key.1 - 1, key.2);
        let xn_key    = (key.0 - 1, key.1, key.2);
        let xp_key    = (key.0 + 1, key.1, key.2);
        let zn_key    = (key.0, key.1, key.2 - 1);
        let zp_key    = (key.0, key.1, key.2 + 1);

        let below_slice = Self::snap_y_slice(chunks, &below_key, sy - 1, sx, sy, sz);
        let xn_slice    = Self::snap_x_slice(chunks, &xn_key,    sx - 1, sx, sy, sz);
        let xp_slice    = Self::snap_x_slice(chunks, &xp_key,    0,      sx, sy, sz);
        let zn_slice    = Self::snap_z_slice(chunks, &zn_key,    sz - 1, sx, sy, sz);
        let zp_slice    = Self::snap_z_slice(chunks, &zp_key,    0,      sx, sy, sz);

        let mut new_levels  = levels_snap.clone();
        let mut new_blocks  = blocks_snap.clone();
        let mut cross_out: Vec<CrossChange> = Vec::new();
        let mut changed = false;
        let mut interaction_changes: Vec<(usize, u8)> = Vec::new();

        // Helper: flat index
        let idx = |x: usize, y: usize, z: usize| x * sy * sz + y * sz + z;

        for x in 0..sx {
            for y in 0..sy {
                for z in 0..sz {
                    let i = idx(x, y, z);
                    let block_id = blocks_snap[i];
                    if !self.is_fluid(block_id) { continue; }

                    let level = levels_snap[i];
                    if level < MIN_LEVEL { continue; }

                    let info = match self.fluid_info.get(&block_id).cloned() {
                        Some(v) => v,
                        None    => continue,
                    };

                    let is_source = level == FLUID_SOURCE_LEVEL;

                    // ========================================================
                    // Priority 1: FLOW DOWN (gravity)
                    // ========================================================
                    let can_flow_down = self.try_flow_down(
                        x, y, z, i, block_id, level, is_source,
                        &blocks_snap, &levels_snap,
                        &mut new_blocks, &mut new_levels,
                        &below_slice, &below_key,
                        sx, sy, sz,
                        &mut cross_out, &mut interaction_changes,
                        &mut changed,
                    );

                    // ========================================================
                    // Priority 2: HORIZONTAL LEVELING
                    // Only when blocked below. For sources always allow spread.
                    // ========================================================
                    let blocked_below = !can_flow_down || {
                        // blocked = solid block or fully-filled same-fluid below
                        if y > 0 {
                            let bi = idx(x, y - 1, z);
                            let bb = blocks_snap[bi];
                            let bl = levels_snap[bi];
                            (bb != 0 && !self.is_fluid(bb))
                                || (bb == block_id && bl == FLUID_SOURCE_LEVEL)
                        } else {
                            match &below_slice {
                                None => true,
                                Some(sl) => {
                                    let (bb, bl) = sl[x * sz + z];
                                    (bb != 0 && !self.is_fluid(bb))
                                        || (bb == block_id && bl == FLUID_SOURCE_LEVEL)
                                }
                            }
                        }
                    };

                    if !blocked_below { continue; }

                    // Slope check: only spread if there is a downward path
                    // within slope_find_distance horizontal steps.
                    let eff_level = new_levels[i];
                    if eff_level < MIN_LEVEL && !is_source { continue; }

                    if info.slope_find_distance > 0 {
                        let has_slope = self.find_slope(
                            x as i32, y as i32, z as i32,
                            info.slope_find_distance as i32,
                            block_id,
                            &blocks_snap, &levels_snap,
                            &xn_slice, &xp_slice, &zn_slice, &zp_slice,
                            &below_slice,
                            &below_key, &xn_key, &xp_key, &zn_key, &zp_key,
                            sx, sy, sz,
                        );

                        if !has_slope && !is_source {
                            // No downward path nearby — spread freely but
                            // only if level is high enough to cross the decrease
                            let min_to_spread = info.level_decrease + 1;
                            if eff_level < min_to_spread { continue; }
                        }
                    }

                    // Collect horizontal neighbours
                    struct HNbr {
                        is_cross:     bool,
                        local_idx:    usize,
                        cross_key:    (i32, i32, i32),
                        cross_idx:    usize,
                        block:        u8,
                        level:        u8,
                    }

                    let mut hn: Vec<HNbr> = Vec::with_capacity(4);

                    macro_rules! push_hn {
                        ($in_bound:expr, $ni:expr, $cross:expr, $ckey:expr, $cidx:expr,
                         $slice:expr, $si:expr) => {
                            if $in_bound {
                                hn.push(HNbr {
                                    is_cross: false, local_idx: $ni,
                                    cross_key: $ckey, cross_idx: 0,
                                    block: blocks_snap[$ni], level: levels_snap[$ni],
                                });
                            } else if let Some(sl) = &$slice {
                                let (nb, nl) = sl[$si];
                                hn.push(HNbr {
                                    is_cross: true, local_idx: 0,
                                    cross_key: $ckey, cross_idx: $cidx,
                                    block: nb, level: nl,
                                });
                            }
                        };
                    }

                    // Cross-chunk index formula: x*sy*sz + y*sz + z with the
                    // boundary coordinate substituted (x=0 / x=sx-1 / z=0 / z=sz-1).
                    // +X
                    push_hn!(x + 1 < sx, idx(x+1,y,z), false, xp_key, y*sz+z,
                             xp_slice, y*sz+z);
                    // -X
                    push_hn!(x > 0, idx(x-1,y,z), false, xn_key, (sx-1)*sy*sz+y*sz+z,
                             xn_slice, y*sz+z);
                    // +Z
                    push_hn!(z + 1 < sz, idx(x,y,z+1), false, zp_key, x*sy*sz+y*sz,
                             zp_slice, x*sy+y);
                    // -Z
                    push_hn!(z > 0, idx(x,y,z-1), false, zn_key, x*sy*sz+y*sz+(sz-1),
                             zn_slice, x*sy+y);

                    let cur_level = new_levels[i];

                    for nbr in &hn {
                        let n_block = nbr.block;
                        let n_level = nbr.level;

                        // Interaction
                        if n_block != 0 && n_block != block_id && self.is_fluid(n_block) {
                            if let Some(product) = self.check_interaction(block_id, n_block) {
                                if nbr.is_cross {
                                    cross_out.push(CrossChange {
                                        chunk_key: nbr.cross_key,
                                        idx: nbr.cross_idx,
                                        new_level: 0,
                                        set_block: product,
                                    });
                                } else {
                                    interaction_changes.push((nbr.local_idx, product));
                                }
                                changed = true;
                            }
                            continue;
                        }

                        // Only spread into air or same-fluid
                        if n_block != 0 && n_block != block_id { continue; }

                        // Compute target level via mechanical leveling
                        let src = if is_source { FLUID_SOURCE_LEVEL } else { cur_level } as i32;
                        let dst = n_level as i32;

                        // The level we'd deliver: src - level_decrease
                        let deliverable = src - info.level_decrease as i32;
                        if deliverable <= 0 { continue; }

                        // Mechanical leveling: average the difference
                        let diff = deliverable - dst;
                        if diff <= 0 { continue; }

                        // Amount to give = diff / 2 (rounding up to dst side)
                        let give = ((diff + 1) / 2).max(1) as u8;
                        let new_nbr_level = (dst as u8).saturating_add(give).min(FLUID_SOURCE_LEVEL - 1);

                        if new_nbr_level <= n_level { continue; }

                        flow_debug_log!(
                            "FluidSimulator", "horizontal",
                            "({},{},{}) level={} → neighbour level={} new={}",
                            x, y, z, cur_level, n_level, new_nbr_level
                        );

                        if nbr.is_cross {
                            cross_out.push(CrossChange {
                                chunk_key: nbr.cross_key,
                                idx:       nbr.cross_idx,
                                new_level: new_nbr_level,
                                set_block: if n_block == 0 { block_id } else { 0 },
                            });
                        } else {
                            if new_blocks[nbr.local_idx] == 0 {
                                new_blocks[nbr.local_idx] = block_id;
                            }
                            if new_nbr_level > new_levels[nbr.local_idx] {
                                new_levels[nbr.local_idx] = new_nbr_level;
                            }
                        }

                        // Source never depletes
                        if !is_source {
                            // Non-source blocks don't decrease their own level
                            // from horizontal spread — the level comes from
                            // propagation (each neighbor gets src-decrease).
                            // The current block keeps its level until a later
                            // tick re-evaluates it naturally.
                        }

                        changed = true;
                    }

                    // Source always restores to 8
                    if is_source { new_levels[i] = FLUID_SOURCE_LEVEL; }
                }
            }
        }

        // --- Re-evaluate non-source levels from neighbors ---
        // Non-source blocks adopt the best level delivered to them from
        // upstream (already written into new_levels above).  If nothing
        // delivered to them and they're isolated, they'll drain next tick.
        for x in 0..sx {
            for y in 0..sy {
                for z in 0..sz {
                    let i = idx(x, y, z);
                    let block_id = blocks_snap[i];
                    if !self.is_fluid(block_id) { continue; }

                    let src_level = levels_snap[i];
                    if src_level == FLUID_SOURCE_LEVEL { continue; } // already handled
                    if src_level < MIN_LEVEL { continue; }

                    let info = match self.fluid_info.get(&block_id) {
                        Some(v) => v,
                        None    => continue,
                    };

                    // Compute the best possible level from any adjacent same-fluid
                    let mut best_supply: u8 = 0;

                    // Check above (falling water provides full level)
                    if y + 1 < sy {
                        let ai = idx(x, y + 1, z);
                        if blocks_snap[ai] == block_id && levels_snap[ai] >= MIN_LEVEL {
                            // Water falling from above → fill to source-1
                            best_supply = best_supply.max(FLUID_SOURCE_LEVEL - 1);
                        }
                    }

                    // Check horizontal neighbors
                    let h_offsets: [(i32, i32); 4] = [(1,0),(-1,0),(0,1),(0,-1)];
                    for (dx, dz) in h_offsets {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx >= 0 && nx < sx as i32 && nz >= 0 && nz < sz as i32 {
                            let ni = idx(nx as usize, y, nz as usize);
                            if blocks_snap[ni] == block_id && levels_snap[ni] >= MIN_LEVEL {
                                let supply = levels_snap[ni].saturating_sub(info.level_decrease);
                                best_supply = best_supply.max(supply);
                            }
                        }
                    }

                    // Update level: take the max of current new_level and best supply
                    let existing_new = new_levels[i];
                    if best_supply > 0 && best_supply != src_level {
                        let target = existing_new.max(best_supply);
                        if target != new_levels[i] {
                            new_levels[i] = target;
                            changed = true;
                        }
                    }
                }
            }
        }

        // --- Apply interaction changes ---
        for (ci, product) in &interaction_changes {
            if new_blocks[*ci] != *product {
                new_blocks[*ci] = *product;
                new_levels[*ci] = 0;
                changed = true;
            }
        }

        // --- Write to chunk ---
        if changed {
            let chunk = chunks.get_mut(key).unwrap();
            for i in 0..new_levels.len() {
                if new_levels[i] != chunk.fluid_levels[i] {
                    chunk.fluid_levels[i] = new_levels[i];
                    chunk.mesh_dirty = true;
                }
                if new_blocks[i] != chunk.blocks[i] {
                    chunk.blocks[i] = new_blocks[i];
                    chunk.mesh_dirty = true;
                }
            }

            // Remove dead fluid cells
            for i in 0..chunk.blocks.len() {
                if self.is_fluid(chunk.blocks[i]) && chunk.fluid_levels[i] < MIN_LEVEL {
                    chunk.blocks[i]       = 0;
                    chunk.fluid_levels[i] = 0;
                    chunk.mesh_dirty = true;
                }
            }

            // Re-queue if still has non-source flowing fluid
            let still_active = chunk.blocks.iter().zip(chunk.fluid_levels.iter())
                .any(|(&b, &l)| self.is_fluid(b) && l >= MIN_LEVEL && l < FLUID_SOURCE_LEVEL);
            if still_active {
                self.dirty_chunks.insert(*key);
            }
        }

        if !cross_out.is_empty() {
            for &(dx, dy, dz) in &NEIGHBOR_OFFSETS {
                let nk = (key.0 + dx, key.1 + dy, key.2 + dz);
                if chunks.contains_key(&nk) { self.dirty_chunks.insert(nk); }
            }
        }

        (changed, cross_out)
    }

    // -----------------------------------------------------------------------
    // Downward flow helper
    // Returns true if the block CAN flow down (even if it already is full below)
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn try_flow_down(
        &self,
        x: usize, y: usize, z: usize, _i: usize,
        block_id: u8, level: u8, is_source: bool,
        blocks_snap: &[u8], levels_snap: &[u8],
        new_blocks: &mut Vec<u8>, new_levels: &mut Vec<u8>,
        below_slice: &Option<Vec<(u8, u8)>>,
        below_key: &(i32, i32, i32),
        _sx: usize, sy: usize, sz: usize,
        cross_out: &mut Vec<CrossChange>,
        interaction_changes: &mut Vec<(usize, u8)>,
        changed: &mut bool,
    ) -> bool {
        let idx = |px: usize, py: usize, pz: usize| px * sy * sz + py * sz + pz;

        let give_level = if is_source { FLUID_SOURCE_LEVEL - 1 } else { level };

        if y > 0 {
            let bi = idx(x, y - 1, z);
            let bb = blocks_snap[bi];
            let bl = levels_snap[bi];

            if bb == 0 {
                // Air below — place fluid
                let new_lvl = give_level.min(FLUID_SOURCE_LEVEL - 1);
                if new_lvl >= MIN_LEVEL && new_lvl > new_levels[bi] {
                    new_blocks[bi] = block_id;
                    new_levels[bi] = new_lvl;
                    if !is_source {
                        // The block above doesn't immediately drain; it will be
                        // re-evaluated by the supply propagation pass.
                    }
                    *changed = true;
                }
                return true;
            } else if bb == block_id {
                // Same fluid below — top up if not full
                if bl < FLUID_SOURCE_LEVEL - 1 {
                    let fill_to = (bl + give_level).min(FLUID_SOURCE_LEVEL - 1);
                    if fill_to > new_levels[bi] {
                        new_levels[bi] = fill_to;
                        *changed = true;
                    }
                }
                return bl < FLUID_SOURCE_LEVEL;
            } else if self.is_fluid(bb) && bb != block_id {
                if let Some(product) = self.check_interaction(block_id, bb) {
                    interaction_changes.push((bi, product));
                    *changed = true;
                }
                return false;
            }
            return false; // solid block below
        }

        // Cross-chunk downward flow
        if let Some(sl) = below_slice {
            let si = x * sz + z;
            let (bb, bl) = sl[si];
            let dest_idx = idx(x, sy - 1, z);

            if bb == 0 {
                let new_lvl = give_level.min(FLUID_SOURCE_LEVEL - 1);
                if new_lvl >= MIN_LEVEL {
                    cross_out.push(CrossChange {
                        chunk_key: *below_key,
                        idx:       dest_idx,
                        new_level: new_lvl,
                        set_block: block_id,
                    });
                    *changed = true;
                }
                return true;
            } else if bb == block_id {
                if bl < FLUID_SOURCE_LEVEL - 1 {
                    let fill_to = (bl + give_level).min(FLUID_SOURCE_LEVEL - 1);
                    if fill_to > bl {
                        cross_out.push(CrossChange {
                            chunk_key: *below_key,
                            idx:       dest_idx,
                            new_level: fill_to,
                            set_block: 0,
                        });
                        *changed = true;
                    }
                }
                return bl < FLUID_SOURCE_LEVEL;
            } else if self.is_fluid(bb) && bb != block_id {
                if let Some(product) = self.check_interaction(block_id, bb) {
                    cross_out.push(CrossChange {
                        chunk_key: *below_key,
                        idx:       dest_idx,
                        new_level: 0,
                        set_block: product,
                    });
                    *changed = true;
                }
                return false;
            }
        }

        false
    }

    // -----------------------------------------------------------------------
    // Slope detection — BFS up to `max_dist` horizontal steps looking for a
    // block from which fluid can fall.
    // Returns true if a path downward exists.
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn find_slope(
        &self,
        ox: i32, oy: i32, oz: i32,
        max_dist: i32,
        block_id: u8,
        blocks_snap: &[u8], levels_snap: &[u8],
        xn_slice: &Option<Vec<(u8, u8)>>,
        xp_slice: &Option<Vec<(u8, u8)>>,
        zn_slice: &Option<Vec<(u8, u8)>>,
        zp_slice: &Option<Vec<(u8, u8)>>,
        below_slice: &Option<Vec<(u8, u8)>>,
        _below_key: &(i32, i32, i32),
        _xn_key: &(i32, i32, i32),
        _xp_key: &(i32, i32, i32),
        _zn_key: &(i32, i32, i32),
        _zp_key: &(i32, i32, i32),
        sx: usize, sy: usize, sz: usize,
    ) -> bool {
        let idx = |x: usize, y: usize, z: usize| x * sy * sz + y * sz + z;

        // Returns (block, passable) for a world-local XZ position at oy height.
        let cell = |wx: i32, wz: i32| -> (u8, u8) {
            let lx = wx - (ox - (ox % sx as i32));
            let lz = wz - (oz - (oz % sz as i32));
            let wy = oy;

            // In-chunk
            if lx >= 0 && lx < sx as i32 && lz >= 0 && lz < sz as i32 {
                let i2 = idx(lx as usize, wy as usize, lz as usize);
                return (blocks_snap[i2], levels_snap[i2]);
            }
            // Cross-chunk (simplified — only check x/z boundary slices)
            if lx < 0 {
                if let Some(sl) = xn_slice {
                    let cly = wy.clamp(0, sy as i32 - 1) as usize;
                    let clz = wz.rem_euclid(sz as i32) as usize;
                    return sl[cly * sz + clz];
                }
            } else if lx >= sx as i32 {
                if let Some(sl) = xp_slice {
                    let cly = wy.clamp(0, sy as i32 - 1) as usize;
                    let clz = wz.rem_euclid(sz as i32) as usize;
                    return sl[cly * sz + clz];
                }
            }
            if lz < 0 {
                if let Some(sl) = zn_slice {
                    let cly = wy.clamp(0, sy as i32 - 1) as usize;
                    let clx = wx.rem_euclid(sx as i32) as usize;
                    return sl[clx * sy + cly];
                }
            } else if lz >= sz as i32 {
                if let Some(sl) = zp_slice {
                    let cly = wy.clamp(0, sy as i32 - 1) as usize;
                    let clx = wx.rem_euclid(sx as i32) as usize;
                    return sl[clx * sy + cly];
                }
            }
            (1, 0) // treat unknown as solid
        };

        let can_pass = |block: u8| -> bool {
            block == 0 || block == block_id
        };

        let has_air_below = |wx: i32, wz: i32| -> bool {
            let wy = oy - 1;
            if wy < 0 {
                // Check below_slice
                if let Some(sl) = below_slice {
                    let lx = wx.rem_euclid(sx as i32) as usize;
                    let lz = wz.rem_euclid(sz as i32) as usize;
                    let (bb, _) = sl[lx * sz + lz];
                    return bb == 0 || bb == block_id;
                }
                return false;
            }
            let (b, _) = cell(wx, wz);
            let _ = b; // silence unused
            // Check the block below at wy
            let lx = wx - (ox - ox.rem_euclid(sx as i32));
            let lz = wz - (oz - oz.rem_euclid(sz as i32));
            if lx >= 0 && lx < sx as i32 && lz >= 0 && lz < sz as i32 {
                let i2 = idx(lx as usize, wy as usize, lz as usize);
                let bb = blocks_snap[i2];
                return bb == 0 || bb == block_id;
            }
            false
        };

        // BFS outward from origin
        let mut visited: HashSet<(i32, i32)> = HashSet::new();
        let mut queue: Vec<(i32, i32, i32)> = vec![(ox, oy, oz)];
        visited.insert((ox, oz));

        while let Some((cx, _cy, cz)) = queue.pop() {
            let dist = (cx - ox).abs().max((cz - oz).abs());

            for (ddx, ddz) in [(1i32,0i32),(-1,0),(0,1),(0,-1)] {
                let nx = cx + ddx;
                let nz = cz + ddz;
                if visited.contains(&(nx, nz)) { continue; }
                visited.insert((nx, nz));

                let (nb, _nl) = cell(nx, nz);
                if !can_pass(nb) { continue; }

                // If there's empty space below this neighbor — slope found!
                if has_air_below(nx, nz) {
                    flow_debug_log!(
                        "FluidSimulator", "find_slope",
                        "slope found at ({},{}) dist={}", nx, nz, dist + 1
                    );
                    return true;
                }

                if dist + 1 < max_dist {
                    queue.push((nx, oy, nz));
                }
            }
        }

        false
    }

    // -----------------------------------------------------------------------
    // Slice snapshot helpers
    // -----------------------------------------------------------------------

    fn snap_y_slice(
        chunks: &HashMap<(i32, i32, i32), Chunk>,
        key: &(i32, i32, i32),
        y: usize,
        sx: usize, sy: usize, sz: usize,
    ) -> Option<Vec<(u8, u8)>> {
        let chunk = chunks.get(key)?;
        let mut out = Vec::with_capacity(sx * sz);
        for x in 0..sx {
            for z in 0..sz {
                let i = x * sy * sz + y * sz + z;
                out.push((chunk.blocks[i], chunk.fluid_levels[i]));
            }
        }
        Some(out)
    }

    fn snap_x_slice(
        chunks: &HashMap<(i32, i32, i32), Chunk>,
        key: &(i32, i32, i32),
        x: usize,
        _sx: usize, sy: usize, sz: usize,
    ) -> Option<Vec<(u8, u8)>> {
        let chunk = chunks.get(key)?;
        let mut out = Vec::with_capacity(sy * sz);
        for y in 0..sy {
            for z in 0..sz {
                let i = x * sy * sz + y * sz + z;
                out.push((chunk.blocks[i], chunk.fluid_levels[i]));
            }
        }
        Some(out)
    }

    fn snap_z_slice(
        chunks: &HashMap<(i32, i32, i32), Chunk>,
        key: &(i32, i32, i32),
        z: usize,
        sx: usize, sy: usize, sz: usize,
    ) -> Option<Vec<(u8, u8)>> {
        let chunk = chunks.get(key)?;
        let mut out = Vec::with_capacity(sx * sy);
        for x in 0..sx {
            for y in 0..sy {
                let i = x * sy * sz + y * sz + z;
                out.push((chunk.blocks[i], chunk.fluid_levels[i]));
            }
        }
        Some(out)
    }
}
