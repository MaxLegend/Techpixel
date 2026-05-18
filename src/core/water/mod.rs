// =============================================================================
// QubePixel — Water Simulation System
// =============================================================================
//
// Cellular automaton water flow:
//   - Gravity: water flows down first (including across chunk boundaries)
//   - Pressure equalization: water spreads horizontally to neighbors with
//     lower levels (including cross-chunk)
//   - Anti-ping-pong: only flows when difference > threshold
//   - Dirty tracking: only simulates active chunks
//
// Cross-chunk flow:
//   The simulation snapshots current chunk + relevant neighbor boundary slices
//   to compute flows (immutable phase), then applies all changes (mutable phase).
//   This avoids borrow-checker conflicts and correctly moves water between chunks.
// =============================================================================

use std::collections::{HashMap, HashSet};
use crate::core::config;
use crate::core::gameobjects::chunk::Chunk;
use crate::debug_log;

// ---------------------------------------------------------------------------
// Simulation constants
// ---------------------------------------------------------------------------

/// Minimum water level — below this the water cell is removed.
const MIN_LEVEL: f32 = 0.005;
/// Minimum level difference to trigger horizontal flow.
const FLOW_THRESHOLD: f32 = 0.01;
/// Fraction of level difference transferred per sub-step (horizontal).
const H_FLOW_RATE: f32 = 0.3;
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
// Pending cross-chunk change
// ---------------------------------------------------------------------------

struct CrossChange {
    chunk_key: (i32, i32, i32),
    idx:       usize,
    delta:     f32,
    set_water: bool, // if true, set block to water_block_id (was air)
}

// ---------------------------------------------------------------------------
// WaterSimulator
// ---------------------------------------------------------------------------

pub struct WaterSimulator {
    water_block_id: u8,
    dirty_chunks:   HashSet<(i32, i32, i32)>,
}

impl WaterSimulator {
    pub fn new(water_block_id: u8) -> Self {
        debug_log!("WaterSimulator", "new",
            "Created water simulator, water_block_id={}", water_block_id);
        Self {
            water_block_id,
            dirty_chunks: HashSet::new(),
        }
    }

    pub fn mark_dirty(&mut self, key: (i32, i32, i32)) {
        self.dirty_chunks.insert(key);
        for &(dx, dy, dz) in &NEIGHBOR_OFFSETS {
            self.dirty_chunks.insert((key.0 + dx, key.1 + dy, key.2 + dz));
        }
    }

    pub fn mark_dirty_local(&mut self, key: (i32, i32, i32)) {
        self.dirty_chunks.insert(key);
    }

    pub fn has_dirty(&self) -> bool {
        !self.dirty_chunks.is_empty()
    }

    // -----------------------------------------------------------------------
    // Main simulation entry point
    // -----------------------------------------------------------------------

    pub fn simulate(&mut self, chunks: &mut HashMap<(i32, i32, i32), Chunk>) {
        for _step in 0..MAX_SUB_STEPS {
            if self.dirty_chunks.is_empty() { return; }

            let dirty: Vec<(i32, i32, i32)> = self.dirty_chunks.drain().collect();
            let mut any_changed = false;
            let mut all_cross: Vec<CrossChange> = Vec::new();

            // --- Compute phase (immutable reads + self-writes) ---
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
                    if cc.set_water && nc.blocks[cc.idx] == 0 {
                        nc.blocks[cc.idx] = self.water_block_id;
                    }
                    nc.mesh_dirty = true;
                    any_changed = true;
                    // Mark this chunk and its neighbors dirty
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
        // We only need one-block-deep slices to determine cross-chunk flow.
        // Below: top row (y=sy-1) of chunk below
        let below_key  = (key.0, key.1 - 1, key.2);
        let above_key  = (key.0, key.1 + 1, key.2);
        let xn_key     = (key.0 - 1, key.1, key.2); // -X
        let xp_key     = (key.0 + 1, key.1, key.2); // +X
        let zn_key     = (key.0, key.1, key.2 - 1); // -Z
        let zp_key     = (key.0, key.1, key.2 + 1); // +Z

        // Each horizontal slice: index = x * sz + z  (below/above)
        // Vertical slices: x-face slice = y * sz + z, z-face slice = x * sy + y
        let below_slice = Self::snap_y_slice(chunks, &below_key, sy - 1, sx, sy, sz);
        let _above_slice = Self::snap_y_slice(chunks, &above_key, 0,      sx, sy, sz);
        let xn_slice    = Self::snap_x_slice(chunks, &xn_key, sx - 1, sx, sy, sz);
        let xp_slice    = Self::snap_x_slice(chunks, &xp_key, 0,      sx, sy, sz);
        let zn_slice    = Self::snap_z_slice(chunks, &zn_key, sz - 1, sx, sy, sz);
        let zp_slice    = Self::snap_z_slice(chunks, &zp_key, 0,      sx, sy, sz);

        // --- Working copies (we mutate these during simulation) ---
        let mut new_levels  = levels_snap.clone();
        let mut new_blocks  = blocks_snap.clone();
        let mut cross_out: Vec<CrossChange> = Vec::new();
        let mut changed = false;

        for x in 0..sx {
            for y in 0..sy {
                for z in 0..sz {
                    let idx = x * sy * sz + y * sz + z;
                    if new_blocks[idx] != self.water_block_id { continue; }

                    let level = new_levels[idx];
                    if level < MIN_LEVEL { continue; }

                    let mut remaining = level;

                    // ====================================================
                    // Priority 1: FLOW DOWN (gravity)
                    // ====================================================
                    if remaining >= MIN_LEVEL {
                        if y > 0 {
                            // Within chunk
                            let bi = x * sy * sz + (y - 1) * sz + z;
                            let bb = new_blocks[bi];
                            if bb == 0 {
                                // Air below
                                let space = 1.0 - new_levels[bi];
                                let flow = remaining.min(space);
                                if flow > MIN_LEVEL {
                                    new_levels[bi] += flow;
                                    new_blocks[bi]  = self.water_block_id;
                                    new_levels[idx] -= flow;
                                    remaining -= flow;
                                    changed = true;
                                }
                            } else if bb == self.water_block_id {
                                // Water below — top-up
                                let space = 1.0 - new_levels[bi];
                                if space > FLOW_THRESHOLD {
                                    let flow = remaining.min(space);
                                    new_levels[bi]  += flow;
                                    new_levels[idx] -= flow;
                                    remaining -= flow;
                                    changed = true;
                                }
                            }
                        } else if let Some(ref sl) = below_slice {
                            // Cross-chunk downward flow
                            let si = x * sz + z;
                            let (bb, bl) = sl[si];
                            if bb == 0 {
                                let space = 1.0 - bl;
                                let flow = remaining.min(space);
                                if flow > MIN_LEVEL {
                                    new_levels[idx] -= flow;
                                    remaining -= flow;
                                    changed = true;
                                    cross_out.push(CrossChange {
                                        chunk_key: below_key,
                                        idx: x * sy * sz + (sy - 1) * sz + z,
                                        delta: flow,
                                        set_water: true,
                                    });
                                }
                            } else if bb == self.water_block_id {
                                let space = 1.0 - bl;
                                if space > FLOW_THRESHOLD {
                                    let flow = remaining.min(space);
                                    new_levels[idx] -= flow;
                                    remaining -= flow;
                                    changed = true;
                                    cross_out.push(CrossChange {
                                        chunk_key: below_key,
                                        idx: x * sy * sz + (sy - 1) * sz + z,
                                        delta: flow,
                                        set_water: false,
                                    });
                                }
                            }
                        }
                    }

                    if remaining < MIN_LEVEL {
                        new_levels[idx] = 0.0;
                        continue;
                    }

                    // ====================================================
                    // Priority 2: Flow DOWN through block above us
                    // (if we have headroom and above block has water flowing in)
                    // — skip, handled when the above block processes itself

                    // Check that the block directly below is full / solid
                    // (horizontal flow only makes sense when can't flow down)
                    let blocked_below = {
                        if y > 0 {
                            let bi = x * sy * sz + (y - 1) * sz + z;
                            let bb = new_blocks[bi];
                            bb != 0 && bb != self.water_block_id
                                || (bb == self.water_block_id && new_levels[bi] >= 1.0 - FLOW_THRESHOLD)
                        } else {
                            // chunk below or no chunk below
                            match &below_slice {
                                None => true,
                                Some(sl) => {
                                    let (bb, bl) = sl[x * sz + z];
                                    bb != 0 && bb != self.water_block_id
                                        || (bb == self.water_block_id && bl >= 1.0 - FLOW_THRESHOLD)
                                }
                            }
                        }
                    };

                    if !blocked_below { continue; } // can still flow down, skip horizontal

                    // ====================================================
                    // Priority 3: FLOW HORIZONTALLY (equalization)
                    // ====================================================

                    // Collect 4 horizontal neighbor levels
                    struct HNeighbor {
                        is_cross: bool,
                        in_chunk_idx: usize,
                        cross_chunk: (i32, i32, i32),
                        cross_idx: usize,
                        block: u8,
                        level: f32,
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

                        let flow = if n_block == 0 {
                            // Air — spread into it
                            let diff = remaining - n_level;
                            if diff > FLOW_THRESHOLD {
                                (diff * H_FLOW_RATE).min(remaining)
                            } else { 0.0 }
                        } else if n_block == self.water_block_id {
                            // Water — equalize
                            let diff = remaining - n_level;
                            if diff > FLOW_THRESHOLD {
                                (diff * H_FLOW_RATE * 0.5).min(remaining)
                            } else { 0.0 }
                        } else {
                            0.0 // Solid
                        };

                        if flow <= FLOW_THRESHOLD { continue; }

                        new_levels[idx] -= flow;
                        remaining -= flow;
                        changed = true;

                        if hn.is_cross {
                            cross_out.push(CrossChange {
                                chunk_key: hn.cross_chunk,
                                idx: hn.cross_idx,
                                delta: flow,
                                set_water: n_block == 0,
                            });
                        } else {
                            new_levels[hn.in_chunk_idx] += flow;
                            if n_block == 0 {
                                new_blocks[hn.in_chunk_idx] = self.water_block_id;
                            }
                        }
                    }

                    if remaining < MIN_LEVEL {
                        new_levels[idx] = 0.0;
                    }
                }
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

            // Remove dead water cells
            for i in 0..chunk.fluid_levels.len() {
                if chunk.fluid_levels[i] < MIN_LEVEL
                    && chunk.blocks[i] == self.water_block_id
                {
                    chunk.blocks[i] = 0;
                    chunk.fluid_levels[i] = 0.0;
                    chunk.mesh_dirty = true;
                }
            }

            // Re-mark if still active
            let still_active = chunk.fluid_levels.iter()
                .zip(chunk.blocks.iter())
                .any(|(&l, &b)| b == self.water_block_id && l > MIN_LEVEL && l < 1.0 - FLOW_THRESHOLD);
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
        sx: usize,
        sy: usize,
        sz: usize,
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
        _sx: usize,
        sy: usize,
        sz: usize,
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
        sx: usize,
        sy: usize,
        sz: usize,
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
