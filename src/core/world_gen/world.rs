// =============================================================================
// QubePixel — World  (chunk streaming + Perlin-noise terrain + LOD)
// =============================================================================
//
// CHANGELOG:
//   • Cylindrical chunk loading (dx²+dz² ≤ rd²) instead of square
//   • View-direction priority for chunk generation
//   • Frustum-prioritized mesh building with per-frame limit
//   • camera_forward piped through WorldWorker for sorting/culling
// =============================================================================

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::Instant;
use glam::Vec3;
use crate::core::world_gen::pipeline::BiomePipeline;
use crate::core::world_gen::nature::NatureLayer;
use crate::core::world_gen::stratification::StratificationLayer;
use noise::NoiseFn;
// Perlin оставьте, если используется в других местах проекта
use noise::{ Perlin};
use rayon::prelude::*;
use rapier3d::prelude::*;
use crate::{debug_log, flow_debug_log};
use crate::core::config;
use crate::core::gameobjects::block::BlockRegistry;
use crate::core::gameobjects::chunk::Chunk;
use crate::screens::game_3d_pipeline::Vertex3D;
use crate::core::gameobjects::texture_atlas::TextureAtlasLayout;
use crate::core::raycast::{dda_raycast, RaycastResult, MAX_RAYCAST_DISTANCE};
use crate::core::physics::PhysicsChunkReady;
use crate::core::vct::voxel_volume::{VoxelSnapshot, VOLUME_SIZE, vol_idx, VOLUME_TOTAL};
use crate::core::fluid::FluidSimulator;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------


/// Staggered loading — max chunks generated per World::update() call.
const MAX_CHUNKS_PER_FRAME: usize = 64;
/// Max chunk meshes rebuilt per build_meshes() call (frustum-priority order).
const MAX_MESH_BUILDS_PER_FRAME: usize = 128;

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------
pub struct World {
    chunks:   HashMap<(i32, i32, i32), Chunk>,
    pub registry: BlockRegistry,
    pipeline: BiomePipeline,
    atlas_layout: TextureAtlasLayout,

    /// Уровень LOD для чанка: 0 = полный, 1 = 2x, 2 = 4x
    chunk_lod: HashMap<(i32, i32, i32), u8>,

    /// Статистика LOD для профилировщика
    last_lod_counts: [usize; 3],

    /// Сид мира (для логов и будущего использования)
    world_seed: u64,

    /// Nature layer — plants trees and other features during chunk generation.
    nature_layer: NatureLayer,

    /// Fluid simulation engine (supports water, lava, etc.)
    fluid_sim: FluidSimulator,

    /// Numeric block ID for water (cached for fast lookup)
    water_block_id: u8,

    /// Optional world save directory. When set, new chunks try to load saved data
    /// after terrain generation, and save_all_chunks() writes .bin files here.
    world_dir: Option<PathBuf>,
}

impl World {
    const DEFAULT_WORLD_SEED: u64 = 12345;

    pub fn new(atlas_layout: TextureAtlasLayout, registry: BlockRegistry) -> Self {
        Self::new_with_seed(atlas_layout, registry, Self::DEFAULT_WORLD_SEED, None)
    }

    pub fn new_with_seed(
        atlas_layout: TextureAtlasLayout,
        registry: BlockRegistry,
        world_seed: u64,
        world_dir: Option<PathBuf>,
    ) -> Self {
        let water_block_id = registry.id_for("water").unwrap_or(0);
        let fluid_sim = FluidSimulator::new(&registry);

        debug_log!(
        "World", "new",
        "Initializing world with seed={}, {} block types registered, water_id={}",
        world_seed,
        registry.len(),
        water_block_id
        );

        let pipeline = BiomePipeline::new(world_seed);
        let nature_layer = NatureLayer::new(world_seed, &registry);

        debug_log!(
        "World", "new",
        "Terrain pipeline initialized with {} biomes",
        pipeline.biome_dict.len()
        );

        Self {
            chunks:         HashMap::new(),
            registry,
            pipeline,
            nature_layer,
            atlas_layout,
            chunk_lod:      HashMap::new(),
            last_lod_counts: [0; 3],
            world_seed,
            fluid_sim,
            water_block_id,
            world_dir,
        }
    }

    fn generate_chunk_fn(
                pipeline: &BiomePipeline,
                nature: &NatureLayer,
                registry: &BlockRegistry,
                cx: i32, cy: i32, cz: i32,
                transition_noise: &Perlin,
                water_block_id: u8,
            ) -> Chunk {
                let cs_x = config::chunk_size_x();
                let cs_y = config::chunk_size_y();
                let cs_z = config::chunk_size_z();

                let mut chunk = Chunk::new(cx, cy, cz);
                chunk.mesh_dirty = true;

                // Sample the XZ column area for this chunk's footprint.
                // Key optimization: we sample geo/climate/geology once per column,
                // then iterate Y for block placement.
                let columns = pipeline.sample_chunk_area(cx, cz, registry);

                for bx in 0..cs_x {
                    for bz in 0..cs_z {
                        let col_idx = bz * cs_x + bx;
                        let col = &columns[col_idx];

                        let wx = cx * cs_x as i32 + bx as i32;
                        let wz = cz * cs_z as i32 + bz as i32;

                        // Noise value for deep granite transition blending.
                        let granite_noise = transition_noise.get([
                            wx as f64 * 0.05,
                            wz as f64 * 0.05,
                        ]);

                        for by in 0..cs_y {
                            let wy = cy * cs_y as i32 + by as i32;

                            // Use the stratification system to determine the block.
                            let block_id = StratificationLayer::get_block_at_y(
                                &col.stratigraphy,
                                wy,
                                granite_noise,
                            );

                            // Water blocks get full water level (1.0)
                            if block_id == water_block_id {
                                chunk.set_gen_water(bx, by, bz, block_id, 1.0);
                            } else {
                                chunk.set_gen(bx, by, bz, block_id);
                            }
                        }
                    }
                }

                // Nature decoration — trees and other features.
                let base_y = cy * cs_y as i32;
                nature.decorate_chunk(&mut chunk, &columns, base_y);

                // Populate per-column biome colours and averaged ambient tint.
                let mut ambient_sum = [0.0f32; 3];
                for bx in 0..cs_x {
                    for bz in 0..cs_z {
                        let col_idx = bz * cs_x + bx; // pipeline Z-major indexing
                        let col = &columns[col_idx];
                        let chunk_col = bx * cs_z + bz; // chunk X-major indexing
                        if let Some(biome_def) = pipeline.biome_dict.get(col.biome_id) {
                            chunk.biome_foliage_colors[chunk_col] = biome_def.foliage_color;
                            chunk.biome_water_colors[chunk_col]   = biome_def.water_color;
                            for i in 0..3 { ambient_sum[i] += biome_def.ambient_tint[i]; }
                        } else {
                            for i in 0..3 { ambient_sum[i] += 1.0; }
                        }
                    }
                }
                let n = (cs_x * cs_z) as f32;
                chunk.biome_ambient_tint = [ambient_sum[0] / n, ambient_sum[1] / n, ambient_sum[2] / n];

                flow_debug_log!(
                    "World", "generate_chunk_fn",
                    "Generated chunk ({}, {}, {}) size=({},{},{})",
                    cx, cy, cz, cs_x, cs_y, cs_z
                );

                chunk
            }

    // -------------------------------------------------------------------
    // update — generate/evict chunks, compute LOD levels
    // -------------------------------------------------------------------

    /// Load chunks around `camera_pos` and unload ones too far away.
    /// Uses cylindrical shape (dx²+dz² ≤ rd²) and prioritizes view direction.
    /// Returns `(changed, evicted, gen_time_us, pending, fresh_chunk_keys)`.
    /// `fresh_chunk_keys` lists chunks that were newly inserted this update —
    /// used by the worker to harvest entity-spawn candidates only once per chunk.
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        camera_forward: Vec3,
    ) -> (bool, Vec<(i32, i32, i32)>, u128, usize, Vec<(i32, i32, i32)>) {
        let rd      = config::render_distance();
        let v_below = config::vertical_below();
        let v_above = config::vertical_above();
        let rd_sq   = rd * rd;

        let cam_cx = (camera_pos.x / config::chunk_size_x() as f32).floor() as i32;
        let cam_cy = (camera_pos.y / config::chunk_size_y() as f32).floor() as i32;
        let cam_cz = (camera_pos.z / config::chunk_size_z() as f32).floor() as i32;

        // --- Camera forward projected onto XZ plane (for priority sorting) ---
        let fwd_xz_len = (camera_forward.x * camera_forward.x
            + camera_forward.z * camera_forward.z).sqrt();
        let fwd_nx = if fwd_xz_len > 1e-6 { camera_forward.x / fwd_xz_len } else { 0.0 };
        let fwd_nz = if fwd_xz_len > 1e-6 { camera_forward.z / fwd_xz_len } else { 0.0 };

        // --- Collect keys within CYLINDRICAL render distance ---
        let mut to_generate: Vec<(i32, i32, i32)> = Vec::new();
        for dx in -rd..=rd {
            for dz in -rd..=rd {
                // Cylindrical filter: skip corners of the square
                if dx * dx + dz * dz > rd_sq { continue; }

                for dy in -v_below..=v_above {
                    let key = (cam_cx + dx, cam_cy + dy, cam_cz + dz);
                    if self.chunks.contains_key(&key) { continue; }
                    to_generate.push(key);
                }
            }
        }

        // --- Priority sort: view-direction weighted distance ---
        // Lower score = higher priority.
        // Chunks in front of camera get a bonus (negative offset).
        to_generate.sort_by(|a, b| {
            let score = |key: &(i32, i32, i32)| -> i64 {
                let dx = (key.0 - cam_cx) as f64;
                let dy = (key.1 - cam_cy) as f64;
                let dz = (key.2 - cam_cz) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let dot = dx * fwd_nx as f64 + dz * fwd_nz as f64;
                // Subtract dot*3 so forward chunks get ~3 chunks distance advantage
                ((dist_sq - dot * 3.0) * 1000.0) as i64
            };
            score(a).cmp(&score(b))
        });

        let total_pending = to_generate.len();
        let gen_count = to_generate.len().min(MAX_CHUNKS_PER_FRAME);
        let was_truncated = to_generate.len() > MAX_CHUNKS_PER_FRAME;
        to_generate.truncate(gen_count);

        let mut changed = false;
        let mut gen_time_us: u128 = 0;
        let mut fresh_chunk_keys: Vec<(i32, i32, i32)> = Vec::new();

        if !to_generate.is_empty() {
            let t0 = Instant::now();

            let pipeline = &self.pipeline;
            let nature   = &self.nature_layer;
            let registry = &self.registry;

            // Единый источник шума для детерминированности в многопоточности
            let transition_noise = Perlin::new((self.world_seed.wrapping_add(999)) as u32);

            let water_block_id = self.water_block_id;
            let new_chunks: Vec<((i32, i32, i32), Chunk)> = to_generate
                .into_par_iter()
                .map(|key| {
                    let chunk = Self::generate_chunk_fn(
                        pipeline, nature, registry,
                        key.0, key.1, key.2,
                        &transition_noise,
                        water_block_id,
                    );
                    (key, chunk)
                })
                .collect();

            gen_time_us = t0.elapsed().as_micros();

            // Mark neighbors dirty
            for &(ref key, _) in &new_chunks {
                let offsets: [(i32, i32, i32); 6] = [
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1),
                ];
                for &(dx, dy, dz) in &offsets {
                    let nk = (key.0 + dx, key.1 + dy, key.2 + dz);
                    if let Some(neighbor) = self.chunks.get_mut(&nk) {
                        neighbor.mesh_dirty = true;
                    }
                }
            }

            for (key, mut chunk) in new_chunks {
                // Apply persisted player edits on top of freshly generated terrain.
                if let Some(ref wdir) = self.world_dir {
                    if let Some(saved) = crate::core::save_system::read_chunk(wdir, key.0, key.1, key.2) {
                        chunk.apply_save_data(&saved);
                    }
                }
                // Mark chunks containing fluid blocks so the simulator
                // knows to process them from the very first frame.
                let has_fluid = chunk.blocks.iter().any(|&b| self.fluid_sim.is_fluid(b));
                if has_fluid {
                    self.fluid_sim.mark_dirty(key);
                }
                fresh_chunk_keys.push(key);
                self.chunks.insert(key, chunk);
            }

            changed = true;
        }

        // --- Evict chunks beyond CYLINDRICAL render distance ---
        let evict_r   = rd + 2;
        let evict_r_sq = evict_r * evict_r;
        let before = self.chunks.len();

        let evict_keys: Vec<(i32, i32, i32)> = self.chunks.keys()
            .copied()
            .filter(|&(cx, cy, cz)| {
                let dx = cx - cam_cx;
                let dz = cz - cam_cz;
                let too_far_h = dx * dx + dz * dz > evict_r_sq;
                let too_far_v = cy < cam_cy - v_below - 1
                    || cy > cam_cy + v_above + 1;
                too_far_h || too_far_v
            })
            .collect();

        for key in &evict_keys {
            self.chunks.remove(key);
            self.chunk_lod.remove(key);
        }

        let evicted: Vec<(i32, i32, i32)> = evict_keys;

        if self.chunks.len() != before {
            changed = true;
        }

        // --- Compute LOD levels for all loaded chunks ---
        let mut lod_counts = [0usize; 3];

        let loaded_keys: Vec<(i32, i32, i32)> = self.chunks.keys().copied().collect();

        for key in loaded_keys {
            let new_lod = config::compute_lod_level(cam_cx, cam_cz, key.0, key.2);
            let old_lod = self.chunk_lod.get(&key).copied().unwrap_or(255);

            if new_lod != old_lod {
                self.chunk_lod.insert(key, new_lod);
                if old_lod != 255 {
                    if let Some(chunk) = self.chunks.get_mut(&key) {
                        chunk.mesh_dirty = true;
                    }
                    changed = true;
                }
            }

            lod_counts[new_lod as usize] += 1;
        }
        self.last_lod_counts = lod_counts;

        (
            changed,
            evicted,
            gen_time_us,
            if was_truncated { total_pending - gen_count } else { 0 },
            fresh_chunk_keys,
        )
    }

    /// Sample the top surface of a chunk for entity-spawn candidates.
    ///
    /// Returns up to `sample_count` entries of `(wx, wy_surface, wz, surface_block_id)`
    /// where `wy_surface` is the world Y of the topmost non-air solid block in
    /// that column, and `surface_block_id` is that block's ID.  Columns with
    /// no surface (or covered by liquid) are skipped.
    pub fn sample_chunk_surface(
        &self,
        chunk_key:    (i32, i32, i32),
        sample_count: usize,
    ) -> Vec<(i32, i32, i32, u8)> {
        let chunk = match self.chunks.get(&chunk_key) {
            Some(c) => c,
            None    => return Vec::new(),
        };
        let csx = config::chunk_size_x();
        let csy = config::chunk_size_y();
        let csz = config::chunk_size_z();
        let wx_base = chunk_key.0 * csx as i32;
        let wy_base = chunk_key.1 * csy as i32;
        let wz_base = chunk_key.2 * csz as i32;

        // Deterministic stride sampling — picks `sample_count` columns spread
        // across the chunk footprint so each chunk gives a stable set.
        let total_cols = csx * csz;
        let stride = (total_cols / sample_count.max(1)).max(1);
        let mut out = Vec::with_capacity(sample_count);

        for n in 0..sample_count {
            let col_idx = (n * stride) % total_cols;
            let bx = col_idx % csx;
            let bz = col_idx / csx;

            // Scan from the top downward for the first solid block.
            let mut found = false;
            for by in (0..csy).rev() {
                let bid = chunk.get(bx, by, bz);
                if bid == 0 { continue; }
                // Skip if covered by another block at by+1 (we want top surface only).
                if by + 1 < csy && chunk.get(bx, by + 1, bz) != 0 { continue; }
                // Skip fluids — sheep shouldn't stand on water.
                if self.fluid_sim.is_fluid(bid) { continue; }
                out.push((
                    wx_base + bx as i32,
                    wy_base + by as i32,
                    wz_base + bz as i32,
                    bid,
                ));
                found = true;
                break;
            }
            let _ = found;
        }

        out
    }

    // -------------------------------------------------------------------
    // build_meshes — frustum-prioritized, limited per frame
    // -------------------------------------------------------------------

    /// Builds meshes for dirty chunks, prioritizing those visible to camera.
    /// At most `MAX_MESH_BUILDS_PER_FRAME` meshes are built; remaining stay dirty.
    /// Returns `(solid_meshes, water_meshes, glass_meshes, model_blocks, dirty_keys, build_time_us, dirty_count)`.
    pub fn build_meshes(
        &mut self,
        camera_pos: Vec3,
        camera_forward: Vec3,
    ) -> (
        Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
        Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
        Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
        Vec<(i32, i32, i32, u8, u8)>,
        Vec<(i32, i32, i32)>,
        u128, usize,
    ) {
        let t0 = Instant::now();

        let cam_cx = (camera_pos.x / config::chunk_size_x() as f32).floor() as i32;
        let cam_cz = (camera_pos.z / config::chunk_size_z() as f32).floor() as i32;

        // Camera forward on XZ plane
        let fwd_xz = {
            let len = (camera_forward.x * camera_forward.x
                + camera_forward.z * camera_forward.z).sqrt();
            if len > 1e-6 {
                (camera_forward.x / len, camera_forward.z / len)
            } else {
                (0.0f32, 0.0f32)
            }
        };

        // Collect all dirty keys
        let mut dirty_keys: Vec<(i32, i32, i32)> = self.chunks.iter()
            .filter_map(|(&key, chunk)| {
                if chunk.mesh_dirty { Some(key) } else { None }
            })
            .collect();

        // Sort: visible-first (by dot with forward), then by distance
        dirty_keys.sort_by(|a, b| {
            let score = |key: &(i32, i32, i32)| -> i64 {
                let dx = (key.0 - cam_cx) as f32;
                let dz = (key.2 - cam_cz) as f32;
                let dist_sq = dx * dx + dz * dz;
                let dot = dx * fwd_xz.0 + dz * fwd_xz.1;
                // Lower score = higher priority; forward chunks get a bonus
                ((dist_sq - dot * 4.0) * 1000.0) as i64
            };
            score(a).cmp(&score(b))
        });

        // Limit per frame — the rest stay dirty for next frame
        dirty_keys.truncate(MAX_MESH_BUILDS_PER_FRAME);

        let dirty_count = dirty_keys.len();

        // Save a copy for model scanning and for WorldResult (dirty_keys is consumed by par_iter)
        let dirty_keys_snap: Vec<(i32, i32, i32)> = dirty_keys.clone();

        // Clear dirty ONLY for chunks we're about to build
        for &key in &dirty_keys {
            if let Some(chunk) = self.chunks.get_mut(&key) {
                chunk.mesh_dirty = false;
            }
        }

        let chunks   = &self.chunks;
        let registry = &self.registry;
        let atlas    = &self.atlas_layout;
        let lod_map  = &self.chunk_lod;
        let csx = config::chunk_size_x() as i32;
        let csy = config::chunk_size_y() as i32;
        let csz = config::chunk_size_z() as i32;
        let water_block_id = self.water_block_id;
        let fluid_ids = self.fluid_sim.fluid_ids().to_vec();

        // Build solid + fluid + glass meshes in parallel.
        type MeshTuple = ((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3]);
        let per_chunk: Vec<(Option<MeshTuple>, Option<MeshTuple>, Option<MeshTuple>)> = dirty_keys
            .into_par_iter()
            .map(|key| -> (Option<MeshTuple>, Option<MeshTuple>, Option<MeshTuple>) {
                let chunk = match chunks.get(&key) {
                    Some(c) => c,
                    None    => return (None, None, None),
                };

                let lod = lod_map.get(&key).copied().unwrap_or(0);

                let get_neighbor = |wx: i32, wy: i32, wz: i32| -> u8 {
                    let cx = wx.div_euclid(csx);
                    let cy = wy.div_euclid(csy);
                    let cz = wz.div_euclid(csz);
                    let lx = wx.rem_euclid(csx) as usize;
                    let ly = wy.rem_euclid(csy) as usize;
                    let lz = wz.rem_euclid(csz) as usize;
                    chunks.get(&(cx, cy, cz))
                        .map_or(0, |c| c.get(lx, ly, lz))
                };

                // Fluid mesh needs both block ID and fluid level for smooth corners.
                let get_nb_fluid = |wx: i32, wy: i32, wz: i32| -> (u8, f32) {
                    let cx = wx.div_euclid(csx);
                    let cy = wy.div_euclid(csy);
                    let cz = wz.div_euclid(csz);
                    let lx = wx.rem_euclid(csx) as usize;
                    let ly = wy.rem_euclid(csy) as usize;
                    let lz = wz.rem_euclid(csz) as usize;
                    chunks.get(&(cx, cy, cz))
                        .map_or((0, 0.0), |c| (c.get(lx, ly, lz), c.get_fluid_level(lx, ly, lz)))
                };

                // Solid mesh (LOD-aware) — excludes transparent solids (glass).
                let (verts, idxs) = chunk.build_mesh_lod(registry, atlas, get_neighbor, lod, water_block_id);

                let sx = csx as f32;
                let sy = csy as f32;
                let sz = csz as f32;
                let min = [chunk.cx as f32 * sx, chunk.cy as f32 * sy, chunk.cz as f32 * sz];
                let max = [min[0] + sx, min[1] + sy, min[2] + sz];

                let solid = if verts.is_empty() { None } else { Some((key, verts, idxs, min, max)) };

                // Fluid mesh (all fluid types, full detail only)
                let (w_verts, w_idxs) = chunk.build_fluid_mesh(
                    registry, atlas, &fluid_ids, get_nb_fluid,
                );
                let water = if w_verts.is_empty() { None } else { Some((key, w_verts, w_idxs, min, max)) };

                // Glass mesh — alpha-blended pass for transparent solid blocks.
                let (g_verts, g_idxs) = chunk.build_transparent_mesh(registry, atlas, get_neighbor);
                let glass = if g_verts.is_empty() { None } else { Some((key, g_verts, g_idxs, min, max)) };

                (solid, water, glass)
            })
            .collect();

        let mut result:        Vec<MeshTuple> = Vec::with_capacity(per_chunk.len());
        let mut water_meshes:  Vec<MeshTuple> = Vec::with_capacity(per_chunk.len());
        let mut glass_meshes:  Vec<MeshTuple> = Vec::with_capacity(per_chunk.len());
        for (s, w, g) in per_chunk {
            if let Some(s) = s { result.push(s); }
            if let Some(w) = w { water_meshes.push(w); }
            if let Some(g) = g { glass_meshes.push(g); }
        }

        // Scan dirty chunks for blocks with custom 3D models
        let mut model_block_positions: Vec<(i32, i32, i32, u8, u8)> = Vec::new();
        for &key in &dirty_keys_snap {
            let chunk = match self.chunks.get(&key) {
                Some(c) => c,
                None => continue,
            };
            let sx = config::chunk_size_x();
            let sy = config::chunk_size_y();
            let sz = config::chunk_size_z();
            let wx_base = key.0 * sx as i32;
            let wy_base = key.1 * sy as i32;
            let wz_base = key.2 * sz as i32;
            for x in 0..sx {
                for y in 0..sy {
                    for z in 0..sz {
                        let bid = chunk.get(x, y, z);
                        if bid != 0 && self.registry.has_model(bid) {
                            model_block_positions.push((
                                wx_base + x as i32,
                                wy_base + y as i32,
                                wz_base + z as i32,
                                bid,
                                chunk.get_rotation(x, y, z),
                            ));
                        }
                    }
                }
            }
        }

        let build_time_us = t0.elapsed().as_micros();

        (result, water_meshes, glass_meshes, model_block_positions, dirty_keys_snap, build_time_us, dirty_count)
    }

    /// Run one frame of fluid simulation.
    pub fn simulate_water(&mut self) {
        self.fluid_sim.simulate(&mut self.chunks);
    }

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    /// Returns true if any loaded chunk still needs its mesh rebuilt.
    pub fn has_dirty_chunks(&self) -> bool {
        self.chunks.values().any(|c| c.mesh_dirty)
    }

    fn mark_dirty(&mut self, cx: i32, cy: i32, cz: i32) {
        if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
            chunk.mesh_dirty = true;
        }
    }

    fn dirty_boundary_neighbors(
        &mut self,
        cx: i32, cy: i32, cz: i32,
        lx: usize, ly: usize, lz: usize,
    ) {
        let sx = config::chunk_size_x();
        let sy = config::chunk_size_y();
        let sz = config::chunk_size_z();
        if lx == 0      { self.mark_dirty(cx - 1, cy, cz); }
        if lx == sx - 1 { self.mark_dirty(cx + 1, cy, cz); }
        if ly == 0      { self.mark_dirty(cx, cy - 1, cz); }
        if ly == sy - 1 { self.mark_dirty(cx, cy + 1, cz); }
        if lz == 0      { self.mark_dirty(cx, cy, cz - 1); }
        if lz == sz - 1 { self.mark_dirty(cx, cy, cz + 1); }
    }

    pub fn chunk_count(&self) -> usize { self.chunks.len() }
    pub fn lod_counts(&self) -> [usize; 3] { self.last_lod_counts }

    /// Return the biome ambient tint for the chunk column at a given camera position.
    /// Looks up the nearest already-loaded chunk at that XZ; returns neutral if not found.
    pub fn biome_ambient_tint_at(&self, cam_pos: glam::Vec3) -> [f32; 3] {
        let cs_x = config::chunk_size_x() as i32;
        let cs_z = config::chunk_size_z() as i32;
        let cx = cam_pos.x as i32 / cs_x;
        let cz = cam_pos.z as i32 / cs_z;
        // Search any Y layer at this XZ column
        for cy in -2..=4 {
            if let Some(chunk) = self.chunks.get(&(cx, cy, cz)) {
                return chunk.biome_ambient_tint;
            }
        }
        [1.0f32; 3]
    }
    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> u8 {
        let (csx, csy, csz) = (
            config::chunk_size_x() as i32,
            config::chunk_size_y() as i32,
            config::chunk_size_z() as i32,
        );
        let cx = wx.div_euclid(csx);
        let cy = wy.div_euclid(csy);
        let cz = wz.div_euclid(csz);
        let lx = wx.rem_euclid(csx) as usize;
        let ly = wy.rem_euclid(csy) as usize;
        let lz = wz.rem_euclid(csz) as usize;
        match self.chunks.get(&(cx, cy, cz)) {
            Some(chunk) => chunk.get(lx, ly, lz),
            None => 0,
        }
    }

    pub fn remove_block(&mut self, wx: i32, wy: i32, wz: i32) {
        let (csx, csy, csz) = (
            config::chunk_size_x() as i32,
            config::chunk_size_y() as i32,
            config::chunk_size_z() as i32,
        );
        let cx = wx.div_euclid(csx);
        let cy = wy.div_euclid(csy);
        let cz = wz.div_euclid(csz);
        let lx = wx.rem_euclid(csx) as usize;
        let ly = wy.rem_euclid(csy) as usize;
        let lz = wz.rem_euclid(csz) as usize;

        if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
            chunk.set(lx, ly, lz, 0);
            // Clear fluid level so neighbouring fluid sees an empty cell (level 0)
            // and flows in.  Without this, the old level (e.g. 1.0 for a water source)
            // remains in the array and adjacent fluid sees no level difference → no flow.
            let sy = csy as usize;
            let sz = csz as usize;
            chunk.fluid_levels[lx * sy * sz + ly * sz + lz] = 0.0;
            debug_log!("World", "remove_block", "Broke block at ({}, {}, {})", wx, wy, wz);
        }
        self.dirty_boundary_neighbors(cx, cy, cz, lx, ly, lz);
        // Removing a block may open space for adjacent fluid to flow in.
        self.fluid_sim.mark_dirty((cx, cy, cz));
    }

    pub fn place_block(&mut self, wx: i32, wy: i32, wz: i32, block_id: u8, rotation: u8) {
        if block_id == 0 { return; }
        let (csx, csy, csz) = (
            config::chunk_size_x() as i32,
            config::chunk_size_y() as i32,
            config::chunk_size_z() as i32,
        );
        let cx = wx.div_euclid(csx);
        let cy = wy.div_euclid(csy);
        let cz = wz.div_euclid(csz);
        let lx = wx.rem_euclid(csx) as usize;
        let ly = wy.rem_euclid(csy) as usize;
        let lz = wz.rem_euclid(csz) as usize;

        if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
            chunk.set_with_rotation(lx, ly, lz, block_id, rotation);
            debug_log!("World", "place_block", "Placed block {} rot={} at ({}, {}, {})", block_id, rotation, wx, wy, wz);
        }
        self.dirty_boundary_neighbors(cx, cy, cz, lx, ly, lz);
        // If a fluid is placed, initialise its fluid_level and start simulation.
        if self.fluid_sim.is_fluid(block_id) {
            if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
                let sy = csy as usize;
                let sz = csz as usize;
                chunk.fluid_levels[lx * sy * sz + ly * sz + lz] = 1.0;
            }
            self.fluid_sim.mark_dirty((cx, cy, cz));
        }
    }

    /// Extract a cubic voxel snapshot centred on the camera for VCT GI.
    ///
    /// Iterates chunk-by-chunk (~64 HashMap lookups) instead of per-voxel (2M lookups),
    /// making this fast enough to run every few frames.
    pub fn extract_voxel_snapshot(&self, camera_pos: Vec3) -> VoxelSnapshot {
        let half = (VOLUME_SIZE / 2) as i32;
        let ox = camera_pos.x.floor() as i32 - half;
        let oy = camera_pos.y.floor() as i32 - half;
        let oz = camera_pos.z.floor() as i32 - half;
        let sz = VOLUME_SIZE as i32;

        let csx = config::chunk_size_x() as i32;
        let csy = config::chunk_size_y() as i32;
        let csz = config::chunk_size_z() as i32;

        // Chunk coordinate range covering the volume
        let cx_min = ox.div_euclid(csx);
        let cx_max = (ox + sz - 1).div_euclid(csx);
        let cy_min = oy.div_euclid(csy);
        let cy_max = (oy + sz - 1).div_euclid(csy);
        let cz_min = oz.div_euclid(csz);
        let cz_max = (oz + sz - 1).div_euclid(csz);

        // Pre-zero the block array (air default)
        let mut blocks = vec![0u8; VOLUME_TOTAL];

        for cx in cx_min..=cx_max {
            for cy in cy_min..=cy_max {
                for cz in cz_min..=cz_max {
                    let chunk = match self.chunks.get(&(cx, cy, cz)) {
                        Some(c) => c,
                        None => continue, // unloaded chunk = air
                    };

                    // World coord of this chunk's (0,0,0) corner
                    let wx0 = cx * csx;
                    let wy0 = cy * csy;
                    let wz0 = cz * csz;

                    // Local voxel range that overlaps [ox, ox+sz) in each axis
                    let lx0 = (ox - wx0).max(0) as usize;
                    let lx1 = (ox + sz - wx0).min(csx).max(0) as usize;
                    let ly0 = (oy - wy0).max(0) as usize;
                    let ly1 = (oy + sz - wy0).min(csy).max(0) as usize;
                    let lz0 = (oz - wz0).max(0) as usize;
                    let lz1 = (oz + sz - wz0).min(csz).max(0) as usize;

                    for lx in lx0..lx1 {
                        for ly in ly0..ly1 {
                            for lz in lz0..lz1 {
                                let bid = chunk.get(lx, ly, lz);
                                if bid == 0 { continue; } // skip air (already zeroed)

                                // Volume-local coords
                                let vx = (wx0 + lx as i32 - ox) as u32;
                                let vy = (wy0 + ly as i32 - oy) as u32;
                                let vz = (wz0 + lz as i32 - oz) as u32;
                                blocks[vol_idx(vx, vy, vz)] = bid;
                            }
                        }
                    }
                }
            }
        }

        VoxelSnapshot {
            origin: [ox, oy, oz],
            size: VOLUME_SIZE,
            blocks,
        }
    }

    /// Write all loaded chunks to `world_dir/chunks/<cx>_<cy>_<cz>.bin`.
    pub fn save_all_chunks(&self, world_dir: &std::path::Path) {
        let dir = world_dir.join("chunks");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            debug_log!("World", "save_all_chunks", "mkdir error: {}", e);
            return;
        }
        let mut count = 0usize;
        for chunk in self.chunks.values() {
            let data = chunk.to_save_data();
            if let Err(e) = crate::core::save_system::write_chunk(world_dir, &data) {
                debug_log!("World", "save_all_chunks", "write error ({},{},{}): {}", chunk.cx, chunk.cy, chunk.cz, e);
            } else {
                count += 1;
            }
        }
        debug_log!("World", "save_all_chunks", "Saved {} chunks to {:?}", count, world_dir);
    }

}

// =============================================================================
// WorldWorker — runs World on a background thread
// =============================================================================

enum WorldRequest {
    Update {
        camera_pos:        Vec3,
        camera_forward:    Vec3,
        ray_dir:           Option<Vec3>,
        block_ops:         Vec<BlockOp>,
        /// Chunks evicted from GPU VRAM — mark dirty so they get re-meshed
        /// when the camera brings them back into view.
        force_dirty_keys:  Vec<(i32, i32, i32)>,
        /// Player foot position for water/swimming detection.
        foot_pos:          Vec3,
    },
    /// Save all currently-loaded chunks to the given world directory.
    SaveAll { world_dir: PathBuf },
    Shutdown,
}
#[derive(Debug, Clone)]
pub enum BlockOp {
    Break { x: i32, y: i32, z: i32 },
    Place { x: i32, y: i32, z: i32, block_id: u8, rotation: u8 },
}

pub struct WorldResult {
    pub meshes: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
    /// Water meshes (separate for transparent rendering pass).
    pub water_meshes: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
    /// Transparent solid (glass) meshes — alpha-blended pbr_vct pass.
    pub transparent_meshes: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
    pub evicted: Vec<(i32, i32, i32)>,
    pub chunk_count: usize,
    pub gen_time_us: u128,
    pub mesh_time_us: u128,
    pub pending: usize,
    pub raycast_result: Option<RaycastResult>,
    /// Pre-built physics colliders — compound shapes already constructed off-thread.
    pub physics_ready: Vec<PhysicsChunkReady>,
    /// Voxel snapshot for VCT GI — cubic volume of block IDs centred on camera.
    pub voxel_snapshot: Option<VoxelSnapshot>,
    // Profiler data
    pub lod_counts: [usize; 3],
    pub dirty_count: usize,
    pub worker_total_us: u128,
    /// Biome ambient tint at the current camera position.
    /// Blended into ambient lighting in game_screen when the toggle is on.
    pub biome_ambient_tint: [f32; 3],
    /// Block ID at player's feet (for water/swimming detection). 0 = air/unloaded.
    pub foot_block: u8,
    /// Block ID one block above player's feet (checks if submerged in deeper water).
    pub foot_block_above: u8,
    /// Positions of blocks with custom 3D models in visible chunks.
    /// Each entry: (world_x, world_y, world_z, block_id, rotation).
    /// These are rendered as entity instances instead of chunk mesh faces.
    pub model_block_positions: Vec<(i32, i32, i32, u8, u8)>,
    /// Chunk keys whose meshes were rebuilt this frame.
    /// Used by GameScreen to refresh block model entity spawns for those chunks.
    pub meshed_chunk_keys: Vec<(i32, i32, i32)>,
    /// Per-chunk top-surface samples for chunks freshly generated this update.
    /// Each entry: `(chunk_key, samples)` where `samples` is a list of
    /// `(wx, wy_surface, wz, surface_block_id)` columns.  Consumed by the
    /// game screen to drive natural entity spawning.
    pub fresh_chunk_surfaces: Vec<((i32, i32, i32), Vec<(i32, i32, i32, u8)>)>,
}

/// Rotate an AABB (block-local coords, origin = block corner, pivot = (0.5, 0, 0.5)) by a
/// block rotation value.  Only Y-axis rotations 1–4 are handled; all others return unchanged.
/// Rotation 3 = +90° CCW: (x,y,z)→(z,y,1-x).  Rotation 4 = -90° CW: (x,y,z)→(1-z,y,x).
fn rotate_aabb_y(rot: u8, mn: [f32; 3], mx: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    match rot {
        1 => ([1.0-mx[0], mn[1], 1.0-mx[2]], [1.0-mn[0], mx[1], 1.0-mn[2]]),
        3 => ([mn[2], mn[1], 1.0-mx[0]], [mx[2], mx[1], 1.0-mn[0]]),
        4 => ([1.0-mx[2], mn[1], mn[0]], [1.0-mn[2], mx[1], mx[0]]),
        _ => (mn, mx),
    }
}

pub struct WorldWorker {
    request_tx:          mpsc::Sender<WorldRequest>,
    result_rx:           mpsc::Receiver<WorldResult>,
    thread:              Option<thread::JoinHandle<()>>,
    busy:                bool,
    last_chunk_count:    usize,
    pending_ops:         Vec<BlockOp>,
    /// Chunk keys evicted from GPU VRAM, buffered until next request_update.
    pending_force_dirty: Vec<(i32, i32, i32)>,
    /// Per-block AABB overrides for model blocks, keyed by block_id.
    /// Values are (min, max) in block-local [0..1] space (same coordinate space
    /// used by `model_aabbs` in GameScreen, i.e. render_offset already applied).
    /// Written by the main thread after models load; read by the worker for
    /// raycast refinement and physics collider sizing.
    pub model_block_aabbs: Arc<RwLock<HashMap<u8, ([f32; 3], [f32; 3])>>>,
    /// Per-block triangle soup for model blocks, keyed by block_id.
    /// Each element is `[v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z]` in
    /// block-local [0..1] space (+0.5 X/Z already applied).
    /// Written by the main thread; read by the worker for exact raycast.
    pub model_block_tris: Arc<RwLock<HashMap<u8, Vec<[f32; 9]>>>>,
}

impl WorldWorker {
    pub fn new(atlas_layout: TextureAtlasLayout, registry: BlockRegistry) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<WorldRequest>();
        let (result_tx,  result_rx)  = mpsc::channel::<WorldResult>();

        // Shared model AABB table — main thread writes, worker thread reads.
        let model_block_aabbs: Arc<RwLock<HashMap<u8, ([f32; 3], [f32; 3])>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let model_block_aabbs_worker = Arc::clone(&model_block_aabbs);

        // Shared triangle soup — main thread writes, worker thread reads.
        let model_block_tris: Arc<RwLock<HashMap<u8, Vec<[f32; 9]>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let model_block_tris_worker = Arc::clone(&model_block_tris);

        let handle = thread::Builder::new()
            .name("world-worker".into())
            .spawn(move || {
        let model_block_aabbs = model_block_aabbs_worker;
        let model_block_tris  = model_block_tris_worker;
                let mut world = World::new(atlas_layout, registry);
                // Track last player chunk position for physics rebuild triggers.
                // (cx, cam_wy_coarse, cz) — coarse Y prevents constant rebuild during walking on flat ground.
                let mut last_phys_cam: (i32, i32, i32) = (i32::MIN, i32::MIN, i32::MIN);
                // Track last VCT snapshot position — extract when camera moves > 8 blocks.
                let mut last_snapshot_cam = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
                loop {
                    match request_rx.recv() {
                        Ok(WorldRequest::Update {
                               camera_pos,
                               camera_forward,
                               ray_dir,
                               block_ops,
                               force_dirty_keys,
                               foot_pos,
                           }) => {
                            let t_total = Instant::now();

                            // Re-mark GPU-evicted chunks dirty so they get re-meshed
                            // next time the camera moves near them.
                            for key in &force_dirty_keys {
                                if let Some(chunk) = world.chunks.get_mut(key) {
                                    chunk.mesh_dirty = true;
                                }
                            }

                            let (changed, evicted, gen_time, pending, fresh_chunk_keys) =
                                world.update(camera_pos, camera_forward);

                            // Harvest top-surface samples from chunks that were just generated.
                            // Done before block_ops so player edits never overlap fresh-chunk spawn data.
                            let fresh_chunk_surfaces: Vec<((i32, i32, i32), Vec<(i32, i32, i32, u8)>)> =
                                fresh_chunk_keys.iter()
                                    .map(|&key| (key, world.sample_chunk_surface(key, 8)))
                                    .collect();

                            for op in &block_ops {
                                match op {
                                    BlockOp::Break { x, y, z } => {
                                        world.remove_block(*x, *y, *z);
                                    }
                                    BlockOp::Place { x, y, z, block_id, rotation } => {
                                        world.place_block(*x, *y, *z, *block_id, *rotation);
                                    }
                                }
                            }
                            let has_block_ops = !block_ops.is_empty();

                            // Run water simulation before the dirty check so that
                            // any chunks the simulator marks dirty are picked up by
                            // build_meshes in the same frame (no one-frame visual lag).
                            world.simulate_water();

                            // Rebuild meshes when: new chunks loaded, block ops performed,
                            // deferred dirty chunks from previous frames, OR water sim
                            // just modified chunk contents.
                            let has_dirty = changed
                                || has_block_ops
                                || world.has_dirty_chunks();

                            let (meshes, water_meshes, transparent_meshes, model_block_positions, meshed_chunk_keys, mesh_time, dirty_count) = if has_dirty {
                                world.build_meshes(camera_pos, camera_forward)
                            } else {
                                (Vec::new(), Vec::new(), Vec::new(), Vec::<(i32, i32, i32, u8, u8)>::new(), Vec::new(), 0, 0)
                            };

                            // Raycast skips fluid blocks (water, lava) — they are transparent to selection.
                            // For model blocks, the ray is refined against per-triangle geometry:
                            // if no triangle is hit the block is treated as air.
                            let tris_ray = model_block_tris.read().unwrap();
                            let raycast_result = ray_dir.and_then(|dir| {
                                dda_raycast(
                                    camera_pos,
                                    dir,
                                    MAX_RAYCAST_DISTANCE,
                                    |wx, wy, wz| {
                                        let id = world.get_block(wx, wy, wz);
                                        if id == 0 || world.registry.is_fluid_block(id) { return 0; }
                                        // If this block has custom model triangles, test each one.
                                        if let Some(tris) = tris_ray.get(&id) {
                                            if !tris.is_empty() {
                                                let bp = Vec3::new(wx as f32, wy as f32, wz as f32);
                                                use crate::core::raycast::ray_triangle_intersect;
                                                let hit = tris.iter().any(|t| {
                                                    ray_triangle_intersect(
                                                        camera_pos, dir,
                                                        bp + Vec3::new(t[0], t[1], t[2]),
                                                        bp + Vec3::new(t[3], t[4], t[5]),
                                                        bp + Vec3::new(t[6], t[7], t[8]),
                                                    )
                                                });
                                                if !hit { return 0; }
                                            }
                                        }
                                        id
                                    },
                                )
                            });
                            drop(tris_ray);

                            // Foot block detection for swimming/water effects.
                            let foot_block = world.get_block(
                                foot_pos.x.floor() as i32,
                                foot_pos.y.floor() as i32,
                                foot_pos.z.floor() as i32,
                            );
                            let foot_block_above = world.get_block(
                                foot_pos.x.floor() as i32,
                                (foot_pos.y + 1.0).floor() as i32,
                                foot_pos.z.floor() as i32,
                            );

                            let chunk_sx = config::chunk_size_x() as f32;
                            let chunk_sy = config::chunk_size_y() as f32;
                            let chunk_sz = config::chunk_size_z() as f32;
                            let phys_sx  = config::chunk_size_x();
                            let phys_sy  = config::chunk_size_y();
                            let phys_sz  = config::chunk_size_z();

                            // ---------------------------------------------------------------------------
                            // Physics — proximity scan + compound build (all off main thread).
                            //
                            // Only the 5×5 chunk area around the player, Y-sliced to ±PHYSICS_Y_MARGIN.
                            // On block-op only the single affected chunk is rebuilt (incremental).
                            // Compound shapes are built here via rayon — main thread just inserts handles.
                            // ---------------------------------------------------------------------------
                            const PHYSICS_H_RADIUS: i32 = 2;
                            const PHYSICS_Y_MARGIN: i32 = 10;

                            let cam_chunk_x = (camera_pos.x / chunk_sx).floor() as i32;
                            let cam_chunk_z = (camera_pos.z / chunk_sz).floor() as i32;
                            let cam_cy      = (camera_pos.y / chunk_sy).floor() as i32;
                            let cam_wy      = camera_pos.y.floor() as i32;

                            let cam_wy_coarse = cam_wy / 4;
                            let player_moved  = (cam_chunk_x, cam_wy_coarse, cam_chunk_z) != last_phys_cam;

                            // Collect affected chunk keys from block-ops for incremental rebuild
                            let mut block_op_chunk_keys: Vec<(i32, i32, i32)> = Vec::new();
                            if has_block_ops {
                                let csx_i = phys_sx as i32;
                                let csy_i = phys_sy as i32;
                                let csz_i = phys_sz as i32;
                                for op in &block_ops {
                                    let (wx, wy, wz) = match op {
                                        BlockOp::Break { x, y, z } => (*x, *y, *z),
                                        BlockOp::Place { x, y, z, .. } => (*x, *y, *z),
                                    };
                                    let key = (
                                        wx.div_euclid(csx_i),
                                        wy.div_euclid(csy_i),
                                        wz.div_euclid(csz_i),
                                    );
                                    if !block_op_chunk_keys.contains(&key) {
                                        block_op_chunk_keys.push(key);
                                    }
                                }
                            }

                            let build_physics = has_dirty || has_block_ops || player_moved;

                            if build_physics {
                                last_phys_cam = (cam_chunk_x, cam_wy_coarse, cam_chunk_z);
                            }

                            let v_below = config::vertical_below() as i32;
                            let v_above = config::vertical_above() as i32;

                            let physics_ready: Vec<PhysicsChunkReady> = if build_physics {
                                // Decide which chunk keys to scan:
                                // • incremental (block-op only) → just the affected chunk(s)
                                // • full → all chunks within PHYSICS_H_RADIUS
                                let keys_to_build: Vec<(i32, i32, i32)> =
                                    if !block_op_chunk_keys.is_empty() && !changed && !player_moved {
                                        // Incremental: only the chunks that were modified
                                        block_op_chunk_keys.clone()
                                    } else {
                                        // Full scan
                                        let mut keys = Vec::new();
                                        for dx in -PHYSICS_H_RADIUS..=PHYSICS_H_RADIUS {
                                            for dz in -PHYSICS_H_RADIUS..=PHYSICS_H_RADIUS {
                                                for dy in -v_below..=v_above {
                                                    keys.push((cam_chunk_x + dx, cam_cy + dy, cam_chunk_z + dz));
                                                }
                                            }
                                        }
                                        keys
                                    };

                                flow_debug_log!(
                                    "WorldWorker", "thread",
                                    "[Physics] rebuild keys={} cam=({},{}) wy={} dirty={} ops={} moved={}",
                                    keys_to_build.len(), cam_chunk_x, cam_chunk_z, cam_wy,
                                    has_dirty, has_block_ops, player_moved
                                );

                                // Capture fluid IDs for physics exclusion (non-solid blocks
                                // must not get collision shapes, otherwise the player walks on fluid).
                                let fluid_ids_phys: Vec<u8> = world.fluid_sim.fluid_ids().to_vec();

                                // Snapshot model AABBs for use inside the parallel closure.
                                // Keyed by block_id; values in block-local [0..1] space.
                                let aabbs_phys: HashMap<u8, ([f32; 3], [f32; 3])> =
                                    model_block_aabbs.read().unwrap().clone();

                                // Build compound shapes in parallel with rayon
                                keys_to_build.into_par_iter()
                                    .filter_map(|key| {
                                        let wy_base = key.1 * phys_sy as i32;
                                        let y_lo = (cam_wy - PHYSICS_Y_MARGIN - wy_base)
                                            .max(0) as usize;
                                        let y_hi = (cam_wy + PHYSICS_Y_MARGIN + 1 - wy_base)
                                            .min(phys_sy as i32).max(0) as usize;
                                        if y_lo >= y_hi { return None; }

                                        let chunk = world.chunks.get(&key)?;
                                        let cuboid_full = SharedShape::cuboid(0.5, 0.5, 0.5);
                                        let mut shapes: Vec<(Pose, SharedShape)> = Vec::new();
                                        for x in 0..phys_sx {
                                            for y in y_lo..y_hi {
                                                for z in 0..phys_sz {
                                                    let bid = chunk.get(x, y, z);
                                                    // Skip air and fluid blocks (water, lava, etc.)
                                                    if bid == 0 || fluid_ids_phys.contains(&bid) {
                                                        continue;
                                                    }
                                                    // Block corner in world space
                                                    let bx = key.0 as f32 * chunk_sx + x as f32;
                                                    let by = key.1 as f32 * chunk_sy + y as f32;
                                                    let bz = key.2 as f32 * chunk_sz + z as f32;
                                                    // Use model AABB if available, otherwise full 1×1×1 cube
                                                    let (pose, shape) = if let Some(&(mn_raw, mx_raw)) = aabbs_phys.get(&bid) {
                                                        let rot = chunk.get_rotation(x, y, z);
                                                        let (mn, mx) = rotate_aabb_y(rot, mn_raw, mx_raw);
                                                        let hw = (mx[0] - mn[0]) * 0.5;
                                                        let hh = (mx[1] - mn[1]) * 0.5;
                                                        let hd = (mx[2] - mn[2]) * 0.5;
                                                        let cx = bx + mn[0] + hw;
                                                        let cy = by + mn[1] + hh;
                                                        let cz = bz + mn[2] + hd;
                                                        (
                                                            Pose::from_translation(rapier3d::prelude::Vector::new(cx, cy, cz)),
                                                            SharedShape::cuboid(hw.max(0.01), hh.max(0.01), hd.max(0.01)),
                                                        )
                                                    } else {
                                                        (
                                                            Pose::from_translation(rapier3d::prelude::Vector::new(bx + 0.5, by + 0.5, bz + 0.5)),
                                                            cuboid_full.clone(),
                                                        )
                                                    };
                                                    shapes.push((pose, shape));
                                                }
                                            }
                                        }
                                        if shapes.is_empty() { return None; }

                                        let compound = SharedShape::compound(shapes);
                                        let collider = ColliderBuilder::new(compound)
                                            .friction(1.0)
                                            .restitution(0.0)
                                            .build();
                                        Some(PhysicsChunkReady { key, collider })
                                    })
                                    .collect()
                            } else {
                                Vec::new()
                            };
                            // Extract voxel snapshot for VCT GI.
                            // Triggers on: world dirty, block ops, OR camera moved > 8 blocks.
                            // Uses chunk-based iteration — fast enough for per-8-block granularity.
                            let snapshot_dist = (camera_pos - last_snapshot_cam).length();
                            let needs_snapshot = has_dirty || has_block_ops || snapshot_dist > 8.0;
                            let voxel_snapshot = if needs_snapshot {
                                last_snapshot_cam = camera_pos;
                                Some(world.extract_voxel_snapshot(camera_pos))
                            } else {
                                None
                            };

                            let worker_total_us = t_total.elapsed().as_micros();
                            let lod_counts = world.lod_counts();

                            let biome_ambient_tint = world.biome_ambient_tint_at(camera_pos);

                            let _ = result_tx.send(WorldResult {
                                meshes,
                                water_meshes,
                                transparent_meshes,
                                evicted,
                                chunk_count: world.chunk_count(),
                                gen_time_us: gen_time,
                                mesh_time_us: mesh_time,
                                pending,
                                raycast_result,
                                physics_ready,
                                voxel_snapshot,
                                lod_counts,
                                dirty_count,
                                worker_total_us,
                                biome_ambient_tint,
                                foot_block,
                                foot_block_above,
                                model_block_positions,
                                meshed_chunk_keys,
                                fresh_chunk_surfaces,
                            });
                        }
                        Ok(WorldRequest::SaveAll { world_dir }) => {
                            world.save_all_chunks(&world_dir);
                        }
                        Ok(WorldRequest::Shutdown) | Err(_) => break,
                    }
                }
                debug_log!("WorldWorker", "thread", "Background thread exiting");
            })
            .expect("Failed to spawn world-worker thread");

        debug_log!("WorldWorker", "new", "Spawned background world thread");

        Self {
            request_tx,
            result_rx,
            thread: Some(handle),
            busy:                false,
            last_chunk_count:    0,
            pending_ops:         Vec::new(),
            pending_force_dirty: Vec::new(),
            model_block_aabbs,
            model_block_tris,
        }
    }

    /// Like `new` but uses a custom seed and optional world save directory.
    pub fn new_with_seed(
        atlas_layout: TextureAtlasLayout,
        registry: BlockRegistry,
        seed: u64,
        world_dir: Option<PathBuf>,
    ) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<WorldRequest>();
        let (result_tx,  result_rx)  = mpsc::channel::<WorldResult>();

        let model_block_aabbs: Arc<RwLock<HashMap<u8, ([f32; 3], [f32; 3])>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let model_block_aabbs_worker = Arc::clone(&model_block_aabbs);

        let model_block_tris: Arc<RwLock<HashMap<u8, Vec<[f32; 9]>>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let model_block_tris_worker = Arc::clone(&model_block_tris);

        let handle = thread::Builder::new()
            .name("world-worker".into())
            .spawn(move || {
                let model_block_aabbs = model_block_aabbs_worker;
                let model_block_tris  = model_block_tris_worker;
                let mut world = World::new_with_seed(atlas_layout, registry, seed, world_dir);
                let mut last_phys_cam: (i32, i32, i32) = (i32::MIN, i32::MIN, i32::MIN);
                let mut last_snapshot_cam = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
                loop {
                    match request_rx.recv() {
                        Ok(WorldRequest::Update {
                               camera_pos,
                               camera_forward,
                               ray_dir,
                               block_ops,
                               force_dirty_keys,
                               foot_pos,
                           }) => {
                            let t_total = Instant::now();
                            for key in &force_dirty_keys {
                                if let Some(chunk) = world.chunks.get_mut(key) {
                                    chunk.mesh_dirty = true;
                                }
                            }
                            let (changed, evicted, gen_time, pending, fresh_chunk_keys) =
                                world.update(camera_pos, camera_forward);
                            let fresh_chunk_surfaces: Vec<((i32, i32, i32), Vec<(i32, i32, i32, u8)>)> =
                                fresh_chunk_keys.iter()
                                    .map(|&key| (key, world.sample_chunk_surface(key, 8)))
                                    .collect();
                            for op in &block_ops {
                                match op {
                                    BlockOp::Break { x, y, z } => { world.remove_block(*x, *y, *z); }
                                    BlockOp::Place { x, y, z, block_id, rotation } => { world.place_block(*x, *y, *z, *block_id, *rotation); }
                                }
                            }
                            let has_block_ops = !block_ops.is_empty();
                            world.simulate_water();
                            let has_dirty = changed || has_block_ops || world.has_dirty_chunks();
                            let (meshes, water_meshes, transparent_meshes, model_block_positions, meshed_chunk_keys, mesh_time, dirty_count) = if has_dirty {
                                world.build_meshes(camera_pos, camera_forward)
                            } else {
                                (Vec::new(), Vec::new(), Vec::new(), Vec::<(i32, i32, i32, u8, u8)>::new(), Vec::new(), 0, 0)
                            };
                            let tris_ray = model_block_tris.read().unwrap();
                            let raycast_result = ray_dir.and_then(|dir| {
                                dda_raycast(camera_pos, dir, MAX_RAYCAST_DISTANCE, |wx, wy, wz| {
                                    let id = world.get_block(wx, wy, wz);
                                    if id == 0 || world.registry.is_fluid_block(id) { return 0; }
                                    if let Some(tris) = tris_ray.get(&id) {
                                        if !tris.is_empty() {
                                            let bp = Vec3::new(wx as f32, wy as f32, wz as f32);
                                            use crate::core::raycast::ray_triangle_intersect;
                                            let hit = tris.iter().any(|t| ray_triangle_intersect(camera_pos, dir, bp + Vec3::new(t[0],t[1],t[2]), bp + Vec3::new(t[3],t[4],t[5]), bp + Vec3::new(t[6],t[7],t[8])));
                                            if !hit { return 0; }
                                        }
                                    }
                                    id
                                })
                            });
                            drop(tris_ray);
                            let foot_block = world.get_block(foot_pos.x.floor() as i32, foot_pos.y.floor() as i32, foot_pos.z.floor() as i32);
                            let foot_block_above = world.get_block(foot_pos.x.floor() as i32, (foot_pos.y + 1.0).floor() as i32, foot_pos.z.floor() as i32);
                            let chunk_sx = config::chunk_size_x() as f32;
                            let chunk_sy = config::chunk_size_y() as f32;
                            let chunk_sz = config::chunk_size_z() as f32;
                            let phys_sx  = config::chunk_size_x();
                            let phys_sy  = config::chunk_size_y();
                            let phys_sz  = config::chunk_size_z();
                            const PHYSICS_H_RADIUS: i32 = 2;
                            const PHYSICS_Y_MARGIN: i32 = 10;
                            let cam_chunk_x = (camera_pos.x / chunk_sx).floor() as i32;
                            let cam_chunk_z = (camera_pos.z / chunk_sz).floor() as i32;
                            let cam_cy      = (camera_pos.y / chunk_sy).floor() as i32;
                            let cam_wy      = camera_pos.y.floor() as i32;
                            let cam_wy_coarse = cam_wy / 4;
                            let player_moved  = (cam_chunk_x, cam_wy_coarse, cam_chunk_z) != last_phys_cam;
                            let mut block_op_chunk_keys: Vec<(i32, i32, i32)> = Vec::new();
                            if has_block_ops {
                                let csx_i = phys_sx as i32; let csy_i = phys_sy as i32; let csz_i = phys_sz as i32;
                                for op in &block_ops {
                                    let (wx, wy, wz) = match op { BlockOp::Break {x,y,z}=>(*x,*y,*z), BlockOp::Place{x,y,z,..}=>(*x,*y,*z) };
                                    let key = (wx.div_euclid(csx_i), wy.div_euclid(csy_i), wz.div_euclid(csz_i));
                                    if !block_op_chunk_keys.contains(&key) { block_op_chunk_keys.push(key); }
                                }
                            }
                            let build_physics = has_dirty || has_block_ops || player_moved;
                            if build_physics { last_phys_cam = (cam_chunk_x, cam_wy_coarse, cam_chunk_z); }
                            let v_below = config::vertical_below() as i32;
                            let v_above = config::vertical_above() as i32;
                            let physics_ready: Vec<PhysicsChunkReady> = if build_physics {
                                let keys_to_build: Vec<(i32,i32,i32)> = if !block_op_chunk_keys.is_empty() && !changed && !player_moved {
                                    block_op_chunk_keys.clone()
                                } else {
                                    let mut keys = Vec::new();
                                    for dx in -PHYSICS_H_RADIUS..=PHYSICS_H_RADIUS {
                                        for dz in -PHYSICS_H_RADIUS..=PHYSICS_H_RADIUS {
                                            for dy in -v_below..=v_above { keys.push((cam_chunk_x+dx, cam_cy+dy, cam_chunk_z+dz)); }
                                        }
                                    }
                                    keys
                                };
                                let fluid_ids_phys: Vec<u8> = world.fluid_sim.fluid_ids().to_vec();
                                let aabbs_phys: HashMap<u8,([f32;3],[f32;3])> = model_block_aabbs.read().unwrap().clone();
                                keys_to_build.into_par_iter().filter_map(|key| {
                                    let wy_base = key.1 * phys_sy as i32;
                                    let y_lo = (cam_wy - PHYSICS_Y_MARGIN - wy_base).max(0) as usize;
                                    let y_hi = (cam_wy + PHYSICS_Y_MARGIN + 1 - wy_base).min(phys_sy as i32).max(0) as usize;
                                    let chunk = world.chunks.get(&key)?;
                                    let cuboid_full = SharedShape::cuboid(0.5,0.5,0.5);
                                    let mut shapes: Vec<(Pose,SharedShape)> = Vec::new();
                                    for x in 0..phys_sx {
                                        for y in y_lo..y_hi {
                                            for z in 0..phys_sz {
                                                let bid = chunk.get(x,y,z);
                                                if bid == 0 || fluid_ids_phys.contains(&bid) { continue; }
                                                let bx = key.0 as f32*chunk_sx + x as f32;
                                                let by = key.1 as f32*chunk_sy + y as f32;
                                                let bz = key.2 as f32*chunk_sz + z as f32;
                                                let (pose,shape) = if let Some(&(mn_raw,mx_raw)) = aabbs_phys.get(&bid) {
                                                    let rot = chunk.get_rotation(x,y,z);
                                                    let (mn,mx) = rotate_aabb_y(rot,mn_raw,mx_raw);
                                                    let hw=(mx[0]-mn[0])*0.5; let hh=(mx[1]-mn[1])*0.5; let hd=(mx[2]-mn[2])*0.5;
                                                    let cx=bx+mn[0]+hw; let cy=by+mn[1]+hh; let cz=bz+mn[2]+hd;
                                                    (Pose::from_translation(rapier3d::prelude::Vector::new(cx,cy,cz)), SharedShape::cuboid(hw.max(0.01),hh.max(0.01),hd.max(0.01)))
                                                } else {
                                                    (Pose::from_translation(rapier3d::prelude::Vector::new(bx+0.5,by+0.5,bz+0.5)), cuboid_full.clone())
                                                };
                                                shapes.push((pose,shape));
                                            }
                                        }
                                    }
                                    if shapes.is_empty() { return None; }
                                    let compound = SharedShape::compound(shapes);
                                    let collider = ColliderBuilder::new(compound).friction(1.0).restitution(0.0).build();
                                    Some(PhysicsChunkReady { key, collider })
                                }).collect()
                            } else { Vec::new() };
                            let snapshot_dist = (camera_pos - last_snapshot_cam).length();
                            let needs_snapshot = has_dirty || has_block_ops || snapshot_dist > 8.0;
                            let voxel_snapshot = if needs_snapshot {
                                last_snapshot_cam = camera_pos;
                                Some(world.extract_voxel_snapshot(camera_pos))
                            } else { None };
                            let worker_total_us = t_total.elapsed().as_micros();
                            let lod_counts = world.lod_counts();
                            let biome_ambient_tint = world.biome_ambient_tint_at(camera_pos);
                            let _ = result_tx.send(WorldResult {
                                meshes, water_meshes, transparent_meshes, evicted,
                                chunk_count: world.chunk_count(),
                                gen_time_us: gen_time, mesh_time_us: mesh_time, pending,
                                raycast_result, physics_ready, voxel_snapshot, lod_counts,
                                dirty_count, worker_total_us, biome_ambient_tint,
                                foot_block, foot_block_above, model_block_positions, meshed_chunk_keys,
                                fresh_chunk_surfaces,
                            });
                        }
                        Ok(WorldRequest::SaveAll { world_dir }) => {
                            world.save_all_chunks(&world_dir);
                        }
                        Ok(WorldRequest::Shutdown) | Err(_) => break,
                    }
                }
                debug_log!("WorldWorker", "new_with_seed thread", "Background thread exiting");
            })
            .expect("Failed to spawn world-worker thread");

        debug_log!("WorldWorker", "new_with_seed", "Spawned seeded world thread seed={}", seed);

        Self {
            request_tx,
            result_rx,
            thread: Some(handle),
            busy:                false,
            last_chunk_count:    0,
            pending_ops:         Vec::new(),
            pending_force_dirty: Vec::new(),
            model_block_aabbs,
            model_block_tris,
        }
    }

    /// Request the background thread to save all loaded chunks to disk.
    pub fn request_save_all(&mut self, world_dir: PathBuf) {
        let _ = self.request_tx.send(WorldRequest::SaveAll { world_dir });
        debug_log!("WorldWorker", "request_save_all", "Save-all request sent");
    }

    pub fn request_update(
        &mut self,
        camera_pos:       Vec3,
        camera_forward:   Vec3,
        ray_dir:          Option<Vec3>,
        block_ops:        Vec<BlockOp>,
        force_dirty_keys: Vec<(i32, i32, i32)>,
        foot_pos:         Vec3,
    ) {
        self.pending_ops.extend(block_ops);
        self.pending_force_dirty.extend(force_dirty_keys);

        if !self.busy {
            let ops         = std::mem::take(&mut self.pending_ops);
            let dirty_keys  = std::mem::take(&mut self.pending_force_dirty);
            let _ = self.request_tx.send(WorldRequest::Update {
                camera_pos,
                camera_forward,
                ray_dir,
                block_ops:        ops,
                force_dirty_keys: dirty_keys,
                foot_pos,
            });
            self.busy = true;
        }
    }

    pub fn try_recv_result(&mut self) -> Option<WorldResult> {
        match self.result_rx.try_recv() {
            Ok(result) => {
                self.busy = false;
                self.last_chunk_count = result.chunk_count;
                Some(result)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.busy = false;
                None
            }
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.last_chunk_count
    }
}

impl Drop for WorldWorker {
    fn drop(&mut self) {
        debug_log!("WorldWorker", "drop", "Shutting down background thread");
        let _ = self.request_tx.send(WorldRequest::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}