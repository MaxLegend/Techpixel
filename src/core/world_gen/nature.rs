//! # Nature Layer
//!
//! Places natural features — trees, shrubs, and other vegetation — during
//! chunk generation. Each column is assigned a `NatureBiome` based on the
//! climate and geography data, which drives what (if anything) gets placed.
//!
//! ## Tree generation
//!
//! Trees are seeded per-column using a deterministic hash.  The algorithm is
//! intentionally kept within-chunk (only columns at least `MAX_CANOPY_R`
//! blocks from the chunk edge can spawn a tree), so no cross-chunk coordination
//! is needed.

use crate::core::gameobjects::block::BlockRegistry;
use crate::core::gameobjects::chunk::Chunk;
use crate::core::world_gen::biome_layer::coord_hash;
use crate::core::world_gen::climate::ClimateType;
use crate::core::world_gen::geography::GeoInfo;
use crate::core::world_gen::pipeline::ColumnData;
use crate::core::config;

// ---------------------------------------------------------------------------
// NatureBiome — what kind of vegetation grows here
// ---------------------------------------------------------------------------

/// The nature (vegetation) biome type for a column.
///
/// Determined from climate + geography; drives tree type and density.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatureBiome {
    /// Open plains — grass but no trees.
    Plains,
    /// Standard mixed forest with moderate tree density.
    Forest,
    /// Thick forest with high tree density.
    DenseForest,
    /// Open woodland with scattered trees.
    SparseForest,
    /// Waterlogged swamp with drooping trees.
    Swamp,
    /// Cold conifer forest (taiga) — tall narrow pines.
    Taiga,
    /// Hot dry savanna — tall flat-topped trees, sparse.
    Savanna,
    /// Tropical dense jungle with large canopies.
    JungleDense,
    /// Frozen tundra — no trees, only sparse low vegetation.
    Tundra,
    /// Cherry blossom grove with pink-leaf trees.
    CherryBlossom,
    /// Mangrove wetland near coastlines.
    Mangrove,
    /// Giant mushroom plains (no normal trees).
    MushroomPlains,
    /// Bamboo-like tall thin tree columns.
    BambooGrove,
    /// Dead forest with bare skeleton trees.
    DeadForest,
}

impl NatureBiome {
    /// Probability (0.0..1.0) that a given column will have a tree origin.
    /// Scaled by the global `tree_density_multiplier` from world gen config.
    pub fn tree_density(self) -> f64 {
        let base = match self {
            NatureBiome::Plains        => 0.0,
            NatureBiome::Forest        => 0.04,
            NatureBiome::DenseForest   => 0.10,
            NatureBiome::SparseForest  => 0.015,
            NatureBiome::Swamp         => 0.025,
            NatureBiome::Taiga         => 0.035,
            NatureBiome::Savanna       => 0.012,
            NatureBiome::JungleDense   => 0.12,
            NatureBiome::Tundra        => 0.0,
            NatureBiome::CherryBlossom => 0.05,
            NatureBiome::Mangrove      => 0.04,
            NatureBiome::MushroomPlains => 0.02,
            NatureBiome::BambooGrove   => 0.08,
            NatureBiome::DeadForest    => 0.025,
        };
        let mult = config::world_gen_config().nature.tree_density_multiplier;
        (base * mult).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// NatureLayer
// ---------------------------------------------------------------------------

pub struct NatureLayer {
    seed: u64,
    // Cached block IDs (0 = not found / no trees of that type)
    oak_log_id:    u8,
    oak_leaves_id: u8,
}

impl NatureLayer {
    pub fn new(seed: u64, registry: &BlockRegistry) -> Self {
        Self {
            seed,
            oak_log_id:    registry.id_for("wood/oak_log").unwrap_or(0),
            oak_leaves_id: registry.id_for("wood/oak_leaves").unwrap_or(0),
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Determine the nature biome for a column.
    pub fn assign_nature_biome(climate_type: ClimateType, geo: &GeoInfo, humidity: f64) -> NatureBiome {
        if geo.is_ocean { return NatureBiome::Plains; }
        if geo.is_river  { return NatureBiome::Plains; }

        // Continentalness bands → broad zones
        let _cont = geo.continentalness;

        match climate_type {
            ClimateType::Arctic | ClimateType::Antarctic =>
                NatureBiome::Tundra,

            ClimateType::Subarctic | ClimateType::Subantarctic => {
                if humidity > 0.5 { NatureBiome::Taiga }
                else               { NatureBiome::Tundra }
            }

            ClimateType::NorthTemperate | ClimateType::SouthTemperate => {
                if humidity > 0.65 { NatureBiome::DenseForest }
                else if humidity > 0.35 { NatureBiome::Forest }
                else { NatureBiome::SparseForest }
            }

            ClimateType::NorthSubtropical | ClimateType::SouthSubtropical => {
                if humidity > 0.6 { NatureBiome::Forest }
                else if humidity > 0.25 { NatureBiome::Savanna }
                else { NatureBiome::Plains }
            }

            ClimateType::NorthTropical | ClimateType::SouthTropical => {
                if humidity > 0.7 { NatureBiome::JungleDense }
                else if humidity > 0.45 { NatureBiome::BambooGrove }
                else { NatureBiome::Savanna }
            }

            ClimateType::NorthSubequatorial | ClimateType::SouthSubequatorial => {
                if humidity > 0.75 { NatureBiome::JungleDense }
                else if humidity > 0.5 { NatureBiome::Mangrove }
                else { NatureBiome::Savanna }
            }

            ClimateType::Equatorial => {
                if humidity > 0.6 { NatureBiome::JungleDense }
                else { NatureBiome::BambooGrove }
            }
        }
    }

    /// Decorate a freshly-generated chunk with trees and other features.
    ///
    /// Called after base terrain is fully filled in.
    /// `columns`: the 16×16 column data for this chunk (order: `bz * cs_x + bx`).
    /// `base_y`: world-Y of the chunk's lowest block (= `cy * cs_y`).
    pub fn decorate_chunk(
        &self,
        chunk: &mut Chunk,
        columns: &[ColumnData],
        base_y: i32,
    ) {
        if self.oak_log_id == 0 && self.oak_leaves_id == 0 {
            return; // no tree blocks registered
        }

        let cs_x = config::chunk_size_x();
        let cs_y = config::chunk_size_y();
        let cs_z = config::chunk_size_z();

        let wx_base = chunk.cx * cs_x as i32;
        let wz_base = chunk.cz * cs_z as i32;

        // Maximum tree canopy radius used for any biome.  Columns within this
        // margin of a chunk edge will not spawn a tree to avoid overflowing.
        const MAX_CANOPY_R: usize = 4;

        for bx in 0..cs_x {
            for bz in 0..cs_z {
                let col_idx = bz * cs_x + bx;
                let col = &columns[col_idx];

                // Skip ocean, river, and frozen surface columns.
                if col.geo.is_ocean || col.geo.is_river { continue; }
                if col.stratigraphy.is_frozen          { continue; }

                let nature_biome = Self::assign_nature_biome(
                    col.climate.climate_type,
                    &col.geo,
                    col.climate.humidity,
                );
                let density = nature_biome.tree_density();
                if density == 0.0 { continue; }

                let wx = wx_base + bx as i32;
                let wz = wz_base + bz as i32;

                // Deterministic per-column roll
                let roll = (coord_hash(self.seed + 7777, wx, wz) >> 11) as f64
                    / (1u64 << 53) as f64;
                if roll >= density { continue; }

                // Surface Y in chunk-local coords
                let surface_world_y = col.stratigraphy.surface_height;
                let surface_local_y = (surface_world_y - base_y) as usize;
                if surface_local_y == 0 || surface_local_y + 1 >= cs_y { continue; }

                // Derive tree dimensions from a secondary hash
                let h2 = coord_hash(self.seed + 8888, wx, wz);
                let (trunk_h, canopy_r, tree_kind) = tree_params(nature_biome, h2);

                // Guard: the full tree must fit within the chunk bounds
                let canopy_top_local = surface_local_y + 1 + trunk_h + canopy_r;
                if canopy_top_local >= cs_y { continue; }
                if bx < canopy_r || bx + canopy_r >= cs_x { continue; }
                if bz < canopy_r || bz + canopy_r >= cs_z { continue; }
                // Also enforce the edge margin from the column check above
                if bx < MAX_CANOPY_R || bx + MAX_CANOPY_R >= cs_x { continue; }
                if bz < MAX_CANOPY_R || bz + MAX_CANOPY_R >= cs_z { continue; }

                self.place_tree(chunk, bx, surface_local_y, bz, trunk_h, canopy_r, tree_kind, cs_x, cs_y, cs_z);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Private tree placer
    // -----------------------------------------------------------------------

    fn place_tree(
        &self,
        chunk: &mut Chunk,
        bx: usize,
        surface_y: usize,
        bz: usize,
        trunk_h: usize,
        canopy_r: usize,
        kind: TreeKind,
        cs_x: usize,
        cs_y: usize,
        cs_z: usize,
    ) {
        let log_id    = self.oak_log_id;
        let leaves_id = self.oak_leaves_id;

        if log_id == 0 && leaves_id == 0 { return; }

        // Trunk
        for ty in 0..trunk_h {
            let y = surface_y + 1 + ty;
            if y < cs_y {
                chunk.set_gen(bx, y, bz, log_id);
            }
        }

        let canopy_base_y = (surface_y + 1 + trunk_h) as i32;

        match kind {
            TreeKind::Oak => {
                // Sphere canopy
                place_sphere_canopy(chunk, bx, bz, canopy_base_y, canopy_r, leaves_id, cs_x, cs_y, cs_z);
            }
            TreeKind::Pine => {
                // Cone canopy — layers from top down, each layer wider
                for layer in 0..=(canopy_r as i32) {
                    let y = canopy_base_y + canopy_r as i32 - layer;
                    if y < 0 || y >= cs_y as i32 { continue; }
                    let r = layer as usize;
                    place_disk(chunk, bx, bz, y as usize, r, leaves_id, cs_x, cs_y, cs_z);
                }
                // Tip
                if canopy_base_y + canopy_r as i32 + 1 < cs_y as i32 {
                    let tip_y = (canopy_base_y + canopy_r as i32 + 1) as usize;
                    if chunk.get(bx, tip_y, bz) == 0 {
                        chunk.set_gen(bx, tip_y, bz, leaves_id);
                    }
                }
            }
            TreeKind::Savanna => {
                // Flat canopy — disk 1 layer thick at the top of the trunk
                place_disk(chunk, bx, bz, canopy_base_y as usize, canopy_r, leaves_id, cs_x, cs_y, cs_z);
                // Add one layer above
                if canopy_base_y + 1 < cs_y as i32 {
                    place_disk(chunk, bx, bz, (canopy_base_y + 1) as usize, canopy_r.saturating_sub(1), leaves_id, cs_x, cs_y, cs_z);
                }
            }
            TreeKind::Swamp => {
                // Wider, shorter sphere — slightly drooping
                let r2 = (canopy_r as i32 + 1) as usize;
                place_sphere_canopy(chunk, bx, bz, canopy_base_y - 1, r2, leaves_id, cs_x, cs_y, cs_z);
            }
            TreeKind::Bamboo => {
                // Very tall single-column trunk, small leaf puffs near top
                for ty in 0..trunk_h {
                    let y = surface_y + 1 + ty;
                    if y < cs_y { chunk.set_gen(bx, y, bz, log_id); }
                }
                for dy in 0..=2i32 {
                    let y = canopy_base_y + dy;
                    if y < 0 || y >= cs_y as i32 { continue; }
                    for dx in -1i32..=1 {
                        for dz in -1i32..=1 {
                            let lx = bx as i32 + dx;
                            let lz = bz as i32 + dz;
                            if lx >= 0 && lx < cs_x as i32 && lz >= 0 && lz < cs_z as i32 {
                                let (lx, ly, lz) = (lx as usize, y as usize, lz as usize);
                                if chunk.get(lx, ly, lz) == 0 {
                                    chunk.set_gen(lx, ly, lz, leaves_id);
                                }
                            }
                        }
                    }
                }
            }
            TreeKind::Dead => {
                // Bare trunk, no leaves
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tree parameter resolution
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum TreeKind {
    Oak,
    Pine,
    Savanna,
    Swamp,
    Bamboo,
    Dead,
}

fn tree_params(biome: NatureBiome, hash: u64) -> (usize, usize, TreeKind) {
    let var = (hash & 0xF) as usize; // 0..15 for variance
    match biome {
        NatureBiome::Forest | NatureBiome::DenseForest | NatureBiome::SparseForest
        | NatureBiome::CherryBlossom => {
            let trunk = 4 + (var % 3);        // 4–6
            let canopy = 2 + (var >> 2 & 1);  // 2–3
            (trunk, canopy, TreeKind::Oak)
        }
        NatureBiome::Taiga => {
            let trunk = 5 + (var % 4);        // 5–8
            let canopy = 2 + (var >> 2 & 2);  // 2–4
            (trunk, canopy, TreeKind::Pine)
        }
        NatureBiome::Savanna => {
            let trunk = 5 + (var % 3);        // 5–7
            let canopy = 3 + (var >> 2 & 1);  // 3–4
            (trunk, canopy, TreeKind::Savanna)
        }
        NatureBiome::JungleDense => {
            let trunk = 6 + (var % 5);        // 6–10
            let canopy = 3 + (var >> 3 & 1);  // 3–4
            (trunk, canopy, TreeKind::Oak)
        }
        NatureBiome::Swamp | NatureBiome::Mangrove => {
            let trunk = 3 + (var % 3);        // 3–5
            let canopy = 2 + (var >> 2 & 1);  // 2–3
            (trunk, canopy, TreeKind::Swamp)
        }
        NatureBiome::BambooGrove => {
            let trunk = 6 + (var % 5);        // 6–10
            (trunk, 1, TreeKind::Bamboo)
        }
        NatureBiome::DeadForest => {
            let trunk = 4 + (var % 4);        // 4–7
            (trunk, 0, TreeKind::Dead)
        }
        NatureBiome::MushroomPlains => {
            let trunk = 4 + (var % 3);
            let canopy = 2 + (var >> 2 & 1);
            (trunk, canopy, TreeKind::Oak)
        }
        _ => (4, 2, TreeKind::Oak),
    }
}

// ---------------------------------------------------------------------------
// Canopy helpers
// ---------------------------------------------------------------------------

fn place_sphere_canopy(
    chunk: &mut Chunk,
    cx: usize, cz: usize,
    center_y: i32,
    radius: usize,
    leaves_id: u8,
    cs_x: usize, cs_y: usize, cs_z: usize,
) {
    let r2 = (radius * radius) as i32;
    for dx in -(radius as i32)..=(radius as i32) {
        for dy in -(radius as i32)..=(radius as i32) {
            for dz in -(radius as i32)..=(radius as i32) {
                if dx * dx + dy * dy + dz * dz > r2 { continue; }
                let lx = cx as i32 + dx;
                let ly = center_y + dy;
                let lz = cz as i32 + dz;
                if lx < 0 || lx >= cs_x as i32 { continue; }
                if ly < 0 || ly >= cs_y as i32 { continue; }
                if lz < 0 || lz >= cs_z as i32 { continue; }
                let (lx, ly, lz) = (lx as usize, ly as usize, lz as usize);
                if chunk.get(lx, ly, lz) == 0 {
                    chunk.set_gen(lx, ly, lz, leaves_id);
                }
            }
        }
    }
}

fn place_disk(
    chunk: &mut Chunk,
    cx: usize, cz: usize,
    y: usize,
    radius: usize,
    leaves_id: u8,
    cs_x: usize, cs_y: usize, cs_z: usize,
) {
    if y >= cs_y { return; }
    let r2 = (radius * radius) as i32;
    for dx in -(radius as i32)..=(radius as i32) {
        for dz in -(radius as i32)..=(radius as i32) {
            if dx * dx + dz * dz > r2 { continue; }
            let lx = cx as i32 + dx;
            let lz = cz as i32 + dz;
            if lx < 0 || lx >= cs_x as i32 { continue; }
            if lz < 0 || lz >= cs_z as i32 { continue; }
            let (lx, lz) = (lx as usize, lz as usize);
            if chunk.get(lx, y, lz) == 0 {
                chunk.set_gen(lx, y, lz, leaves_id);
            }
        }
    }
}
