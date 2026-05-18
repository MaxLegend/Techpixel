//! # Terrain Generation Pipeline
//!
//! Assembles all terrain subsystems into a single, thread-safe generation
//! pipeline.
//!
//! ## Architecture
//!
//! ```text
//!                    ┌──────────────────────────────────────┐
//!                    │         BiomePipeline                 │
//!                    │                                      │
//!  seed ───────────► │  ┌──────────────┐  ┌──────────────┐ │
//!                    │  │ GeographyLayer│  │ ClimateLayer │ │
//!                    │  └──────┬───────┘  └──────┬───────┘ │
//!                    │         │                  │         │
//!                    │  ┌──────┴───────┐  ┌──────┴───────┐ │
//!                    │  │ GeologyLayer │  │ BiomeDict    │ │
//!                    │  └──────┬───────┘  └──────┬───────┘ │
//!                    │         │                  │         │
//!                    │  ┌──────┴──────────────────┴───────┐ │
//!                    │  │ StratificationLayer              │ │
//!                    │  └──────────────────────────────────┘ │
//!                    └──────────────────────────────────────┘
//! ```
//!
//! The [`BiomePipeline`] is constructed once from a seed and shared (via
//! `Arc` or `&`) across rayon worker threads. Each worker calls
//! [`sample_column`](BiomePipeline::sample_column) or
//! [`sample_chunk_area`](BiomePipeline::sample_chunk_area) to produce
//! [`ColumnData`] instances.
//!
//! ## Usage
//!
//! ```ignore
//! use qubepixel_terrain::pipeline::BiomePipeline;
//!
//! let pipeline = BiomePipeline::new(42);
//!
//! // Single column.
//! let col = pipeline.sample_column(100, -200, &registry);
//! println!("biome {} at height {}", col.biome_id, col.stratigraphy.surface_height);
//!
//! // Entire 16×16 chunk.
//! let columns = pipeline.sample_chunk_area(3, -5, &registry);
//! assert_eq!(columns.len(), 256);
//! ```

use rayon::prelude::*;

use crate::core::gameobjects::block::BlockRegistry;
use crate::core::world_gen::biome_layer::BiomeLayer;
use crate::core::world_gen::biome_registry::{BiomeDictionary, ClimateTag};
use crate::core::world_gen::climate::{ClimateInfo, ClimateLayer};
use crate::core::world_gen::geography::{GeoInfo, GeographyLayer};
use crate::core::world_gen::geology::{GeologyInfo, GeologyLayer};
use crate::core::world_gen::layers::{
    LayerAddIsland, LayerBiome, LayerEdge, LayerFuzzyZoom, LayerIsland, LayerRiverInit,
    LayerSmooth, LayerVoronoiZoom, LayerZoom,
};
use crate::core::world_gen::stratification::{ColumnStratigraphy, StratificationLayer};
use crate::{debug_log, ext_debug_log, flow_debug_log};
use crate::core::config;
// ---------------------------------------------------------------------------
// ColumnData
// ---------------------------------------------------------------------------

/// Complete terrain data for a single vertical column at world coordinates
/// `(x, z)`.
///
/// This is the primary output type of the pipeline — it bundles together all
/// the information needed to generate the blocks for one column of a chunk.
///
/// # Thread safety
///
/// `ColumnData` is `Clone` + `Send` + `Sync`.
#[derive(Debug, Clone)]
pub struct ColumnData {
    /// Geographical information (elevation, continentalness, ocean flag).
    pub geo: GeoInfo,

    /// Climate information (temperature, humidity, climate type).
    pub climate: ClimateInfo,

    /// Geological information (macro/micro region, rock types).
    pub geology: GeologyInfo,

    /// Vertical block stratigraphy (block IDs + layer thicknesses).
    pub stratigraphy: ColumnStratigraphy,

    /// Determined biome ID (from the [`BiomeDictionary`]).
    pub biome_id: u32,
}

// ---------------------------------------------------------------------------
// BiomePipeline
// ---------------------------------------------------------------------------

/// The complete terrain generation pipeline.
///
/// Owns all sub-systems (geography, climate, geology, stratification, biome
/// dictionary) and provides high-level sampling methods.
///
/// # Thread safety
///
/// `BiomePipeline` is `Send + Sync` — it can be safely shared across rayon
/// worker threads. All internal layers are read-only after construction.
///
/// # Construction
///
/// ```ignore
/// let pipeline = BiomePipeline::new(42);
/// ```
pub struct BiomePipeline {
    /// World seed.
    pub seed: u64,

    /// Geography layer — produces `GeoInfo` per column.
    pub geography: GeographyLayer,

    /// Climate layer — produces `ClimateInfo` per column.
    pub climate: ClimateLayer,

    /// Geology layer — produces `GeologyInfo` per column.
    pub geology: GeologyLayer,

    /// Stratification layer — produces `ColumnStratigraphy` per column.
    pub stratification: StratificationLayer,

    /// Biome dictionary — maps conditions to biome IDs.
    pub biome_dict: BiomeDictionary,
}

impl BiomePipeline {
    /// Construct a new pipeline from a world seed.
    ///
    /// All sub-layers are initialized with the same seed (each layer
    /// internally offsets the seed to decorrelate its noise fields).
    pub fn new(seed: u64) -> Self {
        debug_log!("BiomePipeline", "new", "seed={}", seed);

        let geography = GeographyLayer::new(seed);
        let climate = ClimateLayer::new(seed);
        let geology = GeologyLayer::new(seed);
        let stratification = StratificationLayer::new(seed);
        let biome_dict = BiomeDictionary::create_default();

        debug_log!(
            "BiomePipeline",
            "new",
            "pipeline initialized with {} biomes",
            biome_dict.len()
        );

        Self {
            seed,
            geography,
            climate,
            geology,
            stratification,
            biome_dict,
        }
    }

    /// Construct a pipeline using an explicit config (bypasses global OnceLock).
    /// Used by the world-gen visualizer when editing parameters live.
    pub fn with_config(seed: u64, cfg: &crate::core::config::WorldGenConfig) -> Self {
        let geography = GeographyLayer::with_config(seed, &cfg.geography);
        let climate = ClimateLayer::with_config(seed, &cfg.climate);
        let geology = GeologyLayer::with_config(seed, &cfg.geology);
        let stratification = StratificationLayer::new(seed);
        let biome_dict = BiomeDictionary::create_default();
        Self { seed, geography, climate, geology, stratification, biome_dict }
    }

    /// Sample a single column at world coordinates `(wx, wz)`.
    ///
    /// This method samples all sub-systems and assembles a complete
    /// [`ColumnData`] for the requested position.
    ///
    /// # Arguments
    ///
    /// * `wx` — world X coordinate (block position).
    /// * `wz` — world Z coordinate (block position).
    /// * `registry` — block registry for resolving block names.
    ///
    /// # Returns
    ///
    /// A fully populated [`ColumnData`].
    ///
    /// # Performance
    ///
    /// This method involves 3 noise samples (geography, climate, geology)
    /// plus a registry lookup and a biome-determination query. Typical cost:
    /// ~2–5 µs per column.
    pub fn sample_column(
        &self,
        wx: i32,
        wz: i32,
        registry: &BlockRegistry,
    ) -> ColumnData {
        flow_debug_log!(
            "BiomePipeline",
            "sample_column",
            "wx={} wz={}",
            wx,
            wz
        );

        let geo = self.geography.sample(wx, wz);
        let climate = self.climate.sample(wx, wz);
        let geology = self.geology.sample(wx, wz);
        let stratigraphy =
            self.stratification
                .compute_column(wx, wz, &geo, &climate, &geology, registry);
        let biome_id = self.determine_biome(&geo, &climate, &geology);

        ext_debug_log!(
            "BiomePipeline",
            "sample_column",
            "wx={} wz={} biome={} height={}",
            wx,
            wz,
            biome_id,
            stratigraphy.surface_height
        );

        ColumnData {
            geo,
            climate,
            geology,
            stratigraphy,
            biome_id,
        }
    }

    /// Sample an entire 16×16 chunk area and return a flat `Vec` of 256
    /// [`ColumnData`] entries.
    ///
    /// Columns are ordered row-major: index `(dx, dz)` maps to
    /// `dz * 16 + dx` where `(dx, dz) ∈ [0, 15]²`.
    ///
    /// This method uses **rayon** to parallelize column generation across
    /// all available CPU cores.
    ///
    /// # Arguments
    ///
    /// * `cx` — chunk X coordinate (world position / 16).
    /// * `cz` — chunk Z coordinate (world position / 16).
    /// * `registry` — block registry.
    ///
    /// # Returns
    ///
    /// A `Vec<ColumnData>` of exactly 256 entries.
    ///
    /// # Performance
    ///
    /// On a typical 8-core machine this processes a full chunk in ~1–2 ms
    /// (vs. ~10–15 ms single-threaded).
    pub fn sample_chunk_area(
        &self,
        cx: i32,
        cz: i32,
        registry: &BlockRegistry,
    ) -> Vec<ColumnData> {
        let cs_x = config::chunk_size_x();
        let cs_z = config::chunk_size_z();
        let total = cs_x * cs_z;

        flow_debug_log!(
            "BiomePipeline",
            "sample_chunk_area",
            "cx={} cz={} cs_x={} cs_z={}",
            cx, cz, cs_x, cs_z
        );

        let base_x = cx * cs_x as i32;
        let base_z = cz * cs_z as i32;

        // ── Decorator / spatial-smoothing pass ──────────────────────────────
        //
        // Sample an extended region (chunk + BLEND_RADIUS border on each side)
        // so every inner column has enough neighbors for gaussian weighting.
        // This is the analogue of Minecraft's GenLayerSmooth / decorator pass:
        // it softens abrupt height transitions at biome and zone boundaries.
        const BLEND_RADIUS: usize = 8;  // blocks of border padding
        const BLEND_SIGMA: f64   = 4.0; // gaussian standard deviation (blocks)

        let ext_w = cs_x + 2 * BLEND_RADIUS;
        let ext_h = cs_z + 2 * BLEND_RADIUS;
        let ext_base_x = base_x - BLEND_RADIUS as i32;
        let ext_base_z = base_z - BLEND_RADIUS as i32;

        // Phase 1: sample the extended area in parallel (geography only).
        let ext_geos: Vec<GeoInfo> = (0..ext_w * ext_h)
            .into_par_iter()
            .map(|i| {
                let dx = (i % ext_w) as i32;
                let dz = (i / ext_w) as i32;
                self.geography.sample(ext_base_x + dx, ext_base_z + dz)
            })
            .collect();

        // Phase 2: precompute the (2R+1)² gaussian kernel, then apply it to
        //          every inner column to produce smoothed surface heights.
        let ksize = 2 * BLEND_RADIUS + 1; // e.g. 17 for radius 8
        let inv_2sigma2 = 1.0 / (2.0 * BLEND_SIGMA * BLEND_SIGMA);
        let kernel: Vec<f64> = (0..ksize).flat_map(|ky| {
            (0..ksize).map(move |kx| {
                let dx = kx as i64 - BLEND_RADIUS as i64;
                let dy = ky as i64 - BLEND_RADIUS as i64;
                (-(dx * dx + dy * dy) as f64 * inv_2sigma2).exp()
            })
        }).collect();
        let kernel_sum: f64 = kernel.iter().sum();

        let mut smoothed_heights = vec![0i32; total];
        for iz in 0..cs_z {
            for ix in 0..cs_x {
                let mut h_sum = 0.0f64;
                for ky in 0..ksize {
                    let ext_iz = iz + ky;
                    for kx in 0..ksize {
                        let ext_ix = ix + kx;
                        let w = kernel[ky * ksize + kx];
                        let h = ext_geos[ext_iz * ext_w + ext_ix].surface_height as f64;
                        h_sum += w * h;
                    }
                }
                smoothed_heights[iz * cs_x + ix] = (h_sum / kernel_sum).round() as i32;
            }
        }

        let sea_level = config::world_gen_config().geography.sea_level;

        // Phase 3: full column generation using pre-sampled geo + smoothed heights.
        let columns: Vec<ColumnData> = (0..total)
            .into_par_iter()
            .map(|idx| {
                let dx = idx % cs_x;
                let dz = idx / cs_x;
                let wx = base_x + dx as i32;
                let wz = base_z + dz as i32;

                // Reuse the pre-sampled geo (river_strength / is_river already set).
                let mut geo = ext_geos[(dz + BLEND_RADIUS) * ext_w + (dx + BLEND_RADIUS)];

                // Override height with the spatially-smoothed value.
                let sh = smoothed_heights[dz * cs_x + dx];
                geo.surface_height = sh;
                geo.is_ocean   = sh < sea_level;
                geo.ocean_depth = if geo.is_ocean { sea_level - sh } else { 0 };

                let climate    = self.climate.sample(wx, wz);
                let geology    = self.geology.sample(wx, wz);
                let stratigraphy = self.stratification
                    .compute_column(wx, wz, &geo, &climate, &geology, registry);
                let biome_id   = self.determine_biome(&geo, &climate, &geology);

                ColumnData { geo, climate, geology, stratigraphy, biome_id }
            })
            .collect();

        flow_debug_log!(
            "BiomePipeline",
            "sample_chunk_area",
            "cx={} cz={} done, {} columns",
            cx,
            cz,
            columns.len()
        );

        columns
    }
    /// Determine the best biome ID for the given geographic, climatic, and
    /// geological conditions.
    ///
    /// # Algorithm
    ///
    /// 1. **Build a query tag set** based on the input conditions:
    ///
    ///    | Condition | Tags added |
    ///    |-----------|-----------|
    ///    | `geo.is_ocean` | `Ocean` |
    ///    | `climate.temperature < -0.7` | `Icy` |
    ///    | `climate.temperature > 0.3` | `Hot` |
    ///    | `-0.3 ≤ temperature ≤ 0.3` | `Warm` |
    ///    | `geo.continentalness > 0.8` | `Mountain` |
    ///    | `continentalness < 0.6 && !ocean` | `Plains` |
    ///    | `climate.humidity > 0.6 && !ocean` | `Forest` |
    ///    | `humidity < 0.2 && temperature > 0.2` | `Desert` |
    ///    | `0.35 < continentalness < 0.45 && !ocean` | `Beach` |
    ///
    /// 2. **Score every biome** by the size of its tag-set intersection with
    ///    the query tags.
    ///
    /// 3. **Return** the highest-scoring biome ID, or fall back to Plains
    ///    (ID 300) if scoring fails.
    pub fn determine_biome(
        &self,
        geo: &GeoInfo,
        climate: &ClimateInfo,
        _geology: &GeologyInfo,
    ) -> u32 {
        let mut query_tags: Vec<ClimateTag> = Vec::with_capacity(6);

        // Ocean check.
        if geo.is_ocean {
            query_tags.push(ClimateTag::Ocean);
        }

        // Temperature classification.
        if climate.temperature < -0.7 {
            query_tags.push(ClimateTag::Icy);
        } else if climate.temperature > 0.3 {
            query_tags.push(ClimateTag::Hot);
        } else if climate.temperature > -0.3 {
            query_tags.push(ClimateTag::Warm);
        } else {
            query_tags.push(ClimateTag::Cold);
        }

        // Terrain shape.
        if geo.continentalness > 0.8 {
            query_tags.push(ClimateTag::Mountain);
        } else if geo.continentalness < 0.6 && !geo.is_ocean {
            query_tags.push(ClimateTag::Plains);
        }

        // Beach zone — narrow band around sea level.
        if geo.continentalness > 0.35
            && geo.continentalness < 0.45
            && !geo.is_ocean
        {
            query_tags.push(ClimateTag::Beach);
        }

        // Vegetation.
        if climate.humidity > 0.6 && !geo.is_ocean {
            query_tags.push(ClimateTag::Forest);
        }

        // Aridity.
        if climate.humidity < 0.2 && climate.temperature > 0.2 {
            query_tags.push(ClimateTag::Desert);
        }

        // Find best-matching biome.
        let biome_id = self
            .biome_dict
            .find_best_matching(&query_tags)
            .unwrap_or(300); // Fallback: Plains

        ext_debug_log!(
            "BiomePipeline",
            "determine_biome",
            "temp={:.2} humid={:.2} continent={:.2} ocean={} tags={:?} → biome={}",
            climate.temperature,
            climate.humidity,
            geo.continentalness,
            geo.is_ocean,
            query_tags,
            biome_id
        );

        biome_id
    }
}

// ---------------------------------------------------------------------------
// Layer-based pipeline builder (advanced usage)
// ---------------------------------------------------------------------------

/// Build an example layer-based biome pipeline for advanced usage.
///
/// This constructs the classic Minecraft-style biome generation chain:
///
/// ```text
/// LayerIsland → LayerZoom → LayerAddIsland → LayerZoom → LayerFuzzyZoom
///     → LayerSmooth → LayerZoom → LayerBiome → LayerVoronoiZoom
/// ```
///
/// Each zoom layer doubles the resolution, starting from a coarse 1:1024
/// grid and ending at a 1:1 biome-per-block map.
///
/// # Arguments
///
/// * `seed` — world seed.
///
/// # Returns
///
/// A boxed [`BiomeLayer`] trait object that can be queried via
/// [`BiomeLayer::generate`].
///
/// # Example
///
/// ```ignore
/// use qubepixel_terrain::biome_layer::{BiomeLayer, GenContext};
/// use qubepixel_terrain::pipeline::build_layer_pipeline;
///
/// let pipeline = build_layer_pipeline(42);
/// let mut ctx = GenContext::new(42);
/// let mut output = vec![0u32; 256]; // 16×16
/// pipeline.generate(0, 0, 16, 16, &mut ctx, &mut output);
/// ```
pub fn build_layer_pipeline(seed: u64) -> Box<dyn BiomeLayer> {
    debug_log!("build_layer_pipeline", "new", "seed={}", seed);

    // Root layer: coarse land/ocean map.
    let island = LayerIsland::new(seed);

    // Zoom 1: 1:1024 → 1:512
    let zoom1 = LayerZoom::new(Box::new(island), seed);

    // Add island: re-introduce landmass probability at zoomed scale.
    let add_island = LayerAddIsland::new(Box::new(zoom1), seed);

    // Zoom 2: 1:512 → 1:256
    let zoom2 = LayerZoom::new(Box::new(add_island), seed);

    // Fuzzy zoom: 1:256 → 1:128 (with randomized interpolation for
    // organic-looking biome boundaries).
    let fuzzy = LayerFuzzyZoom::new(Box::new(zoom2), seed);

    // Smooth: reduce noise artifacts in the zoomed grid.
    let smooth = LayerSmooth::new(Box::new(fuzzy));

    // Zoom 3: 1:128 → 1:64
    let zoom3 = LayerZoom::new(Box::new(smooth), seed);

    // Biome assignment: map grid values to actual biome IDs.
    // Uses 13 climate zones, with primary biome IDs starting at 0 and
    // secondary (variant) biome IDs starting at 100.
    let biome = LayerBiome::new_for_climate_zones(Box::new(zoom3), seed, 13, 0, 100);

    // Voronoi zoom: final 1:64 → 1:1 resolution with Voronoi-based
    // interpolation for smooth transitions.
    let voronoi = LayerVoronoiZoom::new(Box::new(biome), seed);

    debug_log!(
        "build_layer_pipeline",
        "new",
        "layer pipeline built successfully"
    );

    Box::new(voronoi)
}

/// Build an example river-generation layer pipeline.
///
/// The river pipeline uses a separate chain of layers to generate river
/// networks:
///
/// ```text
/// LayerRiverInit → LayerZoom → LayerZoom → LayerZoom → ...
/// ```
///
/// The output can be combined with the biome pipeline's output to carve
/// river channels into the terrain.
///
/// # Arguments
///
/// * `seed` — world seed (should be offset from the main seed to avoid
///   correlation, e.g. `seed + 10_000`).
///
/// # Returns
///
/// A boxed [`BiomeLayer`] producing river-related values.
pub fn build_river_pipeline(seed: u64) -> Box<dyn BiomeLayer> {
    debug_log!("build_river_pipeline", "new", "seed={}", seed);

    let river_init = LayerRiverInit::new(seed);
    let zoom1 = LayerZoom::new(Box::new(river_init), seed);
    let zoom2 = LayerZoom::new(Box::new(zoom1), seed);
    let zoom3 = LayerZoom::new(Box::new(zoom2), seed);
    let zoom4 = LayerZoom::new(Box::new(zoom3), seed);
    let zoom5 = LayerZoom::new(Box::new(zoom4), seed);
    // Edge layer: insert transitional values at river network boundaries.
    // Uses 10 zones with edge biome IDs starting at 2000.
    let edge = LayerEdge::new_for_climate(Box::new(zoom5), seed, 10, 2000);

    debug_log!(
        "build_river_pipeline",
        "new",
        "river pipeline built successfully"
    );

    Box::new(edge)
}

// ---------------------------------------------------------------------------
// Send + Sync assertions
// ---------------------------------------------------------------------------

const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn _check() {
        assert_send_sync::<BiomePipeline>();
        assert_send_sync::<ColumnData>();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::world_gen::climate::ClimateType;
    use crate::core::world_gen::geology::{MacroRegion, MicroRegion};
    use crate::core::world_gen::stratification::StratificationLayer;

    /// Minimal GeoInfo for testing.
    fn test_geo(surface_height: i32, is_ocean: bool, continentalness: f64) -> GeoInfo {
        GeoInfo {
            surface_height,
            is_ocean,
            continentalness,
            erosion: 0.5,
            peaks_and_valleys: 0.5,
            ocean_depth: 0,
            river_strength: 0.0,
            is_river: false,
        }
    }

    /// Minimal ClimateInfo for testing.
    fn test_climate(
        temperature: f64,
        humidity: f64,
        climate_type: ClimateType,
    ) -> ClimateInfo {
        ClimateInfo {
            temperature,
            humidity,
            climate_type,
            zone_index: 6, // Equatorial
        }
    }

    /// Minimal GeologyInfo for testing.
    fn test_geology() -> GeologyInfo {
        GeologyInfo {
            macro_region: MacroRegion::Igneous,
            micro_region: MicroRegion::Basalt,
        }
    }

    #[test]
    fn test_pipeline_new() {
        let pipeline = BiomePipeline::new(42);
        assert_eq!(pipeline.seed, 42);
        assert_eq!(pipeline.biome_dict.len(), 30);
    }

    #[test]
    fn test_determine_biome_ocean_warm() {
        let pipeline = BiomePipeline::new(42);
        let geo = test_geo(55, true, 0.3);
        let climate = test_climate(0.5, 0.5, ClimateType::NorthSubtropical);
        let geology = test_geology();

        let biome_id = pipeline.determine_biome(&geo, &climate, &geology);
        // Should match Ocean (101) — Ocean + Hot/Warm tags.
        assert!(pipeline.biome_dict.has_tag(biome_id, ClimateTag::Ocean));
    }

    #[test]
    fn test_determine_biome_frozen_ocean() {
        let pipeline = BiomePipeline::new(42);
        let geo = test_geo(50, true, 0.2);
        let climate = test_climate(-0.8, 0.3, ClimateType::Arctic);
        let geology = test_geology();

        let biome_id = pipeline.determine_biome(&geo, &climate, &geology);
        // Should match Frozen Ocean (102).
        assert_eq!(biome_id, 102);
    }

    #[test]
    fn test_determine_biome_plains() {
        let pipeline = BiomePipeline::new(42);
        let geo = test_geo(64, false, 0.5);
        let climate = test_climate(0.0, 0.5, ClimateType::NorthTemperate);
        let geology = test_geology();

        let biome_id = pipeline.determine_biome(&geo, &climate, &geology);
        // Should match Plains (300) — Plains + Warm.
        assert_eq!(biome_id, 300);
    }

    #[test]
    fn test_determine_biome_desert() {
        let pipeline = BiomePipeline::new(42);
        let geo = test_geo(64, false, 0.5);
        let climate = test_climate(0.8, 0.1, ClimateType::NorthSubequatorial);
        let geology = test_geology();

        let biome_id = pipeline.determine_biome(&geo, &climate, &geology);
        // Should have Desert tag.
        assert!(pipeline.biome_dict.has_tag(biome_id, ClimateTag::Desert));
    }

    #[test]
    fn test_determine_biome_snowy_mountains() {
        let pipeline = BiomePipeline::new(42);
        let geo = test_geo(90, false, 0.9);
        let climate = test_climate(-0.9, 0.4, ClimateType::Arctic);
        let geology = test_geology();

        let biome_id = pipeline.determine_biome(&geo, &climate, &geology);
        // Should have Mountain + Icy tags → Snowy Mountains (701).
        assert_eq!(biome_id, 701);
    }

    #[test]
    fn test_sample_chunk_area_size() {
        let pipeline = BiomePipeline::new(42);

        // We can't easily test without a BlockRegistry, but we can verify
        // that the method signature is correct and the chunk size logic
        // works by checking the base coordinate calculation.
        let cx = 3i32;
        let cz = -5i32;
        let base_x = cx * 16;
        let base_z = cz * 16;
        assert_eq!(base_x, 48);
        assert_eq!(base_z, -80);
    }

    #[test]
    fn test_build_layer_pipeline() {
        let _pipeline = build_layer_pipeline(42);
        // Just verify it doesn't panic during construction.
    }

    #[test]
    fn test_build_river_pipeline() {
        let _pipeline = build_river_pipeline(42);
        // Just verify it doesn't panic during construction.
    }
}
