// =============================================================================
// QubePixel — AppConfig (global atomic settings shared across threads)
// =============================================================================

use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::OnceLock;
use serde::{Deserialize, Serialize};
use crate::debug_log;

// ---------------------------------------------------------------------------
// Lighting systems removed — see archiveLighting/ for original code.
// LightingMode, voxel params, SSAO, Clustered Forward toggles archived.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Chunk dimensions — loaded once from assets/chunk_config.json
// ---------------------------------------------------------------------------

/// Per-axis chunk size in blocks. Default: 16×256×16 (columnar).
/// Dimensions should be multiples of 4 so LOD steps (×2, ×4) divide evenly.
#[derive(Debug, Deserialize, Clone)]
pub struct ChunkDimensions {
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
}

impl Default for ChunkDimensions {
    fn default() -> Self {
        Self { size_x: 16, size_y: 256, size_z: 16 }
    }
}

static CHUNK_DIMS: OnceLock<ChunkDimensions> = OnceLock::new();

/// Returns the global chunk dimensions, loading from JSON on first call.
pub fn chunk_dims() -> &'static ChunkDimensions {
    CHUNK_DIMS.get_or_init(|| {
        let dims: ChunkDimensions = std::fs::read_to_string("assets/chunk_config.json")
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        debug_log!(
            "Config", "chunk_dims",
            "Chunk dimensions: {}x{}x{}", dims.size_x, dims.size_y, dims.size_z
        );
        dims
    })
}

#[inline(always)] pub fn chunk_size_x() -> usize { chunk_dims().size_x }
#[inline(always)] pub fn chunk_size_y() -> usize { chunk_dims().size_y }
#[inline(always)] pub fn chunk_size_z() -> usize { chunk_dims().size_z }

/// Render distance in chunks (horizontal). Range: 1..=128.
pub static RENDER_DISTANCE: AtomicI32 = AtomicI32::new(12);

/// LOD distance multiplier × 100 (e.g. 100 = 1.0×, 150 = 1.5×).
/// Controls how far LOD0/LOD1 zones extend before switching to coarser LOD.
///   LOD0 (full detail):  Chebyshev dist ≤ LOD_NEAR_BASE  × multiplier
///   LOD1 (2× coarser):   Chebyshev dist ≤ LOD_FAR_BASE   × multiplier
///   LOD2 (4× coarser):   everything beyond LOD1 within render distance
pub static LOD_MULTIPLIER: AtomicU32 = AtomicU32::new(300);

/// Base Chebyshev-distance thresholds (in chunks) before multiplier is applied.
pub const LOD_NEAR_BASE: f32 = 3.0;
pub const LOD_FAR_BASE: f32  = 6.0;
/// Vertical chunk loading range — chunks below camera's chunk Y.
/// Default 0: with 256-height columnar chunks the whole world fits in one vertical layer.
pub static VERTICAL_BELOW: AtomicI32 = AtomicI32::new(0);
/// Vertical chunk loading range — chunks above camera's chunk Y.
pub static VERTICAL_ABOVE: AtomicI32 = AtomicI32::new(0);

/// Whether the biome ambient tint is applied to the camera's ambient lighting.
/// 1 = enabled (default), 0 = disabled.
/// Hot biomes → warm yellow cast; icy biomes → cool blue cast.
pub static BIOME_AMBIENT_TINT_ENABLED: AtomicU32 = AtomicU32::new(1);

/// Whether voxel DDA shadow from the sun/moon is applied. 1 = enabled (default).
pub static SHADOW_SUN_ENABLED: AtomicU32 = AtomicU32::new(1);

/// Whether voxel DDA shadow from block light sources (point/spot lights) is applied. 1 = enabled (default).
pub static SHADOW_BLOCK_ENABLED: AtomicU32 = AtomicU32::new(1);

/// GI quality mode (persisted in user settings).
/// 0 = Off    — skips all compute passes; ~4× GPU saving.
/// 1 = Mono16 — 16 propagation steps, monochromatic (Minecraft-style, ~16 block radius).
/// 2 = Mono32 — 32 propagation steps, monochromatic (~32 block radius).
/// 3 = RGB32  — 32 propagation steps, full colour (~32 block radius).
/// 4 = Full   — 64 propagation steps, full colour (~64 block radius, current maximum).
pub static GI_MODE: AtomicU32 = AtomicU32::new(4);

/// Number of DDA ray-march steps for shadow rays. Valid presets: 8, 16, 32, 64.
pub static SHADOW_QUALITY: AtomicU32 = AtomicU32::new(64);

/// Whether light halos (billboard glow quads around light sources) are rendered. 1 = enabled.
pub static HALO_ENABLED: AtomicU32 = AtomicU32::new(1);

/// Whether volumetric god rays (spot-light beam quads) are rendered. 1 = enabled.
pub static VOLUMETRIC_RAYS_ENABLED: AtomicU32 = AtomicU32::new(1);

/// Mouse sensitivity × 10000 (e.g. 30 = 0.003). Controls camera rotation speed.
pub static MOUSE_SENSITIVITY: AtomicU32 = AtomicU32::new(30);

/// Fly speed in blocks/second (default 12). Persisted in user settings.
pub static FLY_SPEED: AtomicU32 = AtomicU32::new(12);

// ---------------------------------------------------------------------------
// Render distance
// ---------------------------------------------------------------------------
pub fn render_distance() -> i32 {
    RENDER_DISTANCE.load(Ordering::Relaxed).clamp(1, 128)
}

pub fn set_render_distance(v: i32) {
    let clamped = v.clamp(1, 128);
    RENDER_DISTANCE.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_render_distance", "Render distance set to {}", clamped);
}

// ---------------------------------------------------------------------------
// LOD multiplier
// ---------------------------------------------------------------------------

/// Returns LOD multiplier as float (1.0 = default).
pub fn lod_multiplier() -> f32 {
    LOD_MULTIPLIER.load(Ordering::Relaxed) as f32 / 100.0
}

/// Set LOD multiplier. Value is clamped to 0.25..=4.0.
pub fn set_lod_multiplier(v: f32) {
    let clamped = (v * 100.0).clamp(25.0, 400.0) as u32;
    LOD_MULTIPLIER.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_lod_multiplier", "LOD multiplier set to {:.2}", clamped as f32 / 100.0);
}
// ---------------------------------------------------------------------------
// Vertical chunk range
// ---------------------------------------------------------------------------

pub fn vertical_below() -> i32 {
    VERTICAL_BELOW.load(Ordering::Relaxed).clamp(0, 16)
}

pub fn set_vertical_below(v: i32) {
    let clamped = v.clamp(0, 16);
    VERTICAL_BELOW.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_vertical_below", "Vertical below set to {}", clamped);
}

pub fn vertical_above() -> i32 {
    VERTICAL_ABOVE.load(Ordering::Relaxed).clamp(0, 16)
}

pub fn set_vertical_above(v: i32) {
    let clamped = v.clamp(0, 16);
    VERTICAL_ABOVE.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_vertical_above", "Vertical above set to {}", clamped);
}

pub fn biome_ambient_tint_enabled() -> bool {
    BIOME_AMBIENT_TINT_ENABLED.load(Ordering::Relaxed) != 0
}

pub fn set_biome_ambient_tint_enabled(v: bool) {
    BIOME_AMBIENT_TINT_ENABLED.store(v as u32, Ordering::Relaxed);
}

pub fn shadow_sun_enabled() -> bool {
    SHADOW_SUN_ENABLED.load(Ordering::Relaxed) != 0
}

pub fn set_shadow_sun_enabled(v: bool) {
    SHADOW_SUN_ENABLED.store(v as u32, Ordering::Relaxed);
}

pub fn shadow_block_enabled() -> bool {
    SHADOW_BLOCK_ENABLED.load(Ordering::Relaxed) != 0
}

pub fn set_shadow_block_enabled(v: bool) {
    SHADOW_BLOCK_ENABLED.store(v as u32, Ordering::Relaxed);
}

pub fn gi_mode() -> u32 {
    GI_MODE.load(Ordering::Relaxed).clamp(0, 4)
}

pub fn set_gi_mode(v: u32) {
    let clamped = v.clamp(0, 4);
    GI_MODE.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_gi_mode", "GI mode set to {}", clamped);
}

pub fn gi_enabled() -> bool {
    gi_mode() != 0
}

pub fn set_gi_enabled(v: bool) {
    set_gi_mode(if v { 4 } else { 0 });
}

/// DDA shadow ray-march step count. Clamped to [8, 64].
pub fn shadow_quality() -> u32 {
    SHADOW_QUALITY.load(Ordering::Relaxed).clamp(8, 64)
}

pub fn set_shadow_quality(v: u32) {
    SHADOW_QUALITY.store(v.clamp(8, 64), Ordering::Relaxed);
}

pub fn halo_enabled() -> bool {
    HALO_ENABLED.load(Ordering::Relaxed) != 0
}

pub fn set_halo_enabled(v: bool) {
    HALO_ENABLED.store(v as u32, Ordering::Relaxed);
}

pub fn volumetric_rays_enabled() -> bool {
    VOLUMETRIC_RAYS_ENABLED.load(Ordering::Relaxed) != 0
}

pub fn set_volumetric_rays_enabled(v: bool) {
    VOLUMETRIC_RAYS_ENABLED.store(v as u32, Ordering::Relaxed);
}

pub fn mouse_sensitivity() -> f32 {
    MOUSE_SENSITIVITY.load(Ordering::Relaxed) as f32 / 10000.0
}

pub fn set_mouse_sensitivity(v: f32) {
    let clamped = (v * 10000.0).clamp(5.0, 100.0) as u32;
    MOUSE_SENSITIVITY.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_mouse_sensitivity", "Mouse sensitivity set to {:.4}", clamped as f32 / 10000.0);
}

pub fn fly_speed() -> f32 {
    FLY_SPEED.load(Ordering::Relaxed).clamp(2, 100) as f32
}

pub fn set_fly_speed(v: f32) {
    let clamped = v.clamp(2.0, 100.0) as u32;
    FLY_SPEED.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_fly_speed", "Fly speed set to {}", clamped);
}

/// Compute the LOD level for a chunk at the given chunk coordinates.
/// Returns 0 (full), 1 (2× coarser), or 2 (4× coarser).
pub fn compute_lod_level(cam_cx: i32, cam_cz: i32, cx: i32, cz: i32) -> u8 {
    let dist = (cx - cam_cx).abs().max((cz - cam_cz).abs());
    let mult = lod_multiplier();
    let lod0_range = (LOD_NEAR_BASE * mult).ceil() as i32;
    let lod1_range = (LOD_FAR_BASE * mult).ceil() as i32;

    if dist <= lod0_range { 0 }
    else if dist <= lod1_range { 1 }
    else { 2 }
}

// ---------------------------------------------------------------------------
// World generation config — loaded once from assets/world_gen_config.json
// ---------------------------------------------------------------------------

fn default_river_scale() -> f64 { 400.0 }
fn default_river_threshold() -> f64 { 0.06 }
fn default_river_carve_depth() -> f64 { 8.0 }
fn default_tree_density_multiplier() -> f64 { 1.0 }

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GeographyConfig {
    pub sea_level: i32,
    pub terrain_base: i32,
    pub terrain_amplitude: i32,
    pub continent_scale: f64,
    pub erosion_scale: f64,
    pub peaks_scale: f64,
    pub island_scale: f64,
    pub continent_octaves: usize,
    pub erosion_octaves: usize,
    pub peaks_octaves: usize,
    pub island_threshold: f64,
    pub island_boost: f64,
    pub deep_ocean_max: f64,
    pub ocean_max: f64,
    pub lowland_max: f64,
    pub midland_max: f64,
    /// Noise scale for river network (larger = rivers spaced further apart).
    #[serde(default = "default_river_scale")]
    pub river_scale: f64,
    /// Half-band threshold around noise zero-crossings that forms river channels.
    /// Larger values = wider rivers.
    #[serde(default = "default_river_threshold")]
    pub river_threshold: f64,
    /// Maximum depth (blocks) carved into the terrain for river channels.
    #[serde(default = "default_river_carve_depth")]
    pub river_carve_depth: f64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ClimateConfig {
    pub band_scale: f64,
    pub boundary_noise_scale: f64,
    pub boundary_warp_amount: f64,
    pub humidity_noise_scale: f64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct GeologyConfig {
    pub macro_cell_size: i32,
    pub micro_cell_size: i32,
    pub mantle_keep_denominator: u64,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StratificationConfig {
    pub deep_granite_threshold: i32,
    pub deep_granite_transition: i32,
    pub thickness_noise_scale: f64,
    pub dirt_thickness_base: u8,
    pub dirt_thickness_variation: u8,
    pub clay_thickness_base: u8,
    pub clay_thickness_variation: u8,
    pub claystone_thickness_base: u8,
    pub claystone_thickness_variation: u8,
}

/// Nature / vegetation generation parameters.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NatureConfig {
    /// Global multiplier applied to every biome's tree density (0.0 = no trees, 1.0 = default, 2.0 = double).
    #[serde(default = "default_tree_density_multiplier")]
    pub tree_density_multiplier: f64,
}

impl Default for NatureConfig {
    fn default() -> Self {
        Self { tree_density_multiplier: 1.0 }
    }
}

#[derive(Debug, Deserialize, Clone)]
struct TestOverrides {
    geography: Option<serde_json::Value>,
    climate: Option<serde_json::Value>,
    geology: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Clone)]
struct WorldGenConfigFile {
    test_mode: Option<bool>,
    geography: GeographyConfig,
    climate: ClimateConfig,
    geology: GeologyConfig,
    stratification: StratificationConfig,
    nature: Option<NatureConfig>,
    test_overrides: Option<TestOverrides>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorldGenConfig {
    pub test_mode: bool,
    pub geography: GeographyConfig,
    pub climate: ClimateConfig,
    pub geology: GeologyConfig,
    pub stratification: StratificationConfig,
    pub nature: NatureConfig,
}

impl Default for WorldGenConfig {
    fn default() -> Self {
        Self {
            test_mode: false,
            geography: GeographyConfig {
                sea_level: 64, terrain_base: 50, terrain_amplitude: 40,
                continent_scale: 3000.0, erosion_scale: 1500.0, peaks_scale: 3000.0,
                island_scale: 200.0, continent_octaves: 4, erosion_octaves: 4,
                peaks_octaves: 4, island_threshold: 0.4, island_boost: 0.3,
                deep_ocean_max: 0.2, ocean_max: 0.4, lowland_max: 0.6, midland_max: 0.8,
                river_scale: 400.0, river_threshold: 0.06, river_carve_depth: 8.0,
            },
            climate: ClimateConfig {
                band_scale: 16000.0, boundary_noise_scale: 0.00005,
                boundary_warp_amount: 1500.0, humidity_noise_scale: 500.0,
            },
            geology: GeologyConfig {
                macro_cell_size: 2048, micro_cell_size: 512, mantle_keep_denominator: 5,
            },
            stratification: StratificationConfig {
                deep_granite_threshold: 32, deep_granite_transition: 16,
                thickness_noise_scale: 0.02,
                dirt_thickness_base: 4, dirt_thickness_variation: 1,
                clay_thickness_base: 4, clay_thickness_variation: 2,
                claystone_thickness_base: 10, claystone_thickness_variation: 10,
            },
            nature: NatureConfig::default(),
        }
    }
}

static WORLD_GEN_CONFIG: OnceLock<WorldGenConfig> = OnceLock::new();

pub fn world_gen_config() -> &'static WorldGenConfig {
    WORLD_GEN_CONFIG.get_or_init(|| {
        let raw = match std::fs::read_to_string("assets/world_gen_config.json") {
            Ok(s) => s,
            Err(_) => {
                debug_log!("Config", "world_gen_config", "No world_gen_config.json found, using defaults");
                return WorldGenConfig::default();
            }
        };

        let file: WorldGenConfigFile = match serde_json::from_str(&raw) {
            Ok(f) => f,
            Err(e) => {
                debug_log!("Config", "world_gen_config", "Failed to parse world_gen_config.json: {}", e);
                return WorldGenConfig::default();
            }
        };

        let test_mode = file.test_mode.unwrap_or(false);
        let mut cfg = WorldGenConfig {
            test_mode,
            geography: file.geography,
            climate: file.climate,
            geology: file.geology,
            stratification: file.stratification,
            nature: file.nature.unwrap_or_default(),
        };

        // Apply test overrides when test_mode is enabled.
        if test_mode {
            if let Some(overrides) = &file.test_overrides {
                if let Some(geo_val) = &overrides.geography {
                    if let Some(v) = geo_val.get("continent_scale").and_then(|v| v.as_f64()) { cfg.geography.continent_scale = v; }
                    if let Some(v) = geo_val.get("erosion_scale").and_then(|v| v.as_f64()) { cfg.geography.erosion_scale = v; }
                    if let Some(v) = geo_val.get("peaks_scale").and_then(|v| v.as_f64()) { cfg.geography.peaks_scale = v; }
                    if let Some(v) = geo_val.get("island_scale").and_then(|v| v.as_f64()) { cfg.geography.island_scale = v; }
                }
                if let Some(clim_val) = &overrides.climate {
                    if let Some(v) = clim_val.get("band_scale").and_then(|v| v.as_f64()) { cfg.climate.band_scale = v; }
                    if let Some(v) = clim_val.get("boundary_noise_scale").and_then(|v| v.as_f64()) { cfg.climate.boundary_noise_scale = v; }
                    if let Some(v) = clim_val.get("boundary_warp_amount").and_then(|v| v.as_f64()) { cfg.climate.boundary_warp_amount = v; }
                    if let Some(v) = clim_val.get("humidity_noise_scale").and_then(|v| v.as_f64()) { cfg.climate.humidity_noise_scale = v; }
                }
                if let Some(geo_val) = &overrides.geology {
                    if let Some(v) = geo_val.get("macro_cell_size").and_then(|v| v.as_i64()) { cfg.geology.macro_cell_size = v as i32; }
                    if let Some(v) = geo_val.get("micro_cell_size").and_then(|v| v.as_i64()) { cfg.geology.micro_cell_size = v as i32; }
                }
            }
            debug_log!("Config", "world_gen_config", "TEST MODE enabled — scales reduced for debugging");
        }

        debug_log!(
            "Config", "world_gen_config",
            "Loaded: test_mode={} continent_scale={:.0} band_scale={:.0} macro_cell={}",
            cfg.test_mode, cfg.geography.continent_scale,
            cfg.climate.band_scale, cfg.geology.macro_cell_size
        );

        cfg
    })
}
