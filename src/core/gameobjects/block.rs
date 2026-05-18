// =============================================================================
// QubePixel — BlockDefinition + BlockRegistry  (loaded from JSON files)
// =============================================================================
//
// Loading pipeline:
//   1. Read  assets/blocks/block_registry.json  → list of block IDs
//   2. For each ID, read  assets/blocks/<id>.json  → full block definition
//   3. If the assets directory is missing, fall back to embedded defaults
//      so the game always works even without external files.
//
// Adding a new block:
//   1. Open  assets/blocks/block_registry.json  and add the block's ID
//      to the "blocks" array.
//   2. Create  assets/blocks/<id>.json  with the desired properties
//      (see any existing block file for the schema).
//   3. Restart the game — the new block is available immediately.
//
// Block IDs:
//   Numeric ID 0 is always Air (implicit, not in any file).
//   Numeric IDs 1..=255 are assigned in registry order (first entry → 1, …).
//
// Face system (6 faces per block):
//   Index: 0 = +X (East),  1 = -X (West),
//          2 = +Y (Top),   3 = -Y (Bottom),
//          4 = +Z (South), 5 = -Z (North)
//
//   Face texture priority (most specific first):
//     N/S/E/W faces: direction → sides → all → block base
//     Top face:      top → all → block base
//     Bottom face:   bottom → all → block base
//
//   Layer system:
//     Each FaceOverride supports up to 5 alpha-blended layers (layer1–layer5).
//     Layers are composited onto the base texture at load time ("baked").
//     The resulting texture key is stored in `resolved_texture` and used at
//     render time via `texture_for_face()`.
// =============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::debug_log;

// ---------------------------------------------------------------------------
// FaceOverride — per-face texture, color, and layer overrides
// ---------------------------------------------------------------------------
/// A single face configuration with optional texture layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceOverride {
    /// RGB colour (0..1). Falls back to block's base `color` when absent.
    #[serde(default)]
    pub color: [f32; 3],
    /// Name of the texture in the atlas (e.g. "grass_top", "dirt").
    /// If absent, this face uses vertex colour only (white tile in atlas).
    #[serde(default)]
    pub texture: Option<String>,
    /// Alpha-blended overlay layers (1 = closest to base, 5 = topmost).
    /// Each value is a texture name from the atlas. Layers are composited
    /// at load time via `BlockRegistry::bake_layered_textures()` and the
    /// result is stored in `resolved_texture`.
    #[serde(default)]
    pub layer1: Option<String>,
    #[serde(default)]
    pub layer2: Option<String>,
    #[serde(default)]
    pub layer3: Option<String>,
    #[serde(default)]
    pub layer4: Option<String>,
    #[serde(default)]
    pub layer5: Option<String>,
    /// Runtime-only: baked texture key after layer compositing.
    /// Set by `BlockRegistry::bake_layered_textures()`.
    /// When set, `texture_for_face()` returns this instead of `texture`.
    #[serde(skip)]
    pub resolved_texture: Option<String>,
}

/// Per-face overrides.
///
/// Priority resolution for texture lookups (most specific first):
///   - Horizontal faces (N/S/E/W): direction → `sides` → `all` → block base
///   - Vertical faces (Top/Bottom):  direction → `all` → block base
///
/// All fields are optional — missing entries fall back to the next priority
/// level, and ultimately to the block's base `color` / `texture`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct FaceColors {
    /// Apply to ALL 6 faces (lowest face-specific priority).
    #[serde(default)]
    pub all:    Option<FaceOverride>,
    /// +Y face (top).
    #[serde(default)]
    pub top:    Option<FaceOverride>,
    /// -Y face (bottom).
    #[serde(default)]
    pub bottom: Option<FaceOverride>,
    /// -Z face (north).
    #[serde(default)]
    pub north:  Option<FaceOverride>,
    /// +Z face (south).
    #[serde(default)]
    pub south:  Option<FaceOverride>,
    /// +X face (east).
    #[serde(default)]
    pub east:   Option<FaceOverride>,
    /// -X face (west).
    #[serde(default)]
    pub west:   Option<FaceOverride>,
    /// Convenience: applies to all 4 horizontal faces (N/S/E/W).
    /// Lower priority than a specific direction override but higher than `all`.
    #[serde(default)]
    pub sides:  Option<FaceOverride>,
}

// ---------------------------------------------------------------------------
// Face resolution helper
// ---------------------------------------------------------------------------

/// Resolve the effective `FaceOverride` for a given face direction.
///
/// `face`: 0 = +X (East), 1 = -X (West), 2 = +Y (Top), 3 = -Y (Bottom),
///         4 = +Z (South), 5 = -Z (North).
fn resolve_face_override(faces: &FaceColors, face: u8) -> Option<&FaceOverride> {
    match face {
        // Top: top → all
        2 => faces.top.as_ref().or(faces.all.as_ref()),
        // Bottom: bottom → all
        3 => faces.bottom.as_ref().or(faces.all.as_ref()),
        // East (+X): east → sides → all
        0 => faces.east.as_ref()
            .or(faces.sides.as_ref())
            .or(faces.all.as_ref()),
        // West (-X): west → sides → all
        1 => faces.west.as_ref()
            .or(faces.sides.as_ref())
            .or(faces.all.as_ref()),
        // South (+Z): south → sides → all
        4 => faces.south.as_ref()
            .or(faces.sides.as_ref())
            .or(faces.all.as_ref()),
        // North (-Z): north → sides → all
        5 => faces.north.as_ref()
            .or(faces.sides.as_ref())
            .or(faces.all.as_ref()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// MaterialProperties — PBR material parameters
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    /// Base albedo colour for PBR (0..1 per channel).
    #[serde(default = "default_albedo")]
    pub albedo: [f32; 3],
    /// Roughness: 0.0 (mirror-smooth) — 1.0 (fully rough).
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    /// Metalness: 0.0 (dielectric) — 1.0 (metal).
    #[serde(default)]
    pub metalness: f32,
    /// Ambient occlusion multiplier: 0.0 (fully occluded) — 1.0 (none).
    #[serde(default = "default_ao")]
    pub ao: f32,
}

impl Default for MaterialProperties {
    fn default() -> Self {
        Self {
            albedo:    [0.5, 0.5, 0.5],
            roughness: 0.9,
            metalness: 0.0,
            ao:        1.0,
        }
    }
}

fn default_albedo()    -> [f32; 3] { [0.5, 0.5, 0.5] }
fn default_roughness() -> f32      { 0.9 }
fn default_ao()        -> f32      { 1.0 }

// ---------------------------------------------------------------------------
// EmissionProperties — light-emitting block parameters
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionProperties {
    /// Whether this block emits light at all.
    #[serde(default)]
    pub emit_light: bool,
    /// RGB colour of the emitted light (0..1).
    #[serde(default = "default_white")]
    pub light_color: [f32; 3],
    /// Light strength / radius of influence (0..15).
    #[serde(default)]
    pub light_strength: f32,
    /// Brightness intensity multiplier.
    #[serde(default)]
    pub light_intensity: f32,
    /// Maximum distance (in blocks) that emitted light can travel.
    /// Controls how far the VCT flood-fill propagation reaches from this source.
    /// Range: 1..64 blocks. Default 24 matches the legacy propagation step count.
    #[serde(default = "default_light_range")]
    pub light_range: f32,
    /// If true, this block's surface does not receive GI from nearby emissive
    /// probes — prevents the block from tinting itself with its own emitted
    /// colour (e.g. glowstone turning extra-yellow from self-illumination).
    #[serde(default)]
    pub no_self_gi: bool,
}

impl Default for EmissionProperties {
    fn default() -> Self {
        Self {
            emit_light:      false,
            light_color:     [1.0, 1.0, 1.0],
            light_strength:  0.0,
            light_intensity: 0.0,
            light_range:     default_light_range(),
            no_self_gi:      false,
        }
    }
}

fn default_white() -> [f32; 3] { [1.0, 1.0, 1.0] }
fn default_light_range() -> f32 { 24.0 }

// ---------------------------------------------------------------------------
// BlockLightSource — inline point/directional light source on a block
// ---------------------------------------------------------------------------

/// The shape of the emitted light cone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LightKind {
    /// Radiates uniformly in all directions.
    #[default]
    Point,
    /// Emits a cone of light in a specific direction.
    Directional,
}

/// A light source positioned relative to the block's unit cube ([0,1]³).
/// Supports both omni-directional (point) and focused (directional) lights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockLightSource {
    /// Type of light: point (omni) or directional (cone).
    #[serde(default)]
    pub kind: LightKind,

    /// Position relative to the block centre (−0.5..0.5 per axis).
    /// [0,0,0] = block centre; [0,0.5,0] = top face centre.
    #[serde(default)]
    pub position: [f32; 3],

    /// RGB color of the emitted light (0..1 per channel).
    #[serde(default = "default_white")]
    pub color: [f32; 3],

    /// Brightness multiplier (0..10, default: 1.0).
    #[serde(default = "default_light_intensity_one")]
    pub intensity: f32,

    /// Maximum reach in blocks (point lights only, 0 = use global default).
    #[serde(default)]
    pub range: f32,

    /// Normalized direction vector for directional lights (points *toward* lit area).
    /// Ignored for Point lights.
    #[serde(default = "default_down")]
    pub direction: [f32; 3],

    /// Half-angle of the light cone in degrees (0 = pencil, 90 = hemisphere).
    /// Only used for Directional lights.
    #[serde(default = "default_focus")]
    pub focus: f32,

    /// Optional label shown in the editor.
    #[serde(default)]
    pub label: String,
}

impl Default for BlockLightSource {
    fn default() -> Self {
        Self {
            kind:      LightKind::Point,
            position:  [0.0, 0.0, 0.0],
            color:     [1.0, 1.0, 1.0],
            intensity: 1.0,
            range:     0.0,
            direction: [0.0, -1.0, 0.0],
            focus:     30.0,
            label:     String::new(),
        }
    }
}

fn default_light_intensity_one() -> f32 { 1.0 }
fn default_down()  -> [f32; 3] { [0.0, -1.0, 0.0] }
fn default_focus() -> f32      { 30.0 }

// ---------------------------------------------------------------------------
// VolumetricProperties — per-block volumetric light / god-ray parameters
// ---------------------------------------------------------------------------

/// Volumetric glow and god-ray parameters for emissive blocks.
/// All settings are opt-in: set `halo_enabled` or `ray_enabled` to true.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumetricProperties {
    /// Enable a soft additive halo (Gaussian sphere) around the block centre.
    #[serde(default)]
    pub halo_enabled: bool,
    /// World-space radius of the halo in blocks (default: 2.5).
    #[serde(default = "default_halo_radius")]
    pub halo_radius: f32,
    /// Peak intensity of the halo (0..1, default: 0.6).
    #[serde(default = "default_halo_intensity")]
    pub halo_intensity: f32,

    /// Enable directional god rays radiating outward from the block.
    #[serde(default)]
    pub ray_enabled: bool,
    /// Number of rays emitted (default: 4).
    #[serde(default = "default_ray_count")]
    pub ray_count: u32,
    /// Length of each ray in blocks (default: 6.0).
    #[serde(default = "default_ray_length")]
    pub ray_length: f32,
    /// Width of each ray in blocks (default: 0.4).
    #[serde(default = "default_ray_width")]
    pub ray_width: f32,
    /// Peak intensity of each ray (0..1, default: 0.18).
    #[serde(default = "default_ray_intensity")]
    pub ray_intensity: f32,
    /// Falloff exponent along the ray (higher = sharper fade, default: 2.0).
    #[serde(default = "default_ray_falloff")]
    pub ray_falloff: f32,
}

impl Default for VolumetricProperties {
    fn default() -> Self {
        Self {
            halo_enabled:  false,
            halo_radius:   default_halo_radius(),
            halo_intensity: default_halo_intensity(),
            ray_enabled:   false,
            ray_count:     default_ray_count(),
            ray_length:    default_ray_length(),
            ray_width:     default_ray_width(),
            ray_intensity: default_ray_intensity(),
            ray_falloff:   default_ray_falloff(),
        }
    }
}

impl VolumetricProperties {
    /// Returns true if at least one volumetric effect is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.halo_enabled || self.ray_enabled
    }
}

fn default_halo_radius()    -> f32 { 2.5 }
fn default_halo_intensity() -> f32 { 0.6 }
fn default_ray_count()      -> u32 { 4 }
fn default_ray_length()     -> f32 { 6.0 }
fn default_ray_width()      -> f32 { 0.4 }
fn default_ray_intensity()  -> f32 { 0.18 }
fn default_ray_falloff()    -> f32 { 2.0 }

// ---------------------------------------------------------------------------
// GlassProperties — for transparent/translucent solid blocks (glass, stained glass)
// ---------------------------------------------------------------------------

/// Optical properties for transparent solid blocks (e.g. glass, stained glass).
///
/// When a block has `transparent: true` and provides these properties:
/// - Sun/moon DDA shadow rays accumulate `tint_color * (1 - opacity)` transmission
///   through the voxel, producing partial coloured shadows.
/// - GI light propagation passes through the voxel attenuated by the tint colour.
///
/// Block-in-hand and entity shadow rays observe the same tinting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlassProperties {
    /// RGB tint that multiplies any light passing through (0..1 per channel).
    /// Pure white (1,1,1) = clear glass; (1,0.3,0.3) = red stained glass that
    /// only passes red light.
    #[serde(default = "default_white")]
    pub tint_color: [f32; 3],
    /// Opacity in [0..1]: 0 = perfectly clear, 1 = fully opaque.
    /// 0.5 = light is halved; 1.0 = no light passes (acts like opaque block).
    /// When the block has a texture, the per-fragment opacity is multiplied
    /// by the texture's alpha so transparent texels let light through.
    #[serde(default = "default_glass_opacity")]
    pub opacity: f32,
}

impl Default for GlassProperties {
    fn default() -> Self {
        Self {
            tint_color: [1.0, 1.0, 1.0],
            opacity:    default_glass_opacity(),
        }
    }
}

fn default_glass_opacity() -> f32 { 0.15 }

// ---------------------------------------------------------------------------
// FluidProperties — per-block fluid simulation and rendering parameters
// ---------------------------------------------------------------------------

/// Properties for fluid blocks (water, lava, oil, etc.).
///
/// When present on a [`BlockDefinition`], the block is treated as a fluid
/// by the simulation and rendering systems.
///
/// Fluid levels (0.0..1.0) are stored per-voxel in `Chunk::fluid_levels`.
/// Level 1.0 = full source block (never depletes), < 1.0 = flowing.
///
/// # Adding a new fluid
/// ```text
/// 1. Create assets/blocks/<fluid>.json with "fluid": { ... }
/// 2. Add the block ID to assets/blocks/block_registry.json
/// 3. The FluidSimulator will automatically pick it up
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluidProperties {
    /// Horizontal flow rate (0.0..1.0). Higher = faster horizontal spread.
    /// Water: 0.25, Lava: 0.05
    #[serde(default = "default_fluid_flow_rate")]
    pub flow_rate: f32,

    /// Downward flow rate multiplier (0.0..1.0). 1.0 = instant gravity.
    /// Water: 1.0, Lava: 0.6
    #[serde(default = "default_fluid_gravity_rate")]
    pub gravity_rate: f32,

    /// Maximum horizontal spread distance from source in blocks (0 = infinite).
    /// Water: 7, Lava: 3
    #[serde(default = "default_fluid_spread_distance")]
    pub spread_distance: u8,

    /// Fog colour when camera is submerged in this fluid (RGB, 0..1).
    #[serde(default = "default_fluid_fog_color")]
    pub fog_color: [f32; 3],

    /// Fog density when camera is submerged (higher = thicker fog).
    /// Water: 0.04, Lava: 0.2
    #[serde(default = "default_fluid_fog_density")]
    pub fog_density: f32,
}

impl Default for FluidProperties {
    fn default() -> Self {
        Self {
            flow_rate:       default_fluid_flow_rate(),
            gravity_rate:    default_fluid_gravity_rate(),
            spread_distance: default_fluid_spread_distance(),
            fog_color:       default_fluid_fog_color(),
            fog_density:     default_fluid_fog_density(),
        }
    }
}

fn default_fluid_flow_rate()       -> f32      { 0.25 }
fn default_fluid_gravity_rate()    -> f32      { 1.0 }
fn default_fluid_spread_distance() -> u8       { 7 }
fn default_fluid_fog_color()       -> [f32; 3] { [0.1, 0.3, 0.6] }
fn default_fluid_fog_density()     -> f32      { 0.04 }

// ---------------------------------------------------------------------------
// PlacementMode — how a block orients itself when placed by the player
// ---------------------------------------------------------------------------

/// Determines how this block rotates when placed by the player.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlacementMode {
    /// Always placed in its default orientation (no rotation). Default.
    #[default]
    Fixed,
    /// Attaches to any of the 4 horizontal (N/S/E/W) side faces.
    /// Top/bottom clicks fall back to player look direction.
    AttachHorizontal,
    /// Attaches to any of the 6 cube faces (block base against the clicked face).
    AttachFull,
    /// Like `AttachFull` but top/bottom placement also picks one of 4 compass
    /// orientations based on the player's horizontal look direction.
    AttachFullRotatable,
    /// Rotates to one of 4 compass directions based on where the player looks.
    LookHorizontal,
    /// Rotates to face any of 6 cube directions based on the full look vector.
    LookFull,
    /// Aligns the block's axis with the clicked face normal (for logs/pillars).
    /// Clicking top/bottom → Y-axis (rot 0); east/west → X-axis (rot 14); north/south → Z-axis (rot 15).
    LookAxis,
}

/// Face permutation map for block rotation.
///
/// `ROT_FACE_MAP[rotation][world_face]` returns the **block-local face** whose
/// texture/colour should be displayed on that world face.
///
/// Face indices: 0=+X(East), 1=-X(West), 2=+Y(Top), 3=-Y(Bottom),
///               4=+Z(South), 5=-Z(North)
///
/// Rotation values:
///  0  default (floor, facing south)      |  7  wall south: local+Y→world+Z, outward=south
///  1  facing north (180° around Y)       |  8  wall east:  local+Y→world+X, outward=east
///  2  facing south (= default)           |  9  wall west:  local+Y→world-X, outward=west
///  3  facing east  (+90° around Y)       | 10  ceiling + facing north
///  4  facing west  (-90° around Y)       | 11  ceiling + facing south (= 5)
///  5  upside-down / ceiling              | 12  ceiling + facing east
///  6  wall north: local+Y→world-Z,       | 13  ceiling + facing west
///     outward=north                      | 14  axis X (log running east-west)
///                                        | 15  axis Z (log running north-south)
///
/// Wall rows (6-9): outward face always shows local_top; inward face shows local_bottom.
pub const ROT_FACE_MAP: [[u8; 6]; 16] = [
    [0, 1, 2, 3, 4, 5], // 0: default
    [1, 0, 2, 3, 5, 4], // 1: facing north
    [0, 1, 2, 3, 4, 5], // 2: facing south
    [4, 5, 2, 3, 1, 0], // 3: facing east  (local+Z→world+X)
    [5, 4, 2, 3, 0, 1], // 4: facing west  (local+Z→world-X)
    [0, 1, 3, 2, 5, 4], // 5: upside-down / ceiling
    // wall north: block placed N of wall, +Z face against wall; local+Y→world-Z(north=outward)
    // world_top←local_south, world_bottom←local_north, world_south←local_bottom, world_north←local_top
    [0, 1, 4, 5, 3, 2], // 6: wall north  (outward=-Z, local+Y→world-Z, local+Z→world+Y)
    // wall south: block placed S of wall, -Z face against wall; local+Y→world+Z(south=outward)
    // world_top←local_north, world_bottom←local_south, world_south←local_top, world_north←local_bottom
    [0, 1, 5, 4, 2, 3], // 7: wall south  (outward=+Z, local+Y→world+Z, local+Z→world-Y)
    [2, 3, 1, 0, 4, 5], // 8: wall east   (outward=+X, local+Y→world+X)
    [3, 2, 0, 1, 4, 5], // 9: wall west   (outward=-X, local+Y→world-X)
    [1, 0, 3, 2, 4, 5], // 10: ceiling + north
    [0, 1, 3, 2, 5, 4], // 11: ceiling + south (= 5)
    [4, 5, 3, 2, 0, 1], // 12: ceiling + east
    [5, 4, 3, 2, 1, 0], // 13: ceiling + west
    [2, 3, 4, 5, 1, 0], // 14: axis X  (log E-W: ring on E/W, bark rotated +90° on T/B/S/N)
    [0, 1, 4, 5, 2, 3], // 15: axis Z  (log N-S: ring on N/S, bark rotated +90° on T/B/E/W)
];

/// Remap a world-space face index to a block-local face for the given rotation.
///
/// Returns the block-local face index to use for texture/colour lookup when
/// `world_face` is visible on a block stored with `rotation`.
#[inline]
pub fn remap_face_for_rotation(rotation: u8, world_face: u8) -> u8 {
    ROT_FACE_MAP[rotation.min(15) as usize][world_face as usize]
}

// ---------------------------------------------------------------------------
// BlockDefinition — deserialised from an individual <id>.json
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockDefinition {
    /// Unique string identifier, e.g. "grass", "dirt", "stone".
    pub id: String,

    /// Human-readable name shown in tooltips / UI.
    #[serde(default)]
    pub display_name: String,

    /// Short description of the block.
    #[serde(default)]
    pub description: String,

    /// Base RGB colour (0..1). Used when face-specific colours are absent.
    pub color: [f32; 3],

    /// Name of the texture in the atlas (e.g. "grass_top", "dirt", "stone_side").
    /// If absent, this face uses vertex colour only (white tile in atlas).
    #[serde(default)]
    pub texture: Option<String>,

    /// Whether the block blocks movement and is meshed with solid faces.
    #[serde(default = "default_true")]
    pub solid: bool,

    /// Whether light can pass through this block.
    /// When `true` and the block is `solid`, it is rendered in the transparent
    /// pass (alpha-blended) and its voxel writes to the VCT tint volume so the
    /// GI/shadow systems propagate light through it with the configured tint.
    #[serde(default)]
    pub transparent: bool,

    /// Optical tint/opacity for transparent solid blocks (glass, stained glass).
    /// Only consulted when `transparent: true`. Defaults to clear glass with
    /// low opacity (0.15).
    #[serde(default)]
    pub glass: GlassProperties,

    /// Whether this block is a water fluid block with variable height.
    /// **Legacy** — prefer the `fluid` field. Reads from old JSONs but never written.
    #[serde(default, skip_serializing)]
    pub is_water: bool,

    /// Fluid properties — when present, this block is a fluid (water, lava, etc.).
    /// Fluids are simulated by the fluid system and rendered with transparency
    /// and variable surface height (0..1).
    /// Takes precedence over `is_water` when both are set.
    #[serde(default)]
    pub fluid: Option<FluidProperties>,

    /// Legacy light emission level (0..15). Superseded by `emission`. Reads old JSONs, never written.
    #[serde(default, skip_serializing)]
    pub emit_light: u8,

    /// PBR material properties (roughness, metalness, AO).
    #[serde(default)]
    pub material: MaterialProperties,

    /// Light emission properties (colour, strength, intensity).
    #[serde(default)]
    pub emission: EmissionProperties,

    /// Volumetric glow and god-ray visual effects (opt-in per block).
    #[serde(default)]
    pub volumetric: VolumetricProperties,

    /// Optional per-face colour and texture overrides (6-face + convenience keys).
    #[serde(default)]
    pub faces: FaceColors,

    /// Whether this block's vertex colour should be multiplied by the biome's
    /// foliage colour at render time. Set `true` for grass, leaves, ferns, etc.
    /// Blocks with this flag receive the biome tint stored in the chunk column.
    #[serde(default)]
    pub biome_tint: bool,

    /// How this block should be oriented when placed by the player.
    #[serde(default)]
    pub placement_mode: PlacementMode,

    /// Default rotation applied when placed (0 = natural upright orientation).
    /// Used as the fallback when `placement_mode` is `Fixed` or cannot determine a rotation.
    #[serde(default)]
    pub default_rotation: u8,

    /// Path to a custom 3D model file (relative to assets/models/ or absolute).
    /// Supports both Bedrock geometry and Java Edition block model formats.
    /// When set, the block is rendered using this model instead of the default cube.
    #[serde(default)]
    pub model: Option<String>,

    /// Named texture paths for the custom model (relative to assets/textures/ or absolute).
    /// Maps texture variable names to image file paths.
    ///
    /// For Java Edition models, keys match the `textures` map in the model JSON:
    /// ```json
    /// "model_textures": {
    ///     "0": "blocks/spotlight_side.png",
    ///     "1": "blocks/spotlight_top.png"
    /// }
    /// ```
    /// Face references like `"texture": "#0"` will use the "0" entry.
    ///
    /// For Bedrock geometry (single-texture), use a single key like "default":
    /// ```json
    /// "model_textures": {
    ///     "default": "blocks/custom_block.png"
    /// }
    /// ```
    ///
    /// All textures are packed into a single per-model atlas at load time.
    #[serde(default)]
    pub model_textures: HashMap<String, String>,

    /// Creative-mode inventory tab this block belongs to.
    /// Any string value creates/selects that tab (e.g. "Building", "Lighting", "MyMod").
    /// Leave absent to appear only in the "All" tab.
    #[serde(default)]
    pub inventory_tab: Option<String>,

    /// Inline light sources attached to this block (point and/or directional).
    /// Visualized as arrows in the block editor preview.
    #[serde(default)]
    pub light_sources: Vec<BlockLightSource>,

    /// Per-cube shadow AABBs for VCT voxel shadow DDA.
    /// Each entry: [x_min, x_max, y_min, y_max, z_min, z_max] in block-canonical [0,1]^3 space.
    #[serde(skip)]
    pub model_shadow_cubes: Option<Vec<[f32; 6]>>,
}

fn default_true() -> bool { true }

impl BlockDefinition {
    /// Return the colour for a given face direction.
    ///
    /// Resolution priority (most specific first):
    ///   Face 0 (+X East):  east → sides → all → block base `color`
    ///   Face 1 (-X West):  west → sides → all → block base `color`
    ///   Face 2 (+Y Top):   top → all → block base `color`
    ///   Face 3 (-Y Bottom):bottom → all → block base `color`
    ///   Face 4 (+Z South): south → sides → all → block base `color`
    ///   Face 5 (-Z North): north → sides → all → block base `color`
    pub fn color_for_face(&self, face: u8) -> [f32; 3] {
        resolve_face_override(&self.faces, face)
            .map(|f| f.color)
            .unwrap_or(self.color)
    }

    /// Return the atlas texture name for a given face direction.
    /// Returns `None` if this face has no texture (falls back to white tile).
    ///
    /// If the face has baked layers (`resolved_texture`), returns the baked key.
    /// Otherwise returns the raw `texture` field.
    ///
    /// `face`: 0 = +X (East), 1 = -X (West), 2 = +Y (Top), 3 = -Y (Bottom),
    ///         4 = +Z (South), 5 = -Z (North).
    pub fn texture_for_face(&self, face: u8) -> Option<&str> {
        let fo = resolve_face_override(&self.faces, face)?;
        // Prefer baked texture (set after layer compositing)
        fo.resolved_texture.as_deref()
            .or(fo.texture.as_deref())
    }

    /// Returns the base texture key and any overlay layer keys for a face.
    /// Unlike `texture_for_face`, this returns the raw (un-baked) keys so callers
    /// can composite them manually (e.g. the block-editor preview renderer).
    ///
    /// Returns `None` when the face has no texture at all.
    pub fn base_and_layers_for_face(&self, face: u8) -> Option<(&str, Vec<&str>)> {
        let fo = resolve_face_override(&self.faces, face)?;
        let base = fo.texture.as_deref()?;
        let layers: Vec<&str> = [&fo.layer1, &fo.layer2, &fo.layer3, &fo.layer4, &fo.layer5]
            .iter()
            .filter_map(|l| l.as_deref())
            .collect();
        Some((base, layers))
    }

    /// Returns `true` if this block is any kind of fluid (water, lava, etc.).
    /// Checks both the legacy `is_water` flag and the new `fluid` field.
    #[inline]
    pub fn is_fluid(&self) -> bool {
        self.is_water || self.fluid.is_some()
    }

    /// Returns a short label describing the block model type.
    pub fn model_type_label(&self) -> &'static str {
        match &self.model {
            None => "Default Cube",
            Some(path) if path.contains(".geo.") => "Bedrock Geometry",
            Some(_) => "Java Edition",
        }
    }

    /// Returns fluid properties if this block is a fluid, `None` otherwise.
    /// For legacy blocks with `is_water: true` but no `fluid` field, returns
    /// `None` — use `BlockRegistry::fluid_props()` which handles this case.
    #[inline]
    pub fn fluid_props(&self) -> Option<&FluidProperties> {
        self.fluid.as_ref()
    }

    /// Returns true if this block is a transparent solid (glass-like).
    /// Excludes fluids and air; these need to be `solid && transparent`.
    #[inline]
    pub fn is_glass(&self) -> bool {
        self.solid && self.transparent && !self.is_fluid() && self.model.is_none()
    }
}

// ---------------------------------------------------------------------------
// BlockRegistryFile — the top-level block_registry.json schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BlockRegistryFile {
    /// Schema version — currently must be 1.
    format_version: u32,
    /// Ordered list of block IDs.  Order determines numeric IDs (1-indexed).
    blocks: Vec<String>,
}

// ---------------------------------------------------------------------------
// BlockRegistry — holds all loaded block types
//
// Block ID 0 is always Air (implicit, not in any file).
// Block IDs 1..=255 map to definitions[id-1].
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct BlockRegistry {
    definitions: Vec<BlockDefinition>,
    id_to_index: HashMap<String, u8>,
}

/// Default relative path to the block assets directory.
const BLOCKS_DIR: &str = "assets/blocks/";
/// Name of the master registry file inside the blocks directory.
const REGISTRY_FILE: &str = "block_registry.json";

impl BlockRegistry {
    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Load the block registry.
    ///
    /// Tries to read from `<assets_dir>/block_registry.json` and then each
    /// individual `<assets_dir>/<block_id>.json`.  Falls back to embedded
    /// defaults when the directory or files are not found.
    pub fn load() -> Self {
        Self::load_from(BLOCKS_DIR)
    }

    /// Same as `load()` but allows specifying a custom assets directory.
    pub fn load_from(assets_dir: &str) -> Self {
        let dir = PathBuf::from(assets_dir);
        let registry_path = dir.join(REGISTRY_FILE);

        // Attempt filesystem load
        if registry_path.exists() {
            debug_log!(
                "BlockRegistry", "load",
                "Found block registry at {:?}",
                registry_path
            );
            match Self::load_from_fs(&dir, &registry_path) {
                Ok(reg) => return reg,
                Err(e) => {
                    debug_log!(
                        "BlockRegistry", "load",
                        "Filesystem load failed: {}, falling back to embedded defaults",
                        e
                    );
                }
            }
        } else {
            debug_log!(
                "BlockRegistry", "load",
                "Registry file not found at {:?}, using embedded defaults",
                registry_path
            );
        }

        // Fallback: embedded defaults
        Self::load_embedded()
    }

    /// Look up a block by numeric ID.  Returns `None` for Air (id = 0) and
    /// unknown ids.
    pub fn get(&self, block_id: u8) -> Option<&BlockDefinition> {
        if block_id == 0 {
            return None;
        }
        self.definitions.get((block_id - 1) as usize)
    }

    /// Look up numeric ID by string identifier (the registry key, e.g. "rocks/andesite").
    pub fn id_for(&self, name: &str) -> Option<u8> {
        self.id_to_index.get(name).copied()
    }

    /// Reverse lookup: registry key string for a given numeric block ID.
    /// Iterates `id_to_index`; O(n) but called only during inventory init.
    pub fn key_for(&self, id: u8) -> Option<&str> {
        self.id_to_index.iter()
            .find(|&(_, &v)| v == id)
            .map(|(k, _)| k.as_str())
    }

    /// Number of registered block types (excluding Air).
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    /// Returns true if no blocks are registered.
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    /// Returns true if the given block has a custom 3D model.
    pub fn has_model(&self, block_id: u8) -> bool {
        self.get(block_id).map_or(false, |d| d.model.is_some())
    }

    /// Returns the model name (e.g. "block/crafting_station") for a block with a custom model.
    /// Returns None if the block has no model.
    pub fn model_name_for(&self, block_id: u8) -> Option<String> {
        self.get(block_id).map(|d| format!("block/{}", d.id))
    }

    /// Sets the per-cube shadow AABBs for VCT voxel shadow DDA.
    pub fn set_model_shadow_cubes(&mut self, block_id: u8, cubes: Vec<[f32; 6]>) {
        if block_id == 0 { return; }
        if let Some(def) = self.definitions.get_mut((block_id - 1) as usize) {
            def.model_shadow_cubes = Some(cubes);
        }
    }

    /// Returns the per-cube shadow AABBs if available.
    pub fn model_shadow_cubes(&self, block_id: u8) -> Option<&[[f32; 6]]> {
        self.get(block_id).and_then(|d| d.model_shadow_cubes.as_deref())
    }

    // -----------------------------------------------------------------------
    // Layer baking — composites face layers into atlas tiles
    // -----------------------------------------------------------------------

    /// Iterate all block definitions and bake layered textures.
    ///
    /// For each `FaceOverride` that has a base `texture` and at least one
    /// `layer1`–`layer5`, calls `bake_fn(base, &layers)` to composite the
    /// layers into a single atlas tile.  The returned key is stored in
    /// `resolved_texture` on the face override.
    ///
    /// `bake_fn` is typically `TextureAtlas::bake_layered_texture`.
    /// This indirection avoids a direct dependency from `block.rs` to
    /// `texture_atlas.rs`.
    pub fn bake_layered_textures<F>(&mut self, mut bake_fn: F)
    where
        F: FnMut(&str, &[&str]) -> Option<String>,
    {
        let mut baked_count = 0usize;
        for def in &mut self.definitions {
            baked_count += Self::bake_face_override(
                &def.id, &mut def.faces.all, &mut bake_fn
            );
            baked_count += Self::bake_face_override(
                &def.id, &mut def.faces.top, &mut bake_fn
            );
            baked_count += Self::bake_face_override(
                &def.id, &mut def.faces.bottom, &mut bake_fn
            );
            baked_count += Self::bake_face_override(
                &def.id, &mut def.faces.north, &mut bake_fn
            );
            baked_count += Self::bake_face_override(
                &def.id, &mut def.faces.south, &mut bake_fn
            );
            baked_count += Self::bake_face_override(
                &def.id, &mut def.faces.east, &mut bake_fn
            );
            baked_count += Self::bake_face_override(
                &def.id, &mut def.faces.west, &mut bake_fn
            );
            baked_count += Self::bake_face_override(
                &def.id, &mut def.faces.sides, &mut bake_fn
            );
        }
        if baked_count > 0 {
            debug_log!(
                "BlockRegistry", "bake_layered_textures",
                "Total layered face textures baked: {}", baked_count
            );
        }
    }

    /// Bake layers for a single optional `FaceOverride`.
    /// Returns 1 if a bake was performed, 0 otherwise.
    fn bake_face_override<F>(
        block_id: &str,
        fo: &mut Option<FaceOverride>,
        bake_fn: &mut F,
    ) -> usize
    where
        F: FnMut(&str, &[&str]) -> Option<String>,
    {
        let fo = match fo.as_mut() {
            Some(f) => f,
            None => return 0,
        };
        if fo.texture.is_none() {
            return 0;
        }

        // Collect non-None layers in order
        let layers: Vec<&str> = [
            fo.layer1.as_deref(),
            fo.layer2.as_deref(),
            fo.layer3.as_deref(),
            fo.layer4.as_deref(),
            fo.layer5.as_deref(),
        ]
            .into_iter()
            .flatten()
            .collect();

        if layers.is_empty() {
            return 0;
        }

        let base = fo.texture.as_ref().unwrap().clone();
        match bake_fn(&base, &layers) {
            Some(baked_key) => {
                debug_log!(
                    "BlockRegistry", "bake_face_override",
                    "Baked block '{}' face: '{}' + {:?} => '{}'",
                    block_id, base, layers, baked_key
                );
                fo.resolved_texture = Some(baked_key);
                1
            }
            None => {
                debug_log!(
                    "BlockRegistry", "bake_face_override",
                    "WARNING: failed to bake block '{}' face: '{}' + {:?}",
                    block_id, base, layers
                );
                0
            }
        }
    }

    // -----------------------------------------------------------------------
    // Radiance Cascades: optical properties for Block LUT
    // -----------------------------------------------------------------------

    /// Returns optical properties for a block as (opacity, emission_r, emission_g, emission_b).
    /// Used by Radiance Cascades to build the Block LUT texture.
    ///
    /// - opacity: 1000.0 for solid blocks, 0.0 for air/transparent
    /// - emission_rgb: pre-multiplied (color * intensity), zero for non-emissive
    pub fn block_optical_props(&self, block_id: u8) -> Option<(f32, f32, f32, f32)> {
        let def = self.get(block_id)?;
        if !def.solid {
            // Fluid blocks may emit light (e.g. lava)
            if def.is_fluid() && def.emission.emit_light {
                let intensity = def.emission.light_intensity;
                return Some((
                    0.0, // transparent
                    def.emission.light_color[0] * intensity,
                    def.emission.light_color[1] * intensity,
                    def.emission.light_color[2] * intensity,
                ));
            }
            // Air / transparent blocks: fully transparent
            return Some((0.0, 0.0, 0.0, 0.0));
        }
        let opacity = 1000.0;
        let (er, eg, eb) = if def.emission.emit_light {
            (
                def.emission.light_color[0] * def.emission.light_intensity,
                def.emission.light_color[1] * def.emission.light_intensity,
                def.emission.light_color[2] * def.emission.light_intensity,
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        Some((opacity, er, eg, eb))
    }

    /// Number of block types that emit light.
    pub fn emissive_count(&self) -> usize {
        self.definitions
            .iter()
            .filter(|d| d.emission.emit_light)
            .count()
    }

    // -----------------------------------------------------------------------
    // Fluid helpers
    // -----------------------------------------------------------------------

    /// Returns `true` if the given block ID is a fluid block.
    pub fn is_fluid_block(&self, block_id: u8) -> bool {
        self.get(block_id).map_or(false, |d| d.is_fluid())
    }

    /// Returns `true` if the given block ID is a transparent solid (glass).
    #[inline]
    pub fn is_glass_block(&self, block_id: u8) -> bool {
        self.get(block_id).map_or(false, |d| d.is_glass())
    }

    /// Returns fluid properties for a fluid block, or `None` if not a fluid.
    /// Handles legacy `is_water: true` blocks by returning default water props.
    pub fn fluid_props_for(&self, block_id: u8) -> Option<FluidProperties> {
        let def = self.get(block_id)?;
        if let Some(ref fp) = def.fluid {
            Some(fp.clone())
        } else if def.is_water {
            Some(FluidProperties::default())
        } else {
            None
        }
    }

    /// Returns all registered fluid block IDs with their properties.
    /// Includes both new-style (`fluid` field) and legacy (`is_water`) blocks.
    pub fn fluid_blocks(&self) -> Vec<(u8, FluidProperties)> {
        self.definitions
            .iter()
            .enumerate()
            .filter_map(|(i, def)| {
                if def.is_fluid() {
                    let props = def.fluid.clone()
                        .unwrap_or_else(FluidProperties::default);
                    Some(((i + 1) as u8, props))
                } else {
                    None
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Filesystem loader
    // -----------------------------------------------------------------------

    /// Recursively scan `dir` and build a map: relative path → full path.
    fn collect_block_files(dir: &Path) -> HashMap<String, PathBuf> {
        let mut map = HashMap::new();
        Self::scan_block_dir(dir, dir, &mut map);
        debug_log!(
            "BlockRegistry", "collect_block_files",
            "Scanned {:?}: found {} JSON files", dir, map.len()
        );
        map
    }

    fn scan_block_dir(root: &Path, current: &Path, map: &mut HashMap<String, PathBuf>) {
        let entries = match std::fs::read_dir(current) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                Self::scan_block_dir(root, &path, map);
                continue;
            }
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    let rel = match path.strip_prefix(root) {
                        Ok(r) => r.to_str()
                            .unwrap_or("")
                            .strip_suffix(".json")
                            .unwrap_or("")
                            .replace('\\', "/"),
                        Err(_) => stem.to_string(),
                    };
                    if rel == "block_registry" { continue; }
                    map.entry(rel).or_insert(path);
                }
            }
        }
    }

    fn load_from_fs(dir: &Path, registry_path: &Path) -> Result<Self, String> {
        let registry_str = std::fs::read_to_string(registry_path)
            .map_err(|e| format!("Failed to read {:?}: {}", registry_path, e))?;

        let registry: BlockRegistryFile = serde_json::from_str(&registry_str)
            .map_err(|e| format!("Failed to parse block_registry.json: {}", e))?;

        if registry.format_version != 1 {
            return Err(format!(
                "Unsupported block_registry format_version: {} (expected 1)",
                registry.format_version
            ));
        }

        debug_log!(
            "BlockRegistry", "load_from_fs",
            "Registry lists {} block(s): {:?}",
            registry.blocks.len(),
            &registry.blocks
        );

        let mut definitions = Vec::with_capacity(registry.blocks.len());
        let mut id_to_index = HashMap::new();

        let file_map = Self::collect_block_files(dir);

        debug_log!(
            "BlockRegistry", "load_from_fs",
            "File map contains {} entries", file_map.len()
        );

        for (_i, block_id) in registry.blocks.iter().enumerate() {
            let numeric_id = (definitions.len() + 1) as u8;

            let block_path = match file_map.get(block_id.as_str()) {
                Some(p) => p.clone(),
                None => {
                    debug_log!(
                        "BlockRegistry", "load_from_fs",
                        "WARNING: block '{}' not found in recursive scan -- skipping",
                        block_id
                    );
                    continue;
                }
            };

            let block_str = std::fs::read_to_string(&block_path)
                .map_err(|e| {
                    format!("Failed to read block file {:?}: {}", block_path, e)
                })?;

            let def: BlockDefinition = serde_json::from_str(&block_str)
                .map_err(|e| {
                    format!("Failed to parse block '{}': {}", block_id, e)
                })?;

            if def.id != *block_id {
                debug_log!(
                    "BlockRegistry", "load_from_fs",
                    "WARNING: file '{:?}' declares id='{}' but registry says '{}'",
                    block_path, def.id, block_id
                );
            }

            debug_log!(
                "BlockRegistry", "load_from_fs",
                "Registered block '{}' => id {} from {:?}", block_id, numeric_id, block_path
            );

            id_to_index.insert(block_id.clone(), numeric_id);
            definitions.push(def);
        }

        // Auto-populate fluid from legacy is_water flag
        for def in &mut definitions {
            if def.is_water && def.fluid.is_none() {
                def.fluid = Some(FluidProperties::default());
            }
        }

        debug_log!(
            "BlockRegistry", "load_from_fs",
            "Total block types loaded from filesystem: {}",
            definitions.len()
        );

        Ok(Self { definitions, id_to_index })
    }

    fn name_to_color(name: &str) -> [f32; 3] {
        let mut hash: u64 = 0;
        for b in name.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(b as u64);
        }
        let r = ((hash & 0xFF) as f32) / 255.0 * 0.6 + 0.2;
        let g = (((hash >> 8) & 0xFF) as f32) / 255.0 * 0.6 + 0.2;
        let b = (((hash >> 16) & 0xFF) as f32) / 255.0 * 0.6 + 0.2;
        [r, g, b]
    }

    // -----------------------------------------------------------------------
    // Embedded fallback — used when assets/ directory is absent
    // -----------------------------------------------------------------------

    fn load_embedded() -> Self {
        debug_log!(
            "BlockRegistry", "load_embedded",
            "Loading embedded default blocks"
        );

        let defaults = include_str!("../../../assets/blocks/block_registry.json");
        let registry: BlockRegistryFile = serde_json::from_str(defaults)
            .expect("[BlockRegistry][load_embedded] Failed to parse embedded block_registry.json");

        let embedded_defs: &[(&str, &str)] = &[
            ("grass",     include_str!("../../../assets/blocks/grass.json")),
            ("dirt",      include_str!("../../../assets/blocks/dirt.json")),
            ("stone",     include_str!("../../../assets/blocks/stone.json")),
            ("glowstone", include_str!("../../../assets/blocks/glowstone.json")),
            ("spotlight", include_str!("../../../assets/blocks/spotlight.json")),
        ];

        let mut definitions = Vec::with_capacity(registry.blocks.len());
        let mut id_to_index = HashMap::new();

        for (i, block_id) in registry.blocks.iter().enumerate() {
            let numeric_id = (i + 1) as u8;

            if let Some(json_str) = embedded_defs
                .iter()
                .find(|(id, _)| *id == block_id.as_str())
                .map(|(_, json)| *json)
            {
                let def: BlockDefinition = serde_json::from_str(json_str)
                    .expect("[BlockRegistry][load_embedded] Failed to parse embedded block JSON");

                debug_log!(
                    "BlockRegistry", "load_embedded",
                    "Registered block '{}' => id {} (embedded)",
                    def.id, numeric_id
                );

                id_to_index.insert(block_id.clone(), numeric_id);
                definitions.push(def);
            } else {
                let fallback_color = BlockRegistry::name_to_color(block_id);
                let def = BlockDefinition {
                    id: block_id.clone(),
                    display_name: block_id.replace('_', " "),
                    description: format!("Auto-generated fallback for {}", block_id),
                    color: fallback_color,
                    texture: None,
                    solid: true,
                    transparent: false,
                    glass: GlassProperties::default(),
                    is_water: false,
                    fluid: None,
                    emit_light: 0,
                    material: MaterialProperties::default(),
                    emission: EmissionProperties::default(),
                    volumetric: VolumetricProperties::default(),
                    faces: FaceColors::default(),
                    biome_tint: false,
                    placement_mode: PlacementMode::Fixed,
                    default_rotation: 0,
                    model: None,
                    model_textures: HashMap::new(),
                    model_shadow_cubes: None,
                    inventory_tab: None,
                    light_sources: Vec::new(),
                };

                debug_log!(
                    "BlockRegistry", "load_embedded",
                    "Registered block '{}' => id {} (synthetic fallback, color={:?})",
                    def.id, numeric_id, def.color
                );

                id_to_index.insert(block_id.clone(), numeric_id);
                definitions.push(def);
            }
        }

        // Auto-populate fluid from legacy is_water flag
        for def in &mut definitions {
            if def.is_water && def.fluid.is_none() {
                def.fluid = Some(FluidProperties::default());
            }
        }

        debug_log!(
            "BlockRegistry", "load_embedded",
            "Total embedded block types loaded: {}",
            definitions.len()
        );

        Self { definitions, id_to_index }
    }
}