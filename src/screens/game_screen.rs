// =============================================================================
// QubePixel — GameScreen  (3D world + HUD + debug overlay + profiler)
// =============================================================================
// Simplified: no Radiance Cascades, no Voxel GPU Lighting, no Volumetric Lights.
// Only static directional (sun/moon) PBR lighting with ambient.

use crate::core::screen::{Screen, ScreenAction};
use crate::{debug_log, flow_debug_log};
use std::time::Instant;

use crate::screens::debug_overlay::DebugOverlay;
use crate::screens::profiler_overlay::{ProfilerOverlay, ProfilerFrame};
use crate::screens::game_3d_pipeline::{Camera, Game3DPipeline, DebugVertex};
use std::collections::{HashMap, VecDeque};
use crate::core::upload_worker::{UploadWorker, UploadJob, UploadPacked, MeshKind};
use crate::core::world_gen::world::{WorldWorker, WorldResult, BlockOp};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use crate::screens::settings::SettingsScreen;
use crate::core::gameobjects::texture_atlas::TextureAtlas;
use crate::core::save_system::{WorldMetadata, save_world_meta};
use crate::core::raycast::RaycastResult;
use crate::core::physics::PhysicsWorld;
use crate::core::player::PlayerController;
use crate::core::gameobjects::block::{BlockRegistry, BlockLightSource, LightKind, PlacementMode};
use crate::core::lighting::{DayNightCycle, LightingConfig, pack_lighting_uniforms};
use crate::screens::sky_renderer::SkyRenderer;
use glam::Vec3;
use egui::{Color32, FontId, Pos2, Rect, Stroke, Vec2, Align2, Rounding};
use crate::core::config;
use crate::core::user_settings;
use crate::core::vct::{VCTSystem, VoxelSnapshot, SpotLightGPU, PointLightGPU};
use crate::core::player_model::PlayerModel;
use crate::core::entity::{EntityId, EntityManager, EntityTransform};
use crate::screens::player_renderer::{PlayerRenderer, PlayerPose, ViewMode, PLAYER_BONE_ORDER, PLAYER_SIZE};
use crate::screens::light_emitter_renderer::EmitterLight;
use crate::screens::entity_renderer::EntityRenderer;
use crate::screens::block_model_renderer::{BlockModelRenderer, BlockModelLighting};
use crate::screens::inventory::{InventoryUI, InventoryConfig, ItemStack};
use crate::screens::hand_block_renderer::HandBlockRenderer;
use crate::core::model_messenger::ModelMessenger;
use crate::core::block_model::{extract_quad_edges, extract_triangles};

const MAX_GPU_UPLOADS_PER_FRAME: usize = 32;

// =============================================================================
// DebugRenderState — per-feature toggles for the F6 debug overlay
// =============================================================================

pub struct DebugRenderState {
    /// Whether the debug panel window is visible (F6 toggle).
    pub visible: bool,
    /// Draw polygon wireframe over chunk geometry (requires POLYGON_MODE_LINE).
    pub wireframe: bool,
    /// Draw chunk AABB bounds — represents visible-collision geometry.
    pub bbox_visible: bool,
    /// Draw player physics capsule outline — represents physical-collision shape.
    pub bbox_physical: bool,
    /// Draw player AABB hitbox.
    pub bbox_hitboxes: bool,
    /// Visualise VCT GI volume boundary and emissive-block markers.
    pub lighting_debug: bool,
    /// Draw sun and moon direction vectors (shadow cast direction).
    pub shadow_vectors: bool,

    // -- runtime state updated each frame ------------------------------------
    /// Set once the GPU pipeline is created (may be false if feature missing).
    pub wireframe_supported: bool,
    /// Most-recent VCT volume origin (world-space lower corner).
    pub vct_origin: [i32; 3],
    /// Spot light world positions + ranges, extracted from spot_lights each frame.
    pub spot_light_positions: Vec<([f32; 3], f32)>,
    /// Point light world positions + ranges, extracted from point_lights each frame.
    pub point_light_positions: Vec<([f32; 3], f32)>,
}

impl DebugRenderState {
    fn new() -> Self {
        Self {
            visible:              false,
            wireframe:            false,
            bbox_visible:         false,
            bbox_physical:        false,
            bbox_hitboxes:        false,
            lighting_debug:       false,
            shadow_vectors:       false,
            wireframe_supported:  false,
            vct_origin:            [0; 3],
            spot_light_positions:  Vec::new(),
            point_light_positions: Vec::new(),
        }
    }
}

// =============================================================================
// Debug geometry helpers — build DebugVertex line lists on the CPU
// =============================================================================

/// Push 12 edges of an AABB (world-space min/max) as line-list pairs.
fn push_aabb(out: &mut Vec<DebugVertex>, min: [f32; 3], max: [f32; 3], c: [f32; 4]) {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    let corners: [[f32; 3]; 8] = [
        [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
    ];
    let edges: [(usize, usize); 12] = [
        (0,1),(2,3),(4,5),(6,7),  // along X
        (0,2),(1,3),(4,6),(5,7),  // along Y
        (0,4),(1,5),(2,6),(3,7),  // along Z
    ];
    for (a, b) in edges {
        out.push(DebugVertex { pos: corners[a], color: c });
        out.push(DebugVertex { pos: corners[b], color: c });
    }
}

/// Push an approximate capsule outline:
///   • two XZ-circles at the top/bottom of the cylinder portion  (16 segments each)
///   • four vertical lines at cardinal offsets
/// `center` = rigid-body centre,  `radius` = PLAYER_HALF_W,  `half_h` = stance half-height.
fn push_capsule(out: &mut Vec<DebugVertex>, center: Vec3, radius: f32, half_h: f32, c: [f32; 4]) {
    const SEGS: usize = 16;
    let cap_hh = (half_h - radius).max(0.01); // half-height of the cylinder portion

    // Two XZ-circles
    for y_sign in [-1.0f32, 1.0] {
        let y = center.y + y_sign * cap_hh;
        for i in 0..SEGS {
            let a0 = (i as f32 / SEGS as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / SEGS as f32) * std::f32::consts::TAU;
            out.push(DebugVertex {
                pos:   [center.x + radius * a0.cos(), y, center.z + radius * a0.sin()],
                color: c,
            });
            out.push(DebugVertex {
                pos:   [center.x + radius * a1.cos(), y, center.z + radius * a1.sin()],
                color: c,
            });
        }
    }
    // Four vertical lines N/S/E/W
    for (dx, dz) in [(radius, 0.0f32), (-radius, 0.0), (0.0, radius), (0.0, -radius)] {
        out.push(DebugVertex {
            pos:   [center.x + dx, center.y - cap_hh, center.z + dz],
            color: c,
        });
        out.push(DebugVertex {
            pos:   [center.x + dx, center.y + cap_hh, center.z + dz],
            color: c,
        });
    }
}

/// Push a small 3-axis cross centred on `pos`.
fn push_cross(out: &mut Vec<DebugVertex>, pos: [f32; 3], size: f32, c: [f32; 4]) {
    let [x, y, z] = pos;
    for (dx, dy, dz) in [(size,0.0,0.0),(0.0,size,0.0),(0.0,0.0,size)] {
        out.push(DebugVertex { pos: [x-dx, y-dy, z-dz], color: c });
        out.push(DebugVertex { pos: [x+dx, y+dy, z+dz], color: c });
    }
}

/// Push an arrow: shaft from `from` to `from+dir`, plus a small V-head at the tip.
fn push_arrow(out: &mut Vec<DebugVertex>, from: Vec3, dir: Vec3, c: [f32; 4]) {
    if dir.length_squared() < 1e-6 { return; }
    let tip = from + dir;
    // Shaft
    out.push(DebugVertex { pos: [from.x, from.y, from.z], color: c });
    out.push(DebugVertex { pos: [tip.x,  tip.y,  tip.z],  color: c });
    // Arrow-head
    let norm = dir.normalize();
    let perp  = if norm.y.abs() < 0.9 {
        norm.cross(Vec3::Y).normalize()
    } else {
        norm.cross(Vec3::X).normalize()
    };
    let base = tip - norm * 0.6;
    let hs   = 0.3_f32;
    for sign in [1.0f32, -1.0] {
        out.push(DebugVertex { pos: [tip.x, tip.y, tip.z], color: c });
        let p = base + perp * sign * hs;
        out.push(DebugVertex { pos: [p.x, p.y, p.z], color: c });
    }
}

/// Signed angular difference a − b, result in (−π, π].
#[inline]
fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = a - b;
    while d >  std::f32::consts::PI { d -= 2.0 * std::f32::consts::PI; }
    while d < -std::f32::consts::PI { d += 2.0 * std::f32::consts::PI; }
    d
}

/// Wrap angle into (−π, π].
#[inline]
fn wrap_angle(a: f32) -> f32 {
    angle_diff(a, 0.0)
}

/// Cheap deterministic [0,1) hash from a world position + salt.
/// Used as the rarity gate for entity spawning so each candidate column
/// has a stable random roll across runs.
#[inline]
fn position_hash01(x: i32, y: i32, z: i32, salt: u64) -> f32 {
    // FNV-1a-ish mix on the packed coords + salt; final shuffle via splitmix64.
    let mut h: u64 = 0xCBF2_9CE4_8422_2325u64
        ^ (x as i64 as u64).wrapping_mul(0x100000001B3)
        ^ (y as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (z as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ salt.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 30; h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27; h = h.wrapping_mul(0x94D0_49BB_1331_11EBu64);
    h ^= h >> 31;
    (h >> 40) as f32 / (1u32 << 24) as f32
}


pub struct GameScreen {
    // -- 3D ------------------------------------------------------------------
    camera:      Camera,
    pipeline_3d: Option<Game3DPipeline>,
    world_worker: WorldWorker,
    atlas: TextureAtlas,
    pending_world_result: Option<WorldResult>,
    last_upload_us: u128,
    upload_worker: UploadWorker,
    upload_queue: VecDeque<UploadPacked>,
    // -- Debug + Profiler ----------------------------------------------------
    debug:        DebugOverlay,
    profiler:     ProfilerOverlay,
    debug_render: DebugRenderState,

    // -- Profiler accumulators ------------------------------------------------
    last_gen_time_us:      u128,
    last_mesh_time_us:     u128,
    last_gen_count:        usize,
    last_dirty_count:      usize,
    last_upload_count:     usize,
    last_worker_total_us:  u128,
    last_lod_counts:       [usize; 3],
    last_worker_pending:   usize,

    // -- Physics + Player ----------------------------------------------------
    physics_world: PhysicsWorld,
    player:        PlayerController,

    // -- Input ---------------------------------------------------------------
    cursor_x:      f64,
    cursor_y:      f64,
    mouse_dx:      f64,
    mouse_dy:      f64,
    camera_locked: bool,
    #[allow(dead_code)]
    mouse_down:    bool,
    keys:          std::collections::HashSet<PhysicalKey>,
    target_block:  Option<RaycastResult>,
    pending_block_ops: Vec<BlockOp>,
    paused:        bool,
    // -- Lighting (simplified — day/night cycle only) ------------------------
    day_night:       DayNightCycle,
    lighting_config: LightingConfig,
    sky_renderer:    SkyRenderer,
    // -- VCT GI -------------------------------------------------------------
    vct_system:    Option<VCTSystem>,
    pending_voxel_snapshot: Option<VoxelSnapshot>,
    last_voxel_snapshot: Option<VoxelSnapshot>,
    block_registry_for_vct: BlockRegistry,
    /// Dynamic spot lights extracted from the voxel volume when snapshot updates.
    spot_lights: Vec<SpotLightGPU>,
    /// Static block spot lights from light_sources, cached from last voxel snapshot.
    block_spot_lights: Vec<SpotLightGPU>,
    /// Static block point lights from light_sources, cached from last voxel snapshot.
    block_point_lights: Vec<PointLightGPU>,
    /// Dynamic point lights assembled from test entities each frame.
    point_lights: Vec<PointLightGPU>,
    /// Billboard-only mirror of point + spot lights for LightEmitterRenderer.
    /// Excludes the player's own entity lights in first-person view, so the
    /// player doesn't see a glow sphere through their own model.
    emitter_lights: Vec<EmitterLight>,
    /// Test entities with a head point light; spawned via G key.
    test_light_entities: Vec<EntityId>,
    /// Positions queued for test entity spawn; processed at render time (needs device).
    pending_test_entity_spawns: Vec<Vec3>,
    /// Volumetric renderer — halos + god rays for analytical lights.
    volumetric_renderer: Option<crate::core::vct::VolumetricRenderer>,
    /// Billboard glow renderer — bright point at each active light source.
    light_emitter_renderer: Option<crate::screens::light_emitter_renderer::LightEmitterRenderer>,
    /// Player head-mounted spotlight, toggled with L.
    player_spotlight_on: bool,
    // -- Transition ----------------------------------------------------------
    pending_action: ScreenAction,
    // -- Profiler accumulators -----------------------------------------------
    last_sky_render_us:        u128,
    last_vct_upload_us:        u128,
    last_vct_dispatch_us:      u128,
    last_transparent_us:       u128,
    last_entity_render_us:     u128,
    last_block_model_us:       u128,
    last_hand_block_us:        u128,
    last_water_render_us:      u128,
    last_outline_us:           u128,
    last_frame_total_us:       u128,
    last_upload_bytes:         u64,
    last_upload_throughput:    f64,
    /// Chunks evicted from GPU VRAM by LRU budget; passed to world worker next frame.
    gpu_evicted_pending:       Vec<(i32, i32, i32)>,
    /// Wall-clock time at GameScreen creation — used for water animation.
    start_time:                Instant,
    // -- Inventory UI --------------------------------------------------------
    inventory_ui:              InventoryUI,
    /// Cached biome ambient tint at the camera position, updated each WorldResult.
    biome_ambient_tint:        [f32; 3],
    // -- Player model -------------------------------------------------------
    player_model:    Option<PlayerModel>,
    player_renderer: Option<PlayerRenderer>,
    // -- Entity system -------------------------------------------------------
    entity_manager:    EntityManager,
    entity_renderer:   Option<EntityRenderer>,
    player_entity_id:  Option<EntityId>,
    // -- Hand block renderer (block in hand visual) -------------------------
    hand_block_renderer: Option<HandBlockRenderer>,
    // -- Block model loader -------------------------------------------------
    model_messenger: ModelMessenger,
    // -- Block model renderer (dedicated CCW renderer for blocks with 3D models) --
    block_model_renderer: Option<BlockModelRenderer>,
    /// Cached AABBs (min, max) in model-space for each loaded block model.
    /// Key = model_name (e.g. "block/spotlight").
    /// Used for outline rendering and raycast refinement.
    model_aabbs: HashMap<String, (Vec3, Vec3)>,
    /// Cached triangle soup (block-local [0..1] space, +0.5 X/Z applied) per model name.
    /// Used to populate `world_worker.model_block_tris` for per-triangle raycast.
    model_tris: HashMap<String, Vec<[f32; 9]>>,
    // -- View / animation ---------------------------------------------------
    view_mode:       ViewMode,
    /// Body yaw tracked separately from camera (lags with ±90° head range).
    body_yaw:        f32,
    /// Walking animation phase [0, 2π).
    walk_phase:      f32,
    is_walking:      bool,
    last_foot_pos:   Vec3,
    // -- Entity definitions (for per-entity light sources) ------------------
    entity_registry: crate::core::entity_definition::EntityRegistry,
    // -- Natural-spawn entity tracking --------------------------------------
    /// AI state per sheep entity, keyed by entity ID.
    sheep_ai: HashMap<EntityId, crate::core::entity_ai::SheepAI>,
    /// Reverse map: chunk → entity IDs spawned in that chunk (for despawn-on-evict).
    chunk_entities: HashMap<(i32, i32, i32), Vec<EntityId>>,
    /// Counter used to perturb the per-sheep RNG seed.
    sheep_spawn_counter: u64,
    /// True once the sheep model has been registered in EntityRenderer.
    sheep_model_registered: bool,
    /// Pending sheep spawns queued from the worker; processed at render time
    /// (requires `device` for EntityRenderer::spawn_instance).
    pending_sheep_spawns: Vec<((i32, i32, i32), Vec3, f32)>,
    // -- World save ---------------------------------------------------------
    world_metadata:  Option<WorldMetadata>,
    world_dir:       Option<std::path::PathBuf>,
    last_autosave:   Instant,
}

impl GameScreen {
    pub fn new() -> Self {
        Self::new_with_world(None)
    }

    /// Create a GameScreen for a specific named world (seed + save dir).
    pub fn new_with_world(metadata: Option<WorldMetadata>) -> Self {
        debug_log!("GameScreen", "new_with_world", "Creating GameScreen metadata={}", metadata.as_ref().map(|m| m.name.as_str()).unwrap_or("default"));

        let mut atlas   = TextureAtlas::load();
        let mut registry = BlockRegistry::load();

        // --- Bake layered face textures (composite layers into atlas) ---
        registry.bake_layered_textures(|base, layers| {
            atlas.bake_layered_texture(base, layers)
        });
        debug_log!("GameScreen", "new",
            "Layer baking complete, atlas has {} textures",
            atlas.layout().uv_map.len()
        );

        let atlas_layout = atlas.layout().clone();

        let mut physics_world = PhysicsWorld::new();
        let player   = PlayerController::new(&mut physics_world, registry.clone());

        let lighting_config = LightingConfig::load();
        let day_night = DayNightCycle::new(&lighting_config);
        let block_registry_for_vct = registry.clone();

        // Load player model (CPU only — GPU renderer created lazily in render())
        let player_model = std::fs::read_to_string("assets/player_model.json")
            .ok()
            .and_then(|json| PlayerModel::from_json(&json).ok());

        // -- Inventory UI init --
        let mut inventory_ui = {
            let config = InventoryConfig::load("config/inventory.json");
            InventoryUI::new(config)
        };
        // Populate inventory hotbar with solid blocks (first N slots)
        {
            let reg = &block_registry_for_vct;
            let hotbar_count = inventory_ui.hotbar_count();
            let mut slot_i = 0usize;
            for bid in 1u8..=255u8 {
                if slot_i >= hotbar_count { break; }
                if let Some(def) = reg.get(bid) {
                    if def.solid {
                        if let Some(key) = reg.key_for(bid) {
                            let name = if def.display_name.is_empty() { &def.id } else { &def.display_name };
                            let mut top_tex  = def.texture_for_face(2).map(|s| s.to_owned());
                            let mut side_tex = def.texture_for_face(4).map(|s| s.to_owned());
                            // Pre-extract tile images from the atlas and register them
                            if let Some(ref n) = top_tex {
                                if let Some(rgba) = atlas.tile_rgba(n) {
                                    let img = egui::ColorImage::from_rgba_unmultiplied([16, 16], &rgba);
                                    inventory_ui.add_tile_image(n.clone(), img);
                                }
                            }
                            if let Some(ref n) = side_tex {
                                if let Some(rgba) = atlas.tile_rgba(n) {
                                    let img = egui::ColorImage::from_rgba_unmultiplied([16, 16], &rgba);
                                    inventory_ui.add_tile_image(n.clone(), img);
                                }
                            }
                            // Fallback for model blocks: try every model_textures value with
                            // known search prefixes until one loads successfully.
                            if top_tex.is_none() && def.model.is_some() {
                                const MODEL_TEX_PATHS: &[&str] = &[
                                    "assets/textures/modelblocks/",
                                    "assets/models/textures/",
                                    "assets/textures/",
                                ];
                                'tex_search: for filename in def.model_textures.values() {
                                    for prefix in MODEL_TEX_PATHS {
                                        let path = format!("{}{}", prefix, filename);
                                        if let Ok(bytes) = std::fs::read(&path) {
                                            if let Ok(img) = image::load_from_memory(&bytes) {
                                                let rgba = img.to_rgba8();
                                                let (w, h) = rgba.dimensions();
                                                let mut preview = [0u8; 16 * 16 * 4];
                                                for py in 0u32..16 {
                                                    for px in 0u32..16 {
                                                        let sx = (px * w / 16).min(w.saturating_sub(1));
                                                        let sy = (py * h / 16).min(h.saturating_sub(1));
                                                        let src = ((sy * w + sx) * 4) as usize;
                                                        let dst = ((py * 16 + px) * 4) as usize;
                                                        preview[dst..dst + 4].copy_from_slice(&rgba.as_raw()[src..src + 4]);
                                                    }
                                                }
                                                let tile_key = format!("modelblock/{}", def.id);
                                                let color_img = egui::ColorImage::from_rgba_unmultiplied([16, 16], &preview);
                                                inventory_ui.add_tile_image(tile_key.clone(), color_img);
                                                top_tex  = Some(tile_key.clone());
                                                side_tex = Some(tile_key);
                                                break 'tex_search;
                                            }
                                        }
                                    }
                                }
                            }
                            let stack = ItemStack::new(key, name, 64, 64)
                                .with_color(def.color)
                                .with_textures(top_tex, side_tex)
                                .with_tab(def.inventory_tab.clone());
                            inventory_ui.inventory.hotbar_slots[slot_i] =
                                crate::screens::inventory::InventorySlot::occupied(stack);
                            slot_i += 1;
                        }
                    }
                }
            }
        }
        // Populate creative palette from block registry (with colors and textures)
        {
            let reg = block_registry_for_vct.clone();
            let mut creative_items = Vec::new();
            for block_id in 1u8..=255u8 {
                if let Some(def) = reg.get(block_id) {
                    if def.solid || def.transparent {
                        if let Some(key) = reg.key_for(block_id) {
                            let mut top_tex  = def.texture_for_face(2).map(|s| s.to_owned());
                            let mut side_tex = def.texture_for_face(4).map(|s| s.to_owned());
                            if let Some(ref n) = top_tex {
                                if let Some(rgba) = atlas.tile_rgba(n) {
                                    let img = egui::ColorImage::from_rgba_unmultiplied([16, 16], &rgba);
                                    inventory_ui.add_tile_image(n.clone(), img);
                                }
                            }
                            if let Some(ref n) = side_tex {
                                if let Some(rgba) = atlas.tile_rgba(n) {
                                    let img = egui::ColorImage::from_rgba_unmultiplied([16, 16], &rgba);
                                    inventory_ui.add_tile_image(n.clone(), img);
                                }
                            }
                            if top_tex.is_none() && def.model.is_some() {
                                const MODEL_TEX_PATHS: &[&str] = &[
                                    "assets/textures/modelblocks/",
                                    "assets/models/textures/",
                                    "assets/textures/",
                                ];
                                'tex_search: for filename in def.model_textures.values() {
                                    for prefix in MODEL_TEX_PATHS {
                                        let path = format!("{}{}", prefix, filename);
                                        if let Ok(bytes) = std::fs::read(&path) {
                                            if let Ok(img) = image::load_from_memory(&bytes) {
                                                let rgba = img.to_rgba8();
                                                let (w, h) = rgba.dimensions();
                                                let mut preview = [0u8; 16 * 16 * 4];
                                                for py in 0u32..16 {
                                                    for px in 0u32..16 {
                                                        let sx = (px * w / 16).min(w.saturating_sub(1));
                                                        let sy = (py * h / 16).min(h.saturating_sub(1));
                                                        let src = ((sy * w + sx) * 4) as usize;
                                                        let dst = ((py * 16 + px) * 4) as usize;
                                                        preview[dst..dst + 4].copy_from_slice(&rgba.as_raw()[src..src + 4]);
                                                    }
                                                }
                                                let tile_key = format!("modelblock/{}", def.id);
                                                let color_img = egui::ColorImage::from_rgba_unmultiplied([16, 16], &preview);
                                                inventory_ui.add_tile_image(tile_key.clone(), color_img);
                                                top_tex  = Some(tile_key.clone());
                                                side_tex = Some(tile_key);
                                                break 'tex_search;
                                            }
                                        }
                                    }
                                }
                            }
                            creative_items.push(ItemStack::new(
                                key,
                                if def.display_name.is_empty() { &def.id } else { &def.display_name },
                                64,
                                64,
                            ).with_color(def.color).with_textures(top_tex, side_tex)
                             .with_tab(def.inventory_tab.clone()));
                        }
                    }
                }
            }
            inventory_ui.set_creative_items(creative_items);
        }

        let world_dir = metadata.as_ref().map(|m| crate::core::save_system::world_dir(&m.name));
        let seed = metadata.as_ref().map(|m| m.seed).unwrap_or(12345u64);
        let world_worker = WorldWorker::new_with_seed(atlas_layout, registry, seed, world_dir.clone());

        Self {
            camera:      Camera::new(840, 480),
            pipeline_3d: None,
            world_worker,
            pending_world_result: None,
            last_upload_us: 0,
            upload_worker: UploadWorker::new(),
            upload_queue: VecDeque::new(),
            atlas,

            debug:        DebugOverlay::new(),
            profiler:     ProfilerOverlay::new(),
            debug_render: DebugRenderState::new(),

            last_gen_time_us:      0,
            last_mesh_time_us:     0,
            last_gen_count:        0,
            last_dirty_count:      0,
            last_upload_count:     0,
            last_worker_total_us:  0,
            last_lod_counts:       [0; 3],
            last_worker_pending:   0,

            physics_world,
            player,

            cursor_x:      0.0,
            cursor_y:      0.0,
            mouse_dx:      0.0,
            mouse_dy:      0.0,
            camera_locked: false,
            mouse_down:    false,
            keys:          std::collections::HashSet::new(),
            target_block:  None,
            paused:        false,
            pending_block_ops: Vec::new(),

            day_night,
            lighting_config,
            sky_renderer: SkyRenderer::new(),
            vct_system: None,
            pending_voxel_snapshot: None,
            last_voxel_snapshot: None,
            block_registry_for_vct,
            spot_lights: Vec::new(),
            block_spot_lights: Vec::new(),
            block_point_lights: Vec::new(),
            point_lights: Vec::new(),
            emitter_lights: Vec::new(),
            test_light_entities: Vec::new(),
            pending_test_entity_spawns: Vec::new(),
            volumetric_renderer: None,
            light_emitter_renderer: None,
            player_spotlight_on: false,
            pending_action: ScreenAction::None,
            last_sky_render_us:        0,
            last_vct_upload_us:        0,
            last_vct_dispatch_us:      0,
            last_transparent_us:       0,
            last_entity_render_us:     0,
            last_block_model_us:       0,
            last_hand_block_us:        0,
            last_water_render_us:      0,
            last_outline_us:           0,
            last_frame_total_us:       0,
            last_upload_bytes:         0,
            last_upload_throughput:    0.0,
            gpu_evicted_pending:       Vec::new(),
            start_time:                Instant::now(),
            biome_ambient_tint:        [1.0f32; 3],
            player_model,
            player_renderer: None,
            entity_manager:   EntityManager::new(),
            entity_renderer:  None,
            player_entity_id: None,
            hand_block_renderer: None,
            model_messenger: ModelMessenger::new(),
            block_model_renderer: None,
            model_aabbs: HashMap::new(),
            model_tris:  HashMap::new(),
            view_mode:       ViewMode::FirstPerson,
            body_yaw:        -std::f32::consts::FRAC_PI_2,
            walk_phase:      0.0,
            is_walking:      false,
            last_foot_pos:   Vec3::ZERO,
            inventory_ui,
            entity_registry: crate::core::entity_definition::EntityRegistry::load(),
            sheep_ai:                HashMap::new(),
            chunk_entities:          HashMap::new(),
            sheep_spawn_counter:     0,
            sheep_model_registered:  false,
            pending_sheep_spawns:    Vec::new(),
            world_metadata:  metadata,
            world_dir,
            last_autosave:   Instant::now(),
        }
    }

    /// Save all chunks + player state + entities + update world.json last_played.
    fn do_save_world(&mut self) {
        let dir = match self.world_dir.as_ref() {
            Some(d) => d.clone(),
            None => return,
        };

        // Chunks saved on worker thread
        self.world_worker.request_save_all(dir.clone());

        if let Some(meta) = self.world_metadata.as_mut() {
            // --- Player ---
            meta.player = self.player.to_save_data(&self.physics_world);

            // --- Entities (skip the player entity) ---
            let skip: Vec<_> = self.player_entity_id.into_iter().collect();
            meta.entities = self.entity_manager.to_save_data(&skip);

            meta.last_played = chrono::Local::now().to_rfc3339();

            if let Err(e) = save_world_meta(&dir, meta) {
                debug_log!("GameScreen", "do_save_world", "Meta save error: {}", e);
            }
        }

        self.last_autosave = Instant::now();
        debug_log!("GameScreen", "do_save_world", "World save requested to {:?}", dir);
    }

    // ---------------------------------------------------------------------
    // Natural entity spawning
    // ---------------------------------------------------------------------

    /// Run spawn rules against per-chunk surface samples (one batch per WorldResult).
    /// Newly created spawn intents are buffered in `pending_sheep_spawns` and the
    /// GPU instance is created at render time when `device` is available.
    fn process_entity_spawn_candidates(
        &mut self,
        fresh_chunk_surfaces: &[((i32, i32, i32), Vec<(i32, i32, i32, u8)>)],
    ) {
        if fresh_chunk_surfaces.is_empty() { return; }

        // Active sheep count (alive + queued).
        let alive_sheep = self.sheep_ai.len()
            + self.pending_sheep_spawns.len();

        // Iterate every entity definition that has spawn rules.
        let registry_keys: Vec<String> = self.entity_registry.registry_order.clone();
        for key in &registry_keys {
            let def = match self.entity_registry.defs.get(key) { Some(d) => d, None => continue };
            let rule = match def.spawn.as_ref() {
                Some(r) if r.enabled => r,
                _ => continue,
            };

            // Hard global cap for this entity type.
            if key == "sheep" && alive_sheep as u32 >= rule.max_alive { continue; }

            // Pre-resolve block IDs from spawn_on_blocks
            let allowed_ids: Vec<u8> = rule.spawn_on_blocks.iter()
                .filter_map(|n| self.block_registry_for_vct.id_for(n))
                .collect();

            for (chunk_key, samples) in fresh_chunk_surfaces {
                let mut spawned_in_chunk = 0u32;
                for &(wx, wy, wz, bid) in samples {
                    if spawned_in_chunk >= rule.max_per_chunk { break; }
                    if wy < rule.min_y || wy > rule.max_y { continue; }
                    if !allowed_ids.is_empty() && !allowed_ids.contains(&bid) { continue; }

                    // Rarity gate — deterministic hash of position so each
                    // generated chunk produces a stable set of spawns.
                    let r = position_hash01(wx, wy, wz, self.sheep_spawn_counter);
                    if r >= rule.rarity { continue; }

                    let spawn_pos = Vec3::new(wx as f32 + 0.5, (wy + 1) as f32, wz as f32 + 0.5);
                    self.sheep_spawn_counter = self.sheep_spawn_counter.wrapping_add(1);

                    if key == "sheep" {
                        self.pending_sheep_spawns.push((*chunk_key, spawn_pos, 0.0));
                        spawned_in_chunk += 1;
                    }
                }
            }
        }
    }

    /// Free AI state, EntityManager records, and EntityRenderer GPU instances
    /// for every entity associated with the given chunk keys.
    fn despawn_entities_in_chunks(&mut self, chunks: &[(i32, i32, i32)]) {
        for key in chunks {
            if let Some(ids) = self.chunk_entities.remove(key) {
                for id in ids {
                    self.sheep_ai.remove(&id);
                    self.entity_manager.despawn(id);
                    if let Some(ref mut er) = self.entity_renderer {
                        er.despawn_instance(id);
                    }
                    debug_log!("GameScreen", "despawn_entities_in_chunks",
                        "Despawned entity {:?} from evicted chunk {:?}", id, key);
                }
            }
            // Also drop any queued (not-yet-instantiated) spawns inside this chunk.
            self.pending_sheep_spawns.retain(|(ck, _, _)| ck != key);
        }
    }

    /// Advance every sheep AI by `dt` and apply position/yaw + queued block ops.
    fn tick_sheep_ai(&mut self, dt: f32) {
        if self.sheep_ai.is_empty() { return; }
        let player_pos = self.player.foot_position(&self.physics_world);
        let snap = self.last_voxel_snapshot.as_ref();

        // Collect (id, pos, yaw) up front to release the borrow on entity_manager.
        let entries: Vec<(EntityId, Vec3, f32)> = self.sheep_ai.keys()
            .filter_map(|id| {
                self.entity_manager.get(*id).map(|e| (
                    e.id,
                    e.transform.position,
                    e.transform.yaw,
                ))
            })
            .collect();

        for (id, pos, yaw) in entries {
            let Some(ai) = self.sheep_ai.get_mut(&id) else { continue };
            let result = crate::core::entity_ai::tick_sheep(
                ai, pos, yaw, dt, snap, &self.block_registry_for_vct, player_pos,
            );
            if let Some(entity) = self.entity_manager.get_mut(id) {
                entity.transform.position = result.new_position;
                entity.transform.yaw      = result.new_yaw;
            }
            for op in result.block_ops {
                self.pending_block_ops.push(op);
            }
        }
    }

    fn process_input(&mut self, _dt: f64) {
        if self.camera_locked && (self.mouse_dx != 0.0 || self.mouse_dy != 0.0) {
            self.camera.set_sensitivity(config::mouse_sensitivity());
            self.camera.rotate(self.mouse_dx, self.mouse_dy);
        }
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;

        // In third-person with Alt held (orbit mode), movement is relative to body facing.
        let alt_held = self.keys.contains(&PhysicalKey::Code(KeyCode::AltLeft))
                    || self.keys.contains(&PhysicalKey::Code(KeyCode::AltRight));
        let orbit_mode = self.view_mode == ViewMode::ThirdPerson && alt_held;

        let forward_xz = if orbit_mode {
            let (sy, cy) = self.body_yaw.sin_cos();
            Vec3::new(cy, 0.0, sy)
        } else {
            let f = self.camera.forward();
            let flat = Vec3::new(f.x, 0.0, f.z);
            let len = flat.length();
            if len > 1e-6 { flat / len } else { Vec3::ZERO }
        };
        let right_xz = forward_xz.cross(Vec3::Y);

        let mut move_dir = Vec3::ZERO;
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyW)) { move_dir += forward_xz; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyS)) { move_dir -= forward_xz; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyA)) { move_dir -= right_xz; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyD)) { move_dir += right_xz; }

        let move_dir = if move_dir.length_squared() > 1e-6 {
            move_dir.normalize()
        } else {
            Vec3::ZERO
        };

        let shift_held = self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftLeft))
                      || self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftRight));
        let ctrl_held  = self.keys.contains(&PhysicalKey::Code(KeyCode::ControlLeft))
                      || self.keys.contains(&PhysicalKey::Code(KeyCode::ControlRight));

        // Stance switching — only in normal (non-fly) mode
        if !self.player.is_fly_mode() {
            use crate::core::player::PlayerStance;
            let desired = if ctrl_held {
                PlayerStance::Prone
            } else if shift_held {
                PlayerStance::Crouching
            } else {
                PlayerStance::Standing
            };
            self.player.set_stance(&mut self.physics_world, desired);
        }

        let jump     = self.keys.contains(&PhysicalKey::Code(KeyCode::Space));
        // fly_down uses Shift in fly-mode; in normal mode Shift is used for crouch (handled above)
        let fly_down = self.player.is_fly_mode() && shift_held;
        self.player.apply_movement(&mut self.physics_world, move_dir, jump, fly_down);
    }
}

impl Screen for GameScreen {
    fn name(&self) -> &str { "Game" }
    fn load(&mut self) { debug_log!("GameScreen", "load", "Loading resources"); }
    fn start(&mut self) {
        debug_log!("GameScreen", "start", "First-frame init");

        // Restore saved player position + state if this world was loaded from disk.
        // This runs after physics is fully constructed (unlike new_with_world).
        if let Some(meta) = self.world_metadata.clone() {
            // Only restore if position is non-default (i.e. the player actually saved one).
            let saved_pos = meta.player.position;
            let is_default_pos = saved_pos == [8.0_f32, 250.0, 30.0];
            if !is_default_pos {
                self.player.apply_save_data(&mut self.physics_world, &meta.player);
            }

            // Restore world entities (skip the player entity which is special).
            let skip: Vec<_> = self.player_entity_id.into_iter().collect();
            if !meta.entities.is_empty() {
                self.entity_manager.restore_from_save(&meta.entities, &skip);
                debug_log!("GameScreen", "start",
                    "Restored {} world entities", meta.entities.len());
            }
        }
    }

    fn update(&mut self, dt: f64) {
        let t_update_start = Instant::now();

        // Auto-save every 5 minutes when a world_dir is set
        if self.world_dir.is_some()
            && self.last_autosave.elapsed().as_secs() >= 300
        {
            self.do_save_world();
        }

        // Advance day-night cycle
        self.day_night.update(dt);

        self.process_input(dt);

        // Physics step
        self.physics_world.step(dt as f32);
        self.player.update(&mut self.physics_world);

        // ── Walk phase & body yaw tracking ──────────────────────────────────
        {
            let foot_pos = self.player.foot_position(&self.physics_world);
            let moved_xz = {
                let d = foot_pos - self.last_foot_pos;
                (d.x * d.x + d.z * d.z).sqrt()
            };
            self.last_foot_pos = foot_pos;

            self.is_walking = moved_xz > 0.001;
            if self.is_walking {
                self.walk_phase = (self.walk_phase + moved_xz * std::f32::consts::PI) % (2.0 * std::f32::consts::PI);
            }

            // Body yaw: snaps only when head would exceed ±90° from body.
            // Skip updates in orbit mode (Alt held in third-person) so the body stays fixed
            // while the camera orbits freely.
            let alt_held = self.keys.contains(&PhysicalKey::Code(KeyCode::AltLeft))
                        || self.keys.contains(&PhysicalKey::Code(KeyCode::AltRight));
            let orbit_mode = self.view_mode == ViewMode::ThirdPerson && alt_held;

            if !orbit_mode {
                let cam_yaw = self.camera.yaw();
                let head_offset = angle_diff(cam_yaw, self.body_yaw);
                if self.is_walking {
                    // Smoothly rotate body toward camera direction while walking.
                    let speed = (dt as f32 * 8.0).min(1.0);
                    self.body_yaw += head_offset * speed;
                    self.body_yaw = wrap_angle(self.body_yaw);
                } else {
                    use std::f32::consts::FRAC_PI_2;
                    if head_offset > FRAC_PI_2 {
                        self.body_yaw = wrap_angle(cam_yaw - FRAC_PI_2);
                    } else if head_offset < -FRAC_PI_2 {
                        self.body_yaw = wrap_angle(cam_yaw + FRAC_PI_2);
                    }
                }
            }
        }

        // Camera sync
        let eye = self.player.eye_position(&self.physics_world);
        if self.view_mode == ViewMode::ThirdPerson {
            // Place camera behind & slightly above the player.
            let fwd = self.camera.forward();
            self.camera.position = eye - fwd * 4.5 + Vec3::Y * 0.5;
        } else {
            // First-person: push camera forward to the front face of the head cube.
            // Head depth = 8 model units = 0.5 blocks; half = 0.25 blocks, scaled by PLAYER_SIZE.
            let fwd = self.camera.forward();
            let face_offset = 0.25 * 0.85; // half-depth * PLAYER_SIZE
            self.camera.position = eye + Vec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero() * face_offset;
        }

        // World update request
        let cam_fwd  = self.camera.forward();
        let ray_dir  = if self.camera_locked { Some(cam_fwd) } else { None };
        let foot_pos = self.player.foot_position(&self.physics_world);
        self.world_worker.request_update(
            self.camera.position,
            cam_fwd,
            ray_dir,
            std::mem::take(&mut self.pending_block_ops),
            std::mem::take(&mut self.gpu_evicted_pending),
            foot_pos,
        );

        // Non-blocking receive
        if let Some(mut result) = self.world_worker.try_recv_result() {
            // Swimming detection — only in normal (non-fly) mode
            if !self.player.is_fly_mode() {
                self.player.update_swimming(
                    &mut self.physics_world,
                    result.foot_block,
                    result.foot_block_above,
                );
            }

            self.target_block = result.raycast_result;
            self.physics_world.sync_chunks_ready(
                std::mem::take(&mut result.physics_ready),
                &result.evicted,
            );

            // Capture voxel snapshot for VCT GI
            if let Some(snap) = result.voxel_snapshot.take() {
                self.pending_voxel_snapshot = Some(snap);
            }

            // Capture profiler data from WorldResult
            self.last_gen_time_us     = result.gen_time_us;
            self.last_mesh_time_us    = result.mesh_time_us;
            self.last_gen_count       = result.meshes.len();
            self.last_dirty_count     = result.dirty_count;
            self.last_worker_total_us = result.worker_total_us;
            self.last_lod_counts      = result.lod_counts;
            self.last_worker_pending  = result.pending;
            self.biome_ambient_tint   = result.biome_ambient_tint;

            // Process fresh chunk surfaces against entity spawn rules.
            // Surviving spawn candidates are queued for next-frame GPU instantiation.
            self.process_entity_spawn_candidates(&result.fresh_chunk_surfaces);

            // Despawn sheep AI state for chunks that were just evicted.
            if !result.evicted.is_empty() {
                self.despawn_entities_in_chunks(&result.evicted);
            }

            self.pending_world_result = Some(result);
        }

        // Advance sheep AI (independent of WorldResult arrival cadence).
        self.tick_sheep_ai(dt as f32);

        let update_ms = t_update_start.elapsed().as_secs_f64() * 1000.0;

        // Update debug overlay
        let vram = self.pipeline_3d.as_ref().map_or(0, |p| p.vram_usage());
        let cpu_work_ms = update_ms + self.last_frame_total_us as f64 / 1000.0;
        self.debug.update(dt, vram, cpu_work_ms);
        self.debug.update_terrain(
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z,
        );

        // Update look direction in debug overlay
        let fwd = self.camera.forward();
        self.debug.update_look(
            self.camera.yaw(),
            self.camera.pitch(),
            [fwd.x, fwd.y, fwd.z],
        );

        // Update hovered block info in debug overlay
        if let Some(ref r) = self.target_block {
            let id = r.block_id;
            if let Some(def) = self.block_registry_for_vct.get(id) {
                let key = self.block_registry_for_vct.key_for(id).unwrap_or("unknown");
                let placement = format!("{:?}", def.placement_mode);
                let model_type = def.model_type_label();
                self.debug.update_hovered_block(id, key, &placement, model_type);
            }
        } else {
            self.debug.update_hovered_block(0, "", "", "");
        }
    }

    fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view:    &wgpu::TextureView,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        format:  wgpu::TextureFormat,
        width:   u32,
        height:  u32,
    ) {
        let t_frame_start = Instant::now();

        // Lazy-init VCT system + 3D pipeline
        if self.vct_system.is_none() {
            self.vct_system = Some(VCTSystem::new(device, queue));
        }
        if self.volumetric_renderer.is_none() {
            self.volumetric_renderer = Some(crate::core::vct::VolumetricRenderer::new(device, format));
        }
        if self.light_emitter_renderer.is_none() {
            self.light_emitter_renderer = Some(
                crate::screens::light_emitter_renderer::LightEmitterRenderer::new(device, format)
            );
        }
        if self.pipeline_3d.is_none() {
            let vct = self.vct_system.as_ref().unwrap();
            self.pipeline_3d = Some(Game3DPipeline::new(device, queue, format, &vct.frag_bgl));
        }
        // Lazy-init entity renderer + player model registration
        if self.entity_renderer.is_none() {
            if let Some(ref model) = self.player_model {
                let vct = self.vct_system.as_ref().unwrap();
                let mut er = EntityRenderer::new(device, format, &vct.frag_bgl);
                er.register_model(
                    device, queue,
                    "player", model,
                    PLAYER_BONE_ORDER,
                    Some("assets/textures/entity/player/player_skin.png"),
                );
                let pid = self.entity_manager.spawn(
                    EntityTransform::at(Vec3::ZERO),
                    "player",
                );
                er.spawn_instance(device, pid, "player");
                self.player_renderer    = Some(PlayerRenderer::new(model));
                self.player_entity_id   = Some(pid);
                self.entity_renderer    = Some(er);
            }
        }

        // Lazy-register the sheep model once the entity renderer exists.
        if !self.sheep_model_registered && self.entity_renderer.is_some() {
            if let Ok(json) = std::fs::read_to_string("assets/models/entity/sheep_model.json") {
                if let Ok(sheep_model) = PlayerModel::from_json(&json) {
                    if let Some(ref mut er) = self.entity_renderer {
                        let skin = self.entity_registry.defs.get("sheep")
                            .and_then(|d| d.skin_path.clone());
                        er.register_model(
                            device, queue,
                            "sheep", &sheep_model,
                            crate::screens::sheep_renderer::SHEEP_BONE_ORDER,
                            skin.as_deref(),
                        );
                        self.sheep_model_registered = true;
                        debug_log!("GameScreen", "render", "Registered sheep model");
                    }
                } else {
                    debug_log!("GameScreen", "render", "Failed to parse sheep_model.json");
                    self.sheep_model_registered = true; // don't retry every frame
                }
            } else {
                self.sheep_model_registered = true;
            }
        }

        // Process pending sheep spawns (needs device for GPU instance).
        if !self.pending_sheep_spawns.is_empty()
            && self.sheep_model_registered
            && self.entity_renderer.is_some()
        {
            let spawns = std::mem::take(&mut self.pending_sheep_spawns);
            for (chunk_key, pos, yaw) in spawns {
                let mut transform = EntityTransform::at(pos);
                transform.yaw = yaw;
                let eid = self.entity_manager.spawn(transform, "sheep");
                if let Some(ref mut er) = self.entity_renderer {
                    er.spawn_instance(device, eid, "sheep");
                }
                let seed = self.world_metadata.as_ref().map(|m| m.seed).unwrap_or(0);
                let ai = crate::core::entity_ai::SheepAI::new(eid, chunk_key, pos, seed);
                self.sheep_ai.insert(eid, ai);
                self.chunk_entities.entry(chunk_key).or_default().push(eid);
                debug_log!("GameScreen", "render",
                    "Spawned sheep {:?} at [{:.1},{:.1},{:.1}] (chunk {:?})",
                    eid, pos.x, pos.y, pos.z, chunk_key);
            }
        }

        // Process pending test entity spawns (requires device — handled here at render time)
        if !self.pending_test_entity_spawns.is_empty() && self.entity_renderer.is_some() {
            let spawns = std::mem::take(&mut self.pending_test_entity_spawns);
            for pos in spawns {
                let eid = self.entity_manager.spawn(EntityTransform::at(pos), "player");
                self.test_light_entities.push(eid);
                if let Some(ref mut er) = self.entity_renderer {
                    er.spawn_instance(device, eid, "player");
                }
                debug_log!("GameScreen", "render",
                    "Spawned test light entity {:?} at [{:.1},{:.1},{:.1}]", eid, pos.x, pos.y, pos.z);
            }
        }

        // Lazy-init block model renderer
        if self.block_model_renderer.is_none() {
            let vct = self.vct_system.as_ref().unwrap();
            self.block_model_renderer = Some(BlockModelRenderer::new(device, format, &vct.frag_bgl));
        }

        // -- Block model loading via ModelMessenger --------------------------
        // On first frame, scan registry for blocks with custom models
        if self.model_messenger.loaded_count() == 0 && self.model_messenger.pending_count() == 0 {
            self.model_messenger.register_block_models(
                &self.block_registry_for_vct,
                "assets/",
            );
        }
        // Process pending model loads (synchronous disk read + parse)
        self.model_messenger.process_pending();
        // Build per-model texture atlases on GPU
        self.model_messenger.process_pending_gpu(device, queue);

        // Register newly loaded block models with BlockModelRenderer.
        // Collect (model_name, edge_verts) pairs to register with the 3D pipeline
        // after the bmr borrow ends (avoids borrow conflicts with self.pipeline_3d).
        let mut pending_outlines: Vec<(String, Vec<[f32; 3]>)> = Vec::new();

        if let Some(ref mut bmr) = self.block_model_renderer {
            for resp in self.model_messenger.drain_responses() {
                if let (Some(block_id), Some(shadow_cubes)) = (resp.block_id, resp.model_shadow_cubes) {
                    if self.block_registry_for_vct.model_shadow_cubes(block_id).is_none() {
                        self.block_registry_for_vct.set_model_shadow_cubes(block_id, shadow_cubes);
                        // Trigger a repack of the VCT volume since the model's shadow data is now known
                        if let Some(snap) = self.last_voxel_snapshot.as_ref() {
                            self.pending_voxel_snapshot = Some(crate::core::vct::voxel_volume::VoxelSnapshot {
                                origin: snap.origin,
                                size: snap.size,
                                blocks: snap.blocks.clone(),
                            });
                        }
                    }
                }

                if let Some(model_data) = resp.model {
                    if !model_data.is_empty() {
                        // Cache the model's AABB for outline rendering and raycast refinement.
                        // Model vertices are centered at X=0, Z=0 (range ~[-0.5..0.5]).
                        // BlockModelRenderer translates to (wx+0.5, wy, wz+0.5), so we apply
                        // the same +0.5 X/Z shift here to convert to block-corner space [0..1].
                        let render_offset = Vec3::new(0.5, 0.0, 0.5);
                        self.model_aabbs.insert(
                            resp.model_name.clone(),
                            (model_data.aabb_min + render_offset, model_data.aabb_max + render_offset),
                        );

                        // Extract quad edges and shift to block-local space for exact outline.
                        let edges_raw = extract_quad_edges(&model_data.vertices, &model_data.indices);
                        let edges: Vec<[f32; 3]> = edges_raw.iter()
                            .map(|&[x, y, z]| [x + 0.5, y, z + 0.5])
                            .collect();
                        pending_outlines.push((resp.model_name.clone(), edges));

                        // Extract triangles (block-local space) for per-triangle raycast.
                        let tris_raw = extract_triangles(&model_data.vertices, &model_data.indices);
                        let tris: Vec<[f32; 9]> = tris_raw.iter()
                            .map(|&t| [t[0]+0.5, t[1], t[2]+0.5, t[3]+0.5, t[4], t[5]+0.5, t[6]+0.5, t[7], t[8]+0.5])
                            .collect();
                        self.model_tris.insert(resp.model_name.clone(), tris);

                        if let Some((tex, view)) = resp.atlas_texture {
                            bmr.register_model(
                                device,
                                &resp.model_name,
                                &model_data.vertices,
                                &model_data.indices,
                                tex, view,
                            );
                            debug_log!(
                                "GameScreen", "render",
                                "Registered block model '{}' ({} verts, packed atlas) aabb=[{:.2}..{:.2}]",
                                resp.model_name, model_data.vertices.len(),
                                model_data.aabb_min, model_data.aabb_max,
                            );
                        } else {
                            bmr.register_model_fallback(
                                device, queue,
                                &resp.model_name,
                                &model_data.vertices,
                                &model_data.indices,
                            );
                            debug_log!(
                                "GameScreen", "render",
                                "Registered block model '{}' ({} verts, fallback texture) aabb=[{:.2}..{:.2}]",
                                resp.model_name, model_data.vertices.len(),
                                model_data.aabb_min, model_data.aabb_max,
                            );
                        }
                    }
                }
            }
        } else {
            // Consume responses even if renderer isn't ready yet — still cache AABBs
            for resp in self.model_messenger.drain_responses() {
                if let Some(model_data) = resp.model {
                    if !model_data.is_empty() {
                        let render_offset = Vec3::new(0.5, 0.0, 0.5);
                        self.model_aabbs.insert(
                            resp.model_name.clone(),
                            (model_data.aabb_min + render_offset, model_data.aabb_max + render_offset),
                        );
                        let edges_raw = extract_quad_edges(&model_data.vertices, &model_data.indices);
                        let edges: Vec<[f32; 3]> = edges_raw.iter()
                            .map(|&[x, y, z]| [x + 0.5, y, z + 0.5])
                            .collect();
                        pending_outlines.push((resp.model_name.clone(), edges));

                        // Extract triangles for per-triangle raycast.
                        let tris_raw = extract_triangles(&model_data.vertices, &model_data.indices);
                        let tris: Vec<[f32; 9]> = tris_raw.iter()
                            .map(|&t| [t[0]+0.5, t[1], t[2]+0.5, t[3]+0.5, t[4], t[5]+0.5, t[6]+0.5, t[7], t[8]+0.5])
                            .collect();
                        self.model_tris.insert(resp.model_name.clone(), tris);
                    }
                }
            }
        }

        // Upload per-model outline edge VBs to the 3D pipeline.
        let new_models_loaded = !pending_outlines.is_empty();
        if new_models_loaded {
            if let Some(ref mut pipeline) = self.pipeline_3d {
                for (name, edges) in pending_outlines {
                    pipeline.register_model_outline(device, queue, &name, &edges);
                }
            }
        }

        // Rebuild block_id-keyed maps in the world worker whenever new models arrive.
        if new_models_loaded {
            let mut id_aabb_map: HashMap<u8, ([f32; 3], [f32; 3])> = HashMap::new();
            let mut id_tris_map: HashMap<u8, Vec<[f32; 9]>> = HashMap::new();
            for bid in 1u8..=255u8 {
                if let Some(model_name) = self.block_registry_for_vct.model_name_for(bid) {
                    if let Some(&(mn, mx)) = self.model_aabbs.get(&model_name) {
                        id_aabb_map.insert(bid, (mn.to_array(), mx.to_array()));
                    }
                    if let Some(tris) = self.model_tris.get(&model_name) {
                        id_tris_map.insert(bid, tris.clone());
                    }
                }
            }
            *self.world_worker.model_block_aabbs.write().unwrap() = id_aabb_map;
            *self.world_worker.model_block_tris.write().unwrap() = id_tris_map;
        }

        self.atlas.ensure_gpu(device, queue);

        // Lazy-init hand block renderer (GI-lit, attached to right-arm bone)
        if self.hand_block_renderer.is_none() {
            if let Some(atlas_view) = self.atlas.texture_view() {
                let atlas_layout = self.atlas.layout().clone();
                let vct = self.vct_system.as_ref().unwrap();
                self.hand_block_renderer = Some(HandBlockRenderer::new(
                    device, format, atlas_view, atlas_layout, &vct.frag_bgl,
                ));
            }
        }

        // --- Dispatch world result: eviction + submit to packing worker ---
        if let Some(mut result) = self.pending_world_result.take() {
            let pipeline = self.pipeline_3d.as_mut().unwrap();

            // Eviction — remove GPU chunk meshes and despawn any model instances in those chunks
            if !result.evicted.is_empty() {
                pipeline.remove_chunk_meshes(&result.evicted);
                if let Some(ref mut bmr) = self.block_model_renderer {
                    bmr.despawn_in_chunks(&result.evicted);
                }
            }

            // Remeshed chunks — clear old model spawns and stale transparent meshes.
            // Transparent meshes that are now empty (no glass blocks remain) would
            // otherwise persist because an empty result produces no upload job.
            if !result.meshed_chunk_keys.is_empty() {
                pipeline.evict_transparent_for_keys(&result.meshed_chunk_keys);
                if let Some(ref mut bmr) = self.block_model_renderer {
                    bmr.despawn_in_chunks(&result.meshed_chunk_keys);
                }
            }

            // Spawn model instances for blocks found in this frame's dirty chunks
            for &(wx, wy, wz, bid, rotation) in &result.model_block_positions {
                let pos = (wx, wy, wz);
                if self.block_model_renderer.as_ref().map(|b| b.has_instance(pos)).unwrap_or(false) {
                    continue;
                }
                let model_name = match self.block_registry_for_vct.model_name_for(bid) {
                    Some(n) => n,
                    None => continue,
                };
                // Only spawn if the model is already loaded — otherwise it will be spawned
                // next frame when the model finishes loading and the chunk re-meshes dirty
                if !self.model_messenger.is_loaded(&model_name) {
                    continue;
                }
                if let Some(ref mut bmr) = self.block_model_renderer {
                    bmr.spawn(device, pos, &model_name, rotation);
                }
            }

            // New solid meshes → UploadWorker (CPU packing)
            if !result.meshes.is_empty() {
                let jobs: Vec<UploadJob> = result.meshes
                    .into_iter()
                    .map(|(key, vertices, indices, aabb_min, aabb_max)| UploadJob {
                        key, vertices, indices, aabb_min, aabb_max, kind: MeshKind::Opaque,
                    })
                    .collect();
                self.upload_worker.submit(jobs);
            }
            // Water meshes → separate upload queue
            if !result.water_meshes.is_empty() {
                let water_jobs: Vec<UploadJob> = result.water_meshes
                    .into_iter()
                    .map(|(key, vertices, indices, aabb_min, aabb_max)| UploadJob {
                        key, vertices, indices, aabb_min, aabb_max, kind: MeshKind::Water,
                    })
                    .collect();
                self.upload_worker.submit(water_jobs);
            }
            // Glass / transparent solid meshes → glass upload queue
            if !result.transparent_meshes.is_empty() {
                let glass_jobs: Vec<UploadJob> = result.transparent_meshes
                    .into_iter()
                    .map(|(key, vertices, indices, aabb_min, aabb_max)| UploadJob {
                        key, vertices, indices, aabb_min, aabb_max, kind: MeshKind::Glass,
                    })
                    .collect();
                self.upload_worker.submit(glass_jobs);
            }
        }

        // --- Poll packed results into GPU upload queue ---
        {
            let packed_results = self.upload_worker.poll();
            for r in packed_results {
                self.upload_queue.push_back(r);
            }
        }

        // --- Budgeted GPU upload ---
        let upload_time_us;
        let upload_count;
        {
            let pipeline = self.pipeline_3d.as_mut().unwrap();
            let budget = MAX_GPU_UPLOADS_PER_FRAME.min(self.upload_queue.len());

            if budget > 0 {
                let t0 = Instant::now();
                let mut total_bytes = 0u64;

                for _ in 0..budget {
                    let r = self.upload_queue.pop_front().unwrap();

                    let vb_size = r.vertex_data.len() as u64;
                    let ib_size = r.index_data.len() as u64;
                    total_bytes += vb_size + ib_size;

                    let vb = device.create_buffer(&wgpu::BufferDescriptor {
                        label:              Some("Chunk VB (async)"),
                        size:               vb_size,
                        usage:              wgpu::BufferUsages::VERTEX
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    vb.slice(..).get_mapped_range_mut()
                        .copy_from_slice(&r.vertex_data);
                    vb.unmap();

                    let ib = device.create_buffer(&wgpu::BufferDescriptor {
                        label:              Some("Chunk IB (async)"),
                        size:               ib_size,
                        usage:              wgpu::BufferUsages::INDEX
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    ib.slice(..).get_mapped_range_mut()
                        .copy_from_slice(&r.index_data);
                    ib.unmap();

                    match r.kind {
                        MeshKind::Water => {
                            pipeline.insert_water_mesh(
                                device, r.key, vb, ib, r.index_count,
                                vb_size + ib_size,
                                r.aabb_min, r.aabb_max,
                            );
                        }
                        MeshKind::Glass => {
                            pipeline.insert_transparent_mesh(
                                device, r.key, vb, ib, r.index_count,
                                vb_size + ib_size,
                                r.aabb_min, r.aabb_max,
                            );
                        }
                        MeshKind::Opaque => {
                            let evicted = pipeline.insert_chunk_mesh(
                                device, r.key, vb, ib, r.index_count,
                                vb_size + ib_size,
                                r.aabb_min, r.aabb_max,
                            );
                            self.gpu_evicted_pending.extend(evicted);
                        }
                    }
                }

                upload_time_us = t0.elapsed().as_micros();
                upload_count = budget;
                self.last_upload_bytes = total_bytes;
                self.last_upload_throughput = if upload_time_us > 0 {
                    total_bytes as f64 / upload_time_us as f64 / 1.048576
                } else {
                    0.0
                };
            } else {
                self.last_upload_bytes      = 0;
                self.last_upload_throughput = 0.0;
                upload_time_us = 0;
                upload_count = 0;
            }
        }

        self.last_upload_us = upload_time_us;
        self.last_upload_count = upload_count;

        let pipeline = self.pipeline_3d.as_mut().unwrap();
        let chunk_size = config::chunk_size_x() as f32;
        let render_dist = config::render_distance().max(1) as u32;
        self.camera.update_render_distance(chunk_size, render_dist);
        self.camera.update_aspect(width, height);

        // Sky billboard rendering (sun/moon) — before 3D terrain
        {
            let t = Instant::now();
            self.sky_renderer.render(
                encoder, view, device, queue, format,
                &self.camera, &self.day_night,
            );
            self.last_sky_render_us = t.elapsed().as_micros();
        }

        // --- VCT: upload voxel snapshot + dispatch GI compute ---
        let vct = self.vct_system.as_mut().unwrap();
        vct.update_model_shadows(queue, &self.block_registry_for_vct);
        let t_vct_upload = Instant::now();
        if let Some(snap) = self.pending_voxel_snapshot.take() {
            // Cache VCT origin for the lighting-debug visualisation.
            self.debug_render.vct_origin = snap.origin;

            vct.upload_volume(queue, &snap, &self.block_registry_for_vct);
            
            // Keep a clone for when late-loading models need to trigger a repack
            self.last_voxel_snapshot = Some(crate::core::vct::voxel_volume::VoxelSnapshot {
                origin: snap.origin,
                size: snap.size,
                blocks: snap.blocks.clone(),
            });

            // Scan voxel snapshot for blocks with inline light_sources.
            // Builds block_point_lights (PointLightGPU) and block_spot_lights (SpotLightGPU)
            // from the JSON-defined light_sources on each BlockDefinition.
            self.spot_lights.clear();
            self.block_spot_lights.clear();
            self.block_point_lights.clear();

            // Pre-build lookup: block_id → &[BlockLightSource] (skips blocks with no sources)
            let light_source_map: std::collections::HashMap<u8, &[BlockLightSource]> =
                (1u8..=255u8)
                    .filter_map(|id| {
                        let def = self.block_registry_for_vct.get(id)?;
                        if def.light_sources.is_empty() { None }
                        else { Some((id, def.light_sources.as_slice())) }
                    })
                    .collect();

            if !light_source_map.is_empty() {
                use crate::core::vct::voxel_volume::{VOLUME_SIZE, vol_idx};
                let vol_size = VOLUME_SIZE as usize;
                let origin   = snap.origin;

                'scan: for z in 0..vol_size {
                    for y in 0..vol_size {
                        for x in 0..vol_size {
                            let bid = snap.blocks[vol_idx(x as u32, y as u32, z as u32)];
                            if bid == 0 { continue; }
                            let sources = match light_source_map.get(&bid) {
                                Some(s) => *s,
                                None    => continue,
                            };
                            // Block-centre in world space
                            let bx = origin[0] as f32 + x as f32 + 0.5;
                            let by = origin[1] as f32 + y as f32 + 0.5;
                            let bz = origin[2] as f32 + z as f32 + 0.5;

                            for ls in sources {
                                // Light world position = block-centre + per-source offset
                                let lx = bx + ls.position[0];
                                let ly = by + ls.position[1];
                                let lz = bz + ls.position[2];
                                let range = if ls.range > 0.5 { ls.range } else { 16.0 };

                                match ls.kind {
                                    LightKind::Point => {
                                        self.block_point_lights.push(PointLightGPU {
                                            pos_range:       [lx, ly, lz, range],
                                            color_intensity: [ls.color[0], ls.color[1], ls.color[2], ls.intensity],
                                            source_voxel:    [origin[0] + x as i32, origin[1] + y as i32, origin[2] + z as i32, 1],
                                        });
                                    }
                                    LightKind::Directional => {
                                        // focus = outer half-angle in degrees;
                                        // inner = 45% of focus for a bright core.
                                        let inner_cos = (ls.focus * 0.45).to_radians().cos();
                                        let outer_cos = ls.focus.to_radians().cos();
                                        let d   = ls.direction;
                                        let len = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]).sqrt().max(0.001);
                                        self.block_spot_lights.push(SpotLightGPU {
                                            pos_range:       [lx, ly, lz, range],
                                            dir_inner:       [d[0]/len, d[1]/len, d[2]/len, inner_cos],
                                            color_intensity: [ls.color[0], ls.color[1], ls.color[2], ls.intensity],
                                            outer_pad:       [outer_cos, 0.0, 0.0, 0.0],
                                        });
                                    }
                                }

                                // Hard cap: 64 of each type
                                if self.block_point_lights.len() >= 64 || self.block_spot_lights.len() >= 64 {
                                    break 'scan;
                                }
                            }
                        }
                    }
                }

                debug_log!("GameScreen", "render",
                    "Snapshot light scan: {} point, {} spot from block light_sources",
                    self.block_point_lights.len(), self.block_spot_lights.len());
            }
        }
        self.last_vct_upload_us = t_vct_upload.elapsed().as_micros();

        let t_vct_dispatch = Instant::now();
        vct.dispatch_gi(encoder, queue);
        self.last_vct_dispatch_us = t_vct_dispatch.elapsed().as_micros();

        // Sync player entity transform so entity-def lights follow the actual player position.
        if let Some(pid) = self.player_entity_id {
            let foot = self.player.foot_position(&self.physics_world);
            if let Some(entity) = self.entity_manager.get_mut(pid) {
                entity.transform.position = foot;
                entity.transform.yaw      = self.body_yaw;
            }
        }

        // Build per-frame point light list:
        //   1. Static block lights (cached from last snapshot scan)
        //   2. Dynamic entity head lights (test entities spawned with G)
        // Also rebuild emitter_lights in lock-step so billboards stay in sync.
        self.point_lights.clear();
        self.emitter_lights.clear();
        self.point_lights.extend_from_slice(&self.block_point_lights);
        for pl in &self.block_point_lights {
            self.emitter_lights.push(EmitterLight {
                pos:             Vec3::new(pl.pos_range[0], pl.pos_range[1], pl.pos_range[2]),
                color:           [pl.color_intensity[0], pl.color_intensity[1], pl.color_intensity[2]],
                intensity:       pl.color_intensity[3],
                radius_override: 0.0,
            });
        }
        for &eid in &self.test_light_entities {
            if self.point_lights.len() >= 64 { break; }
            if let Some(entity) = self.entity_manager.get(eid) {
                let p = entity.transform.position;
                self.point_lights.push(PointLightGPU {
                    pos_range:       [p.x, p.y + 1.9, p.z, 20.0],
                    color_intensity: [1.0, 0.9, 0.6, 10.0],
                    source_voxel:    [0, 0, 0, 0],
                });
                self.emitter_lights.push(EmitterLight {
                    pos:             Vec3::new(p.x, p.y + 1.9, p.z),
                    color:           [1.0, 0.9, 0.6],
                    intensity:       10.0,
                    radius_override: 0.0,
                });
            }
        }

        // Build per-frame spot light list: static block spot lights (cached from last snapshot scan)
        self.spot_lights.clear();
        if self.spot_lights.len() < 64 {
            let remaining = 64 - self.spot_lights.len();
            let block_count = self.block_spot_lights.len().min(remaining);
            self.spot_lights.extend_from_slice(&self.block_spot_lights[..block_count]);
            for sl in &self.block_spot_lights[..block_count] {
                self.emitter_lights.push(EmitterLight {
                    pos:             Vec3::new(sl.pos_range[0], sl.pos_range[1], sl.pos_range[2]),
                    color:           [sl.color_intensity[0], sl.color_intensity[1], sl.color_intensity[2]],
                    intensity:       sl.color_intensity[3],
                    radius_override: 0.0,
                });
            }
        }

        // --- Entity-definition lights (from EntityRegistry, per live entity) ---
        // Toggled by L key (player_spotlight_on).
        // For the player entity: uses PlayerRenderer::bone_transforms so head-bone lights
        // correctly follow head pitch + relative yaw, and the bone world position includes
        // the PLAYER_SIZE scale that the renderer applies.
        if self.player_spotlight_on {
            use std::f32::consts::{FRAC_PI_2, PI};
            use crate::core::entity_definition::LightKind as ELK;

            // Collect entity data to avoid simultaneous borrow with lights vecs.
            let entity_transforms: Vec<(EntityId, Vec3, f32, String)> = self.entity_manager.iter()
                .map(|e| (e.id, e.transform.position, e.transform.yaw, e.model_id.clone()))
                .collect();

            let reg_count = self.entity_registry.defs.len();
            debug_log!("GameScreen", "render",
                "Entity-def lights: {} entities in manager, {} defs in registry",
                entity_transforms.len(), reg_count);

            // Pre-compute exact bone world transforms for the player entity.
            // This uses the SAME pose that the player renderer uses, including head pitch/yaw.
            let player_bone_mats: Option<(EntityId, std::collections::HashMap<String, glam::Mat4>)> =
                self.player_entity_id.map(|pid| {
                    let head_yaw_offset = angle_diff(self.camera.yaw(), self.body_yaw).clamp(-PI, PI);
                    let head_pitch = self.camera.pitch();
                    let pose = PlayerPose {
                        foot_pos:        self.player.foot_position(&self.physics_world),
                        body_yaw:        self.body_yaw,
                        head_yaw_offset,
                        head_pitch,
                        walk_phase:      self.walk_phase,
                        is_walking:      self.is_walking,
                        view_mode:       self.view_mode,
                    };
                    (pid, PlayerRenderer::bone_transforms(&pose))
                });

            let mut def_point_added = 0usize;
            let mut def_spot_added  = 0usize;

            for (eid, pos, yaw, model_id) in &entity_transforms {
                let def = match self.entity_registry.defs.get(model_id) {
                    Some(d) => d,
                    None    => {
                        debug_log!("GameScreen", "render",
                            "Entity-def lights: no def for model_id '{}' in registry", model_id);
                        continue;
                    }
                };
                if def.light_sources.is_empty() { continue; }

                let is_player = player_bone_mats.as_ref().map_or(false, |(pid, _)| pid == eid);
                let bone_mats = if is_player {
                    player_bone_mats.as_ref().map(|(_, m)| m)
                } else {
                    None
                };

                // Skip the billboard glow for the player's own lights in first-person:
                // the player would otherwise see a sphere floating right in front of them.
                let skip_billboard = is_player && self.view_mode == ViewMode::FirstPerson;

                // Fallback rotation for non-player entities (body yaw only).
                let body_rot = glam::Mat4::from_rotation_y(FRAC_PI_2 - yaw);

                for ls in &def.light_sources {
                    // --- World-space light position ---
                    let world = if !ls.bone.is_empty() {
                        if let Some(bone_mat) = bone_mats.and_then(|b| b.get(&ls.bone)) {
                            // Player entity with known bone: use the bone's exact world transform.
                            // bone_mat = base * bone_local; base already encodes foot_pos + body_yaw + PLAYER_SIZE scale.
                            // ls.position is in block units → divide by PLAYER_SIZE to enter model space.
                            // Do NOT add bone_piv here: the bone_local matrix already rotates around the pivot,
                            // so adding it again would double-count the bone's height offset.
                            bone_mat.transform_point3(Vec3::from(ls.position) / PLAYER_SIZE)
                        } else {
                            // Non-player entity or unknown bone: simple body-yaw, ls.position in block units.
                            *pos + body_rot.transform_point3(Vec3::from(ls.position))
                        }
                    } else {
                        *pos + body_rot.transform_point3(Vec3::from(ls.position))
                    };

                    let range = if ls.range > 0.0 { ls.range } else { 20.0 };
                    match ls.kind {
                        ELK::Point => {
                            if self.point_lights.len() < 64 {
                                debug_log!("GameScreen", "render",
                                    "Entity-def point light '{}': world=[{:.2},{:.2},{:.2}] range={:.1} intensity={:.1}",
                                    ls.label, world.x, world.y, world.z, range, ls.intensity);
                                self.point_lights.push(PointLightGPU {
                                    pos_range:       [world.x, world.y, world.z, range],
                                    color_intensity: [ls.color[0], ls.color[1], ls.color[2], ls.intensity],
                                    source_voxel:    [0, 0, 0, 0],
                                });
                                if !skip_billboard {
                                    self.emitter_lights.push(EmitterLight {
                                        pos:             world,
                                        color:           ls.color,
                                        intensity:       ls.intensity,
                                        radius_override: ls.billboard_radius,
                                    });
                                }
                                def_point_added += 1;
                            }
                        }
                        ELK::Directional => {
                            if self.spot_lights.len() < 64 {
                                // Direction: for player bones, rotate through the bone's full world
                                // transform (includes head pitch + yaw), then normalise (scale cancels).
                                // For others, only body yaw is applied.
                                let dir = if !ls.bone.is_empty() {
                                    if let Some(bone_mat) = bone_mats.and_then(|b| b.get(&ls.bone)) {
                                        bone_mat.transform_vector3(Vec3::from(ls.direction)).normalize_or_zero()
                                    } else {
                                        body_rot.transform_vector3(Vec3::from(ls.direction)).normalize_or_zero()
                                    }
                                } else {
                                    body_rot.transform_vector3(Vec3::from(ls.direction)).normalize_or_zero()
                                };
                                let half = ls.focus.to_radians();
                                let inner_cos = (half * 0.45).cos();
                                let outer_cos = half.cos();
                                debug_log!("GameScreen", "render",
                                    "Entity-def spot light '{}': world=[{:.2},{:.2},{:.2}] dir=[{:.2},{:.2},{:.2}] range={:.1}",
                                    ls.label, world.x, world.y, world.z, dir.x, dir.y, dir.z, range);
                                self.spot_lights.push(SpotLightGPU {
                                    pos_range:       [world.x, world.y, world.z, range],
                                    dir_inner:       [dir.x, dir.y, dir.z, inner_cos],
                                    color_intensity: [ls.color[0], ls.color[1], ls.color[2], ls.intensity],
                                    outer_pad:       [outer_cos, 0.0, 0.0, 0.0],
                                });
                                if !skip_billboard {
                                    self.emitter_lights.push(EmitterLight {
                                        pos:             world,
                                        color:           ls.color,
                                        intensity:       ls.intensity,
                                        radius_override: ls.billboard_radius,
                                    });
                                }
                                def_spot_added += 1;
                            }
                        }
                    }
                }
            }
            debug_log!("GameScreen", "render",
                "Entity-def lights added: {} point, {} spot (total now: {} point, {} spot)",
                def_point_added, def_spot_added, self.point_lights.len(), self.spot_lights.len());
        }

        // Upload point lights and spot lights to GPU
        let point_count = self.point_lights.len() as u32;
        let spot_count  = self.spot_lights.len() as u32;
        vct.update_lights(queue, &self.point_lights, &self.spot_lights);

        // ---- Entity shadow casters ----
        // Player AABB cast into voxel shadow rays so the player drops a
        // sun/moon shadow even though it's not in the voxel grid.
        // Width 0.6, height 1.8 (matches the physics collider).
        let player_aabb = {
            let foot = self.player.foot_position(&self.physics_world);
            const HALF_W: f32 = 0.30;
            const HEIGHT: f32 = 1.80;
            crate::core::vct::dynamic_lights::EntityShadowAABB {
                min_pad:     [foot.x - HALF_W, foot.y,          foot.z - HALF_W, 0.0],
                max_opacity: [foot.x + HALF_W, foot.y + HEIGHT, foot.z + HALF_W, 1.0],
            }
        };
        let entity_aabbs = [player_aabb];
        vct.update_entity_aabbs(queue, &entity_aabbs);
        let entity_aabb_count = entity_aabbs.len() as u32;

        // Update per-frame VCT fragment uniforms (no allocation — bind group pre-built at init).
        vct.prepare_fragment(queue, point_count, spot_count, entity_aabb_count);
        // Mutable vct borrow ends here; re-borrow immutably for the render phase.
        let vct = self.vct_system.as_ref().unwrap();
        let vct_frag_bg = &vct.frag_bg;

        // 3D render — pack lighting uniforms and pass to pipeline
        let atlas_view = self.atlas.texture_view().unwrap();
        let normal_atlas_view = self.atlas.normal_texture_view().unwrap();

        let shadow_view_proj = glam::Mat4::IDENTITY;

        let mut lighting_data = pack_lighting_uniforms(
            &self.camera.view_projection_matrix(),
            self.camera.position,
            &self.day_night,
            &self.lighting_config,
            &shadow_view_proj,
            None,
            256.0,
        );

        // Apply biome ambient tint — blends a subtle colour cast into the ambient light.
        // Strength 0.25: barely perceptible but enough to convey biome atmosphere.
        if config::biome_ambient_tint_enabled() {
            const TINT_STRENGTH: f32 = 0.25;
            let t = self.biome_ambient_tint;
            for i in 0..3 {
                lighting_data[36 + i] *= 1.0 - TINT_STRENGTH + TINT_STRENGTH * t[i];
            }
        }

        // Shadow toggle flags — packed into shadow_params (unused by VCT for legacy shadow maps).
        // shadow_params.x = sun/moon shadow enabled, shadow_params.y = block light shadow enabled.
        lighting_data[40] = if config::shadow_sun_enabled() { 1.0 } else { 0.0 };
        lighting_data[41] = if config::shadow_block_enabled() { 1.0 } else { 0.0 };

        let render_us = pipeline.render(
            encoder, view, device, queue, &self.camera, width, height,
            atlas_view, normal_atlas_view, &lighting_data,
            vct_frag_bg,
        );

        // Transparent solid blocks (glass) — alpha-blended pbr_vct pass.
        // Drawn before water so water always lays correctly on top.
        {
            let t = Instant::now();
            let vct = self.vct_system.as_ref().unwrap();
            pipeline.render_transparent(
                encoder, view, device, queue, &self.camera,
                atlas_view, normal_atlas_view, &lighting_data,
                vct_frag_bg, &vct.frag_bgl,
            );
            self.last_transparent_us = t.elapsed().as_micros();
        }

        // Entity pass — player + future world entities, PBR + VCT GI
        // Rendered before water so depth writes correctly occlude water surface.
        {
            let t = Instant::now();
            if let (Some(pr), Some(er), Some(pid), Some(depth_view)) = (
                self.player_renderer.as_ref(),
                self.entity_renderer.as_ref(),
                self.player_entity_id,
                pipeline.depth_view(),
            ) {
                let foot_pos = self.player.foot_position(&self.physics_world);
                let alt_held = self.keys.contains(&PhysicalKey::Code(KeyCode::AltLeft))
                            || self.keys.contains(&PhysicalKey::Code(KeyCode::AltRight));
                let orbit_mode = self.view_mode == ViewMode::ThirdPerson && alt_held;
                use std::f32::consts::PI;
                let head_yaw_offset = if orbit_mode {
                    0.0
                } else {
                    angle_diff(self.camera.yaw(), self.body_yaw).clamp(-PI, PI)
                };
                let head_pitch = if orbit_mode { 0.0 } else { self.camera.pitch() };
                let pose = PlayerPose {
                    foot_pos,
                    body_yaw:        self.body_yaw,
                    head_yaw_offset,
                    head_pitch,
                    walk_phase:      self.walk_phase,
                    is_walking:      self.is_walking,
                    view_mode:       self.view_mode,
                };
                let player_call = pr.build_render_call(
                    pid,
                    &pose,
                    self.camera.view_projection_matrix(),
                    self.camera.position,
                    &self.day_night,
                    self.lighting_config.ambient.min_level,
                );

                // Build render calls for test light entities (idle T-pose, third-person)
                let mut all_calls = vec![player_call];
                for &eid in &self.test_light_entities {
                    if let Some(entity) = self.entity_manager.get(eid) {
                        let test_pose = PlayerPose {
                            foot_pos:        entity.transform.position,
                            body_yaw:        entity.transform.yaw,
                            head_yaw_offset: 0.0,
                            head_pitch:      0.0,
                            walk_phase:      0.0,
                            is_walking:      false,
                            view_mode:       ViewMode::ThirdPerson,
                        };
                        all_calls.push(pr.build_render_call(
                            eid,
                            &test_pose,
                            self.camera.view_projection_matrix(),
                            self.camera.position,
                            &self.day_night,
                            self.lighting_config.ambient.min_level,
                        ));
                    }
                }

                // Build render calls for sheep entities
                {
                    use crate::screens::sheep_renderer::{SheepRenderer, SheepPose};
                    for (eid, ai) in &self.sheep_ai {
                        if let Some(entity) = self.entity_manager.get(*eid) {
                            let pose = SheepPose {
                                foot_pos:   entity.transform.position,
                                yaw:        entity.transform.yaw,
                                walk_phase: ai.walk_phase,
                                is_walking: ai.is_walking,
                            };
                            all_calls.push(SheepRenderer::build_render_call(
                                *eid,
                                &pose,
                                self.camera.view_projection_matrix(),
                                self.camera.position,
                                &self.day_night,
                                self.lighting_config.ambient.min_level,
                            ));
                        }
                    }
                }

                er.render(encoder, view, depth_view, queue, vct_frag_bg, &all_calls);
            }
            self.last_entity_render_us = t.elapsed().as_micros();
        }

        // Block model pass — static blocks with custom 3D models (crafting stations, etc.)
        {
            let t = Instant::now();
            if let (Some(bmr), Some(depth_view)) = (
                self.block_model_renderer.as_ref(),
                pipeline.depth_view(),
            ) {
                if bmr.instance_count() > 0 {
                    let lighting = BlockModelLighting {
                        view_proj:          self.camera.view_projection_matrix(),
                        sun_dir:            self.day_night.sun_direction(),
                        sun_color:          self.day_night.sun_color(),
                        sun_intensity:      self.day_night.sun_intensity(),
                        moon_dir:           self.day_night.moon_direction(),
                        moon_color:         self.day_night.moon_color(),
                        moon_intensity:     self.day_night.moon_intensity(),
                        ambient:            self.day_night.ambient_color(),
                        ambient_min:        self.lighting_config.ambient.min_level,
                        camera_pos:         self.camera.position,
                        shadow_sun_enabled: config::shadow_sun_enabled(),
                        shadow_offset:      0.3,
                    };
                    bmr.render(encoder, view, depth_view, queue, vct_frag_bg, &lighting);
                }
            }
            self.last_block_model_us = t.elapsed().as_micros();
        }

        // Hand block — anchored to right forearm bone, GI-lit, visible in both
        // first-person and third-person.
        {
            let t = Instant::now();
            if let Some(ref mut hbr) = self.hand_block_renderer {
                if let Some(depth_view) = pipeline.depth_view() {
                    let active_info = self.inventory_ui.inventory.active_item()
                        .map(|item| (
                            Some(item.block_id.as_str()),
                            item.top_texture.as_deref(),
                            item.side_texture.as_deref(),
                        ))
                        .unwrap_or((None, None, None));
                    hbr.update_block(device, active_info.0, active_info.1, active_info.2);

                    // Recompute the right-forearm bone transform for THIS frame —
                    // matches the pose passed to PlayerRenderer above (avoiding a
                    // one-frame lag between hand and arm).
                    let foot_pos = self.player.foot_position(&self.physics_world);
                    use std::f32::consts::PI;
                    let alt_held = self.keys.contains(&PhysicalKey::Code(KeyCode::AltLeft))
                                || self.keys.contains(&PhysicalKey::Code(KeyCode::AltRight));
                    let orbit_mode = self.view_mode == ViewMode::ThirdPerson && alt_held;
                    let head_yaw_offset = if orbit_mode { 0.0 }
                        else { angle_diff(self.camera.yaw(), self.body_yaw).clamp(-PI, PI) };
                    let head_pitch = if orbit_mode { 0.0 } else { self.camera.pitch() };
                    let pose = PlayerPose {
                        foot_pos, body_yaw: self.body_yaw,
                        head_yaw_offset, head_pitch,
                        walk_phase: self.walk_phase, is_walking: self.is_walking,
                        view_mode: self.view_mode,
                    };
                    let bones = PlayerRenderer::bone_transforms(&pose);
                    if let Some(right_hand) = bones.get("rightForearm").or_else(|| bones.get("rightArm")) {
                        hbr.render(
                            encoder, view, depth_view, queue,
                            self.camera.view_projection_matrix(),
                            *right_hand,
                            self.camera.position,
                            &self.day_night,
                            self.lighting_config.ambient.min_level,
                            config::shadow_sun_enabled(),
                            vct_frag_bg,
                        );
                    }
                }
            }
            self.last_hand_block_us = t.elapsed().as_micros();
        }

        // Water (transparent pass — after entities so water correctly occludes submerged geometry)
        {
            let t = Instant::now();
            let elapsed_secs = self.start_time.elapsed().as_secs_f32();
            pipeline.render_water(
                encoder, view, device, queue,
                &self.camera,
                atlas_view,
                &lighting_data,
                elapsed_secs,
            );
            self.last_water_render_us = t.elapsed().as_micros();
        }

        // Block outline — exact geometry for model blocks, AABB cube for standard blocks
        {
            let t = Instant::now();
            let model_name = self.target_block.and_then(|r| {
                self.block_registry_for_vct.model_name_for(r.block_id)
            });
            let outline_aabb = model_name.as_deref().and_then(|n| self.model_aabbs.get(n).copied());
            // Look up rotation for model instances so the outline rotates with the model.
            let outline_rot = self.target_block
                .and_then(|r| {
                    let bp = r.block_pos;
                    self.block_model_renderer.as_ref().map(|bmr| {
                        bmr.get_instance_rotation((bp.x, bp.y, bp.z))
                    })
                })
                .unwrap_or(0);
            pipeline.render_outline(
                encoder, view, device, queue, &self.camera, width, height,
                self.target_block.map(|r| r.block_pos),
                outline_aabb,
                model_name.as_deref(),
                outline_rot,
            );
            self.last_outline_us = t.elapsed().as_micros();
        }

        // Volumetric halos + god rays — additive, depth-tested, no depth write.
        // Rendered after all opaque geometry so they composite correctly.
        if let (Some(vr), Some(depth_view)) = (
            self.volumetric_renderer.as_ref(),
            pipeline.depth_view(),
        ) {
            if !self.point_lights.is_empty() || !self.spot_lights.is_empty() {
                let elapsed_secs = self.start_time.elapsed().as_secs_f32();
                let cam_fwd   = self.camera.forward();
                let cam_right = self.camera.right();
                let cam_up    = cam_right.cross(cam_fwd);
                vr.render(
                    encoder,
                    view,
                    depth_view,
                    queue,
                    self.camera.view_projection_matrix(),
                    cam_right,
                    cam_up,
                    elapsed_secs,
                    &self.point_lights,
                    &self.spot_lights,
                );
            }
        }

        // Billboard glow at each active light source position.
        if let (Some(ler), Some(depth_view)) = (
            self.light_emitter_renderer.as_mut(),
            pipeline.depth_view(),
        ) {
            if !self.emitter_lights.is_empty() {
                let cam_fwd   = self.camera.forward();
                let cam_right = self.camera.right();
                let cam_up    = cam_right.cross(cam_fwd);
                ler.render(
                    encoder,
                    view,
                    depth_view,
                    device,
                    queue,
                    self.camera.view_projection_matrix(),
                    cam_right,
                    cam_up,
                    &self.emitter_lights,
                );
            }
        }

        // ============================================================
        // Debug rendering — wireframe + coloured debug geometry
        // ============================================================
        if self.debug_render.visible {
            // Sync wireframe-supported flag from the pipeline.
            self.debug_render.wireframe_supported = pipeline.wireframe_supported;

            // Cache light positions for the lighting-debug pass.
            self.debug_render.spot_light_positions = self.spot_lights.iter()
                .map(|sl| ([sl.pos_range[0], sl.pos_range[1], sl.pos_range[2]], sl.pos_range[3]))
                .collect();
            self.debug_render.point_light_positions = self.point_lights.iter()
                .map(|pl| ([pl.pos_range[0], pl.pos_range[1], pl.pos_range[2]], pl.pos_range[3]))
                .collect();

            // -- Wireframe overlay ------------------------------------------
            if self.debug_render.wireframe && self.debug_render.wireframe_supported {
                pipeline.render_wireframe(encoder, view, queue, &self.camera);
            }

            // -- Coloured debug lines (AABBs, capsule, light markers, arrows) --
            let need_lines = self.debug_render.bbox_visible
                || self.debug_render.bbox_physical
                || self.debug_render.bbox_hitboxes
                || self.debug_render.lighting_debug
                || self.debug_render.shadow_vectors;

            if need_lines {
                let mut verts: Vec<DebugVertex> = Vec::new();

                // Chunk AABB wireframes (visible / rendered collision)
                if self.debug_render.bbox_visible {
                    for (mn, mx) in pipeline.chunk_aabbs() {
                        push_aabb(&mut verts, mn, mx, [0.0, 1.0, 1.0, 0.65]);
                    }
                }

                // Player physics capsule + hitbox AABB
                if self.debug_render.bbox_physical || self.debug_render.bbox_hitboxes {
                    let eye_off = self.player.stance().eye_offset();
                    let half_h  = self.player.stance().half_h();
                    let body    = self.camera.position - Vec3::Y * eye_off;
                    let hw      = crate::core::player::PLAYER_HALF_W;

                    if self.debug_render.bbox_physical {
                        push_capsule(&mut verts, body, hw, half_h, [0.1, 1.0, 0.2, 0.9]);
                    }
                    if self.debug_render.bbox_hitboxes {
                        push_aabb(&mut verts,
                            [body.x - hw, body.y - half_h, body.z - hw],
                            [body.x + hw, body.y + half_h, body.z + hw],
                            [1.0, 0.15, 0.15, 0.9]);
                    }
                }

                // VCT grid boundary + emissive-block markers
                if self.debug_render.lighting_debug {
                    let [ox, oy, oz] = self.debug_render.vct_origin;
                    let sz = crate::core::vct::VOLUME_SIZE as f32;
                    push_aabb(&mut verts,
                        [ox as f32, oy as f32, oz as f32],
                        [ox as f32 + sz, oy as f32 + sz, oz as f32 + sz],
                        [1.0, 0.85, 0.0, 0.35]);
                    // -- Spot lights: orange cross + range circle + direction arrow --
                    let spot_lights_snap: Vec<_> = self.spot_lights.clone();
                    for sl in &spot_lights_snap {
                        let pos   = [sl.pos_range[0], sl.pos_range[1], sl.pos_range[2]];
                        let range = sl.pos_range[3];
                        push_cross(&mut verts, pos, 0.5, [1.0, 0.5, 0.0, 1.0]);
                        let dir = Vec3::new(sl.dir_inner[0], sl.dir_inner[1], sl.dir_inner[2]);
                        push_arrow(&mut verts, Vec3::from(pos), dir * range.min(8.0), [1.0, 0.5, 0.0, 0.85]);
                        const CIRCLE_SEGS: usize = 24;
                        let [px, py, pz] = pos;
                        for i in 0..CIRCLE_SEGS {
                            let a0 = (i as f32 / CIRCLE_SEGS as f32) * std::f32::consts::TAU;
                            let a1 = ((i+1) as f32 / CIRCLE_SEGS as f32) * std::f32::consts::TAU;
                            verts.push(DebugVertex { pos: [px + range * a0.cos(), py, pz + range * a0.sin()], color: [1.0, 0.35, 0.0, 0.4] });
                            verts.push(DebugVertex { pos: [px + range * a1.cos(), py, pz + range * a1.sin()], color: [1.0, 0.35, 0.0, 0.4] });
                        }
                    }

                    // -- Point lights: cyan cross + range circle (XZ) --
                    let point_positions: Vec<([f32;3], f32)> =
                        self.debug_render.point_light_positions.clone();
                    for (pos, range) in point_positions {
                        push_cross(&mut verts, pos, 0.6, [0.3, 1.0, 1.0, 1.0]);
                        const CIRCLE_SEGS: usize = 24;
                        let [px, py, pz] = pos;
                        for i in 0..CIRCLE_SEGS {
                            let a0 = (i as f32 / CIRCLE_SEGS as f32) * std::f32::consts::TAU;
                            let a1 = ((i+1) as f32 / CIRCLE_SEGS as f32) * std::f32::consts::TAU;
                            verts.push(DebugVertex { pos: [px + range * a0.cos(), py, pz + range * a0.sin()], color: [0.3, 1.0, 1.0, 0.4] });
                            verts.push(DebugVertex { pos: [px + range * a1.cos(), py, pz + range * a1.sin()], color: [0.3, 1.0, 1.0, 0.4] });
                        }
                    }
                }

                // Sun / moon shadow-cast vectors
                if self.debug_render.shadow_vectors {
                    let anchor = self.camera.position + Vec3::Y * 2.5;
                    let sun_d  = self.day_night.sun_direction();
                    let moon_d = self.day_night.moon_direction();
                    // Sun: yellow-orange arrow
                    push_arrow(&mut verts, anchor, sun_d * 10.0, [1.0, 0.85, 0.1, 1.0]);
                    // Moon: light-blue arrow, offset slightly so they don't overlap
                    push_arrow(&mut verts, anchor + Vec3::X * 1.5, moon_d * 10.0,
                               [0.45, 0.65, 1.0, 1.0]);
                }

                if !verts.is_empty() {
                    pipeline.render_debug_lines(
                        encoder, view, device, queue, &self.camera, &verts,
                    );
                }
            }
        }

        self.last_frame_total_us = t_frame_start.elapsed().as_micros();

        // Feed profiler
        let vram = pipeline.vram_usage();
        let visible_chunks = pipeline.gpu_chunk_count() as u32
            - pipeline.culled_last_frame;
        let draw_calls         = pipeline.last_draw_calls;
        let shadow_draw_calls  = pipeline.last_shadow_draw_calls;
        let visible_triangles  = pipeline.last_visible_triangles;
        let visible_vram_bytes = pipeline.last_visible_vram_bytes;
        self.profiler.feed(ProfilerFrame {
            fps:                    self.debug.fps,
            frame_total_us:         self.last_frame_total_us,
            gen_time_us:            self.last_gen_time_us,
            gen_count:              self.last_gen_count,
            mesh_time_us:           self.last_mesh_time_us,
            dirty_count:            self.last_dirty_count,
            upload_time_us:         upload_time_us,
            upload_count:           upload_count,
            sky_render_us:          self.last_sky_render_us,
            vct_upload_us:          self.last_vct_upload_us,
            vct_dispatch_us:        self.last_vct_dispatch_us,
            render_3d_us:           render_us,
            transparent_us:         self.last_transparent_us,
            entity_render_us:       self.last_entity_render_us,
            block_model_us:         self.last_block_model_us,
            hand_block_us:          self.last_hand_block_us,
            water_render_us:        self.last_water_render_us,
            outline_us:             self.last_outline_us,

            total_chunks:           self.world_worker.chunk_count(),
            visible_chunks,
            culled_chunks:          pipeline.culled_last_frame,
            lod0_count:             self.last_lod_counts[0],
            lod1_count:             self.last_lod_counts[1],
            lod2_count:             self.last_lod_counts[2],
            worker_pending:         self.last_worker_pending,
            upload_pending:         self.upload_worker.pending_count(),
            gpu_queue_depth:        self.upload_queue.len(),
            vram_bytes:             vram,
            draw_calls,
            shadow_draw_calls,
            visible_triangles,
            visible_vram_bytes,
            upload_bytes:           self.last_upload_bytes,
            upload_throughput_mb_s: self.last_upload_throughput,
            cam_pos: [
                self.camera.position.x,
                self.camera.position.y,
                self.camera.position.z,
            ],
            get_texture_us:    crate::core::frame_timing::get_texture_us(),
            update_us:         crate::core::frame_timing::update_us(),
            egui_build_us:     crate::core::frame_timing::egui_build_us(),
            egui_render_us:    crate::core::frame_timing::egui_render_us(),
            submit_present_us: crate::core::frame_timing::submit_present_us(),
        });
    }

    fn build_ui(&mut self, ctx: &egui::Context) {
        let screen_rect = ctx.screen_rect();
        let sw = screen_rect.width();
        let sh = screen_rect.height();

        let layer_id = egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("game_hud"),
        );
        let painter = ctx.layer_painter(layer_id);

        // =================== Crosshair ======================================
        let cx = sw / 2.0;
        let cy = sh / 2.0;
        let cross_color = Color32::from_rgba_unmultiplied(255, 255, 255, 128);
        let cross_stroke = Stroke::new(3.0, cross_color);

        painter.line_segment(
            [Pos2::new(cx - 12.0, cy), Pos2::new(cx + 12.0, cy)],
            cross_stroke,
        );
        painter.line_segment(
            [Pos2::new(cx, cy - 12.0), Pos2::new(cx, cy + 12.0)],
            cross_stroke,
        );

        // =================== Cursor lock hint (top center) ==================
        let lock_text = if self.camera_locked {
            "J - unlock cursor"
        } else {
            "J - lock cursor / enable camera"
        };
        let lock_color = if self.camera_locked {
            Color32::from_rgb(102, 255, 102)
        } else {
            Color32::from_rgb(255, 255, 102)
        };
        painter.text(
            Pos2::new(sw / 2.0, 8.0),
            Align2::CENTER_TOP,
            lock_text,
            FontId::proportional(16.0),
            lock_color,
        );

        // =================== Flashlight indicator (top center, below cursor lock) ====
        if self.player_spotlight_on {
            painter.text(
                Pos2::new(sw / 2.0, 28.0),
                Align2::CENTER_TOP,
                "Flashlight ON",
                FontId::proportional(14.0),
                Color32::from_rgba_unmultiplied(255, 240, 180, 180),
            );
        }

        // =================== Controls (bottom right) ========================
        painter.text(
            Pos2::new(sw - 10.0, sh - 24.0),
            Align2::RIGHT_BOTTOM,
            "WASD move  Space jump  L flashlight",
            FontId::proportional(14.0),
            Color32::from_rgba_unmultiplied(255, 255, 255, 102),
        );

        // =================== Inventory (hotbar + window) ====================
        {
            let _render_cmds = self.inventory_ui.build_ui(ctx);
        }

        // =================== Debug overlay (left) ===========================
        self.debug.draw_egui(&painter);

        // =================== Profiler overlay (right) =======================
        self.profiler.draw_egui(&painter, sw);

        // =================== Debug render panel (bottom-right, F6) =========
        self.draw_debug_render_panel(ctx, sw, sh);

        // =================== Time speed HUD (top-right) ====================
        {
            let speed_label = self.day_night.time_speed.label();
            let time = self.day_night.time_of_day();
            let hours = (time * 24.0) as u32;
            let minutes = ((time * 24.0 - hours as f32) * 60.0) as u32;
            let time_text = format!(
                "{:02}:{:02}  Speed: {}  (T/Shift+T/Ctrl+T)",
                hours, minutes, speed_label
            );
            painter.text(
                Pos2::new(sw - 10.0, 8.0),
                Align2::RIGHT_TOP,
                &time_text,
                FontId::proportional(14.0),
                Color32::from_rgba_unmultiplied(255, 255, 200, 180),
            );
        }

        // =================== Pause menu =====================================
        if self.paused {
            let painter_bg = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Middle,
                egui::Id::new("pause_dimmer"),
            ));
            painter_bg.rect_filled(
                screen_rect,
                0.0,
                Color32::from_black_alpha(160),
            );

            let mut open = true;
            egui::Window::new("Pause Menu")
                .title_bar(false)
                .resizable(false)
                .movable(false)
                .collapsible(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .default_width(320.0)
                .frame(egui::Frame {
                    fill: Color32::from_rgb(21, 26, 40),
                    corner_radius: egui::CornerRadius::ZERO,
                    stroke: Stroke::new(2.0, Color32::from_rgb(58, 68, 96)),
                    inner_margin: egui::Margin::same(20),
                    ..Default::default()
                })
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {

                        ui.add_space(10.0);

                        ui.label(
                            egui::RichText::new("Pause")
                                .size(28.0)
                                .color(Color32::WHITE),
                        );

                        ui.add_space(16.0);
                        ui.separator();
                        ui.add_space(12.0);

                        // --- Mouse Sensitivity slider ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(320.0);

                            ui.label(
                                egui::RichText::new("Mouse Sensitivity")
                                    .size(16.0)
                                    .color(Color32::LIGHT_GRAY),
                            );
                            ui.add_space(6.0);

                            let mut sens = config::mouse_sensitivity();
                            let slider = egui::Slider::new(&mut sens, 0.0005..=0.01)
                                .step_by(0.0005)
                                .text("Sensitivity");
                            if ui.add(slider).changed() {
                                config::set_mouse_sensitivity(sens);
                                self.camera.set_sensitivity(sens);
                                user_settings::save_current();
                            }

                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(
                                    "Camera rotation speed. Lower = slower, higher = faster."
                                )
                                    .size(12.0)
                                    .color(egui::Color32::GRAY),
                            );
                        });

                        ui.add_space(16.0);

                        // --- Fly Speed slider ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(320.0);

                            ui.label(
                                egui::RichText::new("Fly Speed")
                                    .size(16.0)
                                    .color(egui::Color32::LIGHT_GRAY),
                            );
                            ui.add_space(6.0);

                            let mut fs = config::fly_speed();
                            let slider = egui::Slider::new(&mut fs, 2.0..=100.0)
                                .step_by(1.0)
                                .suffix(" b/s")
                                .text("Speed");
                            if ui.add(slider).changed() {
                                config::set_fly_speed(fs);
                                user_settings::save_current();
                            }

                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(
                                    "Movement speed in fly mode (N to toggle)."
                                )
                                    .size(12.0)
                                    .color(egui::Color32::GRAY),
                            );
                        });

                        ui.add_space(16.0);

                        // --- Render Distance ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(260.0);

                            ui.label(
                                egui::RichText::new("Render Distance")
                                    .size(16.0)
                                    .color(Color32::LIGHT_GRAY),
                            );
                            ui.add_space(6.0);

                            let mut rd = config::render_distance() as f32;
                            let slider = egui::Slider::new(&mut rd, 1.0..=128.0)
                                .step_by(1.0)
                                .suffix(" chunks");
                            if ui.add(slider).changed() {
                                config::set_render_distance(rd as i32);
                                user_settings::save_current();
                            }
                        });

                        ui.add_space(16.0);

                        // --- LOD Distance Multiplier ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(260.0);

                            ui.label(
                                egui::RichText::new("LOD Distance")
                                    .size(16.0)
                                    .color(Color32::LIGHT_GRAY),
                            );
                            ui.add_space(6.0);

                            let mut lod_m = config::lod_multiplier();
                            let slider = egui::Slider::new(&mut lod_m, 0.25..=4.0)
                                .step_by(0.25)
                                .suffix("x");
                            if ui.add(slider).changed() {
                                config::set_lod_multiplier(lod_m);
                                user_settings::save_current();
                            }

                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(
                                    format!(
                                        "LOD0 \u{2264}{:.0}  LOD1 \u{2264}{:.0}  LOD2 beyond",
                                        config::LOD_NEAR_BASE * lod_m,
                                        config::LOD_FAR_BASE * lod_m,
                                    )
                                )
                                .size(12.0)
                                .color(Color32::GRAY),
                            );
                        });

                        ui.add_space(16.0);

                        // --- Debug / Profiler toggles ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(260.0);

                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.debug.visible, "Debug Overlay");
                                ui.checkbox(&mut self.profiler.visible, "Profiler");
                            });
                        });

                        ui.add_space(20.0);
                        ui.separator();
                        ui.add_space(12.0);

                        // --- Save World ---
                        let save_btn = egui::Button::new(
                            egui::RichText::new("Save World").size(18.0),
                        )
                            .min_size(egui::vec2(220.0, 42.0));

                        if ui.add(save_btn).clicked() {
                            self.do_save_world();
                        }

                        ui.add_space(10.0);

                        // --- Resume ---
                        let resume_btn = egui::Button::new(
                            egui::RichText::new("Resume").size(18.0),
                        )
                            .min_size(egui::vec2(220.0, 42.0));

                        if ui.add(resume_btn).clicked() {
                            self.paused = false;
                        }

                        ui.add_space(10.0);

                        // --- Back to Menu ---
                        let back_btn = egui::Button::new(
                            egui::RichText::new("Back to Menu").size(18.0),
                        )
                            .min_size(egui::vec2(220.0, 42.0));

                        if ui.add(back_btn).clicked() {
                            self.pending_action = ScreenAction::Switch(
                                Box::new(
                                    crate::screens::main_menu::MainMenuScreen::new()
                                ),
                            );
                        }
                    });
                });

            if !open {
                self.paused = false;
            }
        }
    }

    fn post_process(&mut self, _dt: f64) {}

    fn on_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_dx += dx;
        self.mouse_dy += dy;
    }

    fn wants_pointer_lock(&self) -> bool { self.camera_locked && !self.inventory_ui.is_open() }

    fn on_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x;
                self.cursor_y = position.y;
            }

            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p)   => p.y as f32 / 30.0,
                };
                if dy != 0.0 {
                    if !self.inventory_ui.scroll_hotbar(-dy as i32) {
                        self.player.scroll_slot(-dy);
                    }
                }
            }

            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                match state {
                    ElementState::Pressed => {
                        self.mouse_down = true;
                        if self.inventory_ui.was_interacted() {
                            return;
                        }
                        if self.camera_locked {
                            if let Some(op) = PlayerController::break_block(self.target_block.as_ref()) {
                                self.pending_block_ops.push(op);
                            }
                        }
                    }
                    ElementState::Released => {
                        self.mouse_down = false;
                    }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: winit::event::MouseButton::Right,
                ..
            } => {
                if self.inventory_ui.is_open() {
                    return;
                }
                if self.camera_locked {
                    let block_id = self.inventory_ui.active_block_id()
                        .and_then(|name| self.player.registry.id_for(name))
                        .unwrap_or(1);
                    let (placement, default_rot) = self.player.registry.get(block_id)
                        .map(|d| (d.placement_mode.clone(), d.default_rotation))
                        .unwrap_or_default();
                    let look = self.camera.forward();
                    if let Some(op) = self.player.place_block(
                        self.target_block.as_ref(), &self.physics_world,
                        block_id, look, &placement, default_rot,
                    ) {
                        self.pending_block_ops.push(op);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: winit::event::MouseButton::Middle,
                ..
            } => {
                if self.camera_locked {
                    if let Some(ref target) = self.target_block {
                        let block_id = target.block_id;
                        if block_id != 0 {
                            // Try to find matching slot in hotbar
                            let mut found_slot = None;
                            let n = self.inventory_ui.hotbar_count();
                            for i in 0..n {
                                let matches = self.inventory_ui.inventory
                                    .hotbar_slot(i)
                                    .and_then(|s| s.item())
                                    .and_then(|it| self.player.registry.id_for(&it.block_id))
                                    .map(|id| id == block_id)
                                    .unwrap_or(false);
                                if matches {
                                    found_slot = Some(i);
                                    break;
                                }
                            }
                            if let Some(slot_idx) = found_slot {
                                // Block already in hotbar — just select it
                                self.inventory_ui.select_hotbar(slot_idx);
                            } else {
                                // Block not in hotbar — add to active slot
                                if let Some(key) = self.player.registry.key_for(block_id) {
                                    if let Some(def) = self.player.registry.get(block_id) {
                                        let name = if def.display_name.is_empty() { &def.id } else { &def.display_name };
                                        let top_tex = def.texture_for_face(2).map(|s| s.to_owned());
                                        let side_tex = def.texture_for_face(4).map(|s| s.to_owned());
                                        let stack = ItemStack::new(key, name, 64, 64)
                                            .with_color(def.color)
                                            .with_textures(top_tex, side_tex);
                                        let active_idx = self.inventory_ui.inventory.active_hotbar_index;
                                        self.inventory_ui.inventory.hotbar_slots[active_idx] =
                                            crate::screens::inventory::InventorySlot::Occupied(stack);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                match event.state {
                    ElementState::Pressed => {
                        self.keys.insert(event.physical_key);
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                KeyCode::KeyC => {
                                    self.inventory_ui.toggle_creative();
                                    debug_log!("GameScreen", "on_event",
                                        "Creative mode: {}", self.inventory_ui.is_creative());
                                }
                                KeyCode::KeyN => {
                                    self.player.toggle_fly(&mut self.physics_world);
                                }
                                KeyCode::KeyL => {
                                    self.player_spotlight_on = !self.player_spotlight_on;
                                    debug_log!("GameScreen", "on_event",
                                        "Player spotlight: {}", self.player_spotlight_on);
                                }
                                KeyCode::KeyS => {
                                    let ctrl = self.keys.contains(&PhysicalKey::Code(KeyCode::ControlLeft))
                                            || self.keys.contains(&PhysicalKey::Code(KeyCode::ControlRight));
                                    if ctrl {
                                        debug_log!("GameScreen", "on_event", "Ctrl+S: manual save");
                                        self.do_save_world();
                                    }
                                }
                                KeyCode::KeyJ => {
                                    self.camera_locked = !self.camera_locked;
                                    self.mouse_dx = 0.0;
                                    self.mouse_dy = 0.0;
                                    debug_log!("GameScreen","on_event",
                                        "Camera lock: {}", self.camera_locked);
                                }
                                KeyCode::Escape => {
                                    self.paused = !self.paused;
                                    if self.paused {
                                        self.camera_locked = false;
                                        self.mouse_dx = 0.0;
                                        self.mouse_dy = 0.0;
                                    }
                                }
                                KeyCode::F3 => {
                                    self.debug.visible = !self.debug.visible;
                                }
                                KeyCode::F4 => {
                                    self.profiler.visible = !self.profiler.visible;
                                }
                                KeyCode::F5 => {
                                    self.view_mode = match self.view_mode {
                                        ViewMode::FirstPerson => ViewMode::ThirdPerson,
                                        ViewMode::ThirdPerson => ViewMode::FirstPerson,
                                    };
                                }
                                KeyCode::F6 => {
                                    self.debug_render.visible = !self.debug_render.visible;
                                }
                                // G — spawn a test light entity 3 blocks ahead of the player
                                KeyCode::KeyG => {
                                    let foot = self.player.foot_position(&self.physics_world);
                                    let yaw  = self.camera.yaw();
                                    let spawn_pos = foot + Vec3::new(
                                        -yaw.sin() * 3.0,
                                        0.0,
                                        -yaw.cos() * 3.0,
                                    );
                                    self.pending_test_entity_spawns.push(spawn_pos);
                                    debug_log!("GameScreen", "on_event",
                                        "G key: queued test light entity at [{:.1},{:.1},{:.1}]",
                                        spawn_pos.x, spawn_pos.y, spawn_pos.z);
                                }
                                KeyCode::KeyE => {
                                    self.inventory_ui.toggle();
                                    if self.inventory_ui.is_open() {
                                        self.camera_locked = false;
                                        self.mouse_dx = 0.0;
                                        self.mouse_dy = 0.0;
                                    }
                                    debug_log!("GameScreen", "on_event",
                                        "Inventory toggled: open={}", self.inventory_ui.is_open());
                                }
                                KeyCode::KeyQ => {
                                    self.inventory_ui.clear_active_hotbar_slot();
                                    debug_log!("GameScreen", "on_event",
                                        "Q pressed: cleared active hotbar slot {}",
                                        self.inventory_ui.inventory.active_hotbar_index);
                                }
                                KeyCode::Digit1 => { self.inventory_ui.select_hotbar(0); }
                                KeyCode::Digit2 => { self.inventory_ui.select_hotbar(1); }
                                KeyCode::Digit3 => { self.inventory_ui.select_hotbar(2); }
                                KeyCode::Digit4 => { self.inventory_ui.select_hotbar(3); }
                                KeyCode::Digit5 => { self.inventory_ui.select_hotbar(4); }
                                KeyCode::Digit6 => { self.inventory_ui.select_hotbar(5); }
                                KeyCode::Digit7 => { self.inventory_ui.select_hotbar(6); }
                                KeyCode::Digit8 => { self.inventory_ui.select_hotbar(7); }
                                KeyCode::Digit9 => { self.inventory_ui.select_hotbar(8); }
                                // -- Time controls --
                                KeyCode::KeyT => {
                                    let shift = self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftLeft))
                                             || self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftRight));
                                    let ctrl  = self.keys.contains(&PhysicalKey::Code(KeyCode::ControlLeft))
                                             || self.keys.contains(&PhysicalKey::Code(KeyCode::ControlRight));
                                    if shift {
                                        self.day_night.time_speed = self.day_night.time_speed.faster();
                                        debug_log!("GameScreen", "on_event",
                                            "Time speed -> {}", self.day_night.time_speed.label());
                                    } else if ctrl {
                                        self.day_night.time_speed = self.day_night.time_speed.slower();
                                        debug_log!("GameScreen", "on_event",
                                            "Time speed -> {}", self.day_night.time_speed.label());
                                    } else {
                                        // Toggle pause
                                        use crate::core::lighting::TimeSpeed;
                                        if self.day_night.time_speed == TimeSpeed::Paused {
                                            self.day_night.time_speed = TimeSpeed::Normal;
                                        } else {
                                            self.day_night.time_speed = TimeSpeed::Paused;
                                        }
                                        debug_log!("GameScreen", "on_event",
                                            "Time speed -> {}", self.day_night.time_speed.label());
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    ElementState::Released => { self.keys.remove(&event.physical_key); }
                }
            }

            _ => {}
        }
    }

    fn poll_action(&mut self) -> ScreenAction {
        std::mem::replace(&mut self.pending_action, ScreenAction::None)
    }

    fn clear_color(&self) -> wgpu::Color {
        let sky = self.day_night.sky_color();
        wgpu::Color { r: sky[0], g: sky[1], b: sky[2], a: 1.0 }
    }
}

// =============================================================================
// GameScreen — debug panel (outside the Screen trait impl)
// =============================================================================

impl GameScreen {
    fn draw_debug_render_panel(&mut self, ctx: &egui::Context, _sw: f32, sh: f32) {
        if !self.debug_render.visible { return; }

        // Sit above the controls hint text (~24 px from the bottom).
        egui::Window::new("Дебаг-рендер")
            .id(egui::Id::new("debug_render_panel"))
            .anchor(egui::Align2::RIGHT_BOTTOM, [-10.0, -(sh * 0.04 + 30.0)])
            .resizable(false)
            .collapsible(true)
            .default_open(true)
            .show(ctx, |ui| {
                ui.label(egui::RichText::new("Debug Render  [F6]")
                    .color(egui::Color32::from_rgb(100, 200, 255))
                    .strong());
                ui.separator();

                // ── Wireframe ─────────────────────────────────────────────
                if self.debug_render.wireframe_supported {
                    ui.checkbox(&mut self.debug_render.wireframe,
                        "Wireframe (полигональная сетка)");
                } else {
                    ui.add_enabled(false,
                        egui::Checkbox::new(&mut self.debug_render.wireframe,
                            "Wireframe (GPU не поддерживает)"));
                }

                ui.separator();
                ui.label(egui::RichText::new("Баундбоксы / коллизия").strong());

                // ── Bounding boxes ────────────────────────────────────────
                ui.checkbox(&mut self.debug_render.bbox_visible,
                    egui::RichText::new("  Видимая коллизия — чанки  [голубой]")
                        .color(egui::Color32::from_rgb(0, 230, 230)));
                ui.checkbox(&mut self.debug_render.bbox_physical,
                    egui::RichText::new("  Физическая коллизия — капсула  [зелёный]")
                        .color(egui::Color32::from_rgb(30, 230, 80)));
                ui.checkbox(&mut self.debug_render.bbox_hitboxes,
                    egui::RichText::new("  Хитбокс игрока — AABB  [красный]")
                        .color(egui::Color32::from_rgb(230, 60, 60)));

                ui.separator();

                // ── Lighting debug ────────────────────────────────────────
                ui.checkbox(&mut self.debug_render.lighting_debug,
                    egui::RichText::new("Распространение освещения (VCT)")
                        .color(egui::Color32::from_rgb(255, 200, 50)));
                if self.debug_render.lighting_debug {
                    let ns = self.debug_render.spot_light_positions.len();
                    let np = self.debug_render.point_light_positions.len();
                    ui.label(format!("    Точечных: {}  Направленных: {}", np, ns));
                }

                ui.separator();

                // ── Shadow vectors ────────────────────────────────────────
                ui.checkbox(&mut self.debug_render.shadow_vectors,
                    egui::RichText::new("Векторы теней (солнце / луна)")
                        .color(egui::Color32::from_rgb(200, 200, 255)));
                if self.debug_render.shadow_vectors {
                    let sd = self.day_night.sun_direction();
                    let md = self.day_night.moon_direction();
                    ui.label(format!("    Солнце:  [{:.2} {:.2} {:.2}]", sd.x, sd.y, sd.z));
                    ui.label(format!("    Луна:    [{:.2} {:.2} {:.2}]", md.x, md.y, md.z));
                }
            });
    }
}
