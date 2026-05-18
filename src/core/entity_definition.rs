// =============================================================================
// QubePixel — EntityDefinition + EntityLightSource
// =============================================================================
//
// JSON-driven entity definitions loaded from assets/entities/.
// Each definition describes an entity type (model, scale, light sources).
//
// Loading pipeline:
//   1. Read  assets/entities/entity_registry.json  → list of entity IDs
//   2. For each ID, read  assets/entities/<id>.json  → full definition
//
// Adding a new entity type:
//   1. Add the ID to assets/entities/entity_registry.json
//   2. Create  assets/entities/<id>.json  with the desired properties
//   3. The Entity Light Editor picks it up automatically

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use crate::debug_log;

pub const ENTITIES_DIR: &str = "assets/entities";
pub const ENTITY_REGISTRY_FILE: &str = "assets/entities/entity_registry.json";

// ---------------------------------------------------------------------------
// EntityLightSource — light source attached to an entity
// ---------------------------------------------------------------------------

/// The shape of the emitted light cone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LightKind {
    #[default]
    Point,
    Directional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityLightSource {
    #[serde(default)]
    pub kind: LightKind,

    #[serde(default)]
    pub position: [f32; 3],

    #[serde(default = "default_white")]
    pub color: [f32; 3],

    #[serde(default = "default_light_intensity_one")]
    pub intensity: f32,

    #[serde(default)]
    pub range: f32,

    #[serde(default = "default_down")]
    pub direction: [f32; 3],

    #[serde(default = "default_focus")]
    pub focus: f32,

    #[serde(default)]
    pub label: String,

    /// Bone name this light is parented to (empty = entity origin).
    #[serde(default)]
    pub bone: String,

    /// Visual billboard glow radius in blocks. 0.0 = derive from intensity.
    /// Only affects the in-world glow sphere — lighting itself is unchanged.
    #[serde(default)]
    pub billboard_radius: f32,
}

impl Default for EntityLightSource {
    fn default() -> Self {
        Self {
            kind:             LightKind::Point,
            position:         [0.0, 0.0, 0.0],
            color:            [1.0, 1.0, 1.0],
            intensity:        1.0,
            range:            0.0,
            direction:        [0.0, -1.0, 0.0],
            focus:            30.0,
            label:            String::new(),
            bone:             String::new(),
            billboard_radius: 0.0,
        }
    }
}

fn default_white() -> [f32; 3] { [1.0, 1.0, 1.0] }
fn default_light_intensity_one() -> f32 { 1.0 }
fn default_down() -> [f32; 3] { [0.0, -1.0, 0.0] }
fn default_focus() -> f32 { 30.0 }

// ---------------------------------------------------------------------------
// EntityDefinition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDefinition {
    pub id: String,

    #[serde(default)]
    pub display_name: String,

    #[serde(default)]
    pub description: String,

    pub model_path: String,

    #[serde(default)]
    pub skin_path: Option<String>,

    #[serde(default = "default_scale")]
    pub default_scale: f32,

    #[serde(default)]
    pub light_sources: Vec<EntityLightSource>,

    /// World-spawn rule.  When `None`, this entity will not be spawned naturally
    /// by the chunk-based spawner (e.g. the player entity).
    #[serde(default)]
    pub spawn: Option<EntitySpawnRule>,
}

fn default_scale() -> f32 { 1.0 }

/// Rules controlling natural world spawning for an entity type.
/// Loaded from the `spawn` field of an entity definition JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySpawnRule {
    /// Master switch — false disables natural spawn entirely.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Probability (0..1) that a single eligible top-surface column produces
    /// a spawn attempt when a chunk is first loaded. Acts as overall rarity.
    #[serde(default = "default_rarity")]
    pub rarity: f32,

    /// Hard cap on entities that may be spawned per chunk.
    #[serde(default = "default_max_per_chunk")]
    pub max_per_chunk: u32,

    /// Global cap on simultaneously-alive entities of this type.
    #[serde(default = "default_max_alive")]
    pub max_alive: u32,

    /// Inclusive Y range — the surface block must lie within this band.
    #[serde(default = "default_min_y")]
    pub min_y: i32,
    #[serde(default = "default_max_y")]
    pub max_y: i32,

    /// Block keys (registry IDs, e.g. "grass" or "grass/grass_andesite") on
    /// which this entity may stand at spawn time.  Empty = no constraint.
    #[serde(default)]
    pub spawn_on_blocks: Vec<String>,

    /// Optional biome name tokens (substrings matched case-insensitively against
    /// the biome's `name` field).  Empty = no constraint.
    #[serde(default)]
    pub biome_tags: Vec<String>,
}

fn default_true() -> bool { true }
fn default_rarity() -> f32 { 0.02 }
fn default_max_per_chunk() -> u32 { 2 }
fn default_max_alive() -> u32 { 32 }
fn default_min_y() -> i32 { i32::MIN }
fn default_max_y() -> i32 { i32::MAX }

impl EntityDefinition {
    pub fn new(id: &str) -> Self {
        Self {
            id:            id.to_string(),
            display_name:  id.replace('_', " "),
            description:   String::new(),
            model_path:    String::new(),
            skin_path:     None,
            default_scale: 1.0,
            light_sources: Vec::new(),
            spawn:         None,
        }
    }
}

// ---------------------------------------------------------------------------
// EntityRegistry — loads all entity definitions from disk
// ---------------------------------------------------------------------------

pub struct EntityRegistry {
    pub registry_order: Vec<String>,
    pub defs:           HashMap<String, EntityDefinition>,
}

impl EntityRegistry {
    pub fn load() -> Self {
        let mut reg = Self {
            registry_order: Vec::new(),
            defs:           HashMap::new(),
        };

        let reg_path = PathBuf::from(ENTITY_REGISTRY_FILE);
        if !reg_path.exists() {
            debug_log!("EntityRegistry", "load", "Registry file not found at {:?}", reg_path);
            return reg;
        }

        let content = match std::fs::read_to_string(&reg_path) {
            Ok(s) => s,
            Err(e) => {
                debug_log!("EntityRegistry", "load", "Read error: {}", e);
                return reg;
            }
        };

        let val: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                debug_log!("EntityRegistry", "load", "Parse error: {}", e);
                return reg;
            }
        };

        if let Some(arr) = val["entities"].as_array() {
            for entry in arr {
                if let Some(key) = entry.as_str() {
                    reg.registry_order.push(key.to_string());
                    reg.load_entity_def(key);
                }
            }
        }

        debug_log!("EntityRegistry", "load",
            "Loaded {} entity definitions", reg.defs.len());

        reg
    }

    fn load_entity_def(&mut self, key: &str) {
        let base = PathBuf::from(ENTITIES_DIR);
        let path = base.join(format!("{}.json", key));

        let content = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => {
                debug_log!("EntityRegistry", "load_entity_def",
                    "Could not read {:?} for key '{}'", path, key);
                return;
            }
        };

        match serde_json::from_str::<EntityDefinition>(&content) {
            Ok(def) => { self.defs.insert(key.to_string(), def); }
            Err(e) => {
                debug_log!("EntityRegistry", "load_entity_def",
                    "Parse error for '{}': {}", key, e);
            }
        }
    }

    pub fn save_entity_def(def: &EntityDefinition) -> Result<(), String> {
        let path = PathBuf::from(ENTITIES_DIR).join(format!("{}.json", def.id));

        let mut val = serde_json::to_value(def)
            .map_err(|e| format!("serialize: {}", e))?;

        if let Some(obj) = val.as_object_mut() {
            obj.insert("format_version".into(),
                serde_json::Value::Number(1.into()));
        }

        let json = serde_json::to_string_pretty(&val)
            .map_err(|e| format!("stringify: {}", e))?;

        std::fs::write(&path, json)
            .map_err(|e| format!("write {}: {}", path.display(), e))?;

        Ok(())
    }

    pub fn save_registry(order: &[String]) -> Result<(), String> {
        let reg_path = PathBuf::from(ENTITY_REGISTRY_FILE);

        let entities: Vec<serde_json::Value> = order
            .iter()
            .map(|k| serde_json::Value::String(k.clone()))
            .collect();

        let mut map = serde_json::Map::new();
        map.insert("format_version".into(),
            serde_json::Value::Number(1.into()));
        map.insert("description".into(),
            serde_json::Value::String("QubePixel Entity Registry".into()));
        map.insert("entities".into(),
            serde_json::Value::Array(entities));

        let json = serde_json::to_string_pretty(&serde_json::Value::Object(map))
            .map_err(|e| format!("Stringify: {}", e))?;

        std::fs::write(&reg_path, json)
            .map_err(|e| format!("Write: {}", e))?;

        Ok(())
    }
}
