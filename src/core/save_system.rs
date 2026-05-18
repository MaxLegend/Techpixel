// =============================================================================
// QubePixel — SaveSystem  (world save / load)
// =============================================================================
//
// Directory layout:
//   saves/
//     user_settings.json
//     <WorldName>/
//       world.json        ← JSON metadata (seed, player, version)
//       chunks/
//         1_0_2.bin       ← bincode SavedChunk
//

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use bincode::{encode_to_vec, decode_from_slice, config};
use bincode::error::EncodeError;
use crate::debug_log;

pub const SAVE_VERSION: u32 = 1;
const SAVES_DIR: &str = "saves";

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldMetadata {
    pub version:     u32,
    pub name:        String,
    pub seed:        u64,
    pub created_at:  String,
    pub last_played: String,
    pub player:      SavedPlayer,
    /// All non-player world entities (NPCs, dropped items, etc.).
    #[serde(default)]
    pub entities:    Vec<SavedEntity>,
}

/// Serialisable snapshot of one world entity (not the player).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedEntity {
    pub model_id: String,
    pub position: [f32; 3],
    pub yaw:      f32,
    pub scale:    f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedPlayer {
    pub position:      [f32; 3],
    pub fly_mode:      bool,
    pub stance:        u8,          // 0=Standing 1=Crouching 2=Prone
    pub selected_slot: usize,
    pub hotbar_ids:    Vec<u8>,
}

impl Default for SavedPlayer {
    fn default() -> Self {
        Self {
            position:      [8.0, 250.0, 30.0],
            fly_mode:      false,
            stance:        0,
            selected_slot: 0,
            hotbar_ids:    vec![],
        }
    }
}

/// On-disk representation of a single chunk — minimal size.
#[derive(Debug, bincode::Encode, bincode::Decode)]
pub struct SavedChunk {
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
    /// Full block array (id per voxel). Same indexing as Chunk::blocks.
    pub blocks:       Vec<u8>,
    /// Full rotation array.
    pub rotations:    Vec<u8>,
    /// Sparse fluid levels — only non-zero entries stored as (flat_index, value_bits).
    pub fluid_sparse: Vec<(u32, u32)>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the save directory for a named world: `saves/<name>/`.
pub fn world_dir(name: &str) -> PathBuf {
    PathBuf::from(SAVES_DIR).join(name)
}

fn now_iso() -> String {
    chrono::Local::now().to_rfc3339()
}

/// Generate a seed from the current time (no `rand` dependency needed).
pub fn random_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345)
}

// ---------------------------------------------------------------------------
// World list / create / load / save
// ---------------------------------------------------------------------------

/// Returns metadata for every valid world found under `saves/`.
pub fn list_worlds() -> Vec<WorldMetadata> {
    let mut worlds = Vec::new();
    let saves = Path::new(SAVES_DIR);
    if !saves.is_dir() { return worlds; }

    let rd = match std::fs::read_dir(saves) {
        Ok(r) => r,
        Err(e) => {
            debug_log!("SaveSystem", "list_worlds", "read_dir error: {}", e);
            return worlds;
        }
    };

    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Ok(meta) = load_world_meta(&path) {
                worlds.push(meta);
            }
        }
    }

    // Sort by last_played descending (most recent first)
    worlds.sort_by(|a, b| b.last_played.cmp(&a.last_played));
    worlds
}

/// Create a new world directory and write initial metadata. Returns the metadata.
pub fn create_world(name: &str, seed: u64) -> Result<WorldMetadata, String> {
    let dir = world_dir(name);
    std::fs::create_dir_all(&dir).map_err(|e| format!("create_dir: {}", e))?;
    std::fs::create_dir_all(dir.join("chunks")).map_err(|e| format!("create chunks dir: {}", e))?;

    let now = now_iso();
    let meta = WorldMetadata {
        version:     SAVE_VERSION,
        name:        name.to_owned(),
        seed,
        created_at:  now.clone(),
        last_played: now,
        player:      SavedPlayer::default(),
        entities:    Vec::new(),
    };
    save_world_meta(&dir, &meta)?;
    debug_log!("SaveSystem", "create_world", "Created world '{}' seed={} at {:?}", name, seed, dir);
    Ok(meta)
}

/// Load `world.json` from a world directory.
pub fn load_world_meta(world_dir: &Path) -> Result<WorldMetadata, String> {
    let path = world_dir.join("world.json");
    let json = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {}", path.display(), e))?;
    serde_json::from_str(&json)
        .map_err(|e| format!("parse {}: {}", path.display(), e))
}

/// Write `world.json` to the world directory.
pub fn save_world_meta(world_dir: &Path, meta: &WorldMetadata) -> Result<(), String> {
    let path = world_dir.join("world.json");
    let json = serde_json::to_string_pretty(meta)
        .map_err(|e| format!("serialize: {}", e))?;
    std::fs::write(&path, json)
        .map_err(|e| format!("write {}: {}", path.display(), e))
}

// ---------------------------------------------------------------------------
// Chunk save / load
// ---------------------------------------------------------------------------

fn chunk_path(world_dir: &Path, cx: i32, cy: i32, cz: i32) -> PathBuf {
    let name = format!("{}_{}_{}.bin", cx, cy, cz);
    world_dir.join("chunks").join(name)
}

/// Encode and write a `SavedChunk` to disk.
pub fn write_chunk(world_dir: &Path, chunk: &SavedChunk) -> Result<(), EncodeError> {
    let path = chunk_path(world_dir, chunk.cx, chunk.cy, chunk.cz);
    let bytes = encode_to_vec(chunk, config::standard())?;
    std::fs::write(&path, &bytes)
        .map_err(|e| EncodeError::Io { inner: e, index: 0 })?;
    debug_log!(
        "SaveSystem", "write_chunk",
        "Saved chunk ({},{},{}) → {} bytes", chunk.cx, chunk.cy, chunk.cz, bytes.len()
    );
    Ok(())
}

/// Read a `SavedChunk` from disk. Returns `None` if the file does not exist.
pub fn read_chunk(world_dir: &Path, cx: i32, cy: i32, cz: i32) -> Option<SavedChunk> {
    let path = chunk_path(world_dir, cx, cy, cz);
    let bytes = std::fs::read(&path).ok()?;
    match decode_from_slice::<SavedChunk, _>(&bytes, config::standard()) {
        Ok((chunk, _)) => {
            debug_log!(
                "SaveSystem", "read_chunk",
                "Loaded chunk ({},{},{})", cx, cy, cz
            );
            Some(chunk)
        }
        Err(e) => {
            debug_log!("SaveSystem", "read_chunk", "Decode error ({},{},{}): {}", cx, cy, cz, e);
            None
        }
    }
}

/// Delete a world directory and everything in it.
pub fn delete_world(name: &str) -> Result<(), String> {
    let dir = world_dir(name);
    if dir.is_dir() {
        std::fs::remove_dir_all(&dir)
            .map_err(|e| format!("remove_dir_all: {}", e))?;
        debug_log!("SaveSystem", "delete_world", "Deleted world '{}'", name);
    }
    Ok(())
}
