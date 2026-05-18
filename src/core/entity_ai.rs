// =============================================================================
// QubePixel — Entity AI  (sheep wandering, climbing, grass-eating)
// =============================================================================
//
// Per-entity AI state and tick logic for the natural-spawn sheep.
//
// AI runs on the main thread once per game update.  Block reads use the
// camera-centred 128³ voxel snapshot already maintained for VCT GI, which
// guarantees coverage of the player-loaded area.
//
// Behaviour outline:
//   • Idle for 1–3 s, then start a Wander leg with a random yaw.
//   • Wander steps forward at WALK_SPEED while honouring world collisions.
//   • A 1-block-high obstacle triggers a step-up (Y += 1) — taller walls turn.
//   • Gravity pulls the sheep down when the support block is air.
//   • A grass block under the sheep occasionally triggers Eat — the block is
//     replaced with the matching dirt variant via a queued BlockOp::Place.

use glam::{Vec3, IVec3};
use crate::core::entity::EntityId;
use crate::core::vct::voxel_volume::{VoxelSnapshot, VOLUME_SIZE, vol_idx};
use crate::core::world_gen::world::BlockOp;
use crate::core::gameobjects::block::BlockRegistry;
use crate::{debug_log, flow_debug_log};

// ---------------------------------------------------------------------------
// Tiny deterministic RNG (xorshift64) — avoids an extra crate dependency.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct XorShift64 { state: u64 }

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state which yields a constant stream.
        let s = if seed == 0 { 0xDEAD_BEEF_BAAD_F00D } else { seed };
        Self { state: s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // Take the top 24 bits → 2^24 distinct values in [0,1).
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Uniform f32 in [lo, hi).
    #[inline]
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// Horizontal world units per second while wandering.
const WALK_SPEED:        f32 = 1.2;
/// Vertical world units per second pulled by gravity when unsupported.
const GRAVITY:           f32 = 8.0;
/// Idle phase duration range (seconds).
const IDLE_MIN_S:        f32 = 1.0;
const IDLE_MAX_S:        f32 = 3.0;
/// Wander leg duration range (seconds) — one continuous walk before the next decision.
const WANDER_MIN_S:      f32 = 2.0;
const WANDER_MAX_S:      f32 = 5.5;
/// Cooldown between successful grass-eating events.
const EAT_COOLDOWN_S:    f32 = 8.0;
/// Chance per Idle→Wander transition that the sheep decides to eat first if grass is below.
const EAT_PROBABILITY:   f32 = 0.35;
/// Maximum distance from player at which the AI ticks; outside this, sheep idle.
const MAX_AI_DISTANCE:   f32 = 96.0;

// ---------------------------------------------------------------------------
// SheepAI — one instance per live sheep entity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum SheepState {
    Idle,
    Wander,
    Eat { target: IVec3, timer: f32 },
}

pub struct SheepAI {
    pub entity_id:    EntityId,
    /// Chunk this entity was spawned in — used for despawn-on-eviction.
    pub chunk_key:    (i32, i32, i32),
    pub state:        SheepState,
    /// Time remaining in the current state (seconds).
    pub state_timer:  f32,
    /// Direction body is currently facing (radians, 0 = +Z, increases clockwise).
    pub yaw:          f32,
    /// Cooldown until the next eat is permitted (seconds).
    pub eat_cooldown: f32,
    /// Walk animation phase, accumulated while moving.
    pub walk_phase:   f32,
    pub is_walking:   bool,
    /// Cached vertical velocity (used by gravity / step-up).
    pub y_velocity:   f32,
    rng:              XorShift64,
}

impl SheepAI {
    pub fn new(entity_id: EntityId, chunk_key: (i32, i32, i32), spawn_pos: Vec3, seed: u64) -> Self {
        let mut rng = XorShift64::new(seed ^ entity_id.0);
        let yaw = rng.next_f32() * std::f32::consts::TAU;
        Self {
            entity_id,
            chunk_key,
            state:        SheepState::Idle,
            state_timer:  rng.range_f32(IDLE_MIN_S, IDLE_MAX_S),
            yaw,
            eat_cooldown: 0.0,
            walk_phase:   0.0,
            is_walking:   false,
            y_velocity:   0.0,
            rng,
        }
        .with_pos_dbg(spawn_pos)
    }

    fn with_pos_dbg(self, spawn_pos: Vec3) -> Self {
        debug_log!("SheepAI", "new",
            "Sheep {:?} created at [{:.1},{:.1},{:.1}] chunk={:?}",
            self.entity_id, spawn_pos.x, spawn_pos.y, spawn_pos.z, self.chunk_key);
        self
    }
}

// ---------------------------------------------------------------------------
// Voxel snapshot block lookup
// ---------------------------------------------------------------------------

/// Read the block ID at a world-space integer position from the camera-centred
/// VCT snapshot.  Returns `None` if the coordinate is outside the snapshot
/// (treat as "unknown" — caller should typically idle the entity).
fn snap_block(snap: &VoxelSnapshot, wx: i32, wy: i32, wz: i32) -> Option<u8> {
    let lx = wx - snap.origin[0];
    let ly = wy - snap.origin[1];
    let lz = wz - snap.origin[2];
    let size = VOLUME_SIZE as i32;
    if lx < 0 || ly < 0 || lz < 0 || lx >= size || ly >= size || lz >= size {
        return None;
    }
    Some(snap.blocks[vol_idx(lx as u32, ly as u32, lz as u32)])
}

/// True if the block ID is treated as a passable cell for entity movement.
/// Air (0) and fluids are passable; everything else blocks.
fn is_passable(registry: &BlockRegistry, bid: u8) -> bool {
    if bid == 0 { return true; }
    registry.get(bid).map_or(true, |d| !d.solid)
}

/// True if the block ID is a grass surface block (key starts with "grass").
fn is_grass(registry: &BlockRegistry, bid: u8) -> bool {
    registry.key_for(bid).map_or(false, |k| {
        k == "grass" || k.starts_with("grass/")
    })
}

/// Find the corresponding dirt block ID for a grass block ID.
/// Maps "grass/grass_X" → "dirt/dirt_X" and "grass" → "dirt".
fn dirt_for_grass(registry: &BlockRegistry, grass_id: u8) -> Option<u8> {
    let key = registry.key_for(grass_id)?.to_string();
    let dirt_key = if key == "grass" {
        "dirt".to_string()
    } else if let Some(suffix) = key.strip_prefix("grass/grass_") {
        format!("dirt/dirt_{}", suffix)
    } else if let Some(stripped) = key.strip_prefix("grass_") {
        format!("dirt_{}", stripped)
    } else {
        return None;
    };
    registry.id_for(&dirt_key)
}

// ---------------------------------------------------------------------------
// Tick logic
// ---------------------------------------------------------------------------

/// Result of one AI tick — a position update and optional world block ops.
pub struct AITickResult {
    pub new_position: Vec3,
    pub new_yaw:      f32,
    pub block_ops:    Vec<BlockOp>,
}

/// Advance the sheep AI by `dt` seconds.  `current_pos` is the entity's
/// current foot position.  `snap` is the camera-centred voxel snapshot.
/// `player_pos` is used to skip AI for sheep far from the player.
pub fn tick_sheep(
    ai:          &mut SheepAI,
    current_pos: Vec3,
    current_yaw: f32,
    dt:          f32,
    snap:        Option<&VoxelSnapshot>,
    registry:    &BlockRegistry,
    player_pos:  Vec3,
) -> AITickResult {
    // Default: hold position
    let mut new_pos = current_pos;
    let mut new_yaw = current_yaw;
    let mut block_ops = Vec::new();

    // Skip detailed AI when far from player (cheap fallback to idle)
    let dx = current_pos.x - player_pos.x;
    let dz = current_pos.z - player_pos.z;
    if dx * dx + dz * dz > MAX_AI_DISTANCE * MAX_AI_DISTANCE {
        ai.is_walking = false;
        return AITickResult { new_position: new_pos, new_yaw, block_ops };
    }

    let snap = match snap {
        Some(s) => s,
        None => {
            ai.is_walking = false;
            return AITickResult { new_position: new_pos, new_yaw, block_ops };
        }
    };

    ai.eat_cooldown = (ai.eat_cooldown - dt).max(0.0);
    ai.state_timer -= dt;

    // -------- Gravity / vertical adjustment --------
    let foot_x = current_pos.x.floor() as i32;
    let foot_y = current_pos.y.floor() as i32;
    let foot_z = current_pos.z.floor() as i32;

    let block_below = snap_block(snap, foot_x, foot_y - 1, foot_z).unwrap_or(0);
    let on_ground = !is_passable(registry, block_below);

    if !on_ground {
        ai.y_velocity -= GRAVITY * dt;
        new_pos.y += ai.y_velocity * dt;
        // Re-snap to top of any block now under us
        let new_foot_y = new_pos.y.floor() as i32;
        let nb_below = snap_block(snap, foot_x, new_foot_y - 1, foot_z).unwrap_or(0);
        if ai.y_velocity < 0.0 && !is_passable(registry, nb_below) {
            new_pos.y = new_foot_y as f32;
            ai.y_velocity = 0.0;
        }
    } else {
        ai.y_velocity = 0.0;
        new_pos.y = foot_y as f32;
    }

    // -------- State transitions --------
    if ai.state_timer <= 0.0 {
        match ai.state {
            SheepState::Idle => {
                // Decide: maybe eat the grass we're standing on, otherwise wander
                if ai.eat_cooldown <= 0.0 && ai.rng.next_f32() < EAT_PROBABILITY {
                    let below = snap_block(snap, foot_x, foot_y - 1, foot_z).unwrap_or(0);
                    if is_grass(registry, below) {
                        let target = IVec3::new(foot_x, foot_y - 1, foot_z);
                        ai.state = SheepState::Eat { target, timer: 1.4 };
                        ai.state_timer = 1.4;
                        ai.is_walking = false;
                        flow_debug_log!("SheepAI", "tick", "Sheep {:?} starts eating at {:?}", ai.entity_id, target);
                        return AITickResult { new_position: new_pos, new_yaw, block_ops };
                    }
                }
                ai.yaw = ai.rng.next_f32() * std::f32::consts::TAU;
                ai.state = SheepState::Wander;
                ai.state_timer = ai.rng.range_f32(WANDER_MIN_S, WANDER_MAX_S);
            }
            SheepState::Wander => {
                ai.state = SheepState::Idle;
                ai.state_timer = ai.rng.range_f32(IDLE_MIN_S, IDLE_MAX_S);
                ai.is_walking = false;
            }
            SheepState::Eat { target, .. } => {
                // Eating finished — replace the grass with dirt
                if let Some(grass_id) = snap_block(snap, target.x, target.y, target.z) {
                    if is_grass(registry, grass_id) {
                        if let Some(dirt_id) = dirt_for_grass(registry, grass_id) {
                            block_ops.push(BlockOp::Place {
                                x: target.x, y: target.y, z: target.z,
                                block_id: dirt_id, rotation: 0,
                            });
                            debug_log!("SheepAI", "tick",
                                "Sheep {:?} ate grass({}) at {:?} → dirt({})",
                                ai.entity_id, grass_id, target, dirt_id);
                        }
                    }
                }
                ai.eat_cooldown = EAT_COOLDOWN_S;
                ai.state = SheepState::Idle;
                ai.state_timer = ai.rng.range_f32(IDLE_MIN_S, IDLE_MAX_S);
            }
        }
    }

    // -------- Per-state behaviour --------
    if matches!(ai.state, SheepState::Wander) {
        // Move forward along yaw
        let (sy, cy) = ai.yaw.sin_cos();
        // Use (sin, cos) so yaw=0 points +Z (matches model orientation).
        let dir = Vec3::new(sy, 0.0, cy);
        let step = dir * WALK_SPEED * dt;
        let cand = new_pos + step;

        let cx = cand.x.floor() as i32;
        let cy_floor = cand.y.floor() as i32;
        let cz = cand.z.floor() as i32;

        let front_lo = snap_block(snap, cx, cy_floor,     cz).unwrap_or(0);
        let front_hi = snap_block(snap, cx, cy_floor + 1, cz).unwrap_or(0);

        if is_passable(registry, front_lo) && is_passable(registry, front_hi) {
            // Open path — accept the step
            new_pos.x = cand.x;
            new_pos.z = cand.z;
            ai.is_walking = true;
        } else if !is_passable(registry, front_lo) && is_passable(registry, front_hi) {
            // 1-block-high obstacle: check the space TWO blocks above is also clear,
            // then step up.
            let front_hi2 = snap_block(snap, cx, cy_floor + 2, cz).unwrap_or(0);
            if is_passable(registry, front_hi2) {
                new_pos.x  = cand.x;
                new_pos.z  = cand.z;
                new_pos.y  = (cy_floor + 1) as f32;
                ai.is_walking = true;
                flow_debug_log!("SheepAI", "tick", "Sheep {:?} climbed step at [{},{},{}]", ai.entity_id, cx, cy_floor + 1, cz);
            } else {
                // Too tall — turn around
                ai.yaw = (ai.yaw + std::f32::consts::PI + (ai.rng.next_f32() - 0.5) * 1.0)
                    .rem_euclid(std::f32::consts::TAU);
                ai.is_walking = false;
            }
        } else {
            // Blocked above — turn
            ai.yaw = (ai.yaw + std::f32::consts::PI + (ai.rng.next_f32() - 0.5) * 1.0)
                .rem_euclid(std::f32::consts::TAU);
            ai.is_walking = false;
        }

        // Smooth yaw output
        new_yaw = ai.yaw;
    } else {
        ai.is_walking = false;
        new_yaw = ai.yaw;
    }

    if ai.is_walking {
        ai.walk_phase = (ai.walk_phase + dt * 8.0) % std::f32::consts::TAU;
    }

    AITickResult { new_position: new_pos, new_yaw, block_ops }
}
