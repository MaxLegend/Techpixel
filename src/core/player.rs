// =============================================================================
// QubePixel — PlayerController  (движение, прыжок, выбор и взаимодействие с блоками)
// =============================================================================

use glam::{IVec3, Vec3};
use rapier3d::prelude::{RigidBodyHandle, ColliderHandle};

use crate::{debug_log, ext_debug_log, flow_debug_log};
use crate::core::physics::PhysicsWorld;
use crate::core::gameobjects::block::{BlockRegistry, PlacementMode};
use crate::core::raycast::RaycastResult;
use crate::core::world_gen::world::BlockOp;
use crate::core::config;

// ---------------------------------------------------------------------------
// Константы игрока — стоя
// ---------------------------------------------------------------------------
pub const PLAYER_HALF_W: f32      = 0.3;   // полуширина (итого 0.6)
pub const PLAYER_HALF_H: f32      = 0.9;   // полувысота стоя (итого 1.8)
/// Высота глаз над центром тела.  Центр тела на half_h над подошвой,
/// значит глаза на 1.62 над подошвой (Minecraft-стандарт).
pub const PLAYER_EYE_OFFSET: f32  = 0.72;  // = 1.62 - PLAYER_HALF_H

// Константы присядки (Shift)
const CROUCH_HALF_H: f32    = 0.65;  // итого 1.3 м
const CROUCH_EYE: f32       = 0.40;  // над центром (1.05 над подошвой)
const CROUCH_SPEED: f32     = 2.5;

// Константы лёжа (Ctrl) — 0.7 м, пролезает в дырку 1 блок
const PRONE_HALF_H: f32     = 0.35;  // итого 0.7 м
const PRONE_EYE: f32        = 0.15;  // над центром (~0.5 над подошвой)
const PRONE_SPEED: f32      = 1.5;

// Прыжок и скорость
const PLAYER_JUMP_VEL: f32  = 7.5;
const PLAYER_SPEED: f32     = 5.0;

/// Spawn above terrain — surface sits at Y≈128-160.
const SPAWN: Vec3 = Vec3::new(8.0, 250.0, 30.0);

// ---------------------------------------------------------------------------
// PlayerStance
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerStance {
    Standing,
    Crouching,
    Prone,
}

impl PlayerStance {
    pub fn half_h(self) -> f32 {
        match self {
            PlayerStance::Standing  => PLAYER_HALF_H,
            PlayerStance::Crouching => CROUCH_HALF_H,
            PlayerStance::Prone     => PRONE_HALF_H,
        }
    }
    pub fn eye_offset(self) -> f32 {
        match self {
            PlayerStance::Standing  => PLAYER_EYE_OFFSET,
            PlayerStance::Crouching => CROUCH_EYE,
            PlayerStance::Prone     => PRONE_EYE,
        }
    }
    pub fn speed(self) -> f32 {
        match self {
            PlayerStance::Standing  => PLAYER_SPEED,
            PlayerStance::Crouching => CROUCH_SPEED,
            PlayerStance::Prone     => PRONE_SPEED,
        }
    }
}

// ---------------------------------------------------------------------------
// PlayerController
// ---------------------------------------------------------------------------
pub struct PlayerController {
    body_handle:     RigidBodyHandle,
    collider_handle: ColliderHandle,
    on_ground:       bool,
    fly_mode:        bool,
    stance:          PlayerStance,
    /// True when submerged in water deeper than 1 block (swim mode).
    pub is_swimming: bool,

    // -- Хотбар ---------------------------------------------------------
    pub registry:     BlockRegistry,
    selected_slot:    usize,
    hotbar_ids:       Vec<u8>,
    hotbar_names:     Vec<String>,
}

impl PlayerController {
    pub fn new(physics: &mut PhysicsWorld, registry: BlockRegistry) -> Self {
        debug_log!("PlayerController", "new", "Creating player body + hotbar");

        let (body_handle, collider_handle) = physics.create_box_body(
            SPAWN, PLAYER_HALF_W, PLAYER_HALF_H,
        );

        let mut hotbar_ids   = Vec::new();
        let mut hotbar_names = Vec::new();
        for id in 1u8..=255 {
            if let Some(def) = registry.get(id) {
                if def.solid {
                    hotbar_ids.push(id);
                    hotbar_names.push(def.id.clone());
                }
            }
        }
        if hotbar_ids.is_empty() {
            hotbar_ids.push(1);
            hotbar_names.push("block".to_owned());
        }

        Self {
            body_handle,
            collider_handle,
            fly_mode:   false,
            on_ground:  false,
            stance:     PlayerStance::Standing,
            is_swimming: false,
            registry,
            selected_slot: 0,
            hotbar_ids,
            hotbar_names,
        }
    }

    // -----------------------------------------------------------------------
    // Стойка (Stance)
    // -----------------------------------------------------------------------

    pub fn stance(&self) -> PlayerStance { self.stance }

    /// Переключиться в `new_stance`.  Ничего не делает, если уже в этой стойке.
    /// При переходе в стойку с бо́льшим half_h проверяет, есть ли место над головой.
    pub fn set_stance(&mut self, physics: &mut PhysicsWorld, new_stance: PlayerStance) {
        if self.fly_mode || self.stance == new_stance { return; }

        let old_hh = self.stance.half_h();
        let new_hh = new_stance.half_h();

        // При подъёме (встать/присесть) проверяем, нет ли блока над головой
        if new_hh > old_hh && physics.head_blocked(self.body_handle, old_hh, new_hh) {
            debug_log!("PlayerController", "set_stance",
                "Head blocked, can't change stance to {:?}", new_stance);
            return;
        }

        self.collider_handle = physics.replace_player_collider(
            self.body_handle,
            self.collider_handle,
            old_hh,
            PLAYER_HALF_W,
            new_hh,
        );
        self.stance = new_stance;
        debug_log!("PlayerController", "set_stance", "Stance -> {:?}", new_stance);
    }

    // -----------------------------------------------------------------------
    // Плавание (Swimming)
    // -----------------------------------------------------------------------

    /// Обновляет флаг is_swimming на основе данных блоков под ногами.
    /// foot_block / foot_block_above — блоки на уровне подошвы и на 1 выше.
    pub fn update_swimming(&mut self, physics: &mut PhysicsWorld,
                           foot_block: u8, foot_block_above: u8)
    {
        let is_fluid_at_feet  = self.registry.is_fluid_block(foot_block);
        let is_fluid_above    = self.registry.is_fluid_block(foot_block_above);
        let in_deep_water = is_fluid_at_feet && is_fluid_above;

        if in_deep_water != self.is_swimming {
            self.is_swimming = in_deep_water;
            if in_deep_water {
                // Выключаем гравитацию, обнуляем вертикальную скорость
                physics.set_body_gravity_scale(self.body_handle, 0.0);
                physics.set_linvel(self.body_handle,
                    physics.get_linvel(self.body_handle) * Vec3::new(1.0, 0.0, 1.0));
                debug_log!("PlayerController", "update_swimming", "Enter swim mode");
            } else {
                physics.set_body_gravity_scale(self.body_handle, 1.0);
                debug_log!("PlayerController", "update_swimming", "Exit swim mode");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Режим свободной камеры (Fly)
    // -----------------------------------------------------------------------

    pub fn toggle_fly(&mut self, physics: &mut PhysicsWorld) {
        self.fly_mode = !self.fly_mode;
        if self.fly_mode {
            physics.set_body_gravity_scale(self.body_handle, 0.0);
            physics.enable_collider(self.collider_handle, false);
            physics.set_linvel(self.body_handle, Vec3::ZERO);
            // Restore standing stance silently (no block check in fly mode)
            if self.stance != PlayerStance::Standing {
                let old_hh = self.stance.half_h();
                self.collider_handle = physics.replace_player_collider(
                    self.body_handle, self.collider_handle,
                    old_hh, PLAYER_HALF_W, PLAYER_HALF_H,
                );
                self.stance = PlayerStance::Standing;
            }
        } else {
            physics.set_body_gravity_scale(self.body_handle, 1.0);
            physics.enable_collider(self.collider_handle, true);
            physics.set_linvel(self.body_handle, Vec3::ZERO);
        }
    }

    pub fn is_fly_mode(&self) -> bool { self.fly_mode }

    pub fn body_handle(&self) -> RigidBodyHandle { self.body_handle }

    // -----------------------------------------------------------------------
    // Движение и прыжок — вызывать ДО physics.step()
    // -----------------------------------------------------------------------

    pub fn apply_movement(
        &mut self,
        physics:   &mut PhysicsWorld,
        move_dir:  Vec3,
        jump:      bool,
        fly_down:  bool,
    ) {
        if self.fly_mode {
            let mut fly_dir = move_dir;
            if jump     { fly_dir.y += 1.0; }
            if fly_down { fly_dir.y -= 1.0; }

            let fly_speed = config::fly_speed();
            let speed = if fly_dir.length_squared() > 1e-6 {
                fly_dir = fly_dir.normalize();
                fly_speed
            } else { 0.0 };

            physics.set_linvel(self.body_handle,
                Vec3::new(fly_dir.x * speed, fly_dir.y * speed, fly_dir.z * speed));

        } else if self.is_swimming {
            // Плавание: горизонтальное движение + Space = вверх, Shift = вниз
            let mut swim_vel = move_dir * PLAYER_SPEED;
            if jump     { swim_vel.y += 3.5; }
            if fly_down { swim_vel.y -= 3.5; }
            // Плавное затухание вертикали, если нет ввода
            if !jump && !fly_down {
                let cur = physics.get_linvel(self.body_handle);
                swim_vel.y = cur.y * 0.8;
            }
            physics.set_linvel(self.body_handle, swim_vel);

        } else {
            let speed  = if move_dir.length_squared() > 1e-6 {
                self.stance.speed()
            } else { 0.0 };
            let cur_vy = physics.get_linvel(self.body_handle).y;

            let new_vy = if jump && self.on_ground {
                ext_debug_log!("PlayerController", "apply_movement", "Jump! vy={}", PLAYER_JUMP_VEL);
                PLAYER_JUMP_VEL
            } else if self.on_ground && cur_vy.abs() < 0.5 {
                0.0
            } else {
                cur_vy
            };

            // Если игрок прыгает и при этом стоит вплотную к стене
            // в направлении движения — обнуляем горизонтальную скорость,
            // чтобы вектор прыжка не шёл по диагонали в блок.
            let (horiz_x, horiz_z) = if jump && self.on_ground && speed > 0.0 {
                let horiz_dir = Vec3::new(move_dir.x, 0.0, move_dir.z);
                if physics.is_wall_in_direction(self.body_handle, horiz_dir, PLAYER_HALF_W) {
                    ext_debug_log!("PlayerController", "apply_movement",
                        "Wall detected in move direction — zeroing horizontal velocity");
                    (0.0, 0.0)
                } else {
                    (move_dir.x * speed, move_dir.z * speed)
                }
            } else {
                (move_dir.x * speed, move_dir.z * speed)
            };

            physics.set_linvel(self.body_handle,
                Vec3::new(horiz_x, new_vy, horiz_z));
        }
    }

    // -----------------------------------------------------------------------
    // Пост-шаговое обновление — вызывать ПОСЛЕ physics.step()
    // -----------------------------------------------------------------------

    pub fn update(&mut self, physics: &mut PhysicsWorld) {
        if self.fly_mode {
            flow_debug_log!("PlayerController", "update", "FLY pos=({:.1},{:.1},{:.1})",
                self.player_position(physics).x,
                self.player_position(physics).y,
                self.player_position(physics).z);
            return;
        }

        self.on_ground = physics.check_ground(self.body_handle, self.stance.half_h());
        physics.void_respawn(self.body_handle, SPAWN);

        flow_debug_log!("PlayerController", "update",
            "pos=({:.1},{:.1},{:.1}) on_ground={} stance={:?}",
            self.player_position(physics).x,
            self.player_position(physics).y,
            self.player_position(physics).z,
            self.on_ground, self.stance);
    }

    // -----------------------------------------------------------------------
    // Позиция игрока
    // -----------------------------------------------------------------------

    pub fn player_position(&self, physics: &PhysicsWorld) -> Vec3 {
        physics.get_translation(self.body_handle)
    }

    /// Позиция подошвы (низ AABB).
    pub fn foot_position(&self, physics: &PhysicsWorld) -> Vec3 {
        let p = self.player_position(physics);
        Vec3::new(p.x, p.y - self.stance.half_h(), p.z)
    }

    pub fn eye_position(&self, physics: &PhysicsWorld) -> Vec3 {
        self.player_position(physics) + Vec3::new(0.0, self.stance.eye_offset(), 0.0)
    }

    pub fn is_on_ground(&self) -> bool { self.on_ground }

    // -----------------------------------------------------------------------
    // Сохранение / восстановление
    // -----------------------------------------------------------------------

    /// Snapshot the player's runtime state into a serialisable DTO.
    pub fn to_save_data(&self, physics: &PhysicsWorld) -> crate::core::save_system::SavedPlayer {
        let pos = self.player_position(physics);
        crate::core::save_system::SavedPlayer {
            position:      [pos.x, pos.y, pos.z],
            fly_mode:      self.fly_mode,
            stance:        self.stance as u8,
            selected_slot: self.selected_slot,
            hotbar_ids:    self.hotbar_ids.clone(),
        }
    }

    /// Restore player state from a saved DTO (teleport + mode flags + slot).
    pub fn apply_save_data(
        &mut self,
        physics: &mut PhysicsWorld,
        data: &crate::core::save_system::SavedPlayer,
    ) {
        let pos = Vec3::new(data.position[0], data.position[1], data.position[2]);
        physics.set_translation(self.body_handle, pos);
        physics.set_linvel(self.body_handle, Vec3::ZERO);

        // Fly mode
        if data.fly_mode != self.fly_mode {
            self.toggle_fly(physics);
        }

        // Stance (0=Standing, 1=Crouching, 2=Prone)
        let target_stance = match data.stance {
            1 => PlayerStance::Crouching,
            2 => PlayerStance::Prone,
            _ => PlayerStance::Standing,
        };
        if target_stance != self.stance {
            // Skip the head-blocked check during restore (world not yet meshed).
            let old_hh = self.stance.half_h();
            self.collider_handle = physics.replace_player_collider(
                self.body_handle,
                self.collider_handle,
                old_hh,
                PLAYER_HALF_W,
                target_stance.half_h(),
            );
            self.stance = target_stance;
        }

        // Selected hotbar slot
        if data.selected_slot < self.hotbar_ids.len() {
            self.selected_slot = data.selected_slot;
        }

        debug_log!(
            "PlayerController", "apply_save_data",
            "Restored to ({:.1},{:.1},{:.1}) fly={} stance={:?} slot={}",
            pos.x, pos.y, pos.z, data.fly_mode, target_stance, data.selected_slot
        );
    }

    // -----------------------------------------------------------------------
    // Хотбар
    // -----------------------------------------------------------------------

    pub fn selected_block_id(&self) -> u8 {
        self.hotbar_ids.get(self.selected_slot).copied().unwrap_or(1)
    }

    pub fn selected_block_name(&self) -> &str {
        self.hotbar_names.get(self.selected_slot).map(|s| s.as_str()).unwrap_or("?")
    }

    pub fn selected_slot(&self) -> usize { self.selected_slot }
    pub fn slot_count(&self) -> usize     { self.hotbar_ids.len() }

    pub fn slot_name(&self, n: usize) -> &str {
        self.hotbar_names.get(n).map(|s| s.as_str()).unwrap_or("?")
    }

    pub fn scroll_slot(&mut self, delta: f32) {
        if self.hotbar_ids.is_empty() { return; }
        let n = self.hotbar_ids.len();
        if delta > 0.0 {
            self.selected_slot = (self.selected_slot + 1) % n;
        } else if delta < 0.0 {
            self.selected_slot = (self.selected_slot + n - 1) % n;
        }
    }

    pub fn select_slot(&mut self, slot: usize) {
        if slot < self.hotbar_ids.len() {
            self.selected_slot = slot;
        }
    }

    // -----------------------------------------------------------------------
    // Блочные операции
    // -----------------------------------------------------------------------

    pub fn break_block(target: Option<&RaycastResult>) -> Option<BlockOp> {
        target.map(|t| BlockOp::Break { x: t.block_pos.x, y: t.block_pos.y, z: t.block_pos.z })
    }

    pub fn place_block(
        &self,
        target:    Option<&RaycastResult>,
        physics:   &PhysicsWorld,
        block_id:  u8,
        look_dir:  Vec3,
        placement: &PlacementMode,
        default_rotation: u8,
    ) -> Option<BlockOp> {
        let t = target?;
        let pp    = self.player_position(physics);
        let place = t.block_pos + t.normal;
        let px = place.x as f32;
        let py = place.y as f32;
        let pz = place.z as f32;
        let hw = PLAYER_HALF_W;
        let hh = self.stance.half_h();

        let overlaps =
            px + 1.0 > pp.x - hw && px < pp.x + hw
            && py + 1.0 > pp.y - hh && py < pp.y + hh
            && pz + 1.0 > pp.z - hw && pz < pp.z + hw;

        if overlaps { return None; }

        let rotation = compute_block_rotation(placement, default_rotation, t.normal, look_dir);
        Some(BlockOp::Place {
            x: place.x, y: place.y, z: place.z,
            block_id,
            rotation,
        })
    }
}

// ---------------------------------------------------------------------------
// Block rotation computation
// ---------------------------------------------------------------------------

/// Compute the rotation value for a block being placed.
///
/// `placement`: the block's `PlacementMode`
/// `default_rotation`: fallback used for `Fixed` mode
/// `normal`: outward face normal of the block that was clicked (IVec3)
/// `look`: the player's camera forward vector
pub fn compute_block_rotation(
    placement: &PlacementMode,
    default_rotation: u8,
    normal: IVec3,
    look: Vec3,
) -> u8 {
    match placement {
        PlacementMode::Fixed => default_rotation,

        PlacementMode::LookHorizontal => horizontal_look_rotation(look),

        PlacementMode::LookFull => {
            // Vertical: front face follows look direction (same as regular blocks).
            // look up   → front faces up   → rot 6 (local+Z→world+Y via rot_x +90°)
            // look down → front faces down → rot 7 (local+Z→world-Y via rot_x -90°)
            if look.y > 0.65 { 6 }
            else if look.y < -0.65 { 7 }
            else { horizontal_look_rotation(look) }
        }

        PlacementMode::AttachHorizontal => {
            match (normal.x, normal.y, normal.z) {
                (0, 0, -1) => 6,  // north wall
                (0, 0,  1) => 7,  // south wall
                (1, 0,  0) => 8,  // east wall
                (-1, 0, 0) => 9,  // west wall
                _ => horizontal_look_rotation(look),
            }
        }

        PlacementMode::AttachFull => {
            match (normal.x, normal.y, normal.z) {
                (0,  1, 0) => 0,  // top face → place on floor
                (0, -1, 0) => 5,  // bottom face → place on ceiling
                (0,  0, -1) => 6, // north wall
                (0,  0,  1) => 7, // south wall
                (1,  0,  0) => 8, // east wall
                (-1, 0,  0) => 9, // west wall
                _ => default_rotation,
            }
        }

        PlacementMode::AttachFullRotatable => {
            match (normal.x, normal.y, normal.z) {
                (0,  1, 0) => horizontal_look_rotation(look),
                (0, -1, 0) => ceiling_look_rotation(look),
                (0,  0, -1) => 6,
                (0,  0,  1) => 7,
                (1,  0,  0) => 8,
                (-1, 0,  0) => 9,
                _ => default_rotation,
            }
        }

        PlacementMode::LookAxis => {
            // Axis determined by which face was clicked (log aligns with face normal axis).
            if normal.x != 0 { 14 }       // east/west face → X-axis (ring on E/W)
            else if normal.z != 0 { 15 }  // north/south face → Z-axis (ring on N/S)
            else { 0 }                      // top/bottom face → Y-axis (default, ring on T/B)
        }
    }
}

/// Map horizontal look direction → rotation 1–4 (N/S/E/W).
fn horizontal_look_rotation(look: Vec3) -> u8 {
    let yaw = look.x.atan2(-look.z).to_degrees();
    if yaw.abs() < 45.0 { 1 }                      // facing ≈ north
    else if yaw > 45.0 && yaw < 135.0 { 3 }        // facing ≈ east
    else if yaw < -45.0 && yaw > -135.0 { 4 }      // facing ≈ west
    else { 2 }                                       // facing ≈ south
}

/// Map look direction → ceiling+compass rotation 10–13.
fn ceiling_look_rotation(look: Vec3) -> u8 {
    let yaw = look.x.atan2(-look.z).to_degrees();
    if yaw.abs() < 45.0 { 10 }
    else if yaw > 45.0 && yaw < 135.0 { 12 }
    else if yaw < -45.0 && yaw > -135.0 { 13 }
    else { 11 }
}
