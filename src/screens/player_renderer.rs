// =============================================================================
// QubePixel — PlayerRenderer  (pose / animation logic for the player entity)
// =============================================================================
//
// After the Entity system refactor this module only handles CPU-side pose
// computation.  All GPU pipeline, shader compilation, and draw calls are
// delegated to EntityRenderer.
//
// Bone hierarchy and animation amplitudes are kept here for clarity.
//
// Bone hierarchy (transforms computed on CPU each frame):
//   body            — body_yaw
//   head            — body_yaw + head_relative_yaw + head_pitch  (hidden in 1st person)
//   leftArm         — body_yaw + arm swing
//   rightArm        — body_yaw − arm swing
//   leftThigh       — body_yaw + thigh swing
//   leftShin        — leftThigh + knee bend
//   rightThigh      — body_yaw − thigh swing
//   rightShin       — rightThigh + knee bend
// =============================================================================

use std::collections::HashMap;
use glam::{Mat4, Vec3};

use crate::core::entity::EntityId;
use crate::core::player_model::{PlayerModel, MODEL_SCALE};
use crate::core::lighting::DayNightCycle;
use crate::screens::entity_renderer::{BoneDrawData, EntityRenderCall};

/// Overall player model scale (0.85 = 85% of original Minecraft proportions).
pub const PLAYER_SIZE: f32 = 0.85;

/// Walking animation amplitudes (radians).
const WALK_LEG_AMP:   f32 = 0.45;  // thigh swing (~26°)
const WALK_KNEE_AMP:  f32 = 0.55;  // shin bend on back-swing (~31°)
const WALK_ARM_AMP:   f32 = 0.35;  // arm swing (~20°)
const WALK_ELBOW_AMP: f32 = 0.30;  // elbow bend on arm back-swing (~17°)

/// Back-to-front draw order for depth stability with alpha blending.
pub const PLAYER_BONE_ORDER: &[&str] = &[
    "body",
    "leftThigh",  "leftShin",
    "rightThigh", "rightShin",
    "leftArm",    "leftForearm",
    "rightArm",   "rightForearm",
    "head",
];

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    FirstPerson,
    ThirdPerson,
}

/// All animation/pose data needed to compute bone transforms each frame.
pub struct PlayerPose {
    /// World-space foot position (bottom of player AABB).
    pub foot_pos:        Vec3,
    /// Direction the BODY faces (may lag behind camera yaw).
    pub body_yaw:        f32,
    /// Head yaw relative to body, clamped to ±π/2.
    pub head_yaw_offset: f32,
    /// Camera pitch (positive = look up).
    pub head_pitch:      f32,
    /// Walking animation phase [0, 2π).
    pub walk_phase:      f32,
    pub is_walking:      bool,
    pub view_mode:       ViewMode,
}

// ---------------------------------------------------------------------------
// PlayerRenderer — CPU pose only, no GPU resources
// ---------------------------------------------------------------------------

pub struct PlayerRenderer {
    /// Head pivot in model space (block units).  Exposed for camera placement.
    pub head_pivot: Vec3,
}

impl PlayerRenderer {
    /// Build a renderer from the parsed player model.
    /// GPU resources (vertex buffers, pipeline) live in EntityRenderer.
    pub fn new(model: &PlayerModel) -> Self {
        let head_pivot = model.bone_pivots
            .get("head")
            .copied()
            .unwrap_or(Vec3::new(0.0, 24.0 * MODEL_SCALE, 0.0));
        Self { head_pivot }
    }

    // -----------------------------------------------------------------------
    // Bone transform computation
    // -----------------------------------------------------------------------

    fn pivot_rx(pivot: Vec3, angle: f32) -> Mat4 {
        Mat4::from_translation(pivot)
            * Mat4::from_rotation_x(angle)
            * Mat4::from_translation(-pivot)
    }

    fn pivot_ry(pivot: Vec3, angle: f32) -> Mat4 {
        Mat4::from_translation(pivot)
            * Mat4::from_rotation_y(angle)
            * Mat4::from_translation(-pivot)
    }

    /// Compute world-space transform for each bone given the current pose.
    pub fn bone_transforms(pose: &PlayerPose) -> HashMap<String, Mat4> {
        let base = Mat4::from_translation(pose.foot_pos)
            * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2 - pose.body_yaw)
            * Mat4::from_scale(Vec3::splat(PLAYER_SIZE));

        // Bone pivots in (pre-scale) block units.
        let head_piv  = Vec3::new(0.0,                24.0 * MODEL_SCALE, 0.0);
        let l_arm_piv = Vec3::new( 5.0 * MODEL_SCALE, 22.0 * MODEL_SCALE, 0.0);
        let r_arm_piv = Vec3::new(-5.0 * MODEL_SCALE, 22.0 * MODEL_SCALE, 0.0);
        let l_elb_piv = Vec3::new( 5.0 * MODEL_SCALE, 18.0 * MODEL_SCALE, 0.0);
        let r_elb_piv = Vec3::new(-5.0 * MODEL_SCALE, 18.0 * MODEL_SCALE, 0.0);
        let l_hip_piv = Vec3::new( 1.9 * MODEL_SCALE, 12.0 * MODEL_SCALE, 0.0);
        let r_hip_piv = Vec3::new(-1.9 * MODEL_SCALE, 12.0 * MODEL_SCALE, 0.0);
        let l_kne_piv = Vec3::new( 1.9 * MODEL_SCALE,  6.0 * MODEL_SCALE, 0.0);
        let r_kne_piv = Vec3::new(-1.9 * MODEL_SCALE,  6.0 * MODEL_SCALE, 0.0);

        let (leg_amp, knee_amp, arm_amp, elbow_amp) = if pose.is_walking {
            (WALK_LEG_AMP, WALK_KNEE_AMP, WALK_ARM_AMP, WALK_ELBOW_AMP)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let ws = pose.walk_phase.sin();

        let l_thigh  = -leg_amp * ws;
        let r_thigh  =  leg_amp * ws;
        let l_knee   = knee_amp  * (-ws).max(0.0);
        let r_knee   = knee_amp  * ws.max(0.0);
        let l_arm    =  arm_amp  * ws;
        let r_arm    = -arm_amp  * ws;
        let l_elbow  = -(elbow_amp * (-ws).max(0.0));
        let r_elbow  = -(elbow_amp * ws.max(0.0));

        let head_local    = Self::pivot_ry(head_piv, -pose.head_yaw_offset)
                          * Self::pivot_rx(head_piv, -pose.head_pitch);
        let l_arm_local   = Self::pivot_rx(l_arm_piv,  l_arm);
        let r_arm_local   = Self::pivot_rx(r_arm_piv,  r_arm);
        let l_thigh_local = Self::pivot_rx(l_hip_piv, l_thigh);
        let r_thigh_local = Self::pivot_rx(r_hip_piv, r_thigh);

        let mut t = HashMap::new();
        t.insert("body".into(),         base);
        t.insert("head".into(),         base * head_local);
        t.insert("leftArm".into(),      base * l_arm_local);
        t.insert("leftForearm".into(),  base * l_arm_local   * Self::pivot_rx(l_elb_piv, l_elbow));
        t.insert("rightArm".into(),     base * r_arm_local);
        t.insert("rightForearm".into(), base * r_arm_local   * Self::pivot_rx(r_elb_piv, r_elbow));
        t.insert("leftThigh".into(),    base * l_thigh_local);
        t.insert("leftShin".into(),     base * l_thigh_local * Self::pivot_rx(l_kne_piv, l_knee));
        t.insert("rightThigh".into(),   base * r_thigh_local);
        t.insert("rightShin".into(),    base * r_thigh_local * Self::pivot_rx(r_kne_piv, r_knee));
        t
    }

    // -----------------------------------------------------------------------
    // Build render call
    // -----------------------------------------------------------------------

    /// Produce an EntityRenderCall ready to be fed to EntityRenderer::render().
    pub fn build_render_call(
        &self,
        entity_id:   EntityId,
        pose:        &PlayerPose,
        view_proj:   Mat4,
        camera_pos:  Vec3,
        day_night:   &DayNightCycle,
        ambient_min: f32,
    ) -> EntityRenderCall {
        let transforms = Self::bone_transforms(pose);

        let mut skip = Vec::new();
        if pose.view_mode == ViewMode::FirstPerson {
            skip.push("head".to_string());
        }

        EntityRenderCall {
            entity_id,
            model_id:       "player".to_string(),
            bones:          transforms.into_iter()
                                .map(|(n, m)| BoneDrawData { name: n, world_transform: m })
                                .collect(),
            view_proj,
            sun_dir:        day_night.sun_direction(),
            sun_color:      day_night.sun_color(),
            sun_intensity:  day_night.sun_intensity(),
            moon_dir:       day_night.moon_direction(),
            moon_color:     day_night.moon_color(),
            moon_intensity: day_night.moon_intensity(),
            ambient:        day_night.ambient_color(),
            ambient_min,
            camera_pos,
            shadow_sun_enabled: true,
            shadow_offset:      0.3,
            skip_bones:     skip,
        }
    }
}
