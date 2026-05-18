// =============================================================================
// QubePixel — SheepRenderer  (pose / animation logic for sheep entities)
// =============================================================================
//
// Builds an EntityRenderCall for a single sheep instance.  All GPU resources
// live in EntityRenderer (registered as model "sheep").  This module only
// computes per-bone world transforms based on the AI walk phase and yaw.

use std::collections::HashMap;
use glam::{Mat4, Vec3};

use crate::core::entity::EntityId;
use crate::core::lighting::DayNightCycle;
use crate::screens::entity_renderer::{BoneDrawData, EntityRenderCall};
use crate::screens::player_renderer::ViewMode;

/// Overall sheep model scale (block units).  Bedrock sheep are ~14 model units
/// tall at the back of the body; this brings them to roughly 1.2 block height.
pub const SHEEP_SIZE: f32 = 0.95;

/// Walking animation amplitude (radians).
const WALK_LEG_AMP: f32 = 0.55;

/// Back-to-front draw order for depth stability with alpha blending.
pub const SHEEP_BONE_ORDER: &[&str] = &[
    "body",
    "leg0", "leg1", "leg2", "leg3",
    "head",
];

/// Bedrock model unit → block unit conversion (1 pixel = 1/16 block).
const MODEL_SCALE: f32 = 1.0 / 16.0;

/// Minimal pose state required to render a sheep.
pub struct SheepPose {
    pub foot_pos:   Vec3,
    /// Body yaw in radians (0 = facing +Z).
    pub yaw:        f32,
    /// Walk animation phase [0, 2π).
    pub walk_phase: f32,
    pub is_walking: bool,
}

pub struct SheepRenderer;

impl SheepRenderer {
    fn pivot_rx(pivot: Vec3, angle: f32) -> Mat4 {
        Mat4::from_translation(pivot)
            * Mat4::from_rotation_x(angle)
            * Mat4::from_translation(-pivot)
    }

    /// Compute world-space transform for every sheep bone.
    pub fn bone_transforms(pose: &SheepPose) -> HashMap<String, Mat4> {
        // Base: foot position + yaw + uniform scale.
        // Subtract FRAC_PI_2 to align the model's +Z forward with body_yaw=0.
        let base = Mat4::from_translation(pose.foot_pos)
            * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2 - pose.yaw)
            * Mat4::from_scale(Vec3::splat(SHEEP_SIZE));

        // Pivots (in pre-scale block units) — must match the JSON file's `pivot` arrays.
        let leg0_piv = Vec3::new(-3.0 * MODEL_SCALE, 12.0 * MODEL_SCALE,  7.0 * MODEL_SCALE);
        let leg1_piv = Vec3::new( 3.0 * MODEL_SCALE, 12.0 * MODEL_SCALE,  7.0 * MODEL_SCALE);
        let leg2_piv = Vec3::new(-3.0 * MODEL_SCALE, 12.0 * MODEL_SCALE, -5.0 * MODEL_SCALE);
        let leg3_piv = Vec3::new( 3.0 * MODEL_SCALE, 12.0 * MODEL_SCALE, -5.0 * MODEL_SCALE);

        let amp = if pose.is_walking { WALK_LEG_AMP } else { 0.0 };
        let ws  = pose.walk_phase.sin();

        // Opposing leg pairs swing out of phase.
        let leg0_a =  amp * ws;
        let leg1_a = -amp * ws;
        let leg2_a = -amp * ws;
        let leg3_a =  amp * ws;

        let mut t = HashMap::new();
        t.insert("body".into(), base);
        t.insert("head".into(), base);
        t.insert("leg0".into(), base * Self::pivot_rx(leg0_piv, leg0_a));
        t.insert("leg1".into(), base * Self::pivot_rx(leg1_piv, leg1_a));
        t.insert("leg2".into(), base * Self::pivot_rx(leg2_piv, leg2_a));
        t.insert("leg3".into(), base * Self::pivot_rx(leg3_piv, leg3_a));
        t
    }

    /// Build a draw call ready for EntityRenderer::render().
    pub fn build_render_call(
        entity_id:   EntityId,
        pose:        &SheepPose,
        view_proj:   Mat4,
        camera_pos:  Vec3,
        day_night:   &DayNightCycle,
        ambient_min: f32,
    ) -> EntityRenderCall {
        let transforms = Self::bone_transforms(pose);
        EntityRenderCall {
            entity_id,
            model_id:       "sheep".to_string(),
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
            skip_bones:     Vec::new(),
        }
    }
}

// Suppresses an unused-import warning when ViewMode is not referenced elsewhere
// after this re-export anchor; kept for downstream parity with PlayerRenderer.
#[allow(dead_code)]
fn _unused_viewmode_anchor(_v: ViewMode) {}
