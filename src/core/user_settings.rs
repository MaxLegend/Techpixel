// =============================================================================
// QubePixel — UserSettings  (persistent user preferences saved to JSON)
// =============================================================================

use serde::{Deserialize, Serialize};
use crate::{debug_log};
use crate::core::config;

const PATH: &str = "saves/user_settings.json";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UserSettings {
    pub render_distance:           i32,
    pub lod_multiplier_x100:       u32,
    pub vertical_below:            i32,
    pub vertical_above:            i32,
    pub biome_ambient_tint_enabled: bool,
    pub shadow_sun_enabled:        bool,
    pub shadow_block_enabled:      bool,
    pub gi_mode:                   u32,
    pub shadow_quality:            u32,
    pub halo_enabled:              bool,
    pub volumetric_rays_enabled:   bool,
    #[serde(default = "default_mouse_sensitivity")]
    pub mouse_sensitivity_x10000:  u32,
    #[serde(default = "default_fly_speed")]
    pub fly_speed:                 u32,
}

fn default_mouse_sensitivity() -> u32 { 30 }
fn default_fly_speed() -> u32 { 12 }

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            render_distance:           12,
            lod_multiplier_x100:       300,
            vertical_below:            0,
            vertical_above:            0,
            biome_ambient_tint_enabled: true,
            shadow_sun_enabled:        true,
            shadow_block_enabled:      true,
            gi_mode:                   4,
            shadow_quality:            64,
            halo_enabled:              true,
            volumetric_rays_enabled:   true,
            mouse_sensitivity_x10000:  30,
            fly_speed:                 12,
        }
    }
}

/// Read the JSON file (or defaults) and push all values into the global atomics.
pub fn load_and_apply() {
    if let Err(e) = std::fs::create_dir_all("saves") {
        debug_log!("UserSettings", "load_and_apply", "Could not create saves/ dir: {}", e);
    }

    let settings: UserSettings = std::fs::read_to_string(PATH)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    debug_log!(
        "UserSettings", "load_and_apply",
        "Loaded: rd={} lod={} vb={} va={} tint={} sun={} block={} gi_mode={} sq={}",
        settings.render_distance,
        settings.lod_multiplier_x100,
        settings.vertical_below,
        settings.vertical_above,
        settings.biome_ambient_tint_enabled,
        settings.shadow_sun_enabled,
        settings.shadow_block_enabled,
        settings.gi_mode,
        settings.shadow_quality,
    );

    config::set_render_distance(settings.render_distance);
    config::set_lod_multiplier(settings.lod_multiplier_x100 as f32 / 100.0);
    config::set_vertical_below(settings.vertical_below);
    config::set_vertical_above(settings.vertical_above);
    config::set_biome_ambient_tint_enabled(settings.biome_ambient_tint_enabled);
    config::set_shadow_sun_enabled(settings.shadow_sun_enabled);
    config::set_shadow_block_enabled(settings.shadow_block_enabled);
    config::set_gi_mode(settings.gi_mode);
    config::set_shadow_quality(settings.shadow_quality);
    config::set_halo_enabled(settings.halo_enabled);
    config::set_volumetric_rays_enabled(settings.volumetric_rays_enabled);
    config::set_mouse_sensitivity(settings.mouse_sensitivity_x10000 as f32 / 10000.0);
    config::set_fly_speed(settings.fly_speed as f32);
}

/// Read current atomics and write them to the JSON file.
pub fn save_current() {
    let settings = UserSettings {
        render_distance:           config::render_distance(),
        lod_multiplier_x100:       (config::lod_multiplier() * 100.0).round() as u32,
        vertical_below:            config::vertical_below(),
        vertical_above:            config::vertical_above(),
        biome_ambient_tint_enabled: config::biome_ambient_tint_enabled(),
        shadow_sun_enabled:        config::shadow_sun_enabled(),
        shadow_block_enabled:      config::shadow_block_enabled(),
        gi_mode:                   config::gi_mode(),
        shadow_quality:            config::shadow_quality(),
        halo_enabled:              config::halo_enabled(),
        volumetric_rays_enabled:   config::volumetric_rays_enabled(),
        mouse_sensitivity_x10000:  (config::mouse_sensitivity() * 10000.0).round() as u32,
        fly_speed:                 config::fly_speed() as u32,
    };

    match serde_json::to_string_pretty(&settings) {
        Ok(json) => {
            if let Err(e) = std::fs::write(PATH, json) {
                debug_log!("UserSettings", "save_current", "Write error: {}", e);
            } else {
                debug_log!("UserSettings", "save_current", "Settings saved to {}", PATH);
            }
        }
        Err(e) => {
            debug_log!("UserSettings", "save_current", "Serialize error: {}", e);
        }
    }
}
