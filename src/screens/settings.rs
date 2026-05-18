// =============================================================================
// Techpixel — SettingsScreen (egui-based settings)
// =============================================================================

use crate::core::config;
use crate::core::user_settings;
use crate::core::screen::{Screen, ScreenAction};
use crate::{debug_log, flow_debug_log};
use winit::event::WindowEvent;

pub struct SettingsScreen {
    pending_action: ScreenAction,
}

impl SettingsScreen {
    pub fn new() -> Self {
        debug_log!("SettingsScreen", "new", "Creating SettingsScreen (egui)");
        Self {
            pending_action: ScreenAction::None,
        }
    }
}

impl Screen for SettingsScreen {
    fn name(&self) -> &str { "Settings" }
    fn load(&mut self) {
        debug_log!("SettingsScreen", "load", "Loading SettingsScreen");
    }
    fn start(&mut self) {
        debug_log!("SettingsScreen", "start", "First-frame init");
    }
    fn update(&mut self, _dt: f64) {
        flow_debug_log!("SettingsScreen", "update", "Update tick");
    }

    fn render(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        _view:    &wgpu::TextureView,
        _device:  &wgpu::Device,
        _queue:   &wgpu::Queue,
        _format:  wgpu::TextureFormat,
        _width:   u32,
        _height:  u32,
    ) {}

    fn post_process(&mut self, _dt: f64) {}

    fn build_ui(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            debug_log!("SettingsScreen", "build_ui", "Escape -> Pop");
            self.pending_action = ScreenAction::Pop;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // --- Title ---
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.label(
                    egui::RichText::new("Settings")
                        .size(36.0)
                        .color(egui::Color32::WHITE),
                );
                ui.add_space(20.0);
                ui.separator();
                ui.add_space(10.0);
            });

            // --- Two-column settings body ---
            ui.columns(2, |cols| {
                // ===================================================
                // LEFT COLUMN — world / gameplay settings
                // ===================================================
                cols[0].vertical_centered(|ui| {
                    ui.set_width(320.0);

                    // --- Render Distance slider ---
                    ui.label(
                        egui::RichText::new("Render Distance")
                            .size(18.0)
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);

                    let mut rd = config::render_distance() as f32;
                    let slider = egui::Slider::new(&mut rd, 1.0..=128.0)
                        .step_by(1.0)
                        .suffix(" chunks")
                        .text("Distance");
                    if ui.add(slider).changed() {
                        config::set_render_distance(rd as i32);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui",
                            "Render distance -> {}", rd as i32);
                    }
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "Higher values increase world visibility \
                             but may reduce performance."
                        )
                        .size(12.0)
                        .color(egui::Color32::GRAY),
                    );

                    ui.add_space(20.0);

                    // --- Mouse Sensitivity slider ---
                    ui.label(
                        egui::RichText::new("Mouse Sensitivity")
                            .size(18.0)
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);

                    let mut sens = config::mouse_sensitivity();
                    let slider = egui::Slider::new(&mut sens, 0.0005..=0.01)
                        .step_by(0.0005)
                        .text("Sensitivity");
                    if ui.add(slider).changed() {
                        config::set_mouse_sensitivity(sens);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui",
                            "Mouse sensitivity -> {:.4}", sens);
                    }
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "Camera rotation speed. Lower = slower, higher = faster."
                        )
                        .size(12.0)
                        .color(egui::Color32::GRAY),
                    );

                    ui.add_space(20.0);
                    ui.label(
                        egui::RichText::new("Chunks Below Camera")
                            .size(18.0)
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);

                    let mut vb = config::vertical_below() as f32;
                    let slider = egui::Slider::new(&mut vb, 1.0..=16.0)
                        .step_by(1.0)
                        .suffix(" chunks")
                        .text("Below");
                    if ui.add(slider).changed() {
                        config::set_vertical_below(vb as i32);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui",
                            "Vertical below -> {}", vb as i32);
                    }
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "How many chunk layers load below the camera. \
                             Lower = better underground performance."
                        )
                        .size(12.0)
                        .color(egui::Color32::GRAY),
                    );

                    ui.add_space(20.0);

                    // --- Vertical Above slider ---
                    ui.label(
                        egui::RichText::new("Chunks Above Camera")
                            .size(18.0)
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);

                    let mut va = config::vertical_above() as f32;
                    let slider = egui::Slider::new(&mut va, 1.0..=16.0)
                        .step_by(1.0)
                        .suffix(" chunks")
                        .text("Above");
                    if ui.add(slider).changed() {
                        config::set_vertical_above(va as i32);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui",
                            "Vertical above -> {}", va as i32);
                    }
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "How many chunk layers load above the camera. \
                             Lower = fewer sky chunks loaded."
                        )
                        .size(12.0)
                        .color(egui::Color32::GRAY),
                    );

                    ui.add_space(16.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // --- Biome ambient tint toggle ---
                    ui.label(
                        egui::RichText::new("Biome Ambient Tint")
                            .size(16.0)
                            .strong(),
                    );
                    ui.add_space(4.0);

                    let mut tint_on = config::biome_ambient_tint_enabled();
                    if ui.checkbox(&mut tint_on, "Enable biome colour tint on ambient light").changed() {
                        config::set_biome_ambient_tint_enabled(tint_on);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui",
                            "Biome ambient tint -> {}", tint_on);
                    }
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "Tints the ambient lighting based on the current biome. \
                             Hot biomes get a warm yellow cast; icy biomes get a cool blue cast."
                        )
                        .size(12.0)
                        .color(egui::Color32::GRAY),
                    );
                });

                // ===================================================
                // RIGHT COLUMN — graphics settings
                // ===================================================
                cols[1].vertical(|ui| {
                    ui.add_space(4.0);

                    ui.label(
                        egui::RichText::new("Graphics")
                            .size(20.0)
                            .strong()
                            .color(egui::Color32::WHITE),
                    );
                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(12.0);

                    // --- Global Illumination (VCT) ---
                    ui.label(
                        egui::RichText::new("Global Illumination")
                            .size(16.0)
                            .strong()
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);

                    let mut gi_mode = config::gi_mode();
                    let prev_gi_mode = gi_mode;
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut gi_mode, 0u32, "Off");
                        ui.selectable_value(&mut gi_mode, 1u32, "Mono 16");
                        ui.selectable_value(&mut gi_mode, 2u32, "Mono 32");
                        ui.selectable_value(&mut gi_mode, 3u32, "RGB 32");
                        ui.selectable_value(&mut gi_mode, 4u32, "Full");
                    });
                    if gi_mode != prev_gi_mode {
                        config::set_gi_mode(gi_mode);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui", "GI mode -> {}", gi_mode);
                    }
                    ui.add_space(4.0);
                    let gi_desc = match gi_mode {
                        0 => "GI off — highest FPS. Block lights still cast DDA shadows.",
                        1 => "Mono 16 — 16-step monochromatic flood-fill, ~16 block radius. Minecraft-style.",
                        2 => "Mono 32 — 32-step monochromatic flood-fill, ~32 block radius.",
                        3 => "RGB 32 — 32-step coloured flood-fill, ~32 block radius.",
                        _ => "Full — 64-step coloured flood-fill, ~64 block radius. Maximum quality.",
                    };
                    ui.label(
                        egui::RichText::new(gi_desc)
                            .size(11.0)
                            .color(egui::Color32::GRAY),
                    );

                    ui.add_space(16.0);
                    ui.separator();
                    ui.add_space(12.0);

                    // --- DDA Ray Tracing (Shadows) ---
                    ui.label(
                        egui::RichText::new("DDA Ray Tracing")
                            .size(16.0)
                            .strong()
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);

                    let mut sun_shadows = config::shadow_sun_enabled();
                    if ui.checkbox(&mut sun_shadows, "Sun / Moon shadows").changed() {
                        config::set_shadow_sun_enabled(sun_shadows);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui",
                            "Sun shadow -> {}", sun_shadows);
                    }
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "Voxel DDA shadows cast by blocks in the \
                             direction of the sun or moon."
                        )
                        .size(11.0)
                        .color(egui::Color32::GRAY),
                    );

                    ui.add_space(12.0);

                    let mut block_shadows = config::shadow_block_enabled();
                    if ui.checkbox(&mut block_shadows, "Block light shadows").changed() {
                        config::set_shadow_block_enabled(block_shadows);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui",
                            "Block shadow -> {}", block_shadows);
                    }
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "Voxel DDA shadows between a surface and \
                             point/spot light sources (glowstone, etc.)."
                        )
                        .size(11.0)
                        .color(egui::Color32::GRAY),
                    );

                    ui.add_space(12.0);

                    // --- Volumetric effects ---
                    ui.label(
                        egui::RichText::new("Volumetric Effects")
                            .size(16.0)
                            .strong()
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(6.0);

                    let mut halo_on = config::halo_enabled();
                    if ui.checkbox(&mut halo_on, "Light halos").changed() {
                        config::set_halo_enabled(halo_on);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui", "Halo -> {}", halo_on);
                    }
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "Billboard glow quads around point and spot lights."
                        )
                        .size(11.0)
                        .color(egui::Color32::GRAY),
                    );

                    ui.add_space(8.0);

                    let mut rays_on = config::volumetric_rays_enabled();
                    if ui.checkbox(&mut rays_on, "Volumetric god rays").changed() {
                        config::set_volumetric_rays_enabled(rays_on);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui", "Volumetric rays -> {}", rays_on);
                    }
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "Beam quads emitted by spot lights (spotlights, etc.)."
                        )
                        .size(11.0)
                        .color(egui::Color32::GRAY),
                    );

                    ui.add_space(12.0);
                    ui.separator();
                    ui.add_space(12.0);

                    // --- Shadow Quality (DDA step count) ---
                    ui.label(
                        egui::RichText::new("Shadow Quality")
                            .size(14.0)
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(6.0);

                    let mut q = config::shadow_quality();
                    let prev_q = q;
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut q, 8u32,  "Low");
                        ui.selectable_value(&mut q, 16u32, "Medium");
                        ui.selectable_value(&mut q, 32u32, "High");
                        ui.selectable_value(&mut q, 64u32, "Ultra");
                    });
                    if q != prev_q {
                        config::set_shadow_quality(q);
                        user_settings::save_current();
                        debug_log!("SettingsScreen", "build_ui", "Shadow quality -> {} steps", q);
                    }
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(
                            "DDA ray steps per shadow query. \
                             Lower = faster but shallower shadows."
                        )
                        .size(11.0)
                        .color(egui::Color32::GRAY),
                    );
                });
            });

            // --- Back button (centred at the bottom) ---
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);

                let back_btn = egui::Button::new(
                    egui::RichText::new("\u{2190}  Back").size(20.0),
                )
                    .min_size(egui::vec2(240.0, 48.0));

                if ui.add(back_btn).clicked() {
                    debug_log!("SettingsScreen", "build_ui", "Back clicked");
                    self.pending_action = ScreenAction::Pop;
                }
            });
        });
    }

    fn on_event(&mut self, _event: &WindowEvent) {}

    fn poll_action(&mut self) -> ScreenAction {
        std::mem::replace(&mut self.pending_action, ScreenAction::None)
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }
    }
}
