// =============================================================================
// QubePixel — World Generation Visualizer
//
// Interactive 2D preview of terrain generation layers.
// Opened via "World Gen" button in the main menu.
// Uses full-screen panels: left/top controls, right params sidebar, center map.
// =============================================================================

use egui::{Color32, ColorImage, TextureHandle, TextureOptions, Ui};

use crate::core::config::WorldGenConfig;
use crate::core::world_gen::pipeline::BiomePipeline;
use crate::core::world_gen::nature::NatureLayer;

// ---------------------------------------------------------------------------
// Visualization modes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisMode {
    Continentalness,
    Erosion,
    SurfaceHeight,
    Climate,
    Humidity,
    Geology,
    BiomeCombined,
    NatureLayer,
    Rivers,
}

impl VisMode {
    const ALL: [VisMode; 9] = [
        VisMode::BiomeCombined,
        VisMode::NatureLayer,
        VisMode::Rivers,
        VisMode::Continentalness,
        VisMode::Erosion,
        VisMode::SurfaceHeight,
        VisMode::Climate,
        VisMode::Humidity,
        VisMode::Geology,
    ];

    fn label(&self) -> &'static str {
        match self {
            VisMode::Continentalness => "Continentalness",
            VisMode::Erosion => "Erosion",
            VisMode::SurfaceHeight => "Surface Height",
            VisMode::Climate => "Climate Zones",
            VisMode::Humidity => "Humidity",
            VisMode::Geology => "Geology Regions",
            VisMode::BiomeCombined => "Biome (Combined)",
            VisMode::NatureLayer => "Nature / Vegetation",
            VisMode::Rivers => "Rivers",
        }
    }
}

// ---------------------------------------------------------------------------
// Visualizer state
// ---------------------------------------------------------------------------

/// Maximum pixels per axis for the map texture. Limits rebuild cost.
const MAP_MAX: usize = 1024;

pub struct WorldGenVisualizer {
    pub visible: bool,

    // Pipeline for sampling.
    seed: u64,
    pipeline: BiomePipeline,

    // Live-editable params (copy of WorldGenConfig).
    params: WorldGenConfig,
    params_dirty: bool,

    // View params.
    center_x: f32,
    center_z: f32,
    scale: f32, // blocks per pixel

    // Visualization.
    mode: VisMode,
    texture: Option<TextureHandle>,
    needs_rebuild: bool,
    /// Set after rebuild — causes a single GPU upload, then cleared.
    texture_dirty: bool,
    /// Current texture dimensions (updated to match display area each frame).
    map_w: usize,
    map_h: usize,
    pixel_buf: Vec<Color32>,

    // Status message for save feedback
    save_status: Option<String>,
}

impl WorldGenVisualizer {
    pub fn new() -> Self {
        let params = crate::core::config::world_gen_config().clone();
        let seed = 12345;
        Self {
            visible: false,
            seed,
            pipeline: BiomePipeline::with_config(seed, &params),
            params,
            params_dirty: false,
            center_x: 0.0,
            center_z: 0.0,
            scale: 4.0,
            mode: VisMode::BiomeCombined,
            texture: None,
            needs_rebuild: true,
            texture_dirty: true,
            map_w: 512,
            map_h: 512,
            pixel_buf: vec![Color32::BLACK; 512 * 512],
            save_status: None,
        }
    }

    /// Draw the visualizer as full-screen panels (no floating window).
    /// Call this from the dedicated screen's build_ui.
    pub fn draw_egui(&mut self, ctx: &egui::Context) {
        if !self.visible { return; }

        // Rebuild pipeline if params changed.
        if self.params_dirty {
            self.pipeline = BiomePipeline::with_config(self.seed, &self.params);
            self.params_dirty = false;
            self.needs_rebuild = true;
        }

        // Right panel: world gen parameter editor.
        egui::SidePanel::right("worldgen_params_panel")
            .min_width(300.0)
            .max_width(420.0)
            .show(ctx, |ui| {
                self.draw_params_panel(ui);
            });

        // Top panel: mode selector + view controls.
        egui::TopBottomPanel::top("worldgen_controls_panel")
            .show(ctx, |ui| {
                self.draw_controls(ui);
            });

        // Central panel: the map.
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_map(ui, ctx);
        });
    }

    fn draw_controls(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Mode:");
            for mode in &VisMode::ALL {
                if ui.selectable_label(self.mode == *mode, mode.label()).clicked() {
                    self.mode = *mode;
                    self.needs_rebuild = true;
                }
            }

            ui.separator();

            ui.label("Seed:");
            let mut seed_i64 = self.seed as i64;
            if ui.add(egui::DragValue::new(&mut seed_i64).speed(1)).changed() {
                self.seed = seed_i64 as u64;
                self.params_dirty = true;
            }

            ui.separator();

            ui.label("Scale:");
            if ui.add(egui::DragValue::new(&mut self.scale)
                .speed(0.1)
                .range(0.5..=128.0)
                .suffix(" b/px"))
                .changed()
            {
                self.needs_rebuild = true;
            }

            ui.separator();

            ui.label("X:");
            if ui.add(egui::DragValue::new(&mut self.center_x).speed(10.0)).changed() {
                self.needs_rebuild = true;
            }
            ui.label("Z:");
            if ui.add(egui::DragValue::new(&mut self.center_z).speed(10.0)).changed() {
                self.needs_rebuild = true;
            }

            if ui.button("Reset View").clicked() {
                self.center_x = 0.0;
                self.center_z = 0.0;
                self.scale = 4.0;
                self.needs_rebuild = true;
            }
        });
    }

    fn draw_map(&mut self, ui: &mut Ui, ctx: &egui::Context) {
        // 1. Allocate display area.
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(
            available,
            egui::Sense::click_and_drag(),
        );

        // 2. Detect display-size change → resize buffer.
        let new_w = (rect.width()  as usize).min(MAP_MAX).max(64);
        let new_h = (rect.height() as usize).min(MAP_MAX).max(64);
        if new_w != self.map_w || new_h != self.map_h {
            self.map_w = new_w;
            self.map_h = new_h;
            self.pixel_buf = vec![Color32::BLACK; new_w * new_h];
            self.texture = None;
            self.needs_rebuild = true;
        }

        // 3. Rebuild pixel buffer (parallel) only when something changed.
        if self.needs_rebuild {
            self.rebuild_map();
            self.needs_rebuild = false;
            self.texture_dirty = true;
        }

        // 4. Upload to GPU only when the pixel data actually changed.
        if self.texture_dirty {
            let rgba: Vec<u8> = self.pixel_buf.iter()
                .flat_map(|c| [c.r(), c.g(), c.b(), c.a()])
                .collect();
            let image = ColorImage::from_rgba_unmultiplied([self.map_w, self.map_h], &rgba);
            match &mut self.texture {
                Some(t) => { t.set(image, TextureOptions::NEAREST); }
                None    => { self.texture = Some(ctx.load_texture("worldgen_map", image, TextureOptions::NEAREST)); }
            }
            self.texture_dirty = false;
        }

        // 5. Render — just blit the cached texture handle.
        let tex = match &self.texture {
            Some(t) => t.clone(),
            None    => return,
        };

        // Handle drag to pan.
        if response.dragged() {
            let delta = response.drag_delta();
            self.center_x -= delta.x * self.scale;
            self.center_z -= delta.y * self.scale;
            self.needs_rebuild = true;
        }

        // Handle scroll to zoom.
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > 0.1 {
                let factor = if scroll > 0.0 { 0.9 } else { 1.1 };
                self.scale = (self.scale * factor).clamp(0.5, 128.0);
                self.needs_rebuild = true;
            }
        }

        // Draw the texture — fills non-square rect.
        ui.painter().image(
            tex.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );

        // Draw crosshair at center.
        let mid = rect.center();
        let ch = 8.0;
        ui.painter().line_segment(
            [egui::pos2(mid.x - ch, mid.y), egui::pos2(mid.x + ch, mid.y)],
            egui::Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 255, 255, 200)),
        );
        ui.painter().line_segment(
            [egui::pos2(mid.x, mid.y - ch), egui::pos2(mid.x, mid.y + ch)],
            egui::Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 255, 255, 200)),
        );

        // Tooltip at cursor position (not in corner).
        if response.hovered() {
            if let Some(pos) = response.hover_pos() {
                // Map display pixel → texture pixel → world coordinate.
                // rect size may exceed MAP_MAX; clamp to texture bounds.
                let px = ((pos.x - rect.left()) / rect.width()  * self.map_w as f32) as i32;
                let pz = ((pos.y - rect.top())  / rect.height() * self.map_h as f32) as i32;
                let wx = self.center_x as i32 + (px - self.map_w as i32 / 2) * self.scale as i32;
                let wz = self.center_z as i32 + (pz - self.map_h as i32 / 2) * self.scale as i32;

                let geo = self.pipeline.geography.sample(wx, wz);
                let climate = self.pipeline.climate.sample(wx, wz);
                let geology = self.pipeline.geology.sample(wx, wz);
                let biome_id = self.pipeline.determine_biome(&geo, &climate, &geology);
                let biome_name = self.pipeline.biome_dict
                    .get(biome_id)
                    .map(|b| b.name)
                    .unwrap_or("Unknown");
                let nature = NatureLayer::assign_nature_biome(climate.climate_type, &geo, climate.humidity);

                let tip_text = format!(
                    "({}, {})\nBiome: {}\nNature: {:?}{}\nClimate: {}\nGeology: {:?} / {:?}\nHeight: {}\nCont: {:.2}  Eros: {:.2}\nTemp: {:.2}  Humid: {:.2}\nRiver strength: {:.3}",
                    wx, wz,
                    biome_name,
                    nature,
                    if geo.is_river { "  [RIVER]" } else { "" },
                    climate.climate_type.name(),
                    geology.macro_region, geology.micro_region,
                    geo.surface_height,
                    geo.continentalness, geo.erosion,
                    climate.temperature, climate.humidity,
                    geo.river_strength,
                );

                egui::show_tooltip_at_pointer(ctx, ui.layer_id(), egui::Id::new("worldgen_hover_tip"), |ui: &mut Ui| {
                    ui.label(tip_text);
                });
            }
        }
    }

    fn draw_params_panel(&mut self, ui: &mut Ui) {
        ui.heading("World Gen Parameters");
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            // ---- Geography ----
            let mut geo_changed = false;
            egui::CollapsingHeader::new("Geography")
                .default_open(true)
                .show(ui, |ui| {
                    let g = &mut self.params.geography;

                    geo_changed |= drag_i32(ui, "Sea Level",        &mut g.sea_level,         1,  30,  200);
                    geo_changed |= drag_i32(ui, "Terrain Base",     &mut g.terrain_base,       1,  10,  120);
                    geo_changed |= drag_i32(ui, "Terrain Amplitude",&mut g.terrain_amplitude,  1,   5,  120);

                    ui.add_space(4.0);
                    geo_changed |= drag_f64(ui, "Continent Scale",  &mut g.continent_scale,  10.0,  50.0, 20000.0);
                    geo_changed |= drag_f64(ui, "Erosion Scale",    &mut g.erosion_scale,     5.0,  50.0, 10000.0);
                    geo_changed |= drag_f64(ui, "Peaks Scale",      &mut g.peaks_scale,      10.0,  50.0, 20000.0);
                    geo_changed |= drag_f64(ui, "Island Scale",     &mut g.island_scale,      1.0,  10.0,  2000.0);

                    ui.add_space(4.0);
                    geo_changed |= drag_f64(ui, "Island Threshold", &mut g.island_threshold, 0.01, 0.05,   0.95);
                    geo_changed |= drag_f64(ui, "Island Boost",     &mut g.island_boost,     0.01, 0.0,    1.5);

                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("Continentalness thresholds").color(egui::Color32::GRAY).size(11.0));
                    geo_changed |= drag_f64(ui, "Deep Ocean Max", &mut g.deep_ocean_max, 0.01, 0.01, 0.50);
                    geo_changed |= drag_f64(ui, "Ocean Max",      &mut g.ocean_max,      0.01, 0.10, 0.60);
                    geo_changed |= drag_f64(ui, "Lowland Max",    &mut g.lowland_max,    0.01, 0.30, 0.85);
                    geo_changed |= drag_f64(ui, "Midland Max",    &mut g.midland_max,    0.01, 0.50, 0.99);

                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("Rivers").color(egui::Color32::GRAY).size(11.0));
                    geo_changed |= drag_f64(ui, "River Scale",       &mut g.river_scale,       1.0,  50.0, 5000.0);
                    geo_changed |= drag_f64(ui, "River Threshold",   &mut g.river_threshold,   0.001, 0.01, 0.30);
                    geo_changed |= drag_f64(ui, "River Carve Depth", &mut g.river_carve_depth, 0.5,   1.0,  40.0);
                });

            if geo_changed {
                self.params_dirty = true;
            }

            ui.add_space(6.0);

            // ---- Climate ----
            let mut cli_changed = false;
            egui::CollapsingHeader::new("Climate")
                .default_open(true)
                .show(ui, |ui| {
                    let c = &mut self.params.climate;
                    cli_changed |= drag_f64(ui, "Band Scale",          &mut c.band_scale,            10.0,   200.0, 50000.0);
                    cli_changed |= drag_f64(ui, "Humidity Noise Scale",&mut c.humidity_noise_scale,   1.0,    10.0,  5000.0);
                    cli_changed |= drag_f64(ui, "Boundary Warp",       &mut c.boundary_warp_amount,  10.0,    10.0, 10000.0);
                    cli_changed |= drag_f64_sci(ui,"Boundary Noise Sc",&mut c.boundary_noise_scale, 0.000001, 0.000001, 0.1);
                });

            if cli_changed {
                self.params_dirty = true;
            }

            ui.add_space(6.0);

            // ---- Geology ----
            let mut geo2_changed = false;
            egui::CollapsingHeader::new("Geology")
                .default_open(true)
                .show(ui, |ui| {
                    let g = &mut self.params.geology;
                    geo2_changed |= drag_i32(ui, "Macro Cell Size", &mut g.macro_cell_size,    16,  64, 16384);
                    geo2_changed |= drag_i32(ui, "Micro Cell Size", &mut g.micro_cell_size,     4,  16,  4096);
                    let mut denom = g.mantle_keep_denominator as i32;
                    if drag_i32(ui, "Mantle Rarity",  &mut denom, 1, 1, 50) {
                        g.mantle_keep_denominator = denom.max(1) as u64;
                        geo2_changed = true;
                    }
                });

            if geo2_changed {
                self.params_dirty = true;
            }

            ui.add_space(6.0);

            // ---- Nature / Decorator ----
            let mut nat_changed = false;
            egui::CollapsingHeader::new("Nature / Decorator")
                .default_open(true)
                .show(ui, |ui| {
                    let n = &mut self.params.nature;
                    nat_changed |= drag_f64(ui, "Tree Density Mult", &mut n.tree_density_multiplier, 0.05, 0.0, 5.0);
                    ui.label(
                        egui::RichText::new("1.0 = default density per biome. 0 = no trees.")
                            .size(10.0)
                            .color(egui::Color32::GRAY),
                    );
                });

            if nat_changed {
                self.params_dirty = true;
            }

            ui.add_space(10.0);
            ui.separator();

            // ---- Test mode toggle ----
            if ui.checkbox(&mut self.params.test_mode, "Test Mode (small scales)").changed() {
                self.params_dirty = true;
            }
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("In test mode, scales are auto-reduced ~10× so biomes fit in a few chunks.")
                    .size(10.0)
                    .color(egui::Color32::GRAY),
            );

            ui.add_space(10.0);
            ui.separator();

            // ---- Save button ----
            let save_btn = egui::Button::new(
                egui::RichText::new("Save to JSON").size(16.0),
            )
                .min_size(egui::vec2(ui.available_width(), 36.0))
                .fill(egui::Color32::from_rgb(40, 130, 40));

            if ui.add(save_btn).clicked() {
                self.save_status = Some(self.save_to_json());
            }

            if let Some(status) = &self.save_status {
                let color = if status.starts_with("Saved") {
                    egui::Color32::from_rgb(100, 220, 100)
                } else {
                    egui::Color32::from_rgb(220, 80, 80)
                };
                ui.colored_label(color, status.as_str());
            }
        });
    }

    fn save_to_json(&self) -> String {
        let json = match serde_json::to_string_pretty(&self.params) {
            Ok(s) => s,
            Err(e) => return format!("Serialize error: {}", e),
        };
        match std::fs::write("assets/world_gen_config.json", &json) {
            Ok(_) => format!("Saved to assets/world_gen_config.json"),
            Err(e) => format!("Write error: {}", e),
        }
    }

    fn rebuild_map(&mut self) {
        use rayon::prelude::*;

        let half_w = (self.map_w / 2) as f32;
        let half_h = (self.map_h / 2) as f32;
        let cx = self.center_x;
        let cz = self.center_z;
        let sc = self.scale;
        let w  = self.map_w;
        let mode = self.mode;
        // Borrow pipeline separately so pixel_buf can be mutably borrowed.
        let pipeline = &self.pipeline;

        self.pixel_buf
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(pz, row)| {
                for (px, pixel) in row.iter_mut().enumerate() {
                    let wx = (cx + (px as f32 - half_w) * sc) as i32;
                    let wz = (cz + (pz as f32 - half_h) * sc) as i32;
                    *pixel = sample_pixel(pipeline, mode, wx, wz);
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Map sampling (free function so rayon closure can borrow pipeline + pixel_buf)
// ---------------------------------------------------------------------------

fn sample_pixel(pipeline: &BiomePipeline, mode: VisMode, wx: i32, wz: i32) -> Color32 {
    match mode {
        VisMode::Continentalness => {
            let geo = pipeline.geography.sample(wx, wz);
            val_to_gradient(geo.continentalness, &GRADIENT_BLUE_GREEN)
        }
        VisMode::Erosion => {
            let geo = pipeline.geography.sample(wx, wz);
            val_to_gradient(geo.erosion, &GRADIENT_GRAY)
        }
        VisMode::SurfaceHeight => {
            let geo = pipeline.geography.sample(wx, wz);
            let t = ((geo.surface_height as f64 - 30.0) / 70.0).clamp(0.0, 1.0);
            if geo.is_ocean {
                let depth_t = (geo.ocean_depth as f64 / 30.0).clamp(0.0, 1.0);
                Color32::from_rgb(
                    (20.0 - depth_t * 20.0) as u8,
                    (60.0 - depth_t * 40.0) as u8,
                    (180.0 + depth_t * 60.0).min(255.0) as u8,
                )
            } else {
                val_to_gradient(t, &GRADIENT_TERRAIN)
            }
        }
        VisMode::Climate => {
            let climate = pipeline.climate.sample(wx, wz);
            temperature_color(climate.temperature)
        }
        VisMode::Humidity => {
            let climate = pipeline.climate.sample(wx, wz);
            val_to_gradient(climate.humidity, &GRADIENT_DRY_WET)
        }
        VisMode::Geology => {
            let geology = pipeline.geology.sample(wx, wz);
            geology_color(geology.macro_region, geology.micro_region)
        }
        VisMode::BiomeCombined => {
            let geo = pipeline.geography.sample(wx, wz);
            let climate = pipeline.climate.sample(wx, wz);
            let geology = pipeline.geology.sample(wx, wz);
            let biome_id = pipeline.determine_biome(&geo, &climate, &geology);
            biome_color(biome_id, geo.is_ocean, geo.surface_height)
        }
        VisMode::NatureLayer => {
            let geo = pipeline.geography.sample(wx, wz);
            let climate = pipeline.climate.sample(wx, wz);
            if geo.is_ocean {
                return Color32::from_rgb(20, 60, 180);
            }
            let nature = NatureLayer::assign_nature_biome(climate.climate_type, &geo, climate.humidity);
            nature_color(nature, geo.is_river)
        }
        VisMode::Rivers => {
            let geo = pipeline.geography.sample(wx, wz);
            if geo.is_ocean {
                return Color32::from_rgb(20, 60, 180);
            }
            if geo.is_river {
                // River strength drives brightness
                let brightness = (geo.river_strength * 220.0 + 35.0).min(255.0) as u8;
                Color32::from_rgb(20, 80, brightness)
            } else {
                // Land — height-shaded
                let t = ((geo.surface_height as f64 - 30.0) / 70.0).clamp(0.0, 1.0);
                val_to_gradient(t, &GRADIENT_TERRAIN)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Small DragValue helpers
// ---------------------------------------------------------------------------

fn drag_i32(ui: &mut Ui, label: &str, val: &mut i32, speed: i32, min: i32, max: i32) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(egui::DragValue::new(val).speed(speed).range(min..=max)).changed()
        }).inner
    }).inner
}

fn drag_f64(ui: &mut Ui, label: &str, val: &mut f64, speed: f64, min: f64, max: f64) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(egui::DragValue::new(val).speed(speed).range(min..=max)).changed()
        }).inner
    }).inner
}

fn drag_f64_sci(ui: &mut Ui, label: &str, val: &mut f64, speed: f64, min: f64, max: f64) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(egui::DragValue::new(val).speed(speed).range(min..=max)
                .custom_formatter(|v, _| format!("{:.6}", v)))
                .changed()
        }).inner
    }).inner
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

type Gradient = [(f64, [u8; 3])];

const GRADIENT_BLUE_GREEN: [(f64, [u8; 3]); 5] = [
    (0.0, [0, 20, 120]),
    (0.3, [30, 80, 180]),
    (0.5, [60, 160, 60]),
    (0.8, [160, 140, 60]),
    (1.0, [200, 200, 200]),
];

const GRADIENT_GRAY: [(f64, [u8; 3]); 3] = [
    (0.0, [40, 40, 40]),
    (0.5, [128, 128, 128]),
    (1.0, [240, 240, 240]),
];

const GRADIENT_TERRAIN: [(f64, [u8; 3]); 5] = [
    (0.0, [40, 100, 30]),
    (0.25, [80, 160, 50]),
    (0.5, [140, 160, 60]),
    (0.75, [160, 120, 80]),
    (1.0, [220, 220, 220]),
];

const GRADIENT_DRY_WET: [(f64, [u8; 3]); 4] = [
    (0.0, [200, 160, 60]),
    (0.3, [160, 180, 80]),
    (0.6, [60, 140, 60]),
    (1.0, [20, 80, 160]),
];

fn nature_color(
    nature: crate::core::world_gen::nature::NatureBiome,
    is_river: bool,
) -> Color32 {
    use crate::core::world_gen::nature::NatureBiome;
    if is_river { return Color32::from_rgb(40, 100, 220); }
    match nature {
        NatureBiome::Plains        => Color32::from_rgb(180, 220, 100),
        NatureBiome::Forest        => Color32::from_rgb(60, 160, 60),
        NatureBiome::DenseForest   => Color32::from_rgb(20, 100, 20),
        NatureBiome::SparseForest  => Color32::from_rgb(120, 200, 80),
        NatureBiome::Swamp         => Color32::from_rgb(80, 100, 60),
        NatureBiome::Taiga         => Color32::from_rgb(80, 130, 110),
        NatureBiome::Savanna       => Color32::from_rgb(200, 180, 60),
        NatureBiome::JungleDense   => Color32::from_rgb(20, 80, 20),
        NatureBiome::Tundra        => Color32::from_rgb(200, 220, 240),
        NatureBiome::CherryBlossom => Color32::from_rgb(240, 160, 200),
        NatureBiome::Mangrove      => Color32::from_rgb(60, 130, 80),
        NatureBiome::MushroomPlains => Color32::from_rgb(180, 60, 180),
        NatureBiome::BambooGrove   => Color32::from_rgb(140, 200, 60),
        NatureBiome::DeadForest    => Color32::from_rgb(100, 80, 60),
    }
}

fn val_to_gradient(val: f64, gradient: &Gradient) -> Color32 {
    let v = val.clamp(0.0, 1.0);

    if gradient.is_empty() { return Color32::BLACK; }
    if v <= gradient[0].0 { return Color32::from_rgb(gradient[0].1[0], gradient[0].1[1], gradient[0].1[2]); }

    for i in 1..gradient.len() {
        if v <= gradient[i].0 {
            let t = (v - gradient[i-1].0) / (gradient[i].0 - gradient[i-1].0);
            let a = gradient[i-1].1;
            let b = gradient[i].1;
            return Color32::from_rgb(
                (a[0] as f64 + (b[0] as f64 - a[0] as f64) * t) as u8,
                (a[1] as f64 + (b[1] as f64 - a[1] as f64) * t) as u8,
                (a[2] as f64 + (b[2] as f64 - a[2] as f64) * t) as u8,
            );
        }
    }

    let last = gradient.last().unwrap().1;
    Color32::from_rgb(last[0], last[1], last[2])
}

fn temperature_color(temp: f64) -> Color32 {
    let t = ((temp + 1.0) * 0.5).clamp(0.0, 1.0);
    let r = (t * 255.0) as u8;
    let g = ((1.0 - (t - 0.5).abs() * 2.0) * 200.0) as u8;
    let b = ((1.0 - t) * 255.0) as u8;
    Color32::from_rgb(r, g, b)
}

fn geology_color(
    _macro_r: crate::core::world_gen::geology::MacroRegion,
    micro_r: crate::core::world_gen::geology::MicroRegion,
) -> Color32 {
    use crate::core::world_gen::geology::{MicroRegion};
    match micro_r {
        MicroRegion::Gneiss   => Color32::from_rgb(180, 100, 180),
        MicroRegion::Schist   => Color32::from_rgb(160, 80, 160),
        MicroRegion::Marble   => Color32::from_rgb(220, 200, 220),
        MicroRegion::Sandstone  => Color32::from_rgb(200, 180, 100),
        MicroRegion::Limestone  => Color32::from_rgb(180, 180, 160),
        MicroRegion::Shale      => Color32::from_rgb(120, 100, 80),
        MicroRegion::Basalt   => Color32::from_rgb(60, 60, 60),
        MicroRegion::Granite  => Color32::from_rgb(180, 140, 140),
        MicroRegion::Andesite => Color32::from_rgb(140, 140, 140),
        MicroRegion::Lherzolite => Color32::from_rgb(200, 80, 40),
        MicroRegion::Kimberlite => Color32::from_rgb(160, 40, 80),
        MicroRegion::Ophiolite  => Color32::from_rgb(120, 100, 40),
    }
}

fn biome_color(biome_id: u32, is_ocean: bool, surface_height: i32) -> Color32 {
    if is_ocean {
        let d = (64 - surface_height).max(0).min(30);
        return Color32::from_rgb(
            (20 - d / 2) as u8,
            (60 - d) as u8,
            (180 + d * 2).min(240) as u8,
        );
    }
    match biome_id {
        100..=104 => Color32::from_rgb(30, 60, 200),
        200 => Color32::from_rgb(230, 220, 160),
        201 => Color32::from_rgb(220, 230, 240),
        202 => Color32::from_rgb(220, 200, 120),
        300 => Color32::from_rgb(100, 180, 60),
        301 => Color32::from_rgb(200, 220, 230),
        302 => Color32::from_rgb(160, 200, 60),
        303 => Color32::from_rgb(180, 180, 60),
        400 => Color32::from_rgb(40, 120, 30),
        401 => Color32::from_rgb(140, 180, 180),
        402 => Color32::from_rgb(20, 100, 20),
        403 => Color32::from_rgb(60, 100, 80),
        500 => Color32::from_rgb(220, 200, 120),
        501 => Color32::from_rgb(180, 100, 60),
        600 => Color32::from_rgb(60, 100, 60),
        601 => Color32::from_rgb(80, 120, 40),
        700 => Color32::from_rgb(140, 140, 140),
        701 => Color32::from_rgb(220, 220, 240),
        702 => Color32::from_rgb(80, 40, 40),
        703 => Color32::from_rgb(180, 120, 80),
        800 => Color32::from_rgb(180, 200, 200),
        801 => Color32::from_rgb(200, 220, 240),
        900 | 901 => Color32::from_rgb(20, 80, 10),
        1000 => Color32::from_rgb(60, 100, 200),
        1001 => Color32::from_rgb(160, 200, 240),
        _ => Color32::from_rgb(128, 128, 128),
    }
}
