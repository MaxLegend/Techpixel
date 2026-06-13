// =============================================================================
// QubePixel — Block Editor
//
// Full-featured in-game block definition editor.
// Accessible from the main menu → "Block Editor".
//
// Features:
//   • Searchable block list with registry order display
//   • Dedicated wgpu 3D preview cube rendered by BlockPreviewRenderer
//   • Real-time parameter reflection (colors, textures, emission)
//   • Orbit camera via mouse drag on the preview area
//   • Light-emission overlay in the info bar
//   • Tabbed inspector: Basic / Material / Emission / Faces / Fluid / Lights
//   • Save single block JSON + update block_registry.json
//   • Create new blocks from scratch
// =============================================================================

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::path::PathBuf;

use egui::{
    Color32, Pos2, Rect, RichText, Ui, Vec2,
};
use glam::{Mat4, Vec3, Vec4};

use crate::core::gameobjects::block::{
    BlockDefinition, BlockLightSource, EmissionProperties, FaceColors, FaceOverride,
    FluidProperties, GlassProperties, LightKind, MaterialProperties, PlacementMode,
    VolumetricProperties,
};
use crate::debug_log;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BLOCKS_DIR: &str = "assets/blocks";
const REGISTRY_FILE: &str = "assets/blocks/block_registry.json";
const TEXTURES_DIR: &str = "assets/textures/simpleblocks";

// ---------------------------------------------------------------------------
// Editor tab
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditorTab {
    Basic,
    Material,
    Emission,
    Faces,
    Fluid,
    Lights,
}

impl EditorTab {
    const ALL: [EditorTab; 6] = [
        EditorTab::Basic,
        EditorTab::Material,
        EditorTab::Emission,
        EditorTab::Faces,
        EditorTab::Fluid,
        EditorTab::Lights,
    ];

    fn label(self) -> &'static str {
        match self {
            EditorTab::Basic     => "Basic",
            EditorTab::Material  => "Material",
            EditorTab::Emission  => "Emission",
            EditorTab::Faces     => "Faces",
            EditorTab::Fluid     => "Fluid",
            EditorTab::Lights    => "Lights",
        }
    }
}

// ---------------------------------------------------------------------------
// New-block dialog state
// ---------------------------------------------------------------------------

struct NewBlockDialog {
    open: bool,
    id:   String,
    name: String,
    err:  String,
}

impl NewBlockDialog {
    fn new() -> Self {
        Self { open: false, id: String::new(), name: String::new(), err: String::new() }
    }

    fn open(&mut self) {
        self.id.clear();
        self.name.clear();
        self.err.clear();
        self.open = true;
    }
}

// ---------------------------------------------------------------------------
// BlockEditor
// ---------------------------------------------------------------------------

pub struct BlockEditor {
    pub visible: bool,

    // Registry state
    registry_order: Vec<String>,
    block_defs:     HashMap<String, BlockDefinition>,

    // Edit buffer
    selected_key: Option<String>,
    pub edit_def: BlockDefinition,
    edit_key:     String,
    is_new:       bool,
    dirty:        bool,

    // UI
    search:     String,
    active_tab: EditorTab,
    status_msg: String,
    status_ok:  bool,

    // Dialogs
    new_dlg: NewBlockDialog,

    // Preview state — written during build_ui, read during render()
    /// Screen rect (in logical pixels) where the wgpu cube is rendered.
    pub preview_rect:       Option<egui::Rect>,
    /// Orbit camera yaw (Y-axis rotation), updated by mouse drag.
    pub preview_yaw:        f32,
    /// Orbit camera pitch (X-axis rotation), updated by mouse drag.
    pub preview_pitch:      f32,
    /// Current pixels-per-point from egui context.
    pub pixels_per_point:   f32,
}

// ---------------------------------------------------------------------------
// Construction & loading
// ---------------------------------------------------------------------------

impl BlockEditor {
    pub fn new() -> Self {
        debug_log!("BlockEditor", "new", "Creating BlockEditor");
        let mut editor = Self {
            visible:          false,
            registry_order:   Vec::new(),
            block_defs:       HashMap::new(),
            selected_key:     None,
            edit_def:         Self::default_block_def("new_block"),
            edit_key:         String::new(),
            is_new:           false,
            dirty:            false,
            search:           String::new(),
            active_tab:       EditorTab::Basic,
            status_msg:       String::new(),
            status_ok:        true,
            new_dlg:          NewBlockDialog::new(),
            preview_rect:     None,
            preview_yaw:      0.6,
            preview_pitch:    0.4,
            pixels_per_point: 1.0,
        };
        editor.reload_registry();
        editor
    }

    fn default_block_def(id: &str) -> BlockDefinition {
        BlockDefinition {
            id:           id.to_string(),
            display_name: id.replace('_', " "),
            description:  String::new(),
            color:        [0.5, 0.5, 0.5],
            texture:      None,
            solid:        true,
            transparent:  false,
            glass:        GlassProperties::default(),
            is_water:     false,
            fluid:        None,
            emit_light:   0,
            material:     MaterialProperties::default(),
            emission:     EmissionProperties::default(),
            volumetric:   VolumetricProperties::default(),
            faces:        FaceColors::default(),
            biome_tint:   false,
            block_shape:      crate::core::gameobjects::block::BlockShape::Cube,
            placement_mode:   PlacementMode::Fixed,
            default_rotation: 0,
            model:            None,
            model_textures:   HashMap::new(),
            model_shadow_cubes: None,
            inventory_tab:    None,
            light_sources:    Vec::new(),
            tex_has_alpha:    false,
            tex_avg_opacity:  1.0,
        }
    }

    fn reload_registry(&mut self) {
        self.registry_order.clear();
        self.block_defs.clear();

        let reg_path = PathBuf::from(REGISTRY_FILE);
        if !reg_path.exists() {
            debug_log!("BlockEditor", "reload_registry", "Registry file not found");
            return;
        }

        let content = match std::fs::read_to_string(&reg_path) {
            Ok(s)  => s,
            Err(e) => { debug_log!("BlockEditor", "reload_registry", "Read error: {}", e); return; }
        };

        let val: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v)  => v,
            Err(e) => { debug_log!("BlockEditor", "reload_registry", "Parse error: {}", e); return; }
        };

        if let Some(arr) = val["blocks"].as_array() {
            for entry in arr {
                if let Some(key) = entry.as_str() {
                    self.registry_order.push(key.to_string());
                    self.load_block_def(key);
                }
            }
        }

        debug_log!("BlockEditor", "reload_registry",
            "Loaded {} block definitions", self.block_defs.len());
    }

    fn load_block_def(&mut self, key: &str) {
        // Try subfolder path: assets/blocks/<key>.json
        let base = PathBuf::from(BLOCKS_DIR);
        let path = base.join(format!("{}.json", key));

        let content = match std::fs::read_to_string(&path) {
            Ok(s)  => s,
            Err(_) => {
                debug_log!("BlockEditor", "load_block_def",
                    "Could not read {:?} for key '{}'", path, key);
                return;
            }
        };

        match serde_json::from_str::<BlockDefinition>(&content) {
            Ok(def) => { self.block_defs.insert(key.to_string(), def); }
            Err(e)  => {
                debug_log!("BlockEditor", "load_block_def",
                    "Parse error for '{}': {}", key, e);
            }
        }
    }

    fn select_block(&mut self, key: &str) {
        if let Some(def) = self.block_defs.get(key) {
            self.edit_def    = def.clone();
            self.edit_key    = key.to_string();
            self.selected_key = Some(key.to_string());
            self.is_new      = false;
            self.dirty       = false;
            self.status_msg.clear();
            debug_log!("BlockEditor", "select_block", "Selected '{}'", key);
        }
    }
}

// ---------------------------------------------------------------------------
// Main egui draw entry
// ---------------------------------------------------------------------------

impl BlockEditor {
    pub fn draw_egui(&mut self, ctx: &egui::Context) -> bool {

        let mut should_pop = false;

        // Top toolbar
        egui::TopBottomPanel::top("block_editor_toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button(RichText::new("← Back").size(16.0)).clicked() {
                    should_pop = true;
                }
                ui.separator();
                ui.label(RichText::new("Block Editor").size(18.0).strong());
                ui.separator();

                ui.label("Search:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.search)
                        .desired_width(160.0)
                        .hint_text("filter blocks…"),
                );

                ui.separator();

                if ui.button(RichText::new("+ New Block").size(14.0)).clicked() {
                    self.new_dlg.open();
                }

                if self.selected_key.is_some() || self.is_new {
                    ui.separator();
                    let save_label = if self.dirty {
                        RichText::new("💾 Save *").size(14.0).color(Color32::YELLOW)
                    } else {
                        RichText::new("💾 Save").size(14.0)
                    };
                    if ui.button(save_label).clicked() {
                        self.save_current();
                    }

                    if !self.is_new {
                        if ui.button(RichText::new("Duplicate").size(13.0)).clicked() {
                            self.duplicate_current();
                        }
                    }
                }

                // Status message
                if !self.status_msg.is_empty() {
                    ui.separator();
                    let color = if self.status_ok { Color32::GREEN } else { Color32::LIGHT_RED };
                    ui.label(RichText::new(&self.status_msg.clone()).color(color));
                }
            });
        });

        // Left panel: block list
        egui::SidePanel::left("block_editor_list")
            .resizable(true)
            .default_width(200.0)
            .show(ctx, |ui| {
                self.draw_block_list(ui);
            });

        // Right panel: inspector
        egui::SidePanel::right("block_editor_inspector")
            .resizable(true)
            .default_width(380.0)
            .show(ctx, |ui| {
                self.draw_inspector(ui);
            });

        // Center panel: transparent so the wgpu cube render shows through
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                self.draw_preview(ui);
            });

        // New-block dialog (modal window)
        self.draw_new_block_dialog(ctx);

        should_pop
    }

    // -----------------------------------------------------------------------
    // Block list
    // -----------------------------------------------------------------------

    fn draw_block_list(&mut self, ui: &mut Ui) {
        ui.heading("Blocks");
        ui.add_space(4.0);
        ui.label(
            RichText::new(format!("{} registered", self.registry_order.len()))
                .small()
                .color(Color32::GRAY),
        );
        ui.separator();

        let search_lower = self.search.to_lowercase();
        let current_key  = self.selected_key.clone();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let keys: Vec<String> = self.registry_order.clone();
            for (idx, key) in keys.iter().enumerate() {
                // Filter
                let def_name = self.block_defs.get(key)
                    .map(|d| d.display_name.to_lowercase())
                    .unwrap_or_default();
                if !search_lower.is_empty()
                    && !key.to_lowercase().contains(&search_lower)
                    && !def_name.contains(&search_lower)
                {
                    continue;
                }

                let is_selected = current_key.as_deref() == Some(key.as_str());
                let display = self.block_defs.get(key)
                    .map(|d| d.display_name.clone())
                    .unwrap_or_else(|| key.clone());

                // Color chip using block's base color
                let chip_color = self.block_defs.get(key)
                    .map(|d| color_f32_to_egui(d.color))
                    .unwrap_or(Color32::GRAY);

                ui.horizontal(|ui| {
                    // Numeric ID label
                    ui.label(
                        RichText::new(format!("{:3}.", idx + 1))
                            .small()
                            .color(Color32::DARK_GRAY),
                    );
                    // Color square
                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, chip_color);

                    // Selectable row
                    let resp = ui.selectable_label(
                        is_selected,
                        RichText::new(&display).size(13.0),
                    );
                    if resp.clicked() {
                        self.select_block(key);
                    }
                    if resp.hovered() {
                        resp.on_hover_text(key.as_str());
                    }
                });
            }
        });
    }

    // -----------------------------------------------------------------------
    // Preview — allocates the rect for BlockPreviewRenderer and handles
    // mouse-drag orbit. The actual 3D cube is rendered by BlockPreviewRenderer
    // in BlockEditorScreen::render(); this method only draws 2D overlays.
    // -----------------------------------------------------------------------

    fn draw_preview(&mut self, ui: &mut Ui) {
        let avail = ui.available_size();
        let (resp, painter) = ui.allocate_painter(avail, egui::Sense::drag());

        // Communicate rect to render()
        self.preview_rect     = Some(resp.rect);
        self.pixels_per_point = ui.ctx().pixels_per_point();

        // Mouse-drag → orbit camera
        if resp.dragged() {
            let d = resp.drag_delta();
            self.preview_yaw   += d.x * 0.007;
            self.preview_pitch  = (self.preview_pitch + d.y * 0.007).clamp(-1.5, 1.5);
        }

        // Subtle border around the preview area
        painter.rect_stroke(
            resp.rect,
            2.0,
            egui::Stroke::new(1.0, Color32::from_gray(60)),
            egui::StrokeKind::Inside,
        );

        // Rotate hint
        painter.text(
            resp.rect.right_bottom() + Vec2::new(-8.0, -10.0),
            egui::Align2::RIGHT_BOTTOM,
            "Drag to rotate",
            egui::FontId::proportional(10.0),
            Color32::from_rgba_unmultiplied(200, 200, 200, 80),
        );

        if self.selected_key.is_some() || self.is_new {
            self.draw_preview_overlay(&painter, resp.rect);
        } else {
            painter.text(
                resp.rect.center(),
                egui::Align2::CENTER_CENTER,
                "Select a block from the list,\nor create a new one with + New Block",
                egui::FontId::proportional(14.0),
                Color32::from_gray(100),
            );
        }
    }

    // Draws the 2D info bar + emission halo overlay on top of the wgpu cube.
    fn draw_preview_overlay(&self, painter: &egui::Painter, rect: Rect) {
        let def = &self.edit_def;

        // Semi-transparent info bar at the bottom of the preview
        let bar_h   = 62.0;
        let bar_rect = Rect::from_min_max(
            Pos2::new(rect.min.x, rect.max.y - bar_h),
            rect.max,
        );
        painter.rect_filled(
            bar_rect,
            0.0,
            Color32::from_rgba_unmultiplied(10, 10, 20, 185),
        );

        // Block display name
        painter.text(
            Pos2::new(rect.center().x, rect.max.y - bar_h + 8.0),
            egui::Align2::CENTER_TOP,
            &def.display_name,
            egui::FontId::proportional(16.0),
            Color32::WHITE,
        );

        // ID + numeric index
        let num_id = self.registry_order.iter().position(|k| k == &self.edit_key)
            .map(|i| (i + 1).to_string())
            .unwrap_or_else(|| "NEW".into());
        painter.text(
            Pos2::new(rect.center().x, rect.max.y - bar_h + 28.0),
            egui::Align2::CENTER_TOP,
            format!("{}  |  #{}", self.edit_key, num_id),
            egui::FontId::proportional(10.0),
            Color32::GRAY,
        );

        // Property tags row
        let tag_y     = rect.max.y - 10.0;
        let tag_w     = 68.0;
        let tag_h     = 14.0;
        let tags_data: &[(&str, Color32, bool)] = &[
            ("Solid",       Color32::from_rgb(80,  150,  80), def.solid),
            ("Fluid",       Color32::from_rgb(60,  130, 200), def.is_fluid()),
            ("Emits Light", Color32::from_rgb(220, 200,  50), def.emission.emit_light),
            ("Biome Tint",  Color32::from_rgb(80,  180,  80), def.biome_tint),
        ];
        let active_tags: Vec<_> = tags_data.iter().filter(|t| t.2).collect();
        let total_w = active_tags.len() as f32 * (tag_w + 4.0) - 4.0;
        let mut tag_x = rect.center().x - total_w * 0.5;
        for (label, col, _) in &active_tags {
            let tr = Rect::from_min_size(Pos2::new(tag_x, tag_y - tag_h), Vec2::new(tag_w, tag_h));
            painter.rect_filled(tr, 3.0, *col);
            painter.text(
                tr.center(),
                egui::Align2::CENTER_CENTER,
                *label,
                egui::FontId::proportional(9.0),
                Color32::WHITE,
            );
            tag_x += tag_w + 4.0;
        }

        // Emission glow halo painted over the cube center
        if def.emission.emit_light {
            let center   = rect.center() - Vec2::new(0.0, bar_h * 0.3);
            let s        = rect.width().min(rect.height()) * 0.22;
            let col      = color_f32_to_egui(def.emission.light_color);
            let intensity = def.emission.light_intensity.clamp(0.05, 2.5);
            for i in 1..=6u8 {
                let radius = s * 0.9 * i as f32 * intensity;
                let alpha  = ((50.0 - i as f32 * 7.0) * intensity).clamp(0.0, 70.0) as u8;
                painter.circle_filled(
                    center,
                    radius,
                    Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), alpha),
                );
            }
        }

        // Light source arrows
        if !def.light_sources.is_empty() {
            let mvp = preview_mvp(
                self.preview_yaw,
                self.preview_pitch,
                rect.width() / rect.height().max(1.0),
            );
            for ls in &def.light_sources {
                let col = color_f32_to_egui(ls.color);
                let col_bright = Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 220);
                let col_dim    = Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 100);

                let Some(origin_s) = project_to_rect(&mvp, ls.position, rect) else { continue };

                match ls.kind {
                    LightKind::Point => {
                        // Filled circle at position + 4 small radiating tick marks
                        painter.circle_filled(origin_s, 5.0, col_bright);
                        painter.circle_stroke(origin_s, 8.0,
                            egui::Stroke::new(1.0, col_dim));
                        for angle in [0.0, 90.0, 180.0, 270.0_f32] {
                            let rad = angle.to_radians();
                            let tip = origin_s + Vec2::new(rad.cos(), rad.sin()) * 12.0;
                            painter.line_segment([origin_s, tip],
                                egui::Stroke::new(1.5, col_dim));
                        }
                    }
                    LightKind::Directional => {
                        // Arrow from origin along the direction (projected)
                        let tip_pos = [
                            ls.position[0] + ls.direction[0] * 0.4,
                            ls.position[1] + ls.direction[1] * 0.4,
                            ls.position[2] + ls.direction[2] * 0.4,
                        ];
                        if let Some(tip_s) = project_to_rect(&mvp, tip_pos, rect) {
                            painter.arrow(origin_s, tip_s - origin_s,
                                egui::Stroke::new(2.0, col_bright));
                            // Focus cone outline (2 lines at ±focus angle projected)
                            let half_angle = ls.focus.to_radians() * 0.5;
                            let shaft = tip_s - origin_s;
                            let perp  = Vec2::new(-shaft.y, shaft.x).normalized();
                            let dist  = shaft.length() * half_angle.tan();
                            painter.line_segment([origin_s, tip_s + perp * dist],
                                egui::Stroke::new(1.0, col_dim));
                            painter.line_segment([origin_s, tip_s - perp * dist],
                                egui::Stroke::new(1.0, col_dim));
                        }
                        painter.circle_filled(origin_s, 4.0, col_bright);
                    }
                }

                // Label if set
                if !ls.label.is_empty() {
                    painter.text(
                        origin_s + Vec2::new(8.0, -8.0),
                        egui::Align2::LEFT_BOTTOM,
                        &ls.label,
                        egui::FontId::proportional(9.0),
                        col_bright,
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Inspector
    // -----------------------------------------------------------------------

    fn draw_inspector(&mut self, ui: &mut Ui) {
        if self.selected_key.is_none() && !self.is_new {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);
                ui.label(RichText::new("No block selected").color(Color32::GRAY));
            });
            return;
        }

        // Tab bar
        ui.horizontal(|ui| {
            for tab in EditorTab::ALL {
                let active = self.active_tab == tab;
                let label = if active {
                    RichText::new(tab.label()).strong().color(Color32::WHITE)
                } else {
                    RichText::new(tab.label()).color(Color32::GRAY)
                };
                if ui.selectable_label(active, label).clicked() {
                    self.active_tab = tab;
                }
            }
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let tab = self.active_tab;
            match tab {
                EditorTab::Basic     => self.draw_tab_basic(ui),
                EditorTab::Material  => self.draw_tab_material(ui),
                EditorTab::Emission  => self.draw_tab_emission(ui),
                EditorTab::Faces     => self.draw_tab_faces(ui),
                EditorTab::Fluid     => self.draw_tab_fluid(ui),
                EditorTab::Lights    => self.draw_tab_lights(ui),
            }
        });
    }

    // -----------------------------------------------------------------------
    // Tab: Basic
    // -----------------------------------------------------------------------

    fn draw_tab_basic(&mut self, ui: &mut Ui) {
        section_header(ui, "Identity");

        labeled_field(ui, "ID", "Internal registry key (e.g. 'rocks/granite'). Used in block_registry.json and JSON filename.");
        let prev_id = self.edit_def.id.clone();
        if ui.text_edit_singleline(&mut self.edit_def.id).changed() {
            self.dirty = true;
        }
        if self.edit_def.id != prev_id {
            // Update registry key for new blocks
            if self.is_new { self.edit_key = self.edit_def.id.clone(); }
        }

        labeled_field(ui, "Display Name", "Human-readable name shown in inventory, tooltips, and UI.");
        if ui.text_edit_singleline(&mut self.edit_def.display_name).changed() { self.dirty = true; }

        labeled_field(ui, "Description", "One-line description of the block's purpose and appearance.");
        if ui.text_edit_multiline(&mut self.edit_def.description).changed() { self.dirty = true; }

        labeled_field(ui, "Inventory Tab", "Category in the creative inventory. Options: Building, Natural, Decoration, Lighting, Fluids, Ores.");
        let mut tab_str = self.edit_def.inventory_tab.clone().unwrap_or_default();
        if ui.text_edit_singleline(&mut tab_str).changed() {
            self.edit_def.inventory_tab = if tab_str.is_empty() { None } else { Some(tab_str) };
            self.dirty = true;
        }

        ui.add_space(8.0);
        section_header(ui, "Base Color");
        labeled_field(ui, "Color (RGB)", "Base block color (0..1 each). Used when no face-specific color is defined.");
        if color_edit_3(ui, &mut self.edit_def.color) { self.dirty = true; }

        ui.add_space(8.0);
        section_header(ui, "Physics & Rendering");

        toggle_field(
            ui, &mut self.edit_def.solid, "Solid",
            "Whether the block blocks movement and generates solid mesh faces.",
            &mut self.dirty,
        );

        toggle_field(
            ui, &mut self.edit_def.biome_tint, "Biome Tint",
            "Multiply vertex color by the biome's foliage tint color. Use for grass, leaves, ferns.",
            &mut self.dirty,
        );

        ui.add_space(8.0);
        section_header(ui, "Placement");

        labeled_field(ui, "Placement Mode", "How the block orients when placed.\nFixed • AttachHorizontal • AttachFull • AttachFullRotatable • LookHorizontal • LookFull • LookAxis");
        let modes = [
            PlacementMode::Fixed, PlacementMode::AttachHorizontal,
            PlacementMode::AttachFull, PlacementMode::AttachFullRotatable,
            PlacementMode::LookHorizontal, PlacementMode::LookFull,
            PlacementMode::LookAxis,
        ];
        let current_label = format!("{:?}", self.edit_def.placement_mode);
        egui::ComboBox::from_id_salt("placement_mode")
            .selected_text(&current_label)
            .show_ui(ui, |ui| {
                for mode in &modes {
                    let label = format!("{:?}", mode);
                    if ui.selectable_value(&mut self.edit_def.placement_mode, mode.clone(), &label).changed() {
                        self.dirty = true;
                    }
                }
            });

        labeled_field(ui, "Default Rotation", "Rotation index (0..15) applied when placement mode is Fixed.");
        let mut rot = self.edit_def.default_rotation as i32;
        if ui.add(egui::Slider::new(&mut rot, 0..=15)).changed() {
            self.edit_def.default_rotation = rot as u8;
            self.dirty = true;
        }

    }

    // -----------------------------------------------------------------------
    // Tab: Material (PBR)
    // -----------------------------------------------------------------------

    fn draw_tab_material(&mut self, ui: &mut Ui) {
        section_header(ui, "PBR Material Properties");
        ui.label(RichText::new("Controls how the block responds to direct lighting and ambient occlusion.").small().color(Color32::GRAY));
        ui.add_space(4.0);

        labeled_field(ui, "Albedo (RGB)", "Base diffuse color for PBR shading (0..1). Usually matches the block's visual color.");
        if color_edit_3(ui, &mut self.edit_def.material.albedo) { self.dirty = true; }

        labeled_field(ui, "Roughness", "Surface micro-roughness: 0.0 = mirror-smooth, 1.0 = fully diffuse. Most blocks use 0.7–0.95.");
        if ui.add(egui::Slider::new(&mut self.edit_def.material.roughness, 0.0_f32..=1.0)
            .text("roughness")).changed() { self.dirty = true; }

        labeled_field(ui, "Metalness", "Conductor/dielectric: 0.0 = non-metal (stone, wood), 1.0 = metal (iron, gold). Most blocks = 0.");
        if ui.add(egui::Slider::new(&mut self.edit_def.material.metalness, 0.0_f32..=1.0)
            .text("metalness")).changed() { self.dirty = true; }

        labeled_field(ui, "Ambient Occlusion (AO)", "Baked shadow factor: 0.0 = fully darkened crevice, 1.0 = fully lit surface.");
        if ui.add(egui::Slider::new(&mut self.edit_def.material.ao, 0.0_f32..=1.0)
            .text("ao")).changed() { self.dirty = true; }

        ui.add_space(12.0);

        // Visual roughness/metal gauge
        egui::Frame::none()
            .fill(Color32::from_rgb(30, 30, 40))
            .rounding(6.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                ui.label(RichText::new("Material Preview").size(12.0).color(Color32::GRAY));
                ui.horizontal(|ui| {
                    let r = self.edit_def.material.roughness;
                    let m = self.edit_def.material.metalness;
                    let albedo = color_f32_to_egui(self.edit_def.material.albedo);

                    // Specular highlight simulation: rougher = larger/dimmer highlight
                    let highlight_r = 1.0 - r * 0.8;
                    let metal_tint = if m > 0.5 { albedo } else { Color32::WHITE };

                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(60.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 4.0, albedo);
                    // Simulate specular dot
                    let spot_center = Pos2::new(rect.min.x + rect.width() * 0.25, rect.min.y + rect.height() * 0.25);
                    let spot_r = 12.0 * highlight_r;
                    let spot_col = Color32::from_rgba_unmultiplied(
                        metal_tint.r(), metal_tint.g(), metal_tint.b(),
                        (highlight_r * 200.0) as u8,
                    );
                    ui.painter().circle_filled(spot_center, spot_r, spot_col);

                    ui.vertical(|ui| {
                        ui.label(format!("Roughness: {:.2}  →  {}",
                            r, if r < 0.3 { "Mirror-like" } else if r < 0.6 { "Semi-gloss" } else { "Matte" }
                        ));
                        ui.label(format!("Metalness: {:.2}  →  {}",
                            m, if m < 0.2 { "Dielectric (stone/wood)" } else if m < 0.6 { "Mixed" } else { "Metallic" }
                        ));
                    });
                });
            });
    }

    // -----------------------------------------------------------------------
    // Tab: Emission
    // -----------------------------------------------------------------------

    fn draw_tab_emission(&mut self, ui: &mut Ui) {
        section_header(ui, "Light Emission");
        ui.label(RichText::new("Controls global illumination contribution and VCT flood-fill range.").small().color(Color32::GRAY));
        ui.add_space(4.0);

        let gi_mode = crate::core::config::gi_mode();
        let is_mono = gi_mode == 1 || gi_mode == 2;
        let max_range = match gi_mode { 1 => 16.0_f32, 2 | 3 => 32.0, _ => 128.0 };

        if is_mono {
            ui.label(
                RichText::new(format!(
                    "⚠ {} mode active — color editing disabled, range capped to {} blocks.",
                    if gi_mode == 1 { "Mono 16" } else { "Mono 32" },
                    max_range as i32,
                ))
                .size(11.0)
                .color(Color32::from_rgb(255, 200, 60)),
            );
            ui.add_space(4.0);
        }

        toggle_field(
            ui, &mut self.edit_def.emission.emit_light, "Emit Light",
            "Master switch. When true, this block acts as a light source in the VCT GI system.",
            &mut self.dirty,
        );

        if self.edit_def.emission.emit_light {
            labeled_field(ui, "Light Color (RGB)", "Color of emitted light. Pure white = neutral, colored = tinted GI.");
            ui.scope(|ui| {
                ui.set_enabled(!is_mono);
                if color_edit_3(ui, &mut self.edit_def.emission.light_color) { self.dirty = true; }
            });

            labeled_field(ui, "Light Strength (0..15)", "Radius of influence: how many blocks the flood-fill propagates.");
            if ui.add(egui::Slider::new(&mut self.edit_def.emission.light_strength, 0.0_f32..=15.0)
                .text("strength")).changed() { self.dirty = true; }

            labeled_field(ui, "Light Intensity", "Brightness multiplier for the emitted light. Higher = brighter GI contributions.");
            if ui.add(egui::Slider::new(&mut self.edit_def.emission.light_intensity, 0.0_f32..=3.0)
                .text("intensity")).changed() { self.dirty = true; }

            labeled_field(ui, "Light Range (blocks)", "Maximum distance (in blocks) that emitted light can travel.");
            if ui.add(egui::Slider::new(&mut self.edit_def.emission.light_range, 1.0_f32..=max_range)
                .text("range")).changed() { self.dirty = true; }

            toggle_field(
                ui, &mut self.edit_def.emission.no_self_gi, "No Self GI",
                "Prevent this block from receiving GI from its own emission. Stops self-tinting artifacts on glowstone etc.",
                &mut self.dirty,
            );

            ui.add_space(8.0);

            // Live color preview bar
            let em_color = color_f32_to_egui(self.edit_def.emission.light_color);
            let intensity = self.edit_def.emission.light_intensity;
            egui::Frame::none()
                .fill(Color32::from_gray(20))
                .rounding(6.0)
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.label(RichText::new("Emission Preview").small().color(Color32::GRAY));
                    // Draw a glowing circle
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 60.0), egui::Sense::hover());
                    let center = rect.center();
                    for i in (1..=6).rev() {
                        let radius = (i as f32) * 8.0 * intensity.clamp(0.1, 3.0);
                        let alpha = (30.0 / i as f32 * intensity.clamp(0.1, 2.0)) as u8;
                        ui.painter().circle_filled(
                            center, radius,
                            Color32::from_rgba_unmultiplied(em_color.r(), em_color.g(), em_color.b(), alpha),
                        );
                    }
                    ui.painter().circle_filled(center, 8.0 * intensity.clamp(0.1, 3.0), em_color);
                });
        }
    }

    // -----------------------------------------------------------------------
    // Tab: Lights
    // -----------------------------------------------------------------------

    fn draw_tab_lights(&mut self, ui: &mut Ui) {
        section_header(ui, "Block Light Sources");
        ui.label(RichText::new(
            "Attach point or directional lights to this block.\n\
             Positions are relative to the block centre (−0.5..0.5 per axis).\n\
             Arrows are drawn in the preview to visualize each source."
        ).small().color(Color32::GRAY));
        ui.add_space(6.0);

        let gi_mode = crate::core::config::gi_mode();
        let is_mono = gi_mode == 1 || gi_mode == 2;
        let max_range_point = match gi_mode { 1 => 16.0_f32, 2 | 3 => 32.0, _ => 64.0 };

        if is_mono {
            ui.label(
                RichText::new(format!(
                    "⚠ {} mode active — color editing disabled, range capped to {} blocks.",
                    if gi_mode == 1 { "Mono 16" } else { "Mono 32" },
                    max_range_point as i32,
                ))
                .size(11.0)
                .color(Color32::from_rgb(255, 200, 60)),
            );
            ui.add_space(4.0);
        }

        ui.horizontal(|ui| {
            if ui.add(egui::Button::new(egui::RichText::new("+ Add Point Light").size(13.0))
                .min_size(egui::vec2(160.0, 26.0))).clicked()
            {
                self.edit_def.light_sources.push(BlockLightSource {
                    kind: LightKind::Point, ..Default::default()
                });
                self.dirty = true;
            }
            if ui.add(egui::Button::new(egui::RichText::new("+ Add Directional").size(13.0))
                .min_size(egui::vec2(160.0, 26.0))).clicked()
            {
                self.edit_def.light_sources.push(BlockLightSource {
                    kind: LightKind::Directional, ..Default::default()
                });
                self.dirty = true;
            }
        });

        ui.add_space(6.0);

        let mut to_remove: Option<usize> = None;
        let count = self.edit_def.light_sources.len();

        for i in 0..count {
            let ls = &mut self.edit_def.light_sources[i];
            let kind_label = match ls.kind {
                LightKind::Point       => "Point",
                LightKind::Directional => "Directional",
            };
            let header = if ls.label.is_empty() {
                format!("Light #{} ({})", i + 1, kind_label)
            } else {
                format!("#{} '{}' ({})", i + 1, ls.label, kind_label)
            };

            egui::Frame::none()
                .fill(Color32::from_rgb(22, 22, 32))
                .rounding(6.0)
                .inner_margin(8.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.strong(RichText::new(&header).color(Color32::from_rgb(180, 200, 255)));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("✕ Remove").clicked() {
                                to_remove = Some(i);
                            }
                        });
                    });
                    ui.add_space(4.0);

                    // Label
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Label").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("Optional descriptive name for this light source (editor only).");
                        if ui.text_edit_singleline(&mut ls.label).changed() { self.dirty = true; }
                    });

                    // Kind
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Type").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("Point: radiates in all directions.\nDirectional: focused cone toward a specific direction.");
                        egui::ComboBox::from_id_salt(format!("ls_kind_{}", i))
                            .selected_text(kind_label)
                            .show_ui(ui, |ui| {
                                if ui.selectable_value(&mut ls.kind, LightKind::Point,       "Point").changed()       { self.dirty = true; }
                                if ui.selectable_value(&mut ls.kind, LightKind::Directional, "Directional").changed() { self.dirty = true; }
                            });
                    });

                    ui.add_space(4.0);

                    // Position
                    ui.label(RichText::new("Position  (−0.5..0.5 per axis, relative to block centre)").size(11.0).color(Color32::LIGHT_GRAY))
                        .on_hover_text("X = East/West (+X=East), Y = Up/Down (+Y=Up), Z = North/South (+Z=South).\nUse arrows in the preview to adjust visually.");
                    ui.horizontal(|ui| {
                        ui.label("X"); if ui.add(egui::Slider::new(&mut ls.position[0], -0.5_f32..=0.5).text("")).changed() { self.dirty = true; }
                        ui.label("Y"); if ui.add(egui::Slider::new(&mut ls.position[1], -0.5_f32..=0.5).text("")).changed() { self.dirty = true; }
                        ui.label("Z"); if ui.add(egui::Slider::new(&mut ls.position[2], -0.5_f32..=0.5).text("")).changed() { self.dirty = true; }
                    });

                    // Position nudge arrows
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Nudge:").size(11.0).color(Color32::GRAY));
                        let step = 0.05_f32;
                        for (label, axis, sign, tip) in [
                            ("←X", 0usize, -1f32, "Move light West (−X)"),
                            ("→X", 0, 1.0,  "Move light East (+X)"),
                            ("↓Y", 1, -1.0, "Move light Down (−Y)"),
                            ("↑Y", 1,  1.0, "Move light Up (+Y)"),
                            ("←Z", 2, -1.0, "Move light North (−Z)"),
                            ("→Z", 2,  1.0, "Move light South (+Z)"),
                        ] {
                            if ui.small_button(label).on_hover_text(tip).clicked() {
                                ls.position[axis] = (ls.position[axis] + sign * step).clamp(-0.5, 0.5);
                                self.dirty = true;
                            }
                        }
                    });

                    ui.add_space(4.0);

                    // Color
                    ui.label(RichText::new("Color").size(11.0).color(Color32::LIGHT_GRAY))
                        .on_hover_text("RGB color of the emitted light (0..1 per channel).");
                    ui.scope(|ui| {
                        ui.set_enabled(!is_mono);
                        if color_edit_3(ui, &mut ls.color) { self.dirty = true; }
                    });

                    // Intensity
                    ui.label(RichText::new("Intensity").size(11.0).color(Color32::LIGHT_GRAY))
                        .on_hover_text("Brightness multiplier (1.0 = standard, higher = brighter).");
                    if ui.add(egui::Slider::new(&mut ls.intensity, 0.0_f32..=10.0).text("intensity")).changed() { self.dirty = true; }

                    // Range (point only)
                    if ls.kind == LightKind::Point {
                        ui.label(RichText::new("Range (blocks, 0 = default)").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("Maximum distance this light illuminates. 0 uses the global default for the GI system.");
                        if ui.add(egui::Slider::new(&mut ls.range, 0.0_f32..=max_range_point).text("range")).changed() { self.dirty = true; }
                    }

                    // Direction + Focus (directional only)
                    if ls.kind == LightKind::Directional {
                        ui.add_space(4.0);
                        ui.label(RichText::new("Direction  (normalized XYZ)").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("The direction the cone of light points toward.\n[0,−1,0] = straight down, [0,0,1] = south, etc.\nThe vector is auto-normalized.");
                        ui.horizontal(|ui| {
                            let changed = ui.add(egui::Slider::new(&mut ls.direction[0], -1.0_f32..=1.0).text("X")).changed()
                                | ui.add(egui::Slider::new(&mut ls.direction[1], -1.0_f32..=1.0).text("Y")).changed()
                                | ui.add(egui::Slider::new(&mut ls.direction[2], -1.0_f32..=1.0).text("Z")).changed();
                            if changed {
                                // Keep normalized
                                let len = (ls.direction[0].powi(2) + ls.direction[1].powi(2) + ls.direction[2].powi(2)).sqrt().max(1e-6);
                                ls.direction[0] /= len; ls.direction[1] /= len; ls.direction[2] /= len;
                                self.dirty = true;
                            }
                        });

                        ui.label(RichText::new("Focus angle (degrees, 0..180)").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("Half-angle of the light cone.\n0° = laser pencil, 30° = spotlight, 90° = hemisphere, 180° = full sphere.");
                        if ui.add(egui::Slider::new(&mut ls.focus, 0.0_f32..=180.0).text("°")).changed() { self.dirty = true; }
                    }
                });
            ui.add_space(4.0);
        }

        if let Some(idx) = to_remove {
            self.edit_def.light_sources.remove(idx);
            self.dirty = true;
        }

        if count == 0 {
            ui.vertical_centered(|ui| {
                ui.add_space(16.0);
                ui.label(RichText::new("No light sources. Use the buttons above to add one.").color(Color32::DARK_GRAY));
            });
        }
    }

    // -----------------------------------------------------------------------
    // Tab: Faces
    // -----------------------------------------------------------------------

    fn draw_tab_faces(&mut self, ui: &mut Ui) {
        section_header(ui, "Face Texture & Color Overrides");
        ui.label(RichText::new(
            "Priority (most specific first):\n  top/bottom → sides → all → block base color\n  north/south/east/west → sides → all → block base color"
        ).small().color(Color32::GRAY));
        ui.add_space(6.0);

        // Helper macro-like closures
        let dirty = &mut self.dirty;
        let faces  = &mut self.edit_def.faces;

        face_override_section(ui, "All Faces (lowest priority)", &mut faces.all, dirty);
        ui.add_space(4.0);
        face_override_section(ui, "Top (+Y)", &mut faces.top, dirty);
        ui.add_space(4.0);
        face_override_section(ui, "Bottom (-Y)", &mut faces.bottom, dirty);
        ui.add_space(4.0);
        face_override_section(ui, "Sides (N/S/E/W — convenience alias)", &mut faces.sides, dirty);
        ui.add_space(4.0);

        ui.collapsing("Individual directional faces", |ui| {
            face_override_section(ui, "North (-Z)", &mut faces.north, dirty);
            ui.add_space(4.0);
            face_override_section(ui, "South (+Z)", &mut faces.south, dirty);
            ui.add_space(4.0);
            face_override_section(ui, "East (+X)",  &mut faces.east, dirty);
            ui.add_space(4.0);
            face_override_section(ui, "West (-X)",  &mut faces.west, dirty);
        });
    }

    // -----------------------------------------------------------------------
    // Tab: Fluid
    // -----------------------------------------------------------------------

    fn draw_tab_fluid(&mut self, ui: &mut Ui) {
        section_header(ui, "Fluid Simulation");
        ui.label(RichText::new(
            "When fluid properties are present, the block participates in fluid simulation.\n\
             Fluid level (0..1) is stored per-voxel; 1.0 = source block (never depletes)."
        ).small().color(Color32::GRAY));
        ui.add_space(6.0);

        let has_fluid = self.edit_def.fluid.is_some();

        ui.horizontal(|ui| {
            let mut enabled = has_fluid;
            if ui.checkbox(&mut enabled, "Is Fluid Block").changed() {
                if enabled {
                    self.edit_def.fluid = Some(FluidProperties::default());
                } else {
                    self.edit_def.fluid = None;
                }
                self.dirty = true;
            }
            ui.label(
                RichText::new("Enable fluid simulation for this block")
                    .small().color(Color32::GRAY)
            );
        });

        if let Some(ref mut fp) = self.edit_def.fluid {
            ui.add_space(8.0);

            labeled_field(ui, "Level Decrease (per step)", "Levels lost per horizontal spread step. Water = 1 (7 blocks), Lava = 2 (3 blocks).");
            let mut ld = fp.level_decrease as i32;
            if ui.add(egui::Slider::new(&mut ld, 1..=7).text("level_decrease")).changed() {
                fp.level_decrease = ld as u8;
                self.dirty = true;
            }

            labeled_field(ui, "Tick Delay (frames)", "Frames between simulation updates. Water = 5, Lava = 20.");
            let mut td = fp.tick_delay as i32;
            if ui.add(egui::Slider::new(&mut td, 1..=60).text("tick_delay")).changed() {
                fp.tick_delay = td as u32;
                self.dirty = true;
            }

            labeled_field(ui, "Slope Find Distance (blocks)", "How far to search for a downward path before spreading horizontally. Water = Lava = 4.");
            let mut sfd = fp.slope_find_distance as i32;
            if ui.add(egui::Slider::new(&mut sfd, 0..=8).text("slope_find_distance")).changed() {
                fp.slope_find_distance = sfd as u8;
                self.dirty = true;
            }

            ui.add_space(6.0);
            section_header(ui, "Underwater Fog");

            labeled_field(ui, "Fog Color (RGB)", "Color of the fog when camera is submerged in this fluid.");
            if color_edit_3(ui, &mut fp.fog_color) { self.dirty = true; }

            labeled_field(ui, "Fog Density", "Thickness of the underwater fog. Water ≈ 0.04, Lava ≈ 0.2.");
            if ui.add(egui::Slider::new(&mut fp.fog_density, 0.0_f32..=1.0).text("fog_density")).changed() { self.dirty = true; }
        }

    }

    // -----------------------------------------------------------------------
    // New-block dialog
    // -----------------------------------------------------------------------

    fn draw_new_block_dialog(&mut self, ctx: &egui::Context) {
        if !self.new_dlg.open { return; }

        let mut close = false;
        let mut create = false;

        egui::Window::new("Create New Block")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .fixed_size(Vec2::new(340.0, 220.0))
            .show(ctx, |ui| {
                ui.label(RichText::new("New block definition").size(16.0).strong());
                ui.add_space(8.0);

                labeled_field(ui, "Block ID", "Registry key and JSON filename, e.g. 'my_block' or 'folder/my_block'.");
                ui.text_edit_singleline(&mut self.new_dlg.id);

                ui.add_space(6.0);
                labeled_field(ui, "Display Name", "Human-readable name shown in inventory.");
                ui.text_edit_singleline(&mut self.new_dlg.name);

                if !self.new_dlg.err.is_empty() {
                    ui.add_space(4.0);
                    ui.label(RichText::new(&self.new_dlg.err.clone()).color(Color32::LIGHT_RED));
                }

                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button(RichText::new("Create").size(15.0)).clicked() {
                        create = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });

        if create {
            let id = self.new_dlg.id.trim().to_string();
            let name = self.new_dlg.name.trim().to_string();
            if id.is_empty() {
                self.new_dlg.err = "ID cannot be empty.".into();
            } else if self.registry_order.contains(&id) {
                self.new_dlg.err = format!("Block '{}' already exists.", id);
            } else {
                let mut def = Self::default_block_def(&id);
                if !name.is_empty() { def.display_name = name; }
                self.edit_def    = def;
                self.edit_key    = id.clone();
                self.selected_key = None;
                self.is_new      = true;
                self.dirty       = true;
                self.active_tab  = EditorTab::Basic;
                self.new_dlg.open = false;
                self.status_msg  = format!("New block '{}' ready — fill in properties and Save.", id);
                self.status_ok   = true;
                debug_log!("BlockEditor", "create_new", "New block: '{}'", id);
            }
        }

        if close { self.new_dlg.open = false; }
    }

    // -----------------------------------------------------------------------
    // Duplicate
    // -----------------------------------------------------------------------

    fn duplicate_current(&mut self) {
        let mut def = self.edit_def.clone();
        def.id = format!("{}_copy", def.id);
        def.display_name = format!("{} Copy", def.display_name);
        let key = def.id.clone();
        self.edit_def    = def;
        self.edit_key    = key.clone();
        self.selected_key = None;
        self.is_new      = true;
        self.dirty       = true;
        self.status_msg  = format!("Duplicated as '{}' — edit and Save.", key);
        self.status_ok   = true;
    }

    // -----------------------------------------------------------------------
    // Save
    // -----------------------------------------------------------------------

    fn save_current(&mut self) {
        let key = self.edit_key.clone();
        if key.is_empty() {
            self.status_msg = "No block key set — enter an ID in the Basic tab first.".into();
            self.status_ok  = false;
            return;
        }

        // Serialize block definition
        let mut val = match serde_json::to_value(&self.edit_def) {
            Ok(v)  => v,
            Err(e) => {
                self.status_msg = format!("Serialization error: {}", e);
                self.status_ok  = false;
                return;
            }
        };

        // Inject format_version
        if let Some(obj) = val.as_object_mut() {
            let mut ordered = serde_json::Map::new();
            ordered.insert("format_version".into(), serde_json::Value::Number(2.into()));
            for (k, v) in obj.iter() {
                ordered.insert(k.clone(), v.clone());
            }
            val = serde_json::Value::Object(ordered);
        }

        let json = match serde_json::to_string_pretty(&val) {
            Ok(s)  => s,
            Err(e) => {
                self.status_msg = format!("JSON stringify error: {}", e);
                self.status_ok  = false;
                return;
            }
        };

        // Determine output path
        let out_path = PathBuf::from(BLOCKS_DIR).join(format!("{}.json", key));
        if let Some(parent) = out_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                self.status_msg = format!("Failed to create dir: {}", e);
                self.status_ok  = false;
                return;
            }
        }

        if let Err(e) = std::fs::write(&out_path, &json) {
            self.status_msg = format!("Write error: {}", e);
            self.status_ok  = false;
            return;
        }

        // If new block, add to registry
        if self.is_new {
            if !self.registry_order.contains(&key) {
                self.registry_order.push(key.clone());
            }
            if let Err(e) = self.save_registry() {
                self.status_msg = format!("Saved block JSON but registry update failed: {}", e);
                self.status_ok  = false;
                return;
            }
        }

        // Update in-memory map
        self.block_defs.insert(key.clone(), self.edit_def.clone());
        self.selected_key = Some(key.clone());
        self.is_new       = false;
        self.dirty        = false;
        self.status_msg   = format!("Saved '{}'  →  {:?}", key, out_path);
        self.status_ok    = true;

        debug_log!("BlockEditor", "save_current", "Saved '{}' to {:?}", key, out_path);
    }

    fn save_registry(&self) -> Result<(), String> {
        let reg_path = PathBuf::from(REGISTRY_FILE);

        // Read current registry to preserve format_version and description
        let existing: serde_json::Value = if reg_path.exists() {
            let s = std::fs::read_to_string(&reg_path)
                .map_err(|e| format!("Read: {}", e))?;
            serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::Value::Null
        };

        let fv = existing.get("format_version")
            .cloned()
            .unwrap_or(serde_json::Value::Number(1.into()));
        let desc = existing.get("description")
            .cloned()
            .unwrap_or(serde_json::Value::String(
                "QubePixel Block Registry".into()
            ));

        let blocks: Vec<serde_json::Value> = self.registry_order
            .iter()
            .map(|k| serde_json::Value::String(k.clone()))
            .collect();

        let mut map = serde_json::Map::new();
        map.insert("format_version".into(), fv);
        map.insert("description".into(), desc);
        map.insert("blocks".into(), serde_json::Value::Array(blocks));

        let json = serde_json::to_string_pretty(&serde_json::Value::Object(map))
            .map_err(|e| format!("Stringify: {}", e))?;

        std::fs::write(&reg_path, json)
            .map_err(|e| format!("Write: {}", e))?;

        debug_log!("BlockEditor", "save_registry",
            "Registry saved with {} entries", self.registry_order.len());
        Ok(())
    }

}

// =============================================================================
// Standalone UI helpers
// =============================================================================

fn section_header(ui: &mut Ui, label: &str) {
    ui.add_space(2.0);
    ui.label(RichText::new(label).size(13.0).strong().color(Color32::from_rgb(160, 190, 255)));
    ui.add_space(2.0);
}

fn labeled_field(ui: &mut Ui, name: &str, hint: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(name).size(12.0).color(Color32::LIGHT_GRAY));
        ui.label(
            RichText::new("?")
                .size(10.0)
                .color(Color32::DARK_GRAY)
        ).on_hover_text(hint);
    });
}

fn toggle_field(ui: &mut Ui, val: &mut bool, name: &str, hint: &str, dirty: &mut bool) {
    ui.horizontal(|ui| {
        if ui.checkbox(val, RichText::new(name).size(12.0)).changed() {
            *dirty = true;
        }
        ui.label(RichText::new("?").size(10.0).color(Color32::DARK_GRAY))
            .on_hover_text(hint);
    });
}

fn color_edit_3(ui: &mut Ui, color: &mut [f32; 3]) -> bool {
    let mut srgb = [
        (color[0] * 255.0) as u8,
        (color[1] * 255.0) as u8,
        (color[2] * 255.0) as u8,
    ];
    let mut changed = ui.color_edit_button_srgb(&mut srgb).changed();
    if changed {
        color[0] = srgb[0] as f32 / 255.0;
        color[1] = srgb[1] as f32 / 255.0;
        color[2] = srgb[2] as f32 / 255.0;
    }

    // Also show RGB sliders inline
    ui.horizontal(|ui| {
        changed |= ui.add(egui::Slider::new(&mut color[0], 0.0_f32..=1.0).text("R")).changed();
        changed |= ui.add(egui::Slider::new(&mut color[1], 0.0_f32..=1.0).text("G")).changed();
        changed |= ui.add(egui::Slider::new(&mut color[2], 0.0_f32..=1.0).text("B")).changed();
    });

    changed
}

fn face_override_section(ui: &mut Ui, label: &str, face: &mut Option<FaceOverride>, dirty: &mut bool) {
    let has = face.is_some();
    ui.horizontal(|ui| {
        let mut enabled = has;
        if ui.checkbox(&mut enabled, RichText::new(label).size(12.0).strong()).changed() {
            if enabled {
                *face = Some(FaceOverride {
                    color: [1.0, 1.0, 1.0],
                    texture: None,
                    layer1: None, layer2: None, layer3: None, layer4: None, layer5: None,
                    resolved_texture: None,
                });
            } else {
                *face = None;
            }
            *dirty = true;
        }
        ui.label(RichText::new("?").size(10.0).color(Color32::DARK_GRAY))
            .on_hover_text("Enable per-face texture and color override for this face direction.");
    });

    if let Some(fo) = face {
        ui.indent(label, |ui| {
            // Texture path + browse button
            ui.horizontal(|ui| {
                ui.label(RichText::new("Texture").size(12.0).color(Color32::LIGHT_GRAY))
                    .on_hover_text("Texture key (e.g. 'grass_top', 'rocks/granite').\nFile must be at: assets/textures/simpleblocks/<key>.png\nLeave empty to use the block base color only.");
            });
            let mut tex_str = fo.texture.clone().unwrap_or_default();
            ui.horizontal(|ui| {
                let resp = ui.add(egui::TextEdit::singleline(&mut tex_str).desired_width(200.0));
                if resp.changed() {
                    fo.texture = if tex_str.is_empty() { None } else { Some(tex_str.clone()) };
                    *dirty = true;
                }
                if ui.small_button("📂").on_hover_text("Browse for a PNG texture file inside assets/").clicked() {
                    if let Some(key) = pick_texture_key() {
                        fo.texture = Some(key);
                        *dirty = true;
                    }
                }
            });

            // Layer overlays (collapsible)
            ui.collapsing("Overlay Layers (layer1..layer5)", |ui| {
                ui.label(RichText::new(
                    "Alpha-blended overlay layers composited onto the base texture at load time.\n\
                     layer1 is closest to the base; layer5 is the topmost. Each value is a texture key."
                ).small().color(Color32::GRAY));

                let layers: [(&str, &mut Option<String>); 5] = [
                    ("layer1", &mut fo.layer1),
                    ("layer2", &mut fo.layer2),
                    ("layer3", &mut fo.layer3),
                    ("layer4", &mut fo.layer4),
                    ("layer5", &mut fo.layer5),
                ];
                for (name, layer) in layers {
                    let mut s = layer.clone().unwrap_or_default();
                    let mut changed = false;
                    let mut new_key: Option<String> = None;
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{}:", name)).size(11.0).color(Color32::GRAY))
                            .on_hover_text(format!("Overlay texture key for {}. Blended on top of the face texture.", name));
                        if ui.add(egui::TextEdit::singleline(&mut s).desired_width(160.0)).changed() {
                            changed = true;
                        }
                        if ui.small_button("📂").on_hover_text("Browse for overlay texture inside assets/").clicked() {
                            new_key = pick_texture_key();
                        }
                    });
                    if changed {
                        *layer = if s.is_empty() { None } else { Some(s) };
                        *dirty = true;
                    }
                    if let Some(key) = new_key {
                        *layer = Some(key);
                        *dirty = true;
                    }
                }
            });

            // Color tint
            ui.label(RichText::new("Face Color (RGB)").size(12.0).color(Color32::LIGHT_GRAY))
                .on_hover_text("RGB tint multiplied over the texture. White [1,1,1] = no tint; colored = tinted face.");
            let mut rgb = fo.color;
            if color_edit_3(ui, &mut rgb) {
                fo.color = rgb;
                *dirty = true;
            }
        });
    }
}

/// Opens the OS file picker filtered to PNG files inside `assets/`.
/// Returns the texture key relative to `assets/textures/simpleblocks/`, or None.
/// Rejects paths outside the assets directory with a warning.
fn pick_texture_key() -> Option<String> {
    let assets_dir = std::fs::canonicalize("assets").ok()?;
    let file = rfd::FileDialog::new()
        .set_title("Select texture PNG")
        .add_filter("PNG image", &["png"])
        .set_directory(&assets_dir)
        .pick_file()?;

    // Validate that the file is inside assets/
    let canon = std::fs::canonicalize(&file).ok()?;
    if !canon.starts_with(&assets_dir) {
        // Show a warning in the status; caller handles None
        log::warn!("[BlockEditor][pick_texture_key] Rejected path outside assets/: {:?}", file);
        return None;
    }

    // Try to make a texture key relative to assets/textures/simpleblocks/
    let textures_dir = assets_dir.join("textures").join("simpleblocks");
    if let Ok(rel) = canon.strip_prefix(&textures_dir) {
        // Strip .png extension
        let key = rel.with_extension("");
        return Some(key.to_string_lossy().replace('\\', "/"));
    }

    // Fallback: return path relative to assets/ as-is (no extension)
    if let Ok(rel) = canon.strip_prefix(&assets_dir) {
        let key = rel.with_extension("");
        return Some(key.to_string_lossy().replace('\\', "/"));
    }

    None
}

// ---------------------------------------------------------------------------
// Light source projection helpers
// ---------------------------------------------------------------------------

/// Build the same MVP used by BlockPreviewRenderer for a given yaw/pitch/aspect.
fn preview_mvp(yaw: f32, pitch: f32, aspect: f32) -> Mat4 {
    let proj  = Mat4::perspective_rh(50.0_f32.to_radians(), aspect, 0.05, 100.0);
    let eye   = Vec3::new(0.0, 0.0, 2.4);
    let view  = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let model = Mat4::from_rotation_y(yaw) * Mat4::from_rotation_x(pitch);
    proj * view * model
}

/// Project a 3D block-space position to a 2D egui Pos2 inside `rect`.
/// Returns None when the point is behind the camera.
fn project_to_rect(mvp: &Mat4, pos3: [f32; 3], rect: Rect) -> Option<Pos2> {
    let clip = *mvp * Vec4::new(pos3[0], pos3[1], pos3[2], 1.0);
    if clip.w <= 0.0 { return None; }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    // NDC: x ∈ [-1,1] (left→right), y ∈ [-1,1] (bottom→top in wgpu)
    let px = rect.min.x + (ndc_x + 1.0) * 0.5 * rect.width();
    let py = rect.min.y + (1.0 - ndc_y) * 0.5 * rect.height(); // flip Y
    Some(Pos2::new(px, py))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn color_f32_to_egui(c: [f32; 3]) -> Color32 {
    Color32::from_rgb(
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

