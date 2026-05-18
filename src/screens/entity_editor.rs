// =============================================================================
// QubePixel — Entity Light Editor
//
// In-game editor for entity light source definitions.
// Accessible from Main Menu → "Entity Light Editor".
//
// Layout:
//   Left panel   — searchable entity list (registry order)
//   Center panel — 3D preview (rendered by EntityPreviewRenderer) + 2D overlay
//   Right panel  — tabbed inspector: Basic | Lights
// =============================================================================

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use egui::{Color32, Pos2, Rect, RichText, Ui, Vec2};
use glam::{Mat4, Vec3, Vec4};

use crate::core::entity_definition::{
    EntityDefinition, EntityLightSource, EntityRegistry, LightKind,
};
use crate::core::player_model::PlayerModel;
use crate::debug_log;

// ---------------------------------------------------------------------------
// Inspector tabs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditorTab {
    Basic,
    Lights,
}

impl EditorTab {
    const ALL: [EditorTab; 2] = [EditorTab::Basic, EditorTab::Lights];

    fn label(self) -> &'static str {
        match self {
            EditorTab::Basic   => "Basic",
            EditorTab::Lights  => "Lights",
        }
    }
}

// ---------------------------------------------------------------------------
// New-entity dialog
// ---------------------------------------------------------------------------

struct NewEntityDialog {
    open: bool,
    id:   String,
    name: String,
    err:  String,
}

impl NewEntityDialog {
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
// EntityEditor
// ---------------------------------------------------------------------------

pub struct EntityEditor {
    pub visible: bool,

    // Registry state
    registry_order: Vec<String>,
    entity_defs:    HashMap<String, EntityDefinition>,

    // Edit buffer
    selected_key:  Option<String>,
    pub edit_def:  EntityDefinition,
    edit_key:      String,
    is_new:        bool,
    dirty:         bool,

    // UI
    search:     String,
    active_tab: EditorTab,
    status_msg: String,
    status_ok:  bool,

    // Dialogs
    new_dlg: NewEntityDialog,

    // Preview state (written during build_ui, read during render())
    pub preview_rect:      Option<egui::Rect>,
    pub preview_yaw:       f32,
    pub preview_pitch:     f32,
    pub pixels_per_point:  f32,

    // Cached bone names from the currently loaded model (for bone dropdown)
    cached_bones:       Vec<String>,
    cached_model_path:  String,
}

// ---------------------------------------------------------------------------
// Construction & loading
// ---------------------------------------------------------------------------

impl EntityEditor {
    pub fn new() -> Self {
        debug_log!("EntityEditor", "new", "Creating EntityEditor");
        let mut editor = Self {
            visible:          false,
            registry_order:   Vec::new(),
            entity_defs:      HashMap::new(),
            selected_key:     None,
            edit_def:         EntityDefinition::new("new_entity"),
            edit_key:         String::new(),
            is_new:           false,
            dirty:            false,
            search:           String::new(),
            active_tab:       EditorTab::Basic,
            status_msg:       String::new(),
            status_ok:        true,
            new_dlg:          NewEntityDialog::new(),
            preview_rect:     None,
            preview_yaw:      0.4,
            preview_pitch:    0.2,
            pixels_per_point: 1.0,
            cached_bones:      Vec::new(),
            cached_model_path: String::new(),
        };
        editor.reload_registry();
        editor
    }

    fn reload_registry(&mut self) {
        let reg = EntityRegistry::load();
        self.registry_order = reg.registry_order;
        self.entity_defs    = reg.defs;
        debug_log!("EntityEditor", "reload_registry",
            "Loaded {} entity definitions", self.entity_defs.len());
    }

    fn select_entity(&mut self, key: &str) {
        if let Some(def) = self.entity_defs.get(key) {
            self.edit_def    = def.clone();
            self.edit_key    = key.to_string();
            self.selected_key = Some(key.to_string());
            self.is_new      = false;
            self.dirty       = false;
            self.status_msg.clear();
            self.refresh_bones_cache();
            debug_log!("EntityEditor", "select_entity", "Selected '{}'", key);
        }
    }

    fn refresh_bones_cache(&mut self) {
        let path = &self.edit_def.model_path;
        if path == &self.cached_model_path { return; }
        self.cached_bones.clear();
        self.cached_model_path = path.clone();
        if path.is_empty() { return; }
        let json = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => return,
        };
        if let Ok(model) = PlayerModel::from_json(&json) {
            let mut names: Vec<String> = model.bone_pivots.keys().cloned().collect();
            names.sort();
            self.cached_bones = names;
            debug_log!("EntityEditor", "refresh_bones_cache",
                "Loaded {} bones from '{}'", self.cached_bones.len(), path);
        }
    }

    fn save_current(&mut self) {
        // Ensure assets/entities directory exists
        if let Err(e) = std::fs::create_dir_all("assets/entities") {
            self.status_msg = format!("Cannot create directory: {}", e);
            self.status_ok  = false;
            return;
        }

        // Save the entity definition JSON
        if let Err(e) = EntityRegistry::save_entity_def(&self.edit_def) {
            self.status_msg = format!("Save failed: {}", e);
            self.status_ok  = false;
            debug_log!("EntityEditor", "save_current", "Error: {}", e);
            return;
        }

        // If new entity, add to registry order and save registry
        if self.is_new {
            if !self.registry_order.contains(&self.edit_def.id) {
                self.registry_order.push(self.edit_def.id.clone());
            }
            if let Err(e) = EntityRegistry::save_registry(&self.registry_order) {
                self.status_msg = format!("Registry save failed: {}", e);
                self.status_ok  = false;
                return;
            }
        }

        // Update local cache
        let key = self.edit_def.id.clone();
        self.entity_defs.insert(key.clone(), self.edit_def.clone());
        if !self.registry_order.contains(&key) {
            self.registry_order.push(key.clone());
        }
        self.selected_key = Some(key.clone());
        self.edit_key     = key;
        self.is_new       = false;
        self.dirty        = false;
        self.status_msg   = "Saved.".into();
        self.status_ok    = true;

        debug_log!("EntityEditor", "save_current", "Saved '{}'", self.edit_def.id);
    }
}

// ---------------------------------------------------------------------------
// Main egui entry
// ---------------------------------------------------------------------------

impl EntityEditor {
    pub fn draw_egui(&mut self, ctx: &egui::Context) -> bool {
        let mut should_pop = false;

        // Top toolbar
        egui::TopBottomPanel::top("entity_editor_toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button(RichText::new("← Back").size(16.0)).clicked() {
                    should_pop = true;
                }
                ui.separator();
                ui.label(RichText::new("Entity Light Editor").size(18.0).strong());
                ui.separator();

                ui.label("Search:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.search)
                        .desired_width(160.0)
                        .hint_text("filter entities…"),
                );

                ui.separator();
                if ui.button(RichText::new("+ New Entity").size(14.0)).clicked() {
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
                }

                if !self.status_msg.is_empty() {
                    ui.separator();
                    let color = if self.status_ok { Color32::GREEN } else { Color32::LIGHT_RED };
                    ui.label(RichText::new(self.status_msg.clone()).color(color));
                }
            });
        });

        // Left panel: entity list
        egui::SidePanel::left("entity_editor_list")
            .resizable(true)
            .default_width(200.0)
            .show(ctx, |ui| {
                self.draw_entity_list(ui);
            });

        // Right panel: inspector
        egui::SidePanel::right("entity_editor_inspector")
            .resizable(true)
            .default_width(380.0)
            .show(ctx, |ui| {
                self.draw_inspector(ui);
            });

        // Center panel: transparent so the wgpu entity render shows through
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                self.draw_preview(ui);
            });

        // New-entity dialog
        self.draw_new_entity_dialog(ctx);

        should_pop
    }

    // -----------------------------------------------------------------------
    // Entity list (left panel)
    // -----------------------------------------------------------------------

    fn draw_entity_list(&mut self, ui: &mut Ui) {
        ui.heading("Entities");
        ui.add_space(4.0);
        ui.label(
            RichText::new(format!("{} registered", self.registry_order.len()))
                .small()
                .color(Color32::GRAY),
        );
        ui.separator();

        let search_lower  = self.search.to_lowercase();
        let current_key   = self.selected_key.clone();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let keys: Vec<String> = self.registry_order.clone();
            for (idx, key) in keys.iter().enumerate() {
                let def_name = self.entity_defs.get(key)
                    .map(|d| d.display_name.to_lowercase())
                    .unwrap_or_default();
                if !search_lower.is_empty()
                    && !key.to_lowercase().contains(&search_lower)
                    && !def_name.contains(&search_lower)
                {
                    continue;
                }

                let is_selected = current_key.as_deref() == Some(key.as_str());
                let display = self.entity_defs.get(key)
                    .map(|d| d.display_name.clone())
                    .unwrap_or_else(|| key.clone());

                // Light count chip color
                let has_lights = self.entity_defs.get(key)
                    .map(|d| !d.light_sources.is_empty())
                    .unwrap_or(false);
                let chip_color = if has_lights {
                    Color32::from_rgb(220, 180, 60)
                } else {
                    Color32::from_rgb(60, 60, 80)
                };

                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{:3}.", idx + 1))
                            .small()
                            .color(Color32::DARK_GRAY),
                    );
                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, chip_color);

                    let resp = ui.selectable_label(
                        is_selected,
                        RichText::new(&display).size(13.0),
                    );
                    if resp.clicked() {
                        self.select_entity(key);
                    }
                    if resp.hovered() {
                        resp.on_hover_text(key.as_str());
                    }
                });
            }
        });
    }

    // -----------------------------------------------------------------------
    // Preview (center panel)
    // -----------------------------------------------------------------------

    fn draw_preview(&mut self, ui: &mut Ui) {
        let avail = ui.available_size();
        let (resp, painter) = ui.allocate_painter(avail, egui::Sense::drag());

        self.preview_rect     = Some(resp.rect);
        self.pixels_per_point = ui.ctx().pixels_per_point();

        // Mouse drag → orbit camera
        if resp.dragged() {
            let d = resp.drag_delta();
            self.preview_yaw   += d.x * 0.007;
            self.preview_pitch  = (self.preview_pitch + d.y * 0.007).clamp(-1.5, 1.5);
        }

        // Subtle border
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
                "Select an entity from the list,\nor create a new one with + New Entity",
                egui::FontId::proportional(14.0),
                Color32::from_gray(100),
            );
        }
    }

    // Draws the 2D info bar and light-source overlays on top of the wgpu entity render.
    fn draw_preview_overlay(&self, painter: &egui::Painter, rect: Rect) {
        let def = &self.edit_def;

        // Semi-transparent info bar at bottom
        let bar_h    = 56.0;
        let bar_rect = Rect::from_min_max(
            Pos2::new(rect.min.x, rect.max.y - bar_h),
            rect.max,
        );
        painter.rect_filled(bar_rect, 0.0, Color32::from_rgba_unmultiplied(10, 10, 20, 185));

        // Entity display name
        painter.text(
            Pos2::new(rect.center().x, rect.max.y - bar_h + 8.0),
            egui::Align2::CENTER_TOP,
            &def.display_name,
            egui::FontId::proportional(16.0),
            Color32::WHITE,
        );

        // ID + registry index
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

        // Light count tag
        if !def.light_sources.is_empty() {
            let tag_text = format!("{} light(s)", def.light_sources.len());
            let tag_y    = rect.max.y - 10.0;
            let tag_w    = 70.0;
            let tag_h    = 14.0;
            let tag_x    = rect.center().x - tag_w * 0.5;
            let tr       = Rect::from_min_size(
                Pos2::new(tag_x, tag_y - tag_h),
                Vec2::new(tag_w, tag_h),
            );
            painter.rect_filled(tr, 3.0, Color32::from_rgb(200, 160, 40));
            painter.text(
                tr.center(), egui::Align2::CENTER_CENTER,
                tag_text, egui::FontId::proportional(9.0), Color32::WHITE,
            );
        }

        // Light source arrows (projected into 2D)
        if !def.light_sources.is_empty() {
            let mvp = preview_mvp(
                self.preview_yaw,
                self.preview_pitch,
                rect.width() / rect.height().max(1.0),
            );
            for ls in &def.light_sources {
                let col        = color_f32_to_egui(ls.color);
                let col_bright = Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 220);
                let col_dim    = Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 100);

                let Some(origin_s) = project_to_rect(&mvp, ls.position, rect) else { continue };

                match ls.kind {
                    LightKind::Point => {
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
                        let tip_pos = [
                            ls.position[0] + ls.direction[0] * 0.5,
                            ls.position[1] + ls.direction[1] * 0.5,
                            ls.position[2] + ls.direction[2] * 0.5,
                        ];
                        if let Some(tip_s) = project_to_rect(&mvp, tip_pos, rect) {
                            painter.arrow(origin_s, tip_s - origin_s,
                                egui::Stroke::new(2.0, col_bright));
                            let half_angle = ls.focus.to_radians() * 0.5;
                            let shaft      = tip_s - origin_s;
                            if shaft.length() > 2.0 {
                                let perp = Vec2::new(-shaft.y, shaft.x).normalized();
                                let dist = shaft.length() * half_angle.tan();
                                painter.line_segment([origin_s, tip_s + perp * dist],
                                    egui::Stroke::new(1.0, col_dim));
                                painter.line_segment([origin_s, tip_s - perp * dist],
                                    egui::Stroke::new(1.0, col_dim));
                            }
                        }
                        painter.circle_filled(origin_s, 4.0, col_bright);
                    }
                }

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
    // Inspector (right panel)
    // -----------------------------------------------------------------------

    fn draw_inspector(&mut self, ui: &mut Ui) {
        if self.selected_key.is_none() && !self.is_new {
            ui.vertical_centered(|ui| {
                ui.add_space(40.0);
                ui.label(RichText::new("No entity selected").color(Color32::GRAY));
            });
            return;
        }

        // Tab bar
        ui.horizontal(|ui| {
            for tab in EditorTab::ALL {
                let active = self.active_tab == tab;
                let label  = if active {
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
            match self.active_tab {
                EditorTab::Basic   => self.draw_tab_basic(ui),
                EditorTab::Lights  => self.draw_tab_lights(ui),
            }
        });
    }

    // -----------------------------------------------------------------------
    // Tab: Basic
    // -----------------------------------------------------------------------

    fn draw_tab_basic(&mut self, ui: &mut Ui) {
        section_header(ui, "Entity Definition");
        ui.add_space(4.0);

        // ID (read-only for existing, editable for new)
        ui.horizontal(|ui| {
            ui.label(RichText::new("ID").size(11.0).color(Color32::LIGHT_GRAY))
                .on_hover_text("Internal identifier used as filename key. Cannot be changed after saving.");
            if self.is_new {
                if ui.text_edit_singleline(&mut self.edit_def.id).changed() {
                    self.dirty = true;
                }
            } else {
                ui.label(RichText::new(&self.edit_def.id).monospace().color(Color32::LIGHT_GRAY));
            }
        });

        ui.add_space(4.0);

        // Display Name
        ui.horizontal(|ui| {
            ui.label(RichText::new("Display Name").size(11.0).color(Color32::LIGHT_GRAY))
                .on_hover_text("Human-readable name shown in the editor.");
            if ui.text_edit_singleline(&mut self.edit_def.display_name).changed() {
                self.dirty = true;
            }
        });

        // Description
        ui.label(RichText::new("Description").size(11.0).color(Color32::LIGHT_GRAY));
        if ui.text_edit_multiline(&mut self.edit_def.description).changed() {
            self.dirty = true;
        }

        ui.add_space(6.0);
        section_header(ui, "Model & Skin");
        ui.add_space(4.0);

        // Model Path
        ui.label(RichText::new("Model Path  (Bedrock .geo.json)").size(11.0).color(Color32::LIGHT_GRAY))
            .on_hover_text("Path to the Bedrock-format .geo.json model file relative to the project root.");
        ui.horizontal(|ui| {
            if ui.text_edit_singleline(&mut self.edit_def.model_path).changed() {
                self.dirty = true;
                self.cached_model_path.clear(); // force bone cache refresh on next access
            }
            if ui.small_button("Browse…").clicked() {
                if let Some(p) = pick_json_path() {
                    self.edit_def.model_path = p;
                    self.dirty = true;
                    self.cached_model_path.clear();
                }
            }
        });

        // Skin Path
        ui.add_space(4.0);
        ui.label(RichText::new("Skin Path  (optional PNG)").size(11.0).color(Color32::LIGHT_GRAY))
            .on_hover_text("Path to the skin texture PNG. Leave empty to use a grey placeholder.");
        let mut skin_str = self.edit_def.skin_path.clone().unwrap_or_default();
        ui.horizontal(|ui| {
            if ui.text_edit_singleline(&mut skin_str).changed() {
                self.edit_def.skin_path = if skin_str.is_empty() { None } else { Some(skin_str.clone()) };
                self.dirty = true;
            }
            if ui.small_button("Browse…").clicked() {
                if let Some(p) = pick_png_path() {
                    self.edit_def.skin_path = Some(p);
                    self.dirty = true;
                }
            }
            if self.edit_def.skin_path.is_some() {
                if ui.small_button("Clear").clicked() {
                    self.edit_def.skin_path = None;
                    self.dirty = true;
                }
            }
        });

        // Default Scale
        ui.add_space(6.0);
        ui.label(RichText::new("Default Scale").size(11.0).color(Color32::LIGHT_GRAY))
            .on_hover_text("Uniform scale applied to all instances of this entity (1.0 = normal size).");
        if ui.add(egui::Slider::new(&mut self.edit_def.default_scale, 0.1_f32..=4.0)
            .text("scale")).changed()
        {
            self.dirty = true;
        }
    }

    // -----------------------------------------------------------------------
    // Tab: Lights
    // -----------------------------------------------------------------------

    fn draw_tab_lights(&mut self, ui: &mut Ui) {
        section_header(ui, "Entity Light Sources");
        ui.label(RichText::new(
            "Attach point or directional lights to this entity type.\n\
             Positions are relative to the entity origin (Y=0 at feet, Y=1.8 at head).\n\
             Arrows are drawn in the preview to visualize each source."
        ).small().color(Color32::GRAY));
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            if ui.add(egui::Button::new(egui::RichText::new("+ Add Point Light").size(13.0))
                .min_size(egui::vec2(160.0, 26.0))).clicked()
            {
                self.edit_def.light_sources.push(EntityLightSource {
                    kind: LightKind::Point, ..Default::default()
                });
                self.dirty = true;
            }
            if ui.add(egui::Button::new(egui::RichText::new("+ Add Directional").size(13.0))
                .min_size(egui::vec2(160.0, 26.0))).clicked()
            {
                self.edit_def.light_sources.push(EntityLightSource {
                    kind: LightKind::Directional, ..Default::default()
                });
                self.dirty = true;
            }
        });

        ui.add_space(6.0);

        let mut to_remove: Option<usize> = None;
        let count = self.edit_def.light_sources.len();

        // Ensure bone cache is up-to-date before rendering each light entry
        self.refresh_bones_cache();
        let bones_snapshot: Vec<String> = self.cached_bones.clone();

        for i in 0..count {
            let ls = &mut self.edit_def.light_sources[i];
            let kind_label = match ls.kind {
                LightKind::Point       => "Point",
                LightKind::Directional => "Directional",
            };
            let bone_label = if ls.bone.is_empty() { "(entity origin)" } else { &ls.bone };
            let header = if ls.label.is_empty() {
                format!("Light #{} ({}) @ {}", i + 1, kind_label, bone_label)
            } else {
                format!("#{} '{}' ({}) @ {}", i + 1, ls.label, kind_label, bone_label)
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
                            .on_hover_text("Optional descriptive name (editor only).");
                        if ui.text_edit_singleline(&mut ls.label).changed() { self.dirty = true; }
                    });

                    // Bone parent
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Bone").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("Parent bone. Position is relative to the bone pivot.\nEmpty = entity origin.");
                        let bone_selected = if ls.bone.is_empty() { "(entity origin)".to_string() } else { ls.bone.clone() };
                        egui::ComboBox::from_id_salt(format!("els_bone_{}", i))
                            .selected_text(&bone_selected)
                            .show_ui(ui, |ui| {
                                if ui.selectable_value(&mut ls.bone, String::new(), "(entity origin)").changed() {
                                    self.dirty = true;
                                }
                                for bone_name in &bones_snapshot {
                                    if ui.selectable_value(&mut ls.bone, bone_name.clone(), bone_name).changed() {
                                        self.dirty = true;
                                    }
                                }
                            });
                        if bones_snapshot.is_empty() {
                            ui.label(RichText::new("(load model to see bones)").small().color(Color32::DARK_GRAY));
                        }
                    });

                    // Kind
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Type").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("Point: radiates in all directions.\nDirectional: focused cone.");
                        egui::ComboBox::from_id_salt(format!("els_kind_{}", i))
                            .selected_text(kind_label)
                            .show_ui(ui, |ui| {
                                if ui.selectable_value(&mut ls.kind, LightKind::Point,       "Point").changed()       { self.dirty = true; }
                                if ui.selectable_value(&mut ls.kind, LightKind::Directional, "Directional").changed() { self.dirty = true; }
                            });
                    });

                    ui.add_space(4.0);

                    // Position
                    ui.label(RichText::new("Position  (blocks, relative to entity origin)").size(11.0).color(Color32::LIGHT_GRAY))
                        .on_hover_text("X = East/West, Y = Up/Down (Y=0 at feet, Y≈1.8 at head), Z = North/South.\nDrag arrows in the preview for visual feedback.");
                    ui.horizontal(|ui| {
                        ui.label("X"); if ui.add(egui::Slider::new(&mut ls.position[0], -2.0_f32..=2.0).text("")).changed() { self.dirty = true; }
                        ui.label("Y"); if ui.add(egui::Slider::new(&mut ls.position[1], -0.5_f32..=2.5).text("")).changed() { self.dirty = true; }
                        ui.label("Z"); if ui.add(egui::Slider::new(&mut ls.position[2], -2.0_f32..=2.0).text("")).changed() { self.dirty = true; }
                    });

                    // Nudge arrows
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Nudge:").size(11.0).color(Color32::GRAY));
                        let step = 0.1_f32;
                        for (label, axis, sign, tip) in [
                            ("←X", 0usize, -1f32, "Move West (−X)"),
                            ("→X", 0,  1.0,       "Move East (+X)"),
                            ("↓Y", 1, -1.0,       "Move Down (−Y)"),
                            ("↑Y", 1,  1.0,       "Move Up (+Y)"),
                            ("←Z", 2, -1.0,       "Move North (−Z)"),
                            ("→Z", 2,  1.0,       "Move South (+Z)"),
                        ] {
                            if ui.small_button(label).on_hover_text(tip).clicked() {
                                ls.position[axis] = (ls.position[axis] + sign * step).clamp(-2.0, 2.5);
                                self.dirty = true;
                            }
                        }
                    });

                    ui.add_space(4.0);

                    // Color
                    ui.label(RichText::new("Color").size(11.0).color(Color32::LIGHT_GRAY))
                        .on_hover_text("RGB color of the emitted light (0..1 per channel).");
                    if color_edit_3(ui, &mut ls.color) { self.dirty = true; }

                    // Intensity
                    ui.label(RichText::new("Intensity").size(11.0).color(Color32::LIGHT_GRAY))
                        .on_hover_text("Brightness multiplier (1.0 = standard).");
                    if ui.add(egui::Slider::new(&mut ls.intensity, 0.0_f32..=10.0).text("intensity")).changed() { self.dirty = true; }

                    // Billboard glow radius
                    ui.label(RichText::new("Billboard radius (blocks, 0 = auto)").size(11.0).color(Color32::LIGHT_GRAY))
                        .on_hover_text("Visual size of the in-world glow sphere drawn at the light position.\n0 = derived from intensity. Does NOT affect lighting itself.");
                    if ui.add(egui::Slider::new(&mut ls.billboard_radius, 0.0_f32..=4.0).text("radius")).changed() { self.dirty = true; }

                    // Range (point only)
                    if ls.kind == LightKind::Point {
                        ui.label(RichText::new("Range (blocks, 0 = default)").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("Maximum illumination distance. 0 uses the global default.");
                        if ui.add(egui::Slider::new(&mut ls.range, 0.0_f32..=64.0).text("range")).changed() { self.dirty = true; }
                    }

                    // Direction + Focus (directional only)
                    if ls.kind == LightKind::Directional {
                        ui.add_space(4.0);
                        ui.label(RichText::new("Direction  (normalized XYZ)").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("Direction the cone points toward. Auto-normalized.");
                        ui.horizontal(|ui| {
                            let changed =
                                ui.add(egui::Slider::new(&mut ls.direction[0], -1.0_f32..=1.0).text("X")).changed()
                                | ui.add(egui::Slider::new(&mut ls.direction[1], -1.0_f32..=1.0).text("Y")).changed()
                                | ui.add(egui::Slider::new(&mut ls.direction[2], -1.0_f32..=1.0).text("Z")).changed();
                            if changed {
                                let len = (ls.direction[0].powi(2) + ls.direction[1].powi(2) + ls.direction[2].powi(2)).sqrt().max(1e-6);
                                ls.direction[0] /= len; ls.direction[1] /= len; ls.direction[2] /= len;
                                self.dirty = true;
                            }
                        });

                        ui.label(RichText::new("Focus angle (degrees, 0..180)").size(11.0).color(Color32::LIGHT_GRAY))
                            .on_hover_text("Half-angle of the light cone.\n0° = laser, 30° = spotlight, 90° = hemisphere, 180° = full sphere.");
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
    // New-entity dialog
    // -----------------------------------------------------------------------

    fn draw_new_entity_dialog(&mut self, ctx: &egui::Context) {
        if !self.new_dlg.open { return; }

        egui::Window::new("New Entity")
            .collapsible(false)
            .resizable(false)
            .min_width(320.0)
            .show(ctx, |ui| {
                ui.label("Enter a unique identifier for the new entity type.");
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("ID:");
                    ui.text_edit_singleline(&mut self.new_dlg.id);
                });
                ui.horizontal(|ui| {
                    ui.label("Display Name:");
                    ui.text_edit_singleline(&mut self.new_dlg.name);
                });

                if !self.new_dlg.err.is_empty() {
                    ui.colored_label(Color32::LIGHT_RED, &self.new_dlg.err.clone());
                }

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Create").clicked() {
                        let id = self.new_dlg.id.trim().to_string();
                        if id.is_empty() {
                            self.new_dlg.err = "ID cannot be empty.".into();
                        } else if self.entity_defs.contains_key(&id) {
                            self.new_dlg.err = format!("'{}' already exists.", id);
                        } else {
                            let mut def = EntityDefinition::new(&id);
                            if !self.new_dlg.name.trim().is_empty() {
                                def.display_name = self.new_dlg.name.trim().to_string();
                            }
                            self.edit_def     = def;
                            self.edit_key     = id.clone();
                            self.selected_key = None;
                            self.is_new       = true;
                            self.dirty        = true;
                            self.new_dlg.open = false;
                            self.active_tab   = EditorTab::Basic;
                            debug_log!("EntityEditor", "draw_new_entity_dialog",
                                "Created new entity draft '{}'", id);
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        self.new_dlg.open = false;
                    }
                });
            });
    }
}

// ---------------------------------------------------------------------------
// Light-source projection helpers (must match EntityPreviewRenderer MVP)
// ---------------------------------------------------------------------------

fn preview_mvp(yaw: f32, pitch: f32, aspect: f32) -> Mat4 {
    let proj  = Mat4::perspective_rh(50.0_f32.to_radians(), aspect, 0.05, 100.0);
    let eye   = Vec3::new(0.0, 0.0, 3.0);
    let view  = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let model = Mat4::from_rotation_y(yaw)
        * Mat4::from_rotation_x(pitch)
        * Mat4::from_translation(Vec3::new(0.0, -0.9, 0.0));
    proj * view * model
}

fn project_to_rect(mvp: &Mat4, pos3: [f32; 3], rect: Rect) -> Option<Pos2> {
    let clip = *mvp * Vec4::new(pos3[0], pos3[1], pos3[2], 1.0);
    if clip.w <= 0.0 { return None; }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let px = rect.min.x + (ndc_x + 1.0) * 0.5 * rect.width();
    let py = rect.min.y + (1.0 - ndc_y) * 0.5 * rect.height();
    Some(Pos2::new(px, py))
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

fn section_header(ui: &mut Ui, label: &str) {
    ui.separator();
    ui.label(RichText::new(label).strong().size(13.0));
    ui.add_space(2.0);
}

fn color_edit_3(ui: &mut Ui, color: &mut [f32; 3]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui.add(egui::Slider::new(&mut color[0], 0.0_f32..=1.0).text("R")).changed();
        changed |= ui.add(egui::Slider::new(&mut color[1], 0.0_f32..=1.0).text("G")).changed();
        changed |= ui.add(egui::Slider::new(&mut color[2], 0.0_f32..=1.0).text("B")).changed();

        // Color preview swatch
        let col = color_f32_to_egui(*color);
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(18.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 3.0, col);
    });
    changed
}

fn color_f32_to_egui(c: [f32; 3]) -> Color32 {
    Color32::from_rgb(
        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
    )
}

fn pick_json_path() -> Option<String> {
    let assets_dir = std::fs::canonicalize("assets").ok()?;
    let file = rfd::FileDialog::new()
        .set_title("Select Bedrock model (.geo.json)")
        .add_filter("JSON model", &["json"])
        .set_directory(&assets_dir)
        .pick_file()?;
    let canon = std::fs::canonicalize(&file).ok()?;
    if let Ok(rel) = canon.strip_prefix(std::fs::canonicalize(".").ok()?) {
        return Some(rel.to_string_lossy().replace('\\', "/"));
    }
    Some(file.to_string_lossy().replace('\\', "/"))
}

fn pick_png_path() -> Option<String> {
    let assets_dir = std::fs::canonicalize("assets").ok()?;
    let file = rfd::FileDialog::new()
        .set_title("Select skin texture (PNG)")
        .add_filter("PNG image", &["png"])
        .set_directory(&assets_dir)
        .pick_file()?;
    let canon = std::fs::canonicalize(&file).ok()?;
    if let Ok(rel) = canon.strip_prefix(std::fs::canonicalize(".").ok()?) {
        return Some(rel.to_string_lossy().replace('\\', "/"));
    }
    Some(file.to_string_lossy().replace('\\', "/"))
}
