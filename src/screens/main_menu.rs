// =============================================================================
// Techpixel — MainMenuScreen (egui-based main menu with world picker)
// =============================================================================

use crate::core::screen::{Screen, ScreenAction};
use crate::{debug_log, flow_debug_log};
use winit::event::WindowEvent;
use crate::screens::game_screen::GameScreen;
use crate::screens::settings::SettingsScreen;
use crate::screens::worldgen_visualizer_screen::WorldGenVisualizerScreen;
use crate::screens::block_editor_screen::BlockEditorScreen;
use crate::screens::entity_editor_screen::EntityEditorScreen;
use crate::core::save_system::{self, WorldMetadata};

// ---------------------------------------------------------------------------
// UI sub-state
// ---------------------------------------------------------------------------

#[derive(PartialEq, Debug)]
enum MenuView {
    Main,
    WorldList,
    NewWorld,
    DeleteConfirm(String),
}

pub struct MainMenuScreen {
    pending_action: ScreenAction,
    view:           MenuView,

    // New-world form
    new_world_name: String,
    new_world_seed: String,

    // Cached world list (refreshed on every entry into WorldList view)
    worlds:         Vec<WorldMetadata>,

    // Status message (errors, success feedback)
    status_msg:     String,
}

impl MainMenuScreen {
    pub fn new() -> Self {
        debug_log!("MainMenuScreen", "new", "Creating MainMenuScreen");
        Self {
            pending_action: ScreenAction::None,
            view:           MenuView::Main,
            new_world_name: String::new(),
            new_world_seed: String::new(),
            worlds:         Vec::new(),
            status_msg:     String::new(),
        }
    }

    fn enter_world_list(&mut self) {
        self.worlds = save_system::list_worlds();
        self.status_msg.clear();
        self.view = MenuView::WorldList;
        debug_log!("MainMenuScreen", "enter_world_list", "Listed {} worlds", self.worlds.len());
    }

    fn enter_new_world(&mut self) {
        self.new_world_name = String::new();
        self.new_world_seed = save_system::random_seed().to_string();
        self.status_msg.clear();
        self.view = MenuView::NewWorld;
    }

    fn launch_game(&mut self, meta: WorldMetadata) {
        debug_log!("MainMenuScreen", "launch_game", "Launching world '{}'", meta.name);
        self.pending_action = ScreenAction::Switch(Box::new(GameScreen::new_with_world(Some(meta))));
    }
}

// ---------------------------------------------------------------------------
// Screen impl
// ---------------------------------------------------------------------------

impl Screen for MainMenuScreen {
    fn name(&self) -> &str { "MainMenu" }
    fn load(&mut self) {
        debug_log!("MainMenuScreen", "load", "Loading MainMenuScreen");
    }
    fn start(&mut self) {
        debug_log!("MainMenuScreen", "start", "First-frame init");
    }
    fn update(&mut self, _dt: f64) {
        flow_debug_log!("MainMenuScreen", "update", "Update tick");
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
        match self.view {
            MenuView::Main         => self.draw_main(ctx),
            MenuView::WorldList    => self.draw_world_list(ctx),
            MenuView::NewWorld     => self.draw_new_world(ctx),
            MenuView::DeleteConfirm(_) => self.draw_delete_confirm(ctx),
        }
    }

    fn on_event(&mut self, _event: &WindowEvent) {}

    fn poll_action(&mut self) -> ScreenAction {
        std::mem::replace(&mut self.pending_action, ScreenAction::None)
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color { r: 0.08, g: 0.08, b: 0.18, a: 1.0 }
    }
}

// ---------------------------------------------------------------------------
// Per-view draw helpers (called from build_ui)
// ---------------------------------------------------------------------------

impl MainMenuScreen {
    fn draw_main(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);
                ui.label(egui::RichText::new("Techpixel").size(52.0).color(egui::Color32::WHITE));
                ui.add_space(4.0);
                ui.label(egui::RichText::new("A Minecraft-inspired voxel engine").size(14.0).color(egui::Color32::GRAY));
                ui.add_space(40.0);

                // --- Worlds button ---
                if ui.add(egui::Button::new(egui::RichText::new("Start Game").size(22.0))
                    .min_size(egui::vec2(240.0, 50.0))).clicked()
                {
                    debug_log!("MainMenuScreen", "draw_main", "Worlds clicked");
                    self.enter_world_list();
                }

                ui.add_space(12.0);

                // Quick-play with default world (backward compat)
                if ui.add(egui::Button::new(egui::RichText::new("Quick Play Test").size(22.0))
                    .min_size(egui::vec2(240.0, 50.0))).clicked()
                {
                    debug_log!("MainMenuScreen", "draw_main", "Quick Play clicked");
                    self.pending_action = ScreenAction::Switch(Box::new(GameScreen::new()));
                }

                ui.add_space(12.0);

                if ui.add(egui::Button::new(egui::RichText::new("Settings").size(22.0))
                    .min_size(egui::vec2(240.0, 50.0))).clicked()
                {
                    debug_log!("MainMenuScreen", "draw_main", "Settings clicked");
                    self.pending_action = ScreenAction::Push(Box::new(SettingsScreen::new()));
                }

                ui.add_space(12.0);

                if ui.add(egui::Button::new(egui::RichText::new("World Gen Editor").size(22.0))
                    .min_size(egui::vec2(240.0, 50.0))).clicked()
                {
                    debug_log!("MainMenuScreen", "draw_main", "World Gen clicked");
                    self.pending_action = ScreenAction::Push(Box::new(WorldGenVisualizerScreen::new()));
                }

                ui.add_space(12.0);

                if ui.add(egui::Button::new(egui::RichText::new("Block Editor").size(22.0))
                    .min_size(egui::vec2(240.0, 50.0))).clicked()
                {
                    debug_log!("MainMenuScreen", "draw_main", "Block Editor clicked");
                    self.pending_action = ScreenAction::Push(Box::new(BlockEditorScreen::new()));
                }

                ui.add_space(12.0);

                if ui.add(egui::Button::new(egui::RichText::new("Entity Editor").size(22.0))
                    .min_size(egui::vec2(240.0, 50.0))).clicked()
                {
                    debug_log!("MainMenuScreen", "draw_main", "Entity Light Editor clicked");
                    self.pending_action = ScreenAction::Push(Box::new(EntityEditorScreen::new()));
                }

                ui.add_space(24.0);
                ui.label(egui::RichText::new("v0.0.1").size(12.0).color(egui::Color32::DARK_GRAY));
            });
        });
    }

    fn draw_world_list(&mut self, ctx: &egui::Context) {
        let mut action: Option<WorldAction> = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(24.0);
                ui.label(egui::RichText::new("Worlds").size(36.0).color(egui::Color32::WHITE));
                ui.add_space(16.0);

                if !self.status_msg.is_empty() {
                    ui.label(egui::RichText::new(&self.status_msg).color(egui::Color32::LIGHT_RED));
                    ui.add_space(8.0);
                }

                if ui.add(egui::Button::new(egui::RichText::new("+ New World").size(18.0))
                    .min_size(egui::vec2(220.0, 40.0))).clicked()
                {
                    action = Some(WorldAction::OpenNew);
                }

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(8.0);
            });

            // Scrollable world list
            egui::ScrollArea::vertical().show(ui, |ui| {
                if self.worlds.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        ui.label(egui::RichText::new("No saved worlds yet.").color(egui::Color32::GRAY));
                    });
                }
                for meta in &self.worlds {
                    ui.horizontal(|ui| {
                        ui.add_space(40.0);
                        ui.group(|ui| {
                            ui.set_min_width(500.0);
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.label(egui::RichText::new(&meta.name).size(18.0).strong());
                                    ui.label(egui::RichText::new(
                                        format!("Seed: {}  |  Last played: {}", meta.seed,
                                            meta.last_played.get(..10).unwrap_or(&meta.last_played))
                                    ).size(12.0).color(egui::Color32::GRAY));
                                });

                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.add(egui::Button::new(egui::RichText::new("\u{1F5D1}").size(18.0))
                                        .fill(egui::Color32::from_rgb(120, 30, 30))).clicked()
                                    {
                                        action = Some(WorldAction::ConfirmDelete(meta.name.clone()));
                                    }
                                    ui.add_space(4.0);
                                    if ui.add(egui::Button::new(egui::RichText::new("\u{25B6} Play").size(16.0))
                                        .min_size(egui::vec2(70.0, 30.0))).clicked()
                                    {
                                        action = Some(WorldAction::Launch(meta.clone()));
                                    }
                                });
                            });
                        });
                    });
                    ui.add_space(8.0);
                }
            });

            ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                ui.add_space(12.0);
                if ui.add(egui::Button::new(egui::RichText::new("Back").size(18.0))
                    .min_size(egui::vec2(200.0, 40.0))).clicked()
                {
                    action = Some(WorldAction::Back);
                }
            });
        });

        match action {
            Some(WorldAction::OpenNew)           => self.enter_new_world(),
            Some(WorldAction::Launch(meta))      => self.launch_game(meta),
            Some(WorldAction::ConfirmDelete(n))  => { self.view = MenuView::DeleteConfirm(n); }
            Some(WorldAction::Back)              => { self.view = MenuView::Main; }
            None => {}
        }
    }

    fn draw_new_world(&mut self, ctx: &egui::Context) {
        let mut action: Option<NewWorldAction> = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);
                ui.label(egui::RichText::new("New World").size(36.0).color(egui::Color32::WHITE));
                ui.add_space(24.0);

                if !self.status_msg.is_empty() {
                    ui.label(egui::RichText::new(&self.status_msg).color(egui::Color32::LIGHT_RED));
                    ui.add_space(8.0);
                }

                ui.label(egui::RichText::new("World Name").size(16.0).color(egui::Color32::LIGHT_GRAY));
                ui.add_space(4.0);
                ui.add(egui::TextEdit::singleline(&mut self.new_world_name)
                    .hint_text("My World")
                    .desired_width(300.0));

                ui.add_space(16.0);

                ui.label(egui::RichText::new("Seed (number)").size(16.0).color(egui::Color32::LIGHT_GRAY));
                ui.add_space(4.0);
                ui.add(egui::TextEdit::singleline(&mut self.new_world_seed)
                    .desired_width(300.0));

                ui.add_space(24.0);

                if ui.add(egui::Button::new(egui::RichText::new("Create").size(20.0))
                    .min_size(egui::vec2(200.0, 44.0))).clicked()
                {
                    action = Some(NewWorldAction::Create);
                }

                ui.add_space(12.0);

                if ui.add(egui::Button::new(egui::RichText::new("\u{2190}  Back").size(18.0))
                    .min_size(egui::vec2(200.0, 40.0))).clicked()
                {
                    action = Some(NewWorldAction::Back);
                }
            });
        });

        match action {
            Some(NewWorldAction::Create) => {
                let name = self.new_world_name.trim().to_owned();
                if name.is_empty() {
                    self.status_msg = "World name cannot be empty.".into();
                    return;
                }
                let seed: u64 = self.new_world_seed.trim()
                    .parse()
                    .unwrap_or_else(|_| save_system::random_seed());
                match save_system::create_world(&name, seed) {
                    Ok(meta) => {
                        debug_log!("MainMenuScreen", "draw_new_world", "Created '{}' seed={}", name, seed);
                        self.launch_game(meta);
                    }
                    Err(e) => {
                        self.status_msg = format!("Error: {}", e);
                    }
                }
            }
            Some(NewWorldAction::Back) => {
                self.enter_world_list();
            }
            None => {}
        }
    }

    fn draw_delete_confirm(&mut self, ctx: &egui::Context) {
        let world_name = if let MenuView::DeleteConfirm(ref n) = self.view {
            n.clone()
        } else {
            self.view = MenuView::WorldList;
            return;
        };
        let mut action: Option<DeleteAction> = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(120.0);
                ui.label(egui::RichText::new("Delete World?").size(30.0).color(egui::Color32::LIGHT_RED));
                ui.add_space(12.0);
                ui.label(egui::RichText::new(format!("\"{}\" will be permanently deleted.", world_name))
                    .size(16.0).color(egui::Color32::GRAY));
                ui.add_space(24.0);

                if !self.status_msg.is_empty() {
                    ui.label(egui::RichText::new(&self.status_msg).color(egui::Color32::LIGHT_RED));
                    ui.add_space(8.0);
                }

                if ui.add(egui::Button::new(egui::RichText::new("Delete").size(20.0))
                    .fill(egui::Color32::from_rgb(160, 40, 40))
                    .min_size(egui::vec2(180.0, 44.0))).clicked()
                {
                    action = Some(DeleteAction::Confirm);
                }
                ui.add_space(12.0);
                if ui.add(egui::Button::new(egui::RichText::new("Cancel").size(18.0))
                    .min_size(egui::vec2(180.0, 40.0))).clicked()
                {
                    action = Some(DeleteAction::Cancel);
                }
            });
        });

        match action {
            Some(DeleteAction::Confirm) => {
                match save_system::delete_world(&world_name) {
                    Ok(_)  => { self.enter_world_list(); }
                    Err(e) => { self.status_msg = format!("Delete failed: {}", e); }
                }
            }
            Some(DeleteAction::Cancel) => {
                self.enter_world_list();
            }
            None => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Action enums (local, not exported)
// ---------------------------------------------------------------------------

enum WorldAction {
    OpenNew,
    Launch(WorldMetadata),
    ConfirmDelete(String),
    Back,
}

enum NewWorldAction {
    Create,
    Back,
}

enum DeleteAction {
    Confirm,
    Cancel,
}
