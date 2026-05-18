#![allow(unused_mut)]
#![allow(dead_code)]
#![allow(deprecated)]
mod core;
mod screens;

use crate::screens::main_menu::MainMenuScreen;
use std::time::Instant;

use winit::event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Fullscreen;
use winit::event_loop::{ControlFlow, EventLoop};


use core::logging;
use core::input_controller::InputController;
use core::screen_manager::ScreenManager;
use crate::core::renderer::Renderer;
use crate::core::egui_manager::EguiManager;

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
struct App {
    window:           Option<winit::window::Window>,
    screen_manager:   ScreenManager,
    #[allow(dead_code)]
    input_controller: InputController,
    renderer:         Renderer,
    egui_manager: EguiManager,
    cursor_locked:    bool,
    tokio_runtime:    tokio::runtime::Runtime,

}

impl App {
    fn new(
        window: winit::window::Window,
        renderer: Renderer,
        tokio_runtime: tokio::runtime::Runtime,
    ) -> Self {
        let mut screen_manager = ScreenManager::new();
        let egui_manager = EguiManager::new(&window, &renderer.device(), renderer.surface_format());

        screen_manager.push(Box::new(MainMenuScreen::new()));
        // Создаём EguiManager ПЕРЕД Self {}, потому что нужен device и window

        Self {
            window: Some(window),
            screen_manager,
            input_controller: InputController::new(),
            renderer,
            egui_manager,              // <--- ДОБАВИТЬ
            cursor_locked: false,
            tokio_runtime,
        }
    }

    /// Apply cursor grab / visibility based on what the active screen requests.
    /// Only calls the OS API when the lock state actually changes.
    fn apply_cursor_lock(&mut self) {
        let wants = self.screen_manager.wants_pointer_lock();
        if wants == self.cursor_locked {
            return; // no change
        }

        if let Some(window) = self.window.as_ref() {
            if wants {
                // Try Locked first (hides and centres cursor), fall back to Confined.
                let result = window
                    .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined));

                if let Err(e) = result {
                    debug_log!("Main", "apply_cursor_lock", "set_cursor_grab failed: {}", e);
                } else {
                    window.set_cursor_visible(false);
                    self.cursor_locked = true;
                    debug_log!("Main", "apply_cursor_lock", "Cursor locked");
                }
            } else {
                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                window.set_cursor_visible(true);
                self.cursor_locked = false;
                debug_log!("Main", "apply_cursor_lock", "Cursor released");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
fn main() {
    if let Err(e) = core::logging::init_logger() {
        eprintln!("Ошибка инициализации логгера: {}", e);
    }

    debug_log!("Main", "init", "Starting Techpixel v0.0.1");

    // Load and apply persisted user settings (render distance, shadows, etc.)
    core::user_settings::load_and_apply();

    let event_loop = EventLoop::new()
        .expect("Failed to create winit event loop");

    debug_log!("Main", "init", "EventLoop created");

    let window = event_loop
        .create_window(
            winit::window::WindowAttributes::default()
                .with_title("Techpixel")
                .with_inner_size(winit::dpi::LogicalSize::new(840, 480)),
        )
        .expect("Failed to create winit window");

    debug_log!("Main", "init", "Window created");

    // --- Load application icon (assets/icon.png, 32×32 or 64×64 PNG) ---
    const ICON_PATH: &str = "assets/icon.png";
    if std::path::Path::new(ICON_PATH).exists() {
        match image::open(ICON_PATH) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                // winit 0.30: set_window_icon takes Option<Icon> built from rgba bytes
                let icon = winit::window::Icon::from_rgba(rgba.into_raw(), w, h);
                match icon {
                    Ok(ic) => {
                        window.set_window_icon(Some(ic));
                        debug_log!("Main", "init", "App icon loaded from {}", ICON_PATH);
                    }
                    Err(e) => {
                        debug_log!("Main", "init", "Failed to create Icon: {}", e);
                    }
                }
            }
            Err(e) => {
                debug_log!("Main", "init", "Failed to load icon {}: {}", ICON_PATH, e);
            }
        }
    } else {
        debug_log!("Main", "init", "No icon at {}, skipping", ICON_PATH);
    }

    let tokio_runtime = tokio::runtime::Runtime::new()
        .expect("Failed to create tokio runtime");
    let renderer = tokio_runtime.block_on(Renderer::new(&window));
    let mut app  = App::new(window, renderer, tokio_runtime);
    let mut last_time = Instant::now();

    debug_log!("Main", "init", "App initialised, entering event loop");

    event_loop
        .run(move |event, target| {
            target.set_control_flow(ControlFlow::Wait);
            // ---- Feed ALL events to egui first ----

            match event {
                // ---- Device events (raw mouse motion when cursor is locked) ----
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                    ..
                } => {
                    // Forward raw delta to the active screen regardless of window focus.
                    // The screen decides whether to use it (camera_locked check).
                    app.screen_manager.on_mouse_motion(dx, dy);
                }

                // ---- Window events -----------------------------------------------
                Event::WindowEvent { event, window_id } => {
                    let is_ours = app
                        .window
                        .as_ref()
                        .map(|w| w.id() == window_id)
                        .unwrap_or(false);
                    if !is_ours { return; }
                    if let Some(window) = app.window.as_ref() {
                        let _ = app.egui_manager.on_window_event(window, &event);
                    }
                    match event {
                        WindowEvent::CloseRequested => {
                            debug_log!("Main", "window_event", "CloseRequested");
                            target.exit();
                        }

                        WindowEvent::Resized(size) => {
                            debug_log!(
                                "Main", "window_event",
                                "Resized: {}x{}", size.width, size.height
                            );
                            app.renderer.resize(size);
                            let scale = app.window.as_ref()
                                .map(|w| w.scale_factor() as f32)
                                .unwrap_or(1.0);
                            app.egui_manager.update_screen_size(
                                size.width, size.height, scale,
                            );
                        }

                        WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                            debug_log!(
                                "Main", "window_event",
                                "Scale factor: {}", scale_factor
                            );
                            let s = app.renderer.size();
                            app.egui_manager.update_screen_size(
                                s.width, s.height, scale_factor as f32,
                            );
                        }

                        WindowEvent::RedrawRequested => {
                            let now = Instant::now();
                            let dt  = now.duration_since(last_time).as_secs_f64();
                            last_time = now;

                            // ── Phase 1: CPU work ─────────────────────────────────
                            // Runs while the GPU is executing the PREVIOUS frame.
                            // The GPU has already been given all commands; doing CPU
                            // work here lets us overlap and reduces get_tex() wait time.

                            let t_update = Instant::now();
                            app.screen_manager.update(dt);
                            app.screen_manager.post_process(dt);
                            crate::core::frame_timing::set_update(t_update.elapsed().as_micros());

                            app.apply_cursor_lock();

                            if let Some(window) = app.window.as_ref() {
                                if let Some(screen) = app.screen_manager.active_screen() {
                                    window.set_title(&format!("Techpixel — {}", screen.name()));
                                }
                            }

                            // Build egui UI + tessellate on CPU.
                            let t_egui = Instant::now();
                            let clipped_primitives = {
                                let size = app.renderer.size();
                                let scale = app.window.as_ref()
                                    .map(|w| w.scale_factor() as f32)
                                    .unwrap_or(1.0);
                                app.egui_manager.update_screen_size(size.width, size.height, scale);
                                if let Some(window) = app.window.as_ref() {
                                    app.egui_manager.update_viewport_info(window);
                                }
                                let raw_input = app.egui_manager
                                    .take_egui_input(app.window.as_ref().unwrap());
                                let egui_ctx = app.egui_manager.ctx().clone();
                                let full_output = egui_ctx.run(raw_input, |ctx| {
                                    app.screen_manager.build_ui(ctx);
                                });
                                app.egui_manager.end_frame_and_tessellate(
                                    app.renderer.device(),
                                    app.renderer.queue(),
                                    full_output,
                                )
                            };
                            crate::core::frame_timing::set_egui_build(t_egui.elapsed().as_micros());

                            // ── Phase 2: GPU sync ─────────────────────────────────
                            // Acquire next swapchain image. May block waiting for GPU
                            // to finish the previous frame and release a buffer slot.
                            // CPU has already done useful work above, so GPU has had
                            // more time to finish → expected shorter block than before.
                            if let Some(frame) = app.renderer.acquire_frame() {

                                // ── Phase 3: Encode ───────────────────────────────
                                let clear_color = app.screen_manager.clear_color();
                                let size        = app.renderer.size();
                                let mut encoder = app.renderer.begin_encoder();
                                app.renderer.clear_pass(&mut encoder, &frame.view, clear_color);

                                // 3D scene
                                app.screen_manager.render(
                                    &mut encoder, &frame.view,
                                    app.renderer.device(), app.renderer.queue(),
                                    app.renderer.surface_format(),
                                    size.width, size.height,
                                );

                                // egui HUD on top
                                let t_egui_render = Instant::now();
                                app.egui_manager.render_draw(
                                    &mut encoder, &frame.view,
                                    app.renderer.device(), app.renderer.queue(),
                                    &clipped_primitives,
                                );
                                crate::core::frame_timing::set_egui_render(
                                    t_egui_render.elapsed().as_micros()
                                );

                                // 3D on top of egui (inventory model previews, etc.)
                                let ppp = app.egui_manager.ctx().pixels_per_point();
                                app.screen_manager.render_post_ui(
                                    &mut encoder, &frame.view,
                                    app.renderer.device(), app.renderer.queue(),
                                    app.renderer.surface_format(),
                                    size.width, size.height, ppp,
                                );

                                // ── Phase 4: Submit ───────────────────────────────
                                // GPU starts executing immediately; CPU returns fast.
                                // Next iteration begins with Phase 1 CPU work while GPU renders.
                                app.renderer.submit_frame(encoder, frame);
                            }

                            if let Some(window) = app.window.as_ref() {
                                window.request_redraw();
                            }
                        }

                        // F11 — toggle borderless fullscreen
                        WindowEvent::KeyboardInput {
                            event: KeyEvent {
                                physical_key: PhysicalKey::Code(KeyCode::F11),
                                state: ElementState::Pressed,
                                repeat: false,
                                ..
                            },
                            ..
                        } => {
                            if let Some(window) = app.window.as_ref() {
                                let next = if window.fullscreen().is_some() {
                                    None
                                } else {
                                    Some(Fullscreen::Borderless(None))
                                };
                                window.set_fullscreen(next);
                                debug_log!("Main", "window_event",
                                    "F11: fullscreen={}", window.fullscreen().is_some());
                            }
                        }

                        other => {
                            app.screen_manager.on_event(&other);
                        }
                    }
                }

                Event::AboutToWait => {
                    if let Some(window) = app.window.as_ref() {
                        window.request_redraw();
                    }
                }

                _ => {}
            }
        })
        .expect("Event loop error");
}