// =============================================================================
// QubePixel — ScreenManager (stack-based state machine)
// =============================================================================

use crate::{debug_log, ext_debug_log};
use crate::core::screen::{Screen, ScreenAction};
use winit::event::WindowEvent;

struct ScreenEntry {
    screen:     Box<dyn Screen>,
    is_started: bool,
}

pub struct ScreenManager {
    entries: Vec<ScreenEntry>,
}

impl ScreenManager {
    pub fn new() -> Self {
        debug_log!("ScreenManager", "new", "ScreenManager created");
        Self { entries: Vec::new() }
    }

    // -----------------------------------------------------------------------
    // Transitions
    // -----------------------------------------------------------------------

    pub fn push(&mut self, mut screen: Box<dyn Screen>) {
        let name = screen.name().to_owned();
        debug_log!("ScreenManager", "push", "Pushing screen: {}", name);
        screen.load();
        self.entries.push(ScreenEntry { screen, is_started: false });
        debug_log!(
            "ScreenManager", "push",
            "Stack size: {}", self.entries.len()
        );
    }

    pub fn pop(&mut self) {
        if let Some(mut entry) = self.entries.pop() {
            let name = entry.screen.name().to_owned();
            debug_log!("ScreenManager", "pop", "Popping screen: {}", name);
            entry.screen.unload();
            debug_log!(
                "ScreenManager", "pop",
                "Stack size: {}", self.entries.len()
            );
        } else {
            debug_log!("ScreenManager", "pop", "Attempted to pop from empty stack");
        }
    }

    pub fn switch(&mut self, mut screen: Box<dyn Screen>) {
        let name = screen.name().to_owned();
        debug_log!("ScreenManager", "switch", "Switching to screen: {}", name);
        while let Some(mut entry) = self.entries.pop() {
            debug_log!(
                "ScreenManager", "switch",
                "Unloading: {}", entry.screen.name()
            );
            entry.screen.unload();
        }
        screen.load();
        self.entries.push(ScreenEntry { screen, is_started: false });
        debug_log!(
            "ScreenManager", "switch",
            "Switch complete, stack size: {}", self.entries.len()
        );
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    pub fn update(&mut self, dt: f64) {
        let action = {
            if let Some(entry) = self.entries.last_mut() {
                if !entry.is_started {
                    ext_debug_log!(
                        "ScreenManager", "update",
                        "First frame — calling start() on: {}", entry.screen.name()
                    );
                    entry.screen.start();
                    entry.is_started = true;
                }
                entry.screen.update(dt);
                entry.screen.poll_action()
            } else {
                ScreenAction::None
            }
        };

        match action {
            ScreenAction::None    => {}
            ScreenAction::Push(s) => {
                debug_log!("ScreenManager", "update", "Screen requested Push");
                self.push(s);
            }
            ScreenAction::Pop => {
                debug_log!("ScreenManager", "update", "Screen requested Pop");
                self.pop();
            }
            ScreenAction::Switch(s) => {
                debug_log!("ScreenManager", "update", "Screen requested Switch");
                self.switch(s);
            }
        }
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view:    &wgpu::TextureView,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        format:  wgpu::TextureFormat,
        width:   u32,
        height:  u32,
    ) {
        if let Some(entry) = self.entries.last_mut() {
            entry.screen.render(encoder, view, device, queue, format, width, height);
        }
    }

    pub fn post_process(&mut self, dt: f64) {
        if let Some(entry) = self.entries.last_mut() {
            entry.screen.post_process(dt);
        }
    }
    /// Build egui UI for the active screen.
    pub fn build_ui(&mut self, ctx: &egui::Context) {
        if let Some(entry) = self.entries.last_mut() {
            entry.screen.build_ui(ctx);
        }
    }

    pub fn render_post_ui(
        &mut self,
        encoder:          &mut wgpu::CommandEncoder,
        view:             &wgpu::TextureView,
        device:           &wgpu::Device,
        queue:            &wgpu::Queue,
        format:           wgpu::TextureFormat,
        width:            u32,
        height:           u32,
        pixels_per_point: f32,
    ) {
        if let Some(entry) = self.entries.last_mut() {
            entry.screen.render_post_ui(encoder, view, device, queue, format, width, height, pixels_per_point);
        }
    }
    pub fn on_event(&mut self, event: &WindowEvent) {
        if let Some(entry) = self.entries.last_mut() {
            entry.screen.on_event(event);
        }
    }

    /// Forward raw mouse motion delta to the active screen.
    /// Called from `main.rs` on `DeviceEvent::MouseMotion`.
    pub fn on_mouse_motion(&mut self, dx: f64, dy: f64) {
        if let Some(entry) = self.entries.last_mut() {
            entry.screen.on_mouse_motion(dx, dy);
        }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    pub fn active_screen(&self) -> Option<&dyn Screen> {
        self.entries.last().map(|e| e.screen.as_ref())
    }

    #[allow(dead_code)]
    pub fn screen_count(&self) -> usize { self.entries.len() }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    pub fn clear_color(&self) -> wgpu::Color {
        self.entries
            .last()
            .map(|e| e.screen.clear_color())
            .unwrap_or(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 })
    }

    /// Returns `true` while the active screen wants the cursor locked.
    pub fn wants_pointer_lock(&self) -> bool {
        self.entries
            .last()
            .map(|e| e.screen.wants_pointer_lock())
            .unwrap_or(false)
    }
}