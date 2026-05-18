// =============================================================================
// QubePixel — Screen trait and ScreenAction
// =============================================================================

use winit::event::WindowEvent;

#[allow(dead_code)]
pub enum ScreenAction {
    None,
    Push(Box<dyn Screen>),
    Pop,
    Switch(Box<dyn Screen>),
}

/// Lifecycle trait that every game screen must implement.
pub trait Screen {
    fn name(&self) -> &str;

    fn load(&mut self);
    fn start(&mut self);

    /// Called every frame.  `dt` is elapsed seconds since the last frame.
    fn update(&mut self, dt: f64);

    fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view:    &wgpu::TextureView,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        format:  wgpu::TextureFormat,
        width:   u32,
        height:  u32,
    ) {
        let _ = (encoder, view, device, queue, format, width, height);
    }

    fn post_process(&mut self, dt: f64);

    fn on_event(&mut self, event: &WindowEvent);

    /// Called with raw mouse motion delta from `DeviceEvent::MouseMotion`.
    ///
    /// This fires even when the cursor is locked/invisible, making it the
    /// correct source for first-person camera rotation.
    /// Default: no-op.
    fn on_mouse_motion(&mut self, _dx: f64, _dy: f64) {}

    /// Return `true` while this screen wants the OS cursor to be locked and
    /// hidden (first-person camera mode).  `main.rs` reads this every frame
    /// and calls `window.set_cursor_grab` / `set_cursor_visible` accordingly.
    /// Default: `false`.
    fn wants_pointer_lock(&self) -> bool {
        false
    }

    fn poll_action(&mut self) -> ScreenAction {
        ScreenAction::None
    }

    fn unload(&mut self) {}
    /// Build egui UI for this screen. Called every frame after 3D rendering.
    /// Override in screens that use egui for menus, HUD, popups, etc.
    fn build_ui(&mut self, _ctx: &egui::Context) {}
    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
    }
    /// Render on top of the egui overlay (after egui has drawn).
    /// Used for 3D content that must appear above the UI (e.g. inventory model previews).
    fn render_post_ui(
        &mut self,
        _encoder:          &mut wgpu::CommandEncoder,
        _view:             &wgpu::TextureView,
        _device:           &wgpu::Device,
        _queue:            &wgpu::Queue,
        _format:           wgpu::TextureFormat,
        _width:            u32,
        _height:           u32,
        _pixels_per_point: f32,
    ) {}
}