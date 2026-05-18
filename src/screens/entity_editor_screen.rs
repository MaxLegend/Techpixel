// =============================================================================
// QubePixel — EntityEditorScreen
//
// Screen wrapper for the entity light definition editor.
// Owns the EntityPreviewRenderer and forwards wgpu resources every frame.
// =============================================================================

use crate::core::screen::{Screen, ScreenAction};
use crate::screens::entity_editor::EntityEditor;
use crate::screens::entity_preview_renderer::EntityPreviewRenderer;
use crate::{debug_log, flow_debug_log};
use winit::event::WindowEvent;

pub struct EntityEditorScreen {
    editor:         EntityEditor,
    preview:        Option<EntityPreviewRenderer>,
    pending_action: ScreenAction,
}

impl EntityEditorScreen {
    pub fn new() -> Self {
        debug_log!("EntityEditorScreen", "new", "Creating EntityEditorScreen");
        Self {
            editor:         EntityEditor::new(),
            preview:        None,
            pending_action: ScreenAction::None,
        }
    }
}

impl Screen for EntityEditorScreen {
    fn name(&self) -> &str { "EntityEditor" }

    fn load(&mut self) {
        debug_log!("EntityEditorScreen", "load", "Loading EntityEditorScreen");
    }

    fn start(&mut self) {
        debug_log!("EntityEditorScreen", "start", "First-frame init");
        self.editor.visible = true;
    }

    fn update(&mut self, _dt: f64) {
        flow_debug_log!("EntityEditorScreen", "update", "Update tick");
    }

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
        // Lazy-initialize the renderer on first call when we have real device/format.
        if self.preview.is_none() {
            debug_log!("EntityEditorScreen", "render", "Initializing EntityPreviewRenderer");
            self.preview = Some(EntityPreviewRenderer::new(device, queue, format));
        }

        let rect = match self.editor.preview_rect {
            Some(r) if r.width() > 4.0 && r.height() > 4.0 => r,
            _ => return,
        };

        let renderer = self.preview.as_mut().unwrap();

        // Sync orbit camera from editor UI state
        renderer.yaw   = self.editor.preview_yaw;
        renderer.pitch = self.editor.preview_pitch;

        renderer.render(
            encoder,
            view,
            device,
            queue,
            &self.editor.edit_def,
            rect,
            self.editor.pixels_per_point,
            width,
            height,
        );
    }

    fn post_process(&mut self, _dt: f64) {}

    fn build_ui(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            debug_log!("EntityEditorScreen", "build_ui", "Escape -> Pop");
            self.pending_action = ScreenAction::Pop;
            return;
        }

        let should_pop = self.editor.draw_egui(ctx);
        if should_pop {
            debug_log!("EntityEditorScreen", "build_ui", "Back button -> Pop");
            self.pending_action = ScreenAction::Pop;
        }
    }

    fn on_event(&mut self, _event: &WindowEvent) {}

    fn poll_action(&mut self) -> ScreenAction {
        std::mem::replace(&mut self.pending_action, ScreenAction::None)
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color { r: 0.04, g: 0.04, b: 0.09, a: 1.0 }
    }
}
