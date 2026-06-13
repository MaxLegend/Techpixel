// =============================================================================
// QubePixel — Renderer (wgpu surface, device, queue, clear pass)
// =============================================================================

use crate::debug_log;
use std::marker::PhantomData;
use std::time::Instant;
use winit::window::Window;

/// Acquired swapchain image — holds the surface texture + view for one frame.
/// Created by `Renderer::acquire_frame()`, consumed by `Renderer::submit_frame()`.
pub struct AcquiredFrame {
    pub surface_texture: wgpu::SurfaceTexture,
    pub view:            wgpu::TextureView,
}

/// In-flight screenshot readback: GPU buffer + layout needed to decode it.
struct ScreenshotCopy {
    buffer:     wgpu::Buffer,
    padded_bpr: u32,
    width:      u32,
    height:     u32,
    format:     wgpu::TextureFormat,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    pending_size: Option<winit::dpi::PhysicalSize<u32>>,
    /// True when the surface supports COPY_SRC (required for screenshots).
    screenshot_supported: bool,
    /// Set by `request_screenshot()`; consumed on the next `submit_frame()`.
    screenshot_requested: bool,
    _window_ref: PhantomData<&'static Window>,
}

impl Renderer {
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        debug_log!(
            "Renderer", "new",
            "Initialising wgpu renderer: {}x{}", size.width, size.height
        );

        // ── Instance ──────────────────────────────────────────────────────

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags:    wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        // ── Surface ───────────────────────────────────────────────────────
        let surface = instance.create_surface(window).expect("Failed to create wgpu surface");
        let surface: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };

        debug_log!("Renderer", "new", "Surface created");

        // ── Adapter ───────────────────────────────────────────────────────
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find suitable GPU adapter");

        debug_log!("Renderer", "new", "Adapter selected: {:?}", adapter.get_info());

        // ── Device + Queue ────────────────────────────────────────────────
        // Request POLYGON_MODE_LINE if the adapter exposes it (enables wireframe debug).
        let optional_features =
            adapter.features() & wgpu::Features::POLYGON_MODE_LINE;

        // wgpu 22: request_device takes ONE argument (no trace path)
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("GPU Device"),
                    required_features: optional_features,
                    required_limits:   wgpu::Limits::default(),
                    // wgpu 22: new memory_hints field
                    experimental_features: Default::default(),
                    memory_hints:      Default::default(),
                    trace: Default::default(),
                },
            )
            .await
            .expect("Failed to create GPU device");

        debug_log!("Renderer", "new", "Device and queue created");

        // ── Surface configuration ─────────────────────────────────────────
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("Failed to get default surface config");

        config.desired_maximum_frame_latency = 3;
        // Disable vsync: CPU was blocking 4-5 ms at get_current_texture() waiting for
        // the FIFO vsync boundary despite the GPU being at <50% utilisation.
        // AutoNoVsync uses Mailbox where available, falls back to Immediate.
        config.present_mode = wgpu::PresentMode::AutoNoVsync;

        // Enable COPY_SRC on the swapchain when the backend allows it, so the
        // presented frame can be copied out for screenshots (F2).
        let screenshot_supported = surface
            .get_capabilities(&adapter)
            .usages
            .contains(wgpu::TextureUsages::COPY_SRC);
        if screenshot_supported {
            config.usage |= wgpu::TextureUsages::COPY_SRC;
        }
        surface.configure(&device, &config);

        debug_log!(
            "Renderer", "new",
            "Surface configured: {}x{}, format={:?}, present_mode={:?}",
            config.width, config.height, config.format, config.present_mode
        );

        Self {
            surface,
            device,
            queue,
            config,
            size,
            screenshot_supported,
            screenshot_requested: false,
            _window_ref: PhantomData,
            pending_size: None,
        }
    }

    // -----------------------------------------------------------------------
    // Resize
    // -----------------------------------------------------------------------

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size        = new_size;
            self.pending_size = Some(new_size);
        }
    }
    /// Apply any pending surface resize. Call before acquiring a frame.
    pub fn process_pending_resize(&mut self) {
        if let Some(size) = self.pending_size.take() {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
            debug_log!(
                "Renderer", "process_pending_resize",
                "Surface resized to {}x{}", size.width, size.height
            );
        }
    }
    // -----------------------------------------------------------------------
    // Split-phase API: acquire → encode → submit
    // ---------------------------------------------------------------------------
    // Use this instead of the closure-based render() to overlap CPU work with GPU:
    //
    //   update()              ← runs while GPU executes previous frame
    //   egui_build()          ← ditto
    //   acquire_frame()       ← may block waiting for GPU; already got a head start
    //   encode 3D + egui
    //   submit_frame()        ← GPU starts next frame immediately
    //
    // -----------------------------------------------------------------------

    /// Acquire the next swapchain image.
    /// May block for several ms waiting for the GPU to finish the previous frame.
    /// Returns `None` if the surface is lost/outdated (surface is reconfigured internally).
    pub fn acquire_frame(&mut self) -> Option<AcquiredFrame> {
        if let Some(size) = self.pending_size.take() {
            self.config.width  = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }

        let t = Instant::now();
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(st)
            | wgpu::CurrentSurfaceTexture::Suboptimal(st) => st,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(self.size);
                return None;
            }
            wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return None,
        };
        crate::core::frame_timing::set_get_texture(t.elapsed().as_micros());

        let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some(AcquiredFrame { surface_texture, view })
    }

    /// Create a fresh command encoder for the current frame.
    pub fn begin_encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Frame Encoder"),
        })
    }

    /// Begin the mandatory clear render pass and immediately drop it
    /// (clears the colour buffer before 3D content is drawn).
    pub fn clear_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        clear_color: wgpu::Color,
    ) {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Clear Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(clear_color),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
            multiview_mask:           None,
        });
    }

    /// Submit the encoded commands and present the frame.
    /// `queue.submit()` returns immediately — the GPU starts executing asynchronously.
    /// `present()` hands the image to the display compositor.
    pub fn submit_frame(&mut self, mut encoder: wgpu::CommandEncoder, frame: AcquiredFrame) {
        // Screenshot: append a texture→buffer copy of the finished frame
        // before submit, read it back after submit (one-frame hitch is fine).
        let screenshot_copy = if self.screenshot_requested {
            self.screenshot_requested = false;
            self.encode_screenshot_copy(&mut encoder, &frame.surface_texture.texture)
        } else {
            None
        };

        let t = Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.surface_texture.present();
        crate::core::frame_timing::set_submit_present(t.elapsed().as_micros());

        if let Some(copy) = screenshot_copy {
            self.finish_screenshot(copy);
        }
    }

    // -----------------------------------------------------------------------
    // Screenshot capture (F2)
    // -----------------------------------------------------------------------

    /// Request that the next presented frame be saved as a PNG into `screenshots/`.
    pub fn request_screenshot(&mut self) {
        if !self.screenshot_supported {
            debug_log!("Renderer", "request_screenshot",
                "Screenshot not supported: surface lacks COPY_SRC usage");
            return;
        }
        self.screenshot_requested = true;
        debug_log!("Renderer", "request_screenshot", "Screenshot requested for next frame");
    }

    /// Encode a copy of the swapchain texture into a freshly created readback buffer.
    /// Returns the buffer plus layout info needed to decode it after submission.
    fn encode_screenshot_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) -> Option<ScreenshotCopy> {
        let width  = self.config.width;
        let height = self.config.height;
        if width == 0 || height == 0 {
            return None;
        }

        // Rows in a texture→buffer copy must be 256-byte aligned.
        let bytes_per_pixel  = 4u32;
        let unpadded_bpr     = width * bytes_per_pixel;
        let padded_bpr       = unpadded_bpr.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                             * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Readback"),
            size:  (padded_bpr as u64) * (height as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        Some(ScreenshotCopy {
            buffer,
            padded_bpr,
            width,
            height,
            format: self.config.format,
        })
    }

    /// Map the readback buffer (blocking) and hand pixel data to a background
    /// thread that encodes and writes the PNG.
    fn finish_screenshot(&self, copy: ScreenshotCopy) {
        let slice = copy.buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        // Blocking wait — acceptable one-off hitch for a manual screenshot.
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((copy.width * copy.height * 4) as usize);
        let row_len = (copy.width * 4) as usize;
        for row in 0..copy.height as usize {
            let start = row * copy.padded_bpr as usize;
            pixels.extend_from_slice(&mapped[start..start + row_len]);
        }
        drop(mapped);
        copy.buffer.unmap();

        // Swapchain formats are usually BGRA — convert to RGBA for the PNG encoder.
        let is_bgra = matches!(
            copy.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        if is_bgra {
            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        // Alpha channel of the swapchain is undefined for an opaque window — force 255.
        for px in pixels.chunks_exact_mut(4) {
            px[3] = 255;
        }

        let (width, height) = (copy.width, copy.height);
        // PNG encoding + disk write off the render thread.
        std::thread::spawn(move || {
            let dir = std::path::PathBuf::from("screenshots");
            if let Err(e) = std::fs::create_dir_all(&dir) {
                log::error!("[Renderer][finish_screenshot] Failed to create screenshots dir: {e}");
                return;
            }
            let name = format!(
                "screenshot_{}.png",
                chrono::Local::now().format("%Y-%m-%d_%H-%M-%S")
            );
            let path = dir.join(name);
            match image::save_buffer(&path, &pixels, width, height, image::ColorType::Rgba8) {
                Ok(())  => debug_log!("Renderer", "finish_screenshot",
                    "Screenshot saved to {}", path.display()),
                Err(e) => log::error!(
                    "[Renderer][finish_screenshot] Failed to save screenshot: {e}"),
            }
        });
    }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    pub fn render<F>(
        &mut self,
        clear_color: wgpu::Color,
        draw_fn: F,
    )
    where
        F: FnOnce(
            &mut wgpu::CommandEncoder,
            &wgpu::TextureView,
            &wgpu::Device,
            &wgpu::Queue,
            wgpu::TextureFormat,
        ),
    {
        self.render_timed(clear_color, draw_fn);
    }

    /// Like `render` but instruments get_current_texture and submit+present
    /// into crate::core::frame_timing globals for the profiler.
    pub fn render_timed<F>(
        &mut self,
        clear_color: wgpu::Color,
        draw_fn: F,
    )
    where
        F: FnOnce(
            &mut wgpu::CommandEncoder,
            &wgpu::TextureView,
            &wgpu::Device,
            &wgpu::Queue,
            wgpu::TextureFormat,
        ),
    {
        if let Some(size) = self.pending_size.take() {
            self.config.width  = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }

        // --- get_current_texture: blocks until GPU finishes the previous frame ---
        let t_tex = Instant::now();
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(self.size);
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        crate::core::frame_timing::set_get_texture(t_tex.elapsed().as_micros());

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
                multiview_mask:           None,
            });
        }

        draw_fn(&mut encoder, &view, &self.device, &self.queue, self.config.format);

        // --- submit + present: CPU returns immediately, GPU starts async work ---
        let t_submit = Instant::now();
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        crate::core::frame_timing::set_submit_present(t_submit.elapsed().as_micros());
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    #[allow(dead_code)]
    pub fn device(&self) -> &wgpu::Device { &self.device }

    #[allow(dead_code)]
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }

    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> { self.size }

    #[allow(dead_code)]
    pub fn surface_format(&self) -> wgpu::TextureFormat { self.config.format }

    #[allow(dead_code)]
    pub fn config(&self) -> &wgpu::SurfaceConfiguration { &self.config }

    /// Returns true when the GPU/driver exposes PolygonMode::Line,
    /// which enables the wireframe debug overlay.
    pub fn wireframe_supported(&self) -> bool {
        self.device.features().contains(wgpu::Features::POLYGON_MODE_LINE)
    }
}