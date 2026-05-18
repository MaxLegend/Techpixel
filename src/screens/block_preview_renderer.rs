// =============================================================================
// QubePixel — BlockPreviewRenderer
//
// Dedicated wgpu mini-renderer that draws an isometric/3D textured block cube
// directly into the swapchain surface at a specified screen rect.
//
// Features:
//   • Full wgpu render pipeline with WGSL shaders
//   • Lambert + Blinn-Phong + emission lighting
//   • 3 separate face texture groups (top / bottom / sides)
//   • Per-face color tint encoded in vertex data (updated CPU-side each frame)
//   • Orbit camera controlled by yaw / pitch
//   • Viewport + scissor constrains rendering to the preview panel rect
//   • Depth texture lives for the full framebuffer, re-created on resize
// =============================================================================

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::path::PathBuf;
use wgpu::util::DeviceExt;

use crate::core::gameobjects::block::BlockDefinition;
use crate::debug_log;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TEXTURES_DIR: &str = "assets/textures/simpleblocks";

// ---------------------------------------------------------------------------
// WGSL Shader
// ---------------------------------------------------------------------------

const CUBE_SHADER: &str = r#"
struct Uniforms {
    mvp:       mat4x4<f32>,   // offset   0 (64 bytes)
    model:     mat4x4<f32>,   // offset  64 (64 bytes)
    light_dir: vec4<f32>,     // offset 128 (16 bytes)  xyz = direction, w unused
    emit:      vec4<f32>,     // offset 144 (16 bytes)  rgb = color, a = intensity
    mat_props: vec4<f32>,     // offset 160 (16 bytes)  x=roughness, y=metalness
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var t_face: texture_2d<f32>;
@group(1) @binding(1) var s_face: sampler;

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) uv:       vec2<f32>,
    @location(2) normal:   vec3<f32>,
    @location(3) color:    vec4<f32>,
}

struct VertOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv:         vec2<f32>,
    @location(1) world_n:    vec3<f32>,
    @location(2) color:      vec4<f32>,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    out.clip    = u.mvp * vec4<f32>(in.position, 1.0);
    out.world_n = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv      = in.uv;
    out.color   = in.color;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    // Sample face texture
    let tex = textureSample(t_face, s_face, in.uv);

    // Lambert diffuse
    let L        = normalize(u.light_dir.xyz);
    let ndotl    = max(dot(in.world_n, L), 0.0);
    let ambient  = 0.30;
    let diffuse  = ambient + ndotl * 0.70;

    // Blinn-Phong specular
    let roughness = u.mat_props.x;
    let metalness = u.mat_props.y;
    let shininess = pow(1.0 - roughness, 3.0) * 96.0 + 2.0;
    let V         = vec3<f32>(0.0, 0.0, 1.0);   // approx fixed viewer direction
    let H         = normalize(L + V);
    let ndoth     = max(dot(in.world_n, H), 0.0);
    let spec_fac  = pow(ndoth, shininess) * (1.0 - roughness) * 0.5;
    // Metal tints specular by albedo; dielectric uses white
    let spec_col  = mix(vec3<f32>(1.0), tex.rgb * in.color.rgb, metalness);

    // Assemble lit color
    let base     = tex.rgb * in.color.rgb * diffuse;
    let specular = spec_col * spec_fac;
    let emissive = u.emit.rgb * u.emit.a;

    return vec4<f32>(base + specular + emissive, tex.a * in.color.a);
}
"#;

// ---------------------------------------------------------------------------
// Vertex struct
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone)]
struct CubeVertex {
    position: [f32; 3],
    uv:       [f32; 2],
    normal:   [f32; 3],
    color:    [f32; 4],   // RGBA face tint (same for all 4 verts in a face)
}

impl CubeVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x2,  // uv
        2 => Float32x3,  // normal
        3 => Float32x4,  // color
    ];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CubeVertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRIBS,
        }
    }
}

// ---------------------------------------------------------------------------
// Uniform buffer
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default)]
struct CubeUniforms {
    mvp:        [[f32; 4]; 4],  // 64
    model:      [[f32; 4]; 4],  // 64
    light_dir:  [f32; 4],       // 16
    emit:       [f32; 4],       // 16 (rgb + intensity)
    mat_props:  [f32; 4],       // 16 (roughness, metalness, pad, pad)
}
// Total: 176 bytes — 16-byte aligned

// ---------------------------------------------------------------------------
// Texture tracking (to avoid redundant reloads)
// ---------------------------------------------------------------------------

/// Tracks the full "base+layer1+layer2+..." key for each face group.
/// When any key changes, the corresponding texture is reloaded and
/// layers are composited in software.
#[derive(Default, Clone, PartialEq)]
struct TexKeys {
    top:    String,
    bottom: String,
    side:   String,
}

/// Build a deterministic dirty-tracking key from base + layers.
fn layer_combo_key(base: &str, layers: &[&str]) -> String {
    if layers.is_empty() {
        base.to_string()
    } else {
        format!("{}+{}", base, layers.join("+"))
    }
}

/// Load a PNG from assets/textures/simpleblocks/<key>.png into an RgbaImage.
fn load_png(key: &str) -> Option<image::RgbaImage> {
    let path = PathBuf::from(TEXTURES_DIR).join(format!("{}.png", key));
    image::open(&path).ok().map(|img| img.to_rgba8())
}

/// Composite `layers` on top of `base` using Porter-Duff "source over".
/// All images are bilinear-scaled to the base resolution before blending.
fn composite_layers(base: image::RgbaImage, layers: &[&str]) -> image::RgbaImage {
    let (w, h) = base.dimensions();
    let mut result = base;
    for &key in layers {
        let Some(layer_img) = load_png(key) else { continue };
        // Scale layer to match base size if needed
        let layer = if layer_img.dimensions() != (w, h) {
            image::imageops::resize(&layer_img, w, h, image::imageops::FilterType::Nearest)
        } else {
            layer_img
        };
        // Porter-Duff source-over blend
        for (px, py, lp) in layer.enumerate_pixels() {
            let a = lp[3] as f32 / 255.0;
            if a < 0.001 { continue; }
            let bp = result.get_pixel_mut(px, py);
            let inv_a = 1.0 - a;
            bp[0] = (bp[0] as f32 * inv_a + lp[0] as f32 * a).round().min(255.0) as u8;
            bp[1] = (bp[1] as f32 * inv_a + lp[1] as f32 * a).round().min(255.0) as u8;
            bp[2] = (bp[2] as f32 * inv_a + lp[2] as f32 * a).round().min(255.0) as u8;
            bp[3] = (bp[3] as f32 * inv_a + 255.0 * a).round().min(255.0) as u8;
        }
    }
    result
}

/// Build a dirty-tracking key for a single face: "base" or "base+layer1+layer2+...".
/// Uses `base_and_layers_for_face()` so it always matches what `load_face_bg_layered` loads.
fn face_combo_key(def: &BlockDefinition, face: u8) -> String {
    match def.base_and_layers_for_face(face) {
        None => String::new(),
        Some((base, layers)) => layer_combo_key(base, &layers),
    }
}

// ---------------------------------------------------------------------------
// BlockPreviewRenderer
// ---------------------------------------------------------------------------

pub struct BlockPreviewRenderer {
    // wgpu resources
    pipeline:         wgpu::RenderPipeline,
    vertex_buf:       wgpu::Buffer,   // VERTEX | COPY_DST (updated each frame)
    index_buf:        wgpu::Buffer,   // VERTEX (static)
    uniform_buf:      wgpu::Buffer,
    uniform_bg:       wgpu::BindGroup,
    tex_bg_layout:    wgpu::BindGroupLayout,
    sampler:          wgpu::Sampler,
    white_view:       wgpu::TextureView,  // 1×1 white fallback

    // Texture bind groups (top / bottom / side)
    top_bg:           wgpu::BindGroup,
    bottom_bg:        wgpu::BindGroup,
    side_bg:          wgpu::BindGroup,
    tex_keys:         TexKeys,

    // Depth texture (full framebuffer size)
    depth_view:       wgpu::TextureView,
    depth_size:       (u32, u32),

    // Camera orbit state (public so screen can set from mouse drag)
    pub yaw:          f32,
    pub pitch:        f32,
}

impl BlockPreviewRenderer {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    pub fn new(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Self {
        debug_log!("BlockPreviewRenderer", "new", "Initializing block preview wgpu renderer");

        // ---- Shader ----
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("block_preview_shader"),
            source: wgpu::ShaderSource::Wgsl(CUBE_SHADER.into()),
        });

        // ---- Bind group layouts ----
        let uniform_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("preview_uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let tex_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("preview_tex_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ---- Pipeline ----
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:               Some("preview_pipeline_layout"),
            bind_group_layouts:  &[Some(&uniform_bg_layout), Some(&tex_bg_layout)],
            immediate_size:      0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("block_preview_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:     &shader,
                entry_point: Some("vs_main"),
                buffers:     &[CubeVertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:     &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:  Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Option::from(true),
                depth_compare:       Some(wgpu::CompareFunction::Less),
                stencil:             Default::default(),
                bias:                Default::default(),
            }),
            multisample:     wgpu::MultisampleState::default(),
            multiview_mask:  None,
            cache:           None,
        });

        // ---- Geometry ----
        let (vertices, indices) = build_cube_vertices([1.0; 4], [1.0; 4], [1.0; 4]);
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("preview_vb"),
            contents: bytemuck::cast_slice(&vertices),
            usage:    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("preview_ib"),
            contents: bytemuck::cast_slice(&indices),
            usage:    wgpu::BufferUsages::INDEX,
        });

        // ---- Uniform buffer ----
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("preview_uniforms"),
            size:               std::mem::size_of::<CubeUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("preview_uniform_bg"),
            layout:  &uniform_bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ---- White fallback texture (1×1) ----
        let white_data: [u8; 4] = [255, 255, 255, 255];
        let white_tex = create_rgba_texture(device, queue, 1, 1, &white_data);
        let white_view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // ---- Sampler ----
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:            Some("preview_sampler"),
            address_mode_u:   wgpu::AddressMode::Repeat,
            address_mode_v:   wgpu::AddressMode::Repeat,
            mag_filter:       wgpu::FilterMode::Nearest,
            min_filter:       wgpu::FilterMode::Linear,
            mipmap_filter:    wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Default bind groups using white texture
        let top_bg    = make_tex_bg(device, &tex_bg_layout, &white_view, &sampler);
        let bottom_bg = make_tex_bg(device, &tex_bg_layout, &white_view, &sampler);
        let side_bg   = make_tex_bg(device, &tex_bg_layout, &white_view, &sampler);

        // ---- Depth texture (placeholder 1×1, resized in render) ----
        let (_, depth_view) = create_depth_texture(device, 1, 1);

        Self {
            pipeline,
            vertex_buf,
            index_buf,
            uniform_buf,
            uniform_bg,
            tex_bg_layout,
            sampler,
            white_view,
            top_bg,
            bottom_bg,
            side_bg,
            tex_keys: TexKeys::default(),
            depth_view,
            depth_size: (1, 1),
            yaw:   0.6,
            pitch: 0.4,
        }
    }

    // -----------------------------------------------------------------------
    // Main render entry
    // -----------------------------------------------------------------------

    pub fn render(
        &mut self,
        encoder:          &mut wgpu::CommandEncoder,
        surface_view:     &wgpu::TextureView,
        device:           &wgpu::Device,
        queue:            &wgpu::Queue,
        def:              &BlockDefinition,
        preview_rect:     egui::Rect,
        pixels_per_point: f32,
        fb_width:         u32,
        fb_height:        u32,
    ) {
        // ---- Recreate depth texture on framebuffer resize ----
        if self.depth_size != (fb_width, fb_height) {
            let (_, dv) = create_depth_texture(device, fb_width, fb_height);
            self.depth_view = dv;
            self.depth_size = (fb_width, fb_height);
            debug_log!("BlockPreviewRenderer", "render",
                "Depth texture resized to {}×{}", fb_width, fb_height);
        }

        // ---- Reload face textures if keys changed ----
        self.sync_textures(device, queue, def);

        // ---- Update vertex buffer with current face colors ----
        let tc = def.color_for_face(2);
        let bc = def.color_for_face(3);
        let sc = def.color_for_face(0); // sides
        let (verts, _) = build_cube_vertices(
            [tc[0], tc[1], tc[2], 1.0],
            [bc[0], bc[1], bc[2], 0.85],  // slightly darker bottom
            [sc[0], sc[1], sc[2], 1.0],
        );
        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&verts));

        // ---- Update uniforms ----
        let uniforms = self.compute_uniforms(def, preview_rect.width(), preview_rect.height());
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // ---- Compute physical-pixel viewport ----
        let x = (preview_rect.min.x * pixels_per_point).round() as u32;
        let y = (preview_rect.min.y * pixels_per_point).round() as u32;
        let w = (preview_rect.width()  * pixels_per_point).round() as u32;
        let h = (preview_rect.height() * pixels_per_point).round() as u32;

        // Guard against zero or out-of-bounds rects
        if w == 0 || h == 0 { return; }
        let x = x.min(fb_width.saturating_sub(1));
        let y = y.min(fb_height.saturating_sub(1));
        let w = w.min(fb_width.saturating_sub(x)).max(1);
        let h = h.min(fb_height.saturating_sub(y)).max(1);

        // ---- Render pass ----
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("block_preview_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:      &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
        rpass.set_scissor_rect(x, y, w, h);
        rpass.set_bind_group(0, &self.uniform_bg, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);

        // Draw top face
        rpass.set_bind_group(1, &self.top_bg, &[]);
        rpass.draw_indexed(0..6, 0, 0..1);

        // Draw bottom face
        rpass.set_bind_group(1, &self.bottom_bg, &[]);
        rpass.draw_indexed(6..12, 0, 0..1);

        // Draw 4 side faces
        rpass.set_bind_group(1, &self.side_bg, &[]);
        rpass.draw_indexed(12..36, 0, 0..1);
    }

    // -----------------------------------------------------------------------
    // Texture sync
    // -----------------------------------------------------------------------

    fn sync_textures(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, def: &BlockDefinition) {
        // Build layer-aware combo keys: "base+layer1+layer2+..." for dirty tracking.
        let new_keys = TexKeys {
            top:    face_combo_key(def, 2),
            bottom: face_combo_key(def, 3),
            side:   face_combo_key(def, 0),
        };

        if new_keys.top != self.tex_keys.top {
            self.top_bg = self.load_face_bg_layered(device, queue, def, 2);
        }
        if new_keys.bottom != self.tex_keys.bottom {
            self.bottom_bg = self.load_face_bg_layered(device, queue, def, 3);
        }
        if new_keys.side != self.tex_keys.side {
            self.side_bg = self.load_face_bg_layered(device, queue, def, 0);
        }
        self.tex_keys = new_keys;
    }

    /// Load (and composite layers for) one face, upload as a wgpu BindGroup.
    fn load_face_bg_layered(
        &self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        def:    &BlockDefinition,
        face:   u8,
    ) -> wgpu::BindGroup {
        let Some((base_key, layers)) = def.base_and_layers_for_face(face) else {
            return make_tex_bg(device, &self.tex_bg_layout, &self.white_view, &self.sampler);
        };

        let Some(base_img) = load_png(base_key) else {
            debug_log!("BlockPreviewRenderer", "load_face_bg_layered",
                "Base texture '{}' not found, using white", base_key);
            return make_tex_bg(device, &self.tex_bg_layout, &self.white_view, &self.sampler);
        };

        let composited = if layers.is_empty() {
            debug_log!("BlockPreviewRenderer", "load_face_bg_layered",
                "Loaded face {} base '{}'", face, base_key);
            base_img
        } else {
            debug_log!("BlockPreviewRenderer", "load_face_bg_layered",
                "Compositing face {} '{}' + {:?}", face, base_key, layers);
            composite_layers(base_img, &layers)
        };

        let (w, h) = composited.dimensions();
        let tex  = create_rgba_texture(device, queue, w, h, composited.as_raw());
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        make_tex_bg(device, &self.tex_bg_layout, &view, &self.sampler)
    }

    // -----------------------------------------------------------------------
    // Uniform computation
    // -----------------------------------------------------------------------

    fn compute_uniforms(&self, def: &BlockDefinition, w_logical: f32, h_logical: f32) -> CubeUniforms {
        let aspect = (w_logical / h_logical).max(0.01);

        // Perspective projection (wgpu uses right-handed coords, depth [0..1])
        let proj = Mat4::perspective_rh(50.0_f32.to_radians(), aspect, 0.05, 100.0);

        // Camera: fixed distance, looking at origin
        let eye    = Vec3::new(0.0, 0.0, 2.4);
        let view   = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);

        // Model: orbit rotation from yaw/pitch
        let model  = Mat4::from_rotation_y(self.yaw) * Mat4::from_rotation_x(self.pitch);

        let mvp = proj * view * model;

        // Fixed key light direction (top-left-front in world space)
        let light = Vec3::new(0.45, 0.80, 0.55).normalize();

        let emit_intensity = if def.emission.emit_light {
            def.emission.light_intensity.clamp(0.0, 3.0) * 0.2  // scale down for preview
        } else {
            0.0
        };

        CubeUniforms {
            mvp:       mat4_cols(mvp),
            model:     mat4_cols(model),
            light_dir: [light.x, light.y, light.z, 0.0],
            emit:      [
                def.emission.light_color[0],
                def.emission.light_color[1],
                def.emission.light_color[2],
                emit_intensity,
            ],
            mat_props: [def.material.roughness, def.material.metalness, 0.0, 0.0],
        }
    }
}

// =============================================================================
// Geometry builder
// =============================================================================

/// Build cube vertex/index data.
/// `top_col`, `bot_col`, `side_col` are [f32;4] RGBA face tints.
/// Returns (vertices, indices).
///
/// Face layout in index buffer:
///   Top:   [0..6]   → 1 face (4 verts)
///   Bot:   [6..12]  → 1 face (4 verts)
///   Sides: [12..36] → 4 faces (4 verts each)
fn build_cube_vertices(
    top_col:  [f32; 4],
    bot_col:  [f32; 4],
    side_col: [f32; 4],
) -> (Vec<CubeVertex>, Vec<u16>) {
    let mut verts: Vec<CubeVertex>  = Vec::with_capacity(24);
    let mut idxs:  Vec<u16>         = Vec::with_capacity(36);

    // Helper to push one quad (4 verts, CCW winding: 0-1-2, 0-2-3)
    let mut push_quad = |
        v: [[f32; 3]; 4],
        uvs: [[f32; 2]; 4],
        n: [f32; 3],
        col: [f32; 4],
        verts: &mut Vec<CubeVertex>,
        idxs:  &mut Vec<u16>,
    | {
        let base = verts.len() as u16;
        for i in 0..4 {
            verts.push(CubeVertex { position: v[i], uv: uvs[i], normal: n, color: col });
        }
        idxs.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    };

    // ---- Top face (+Y), vertices CCW when viewed from +Y (outside) ----
    push_quad(
        [[-0.5, 0.5,  0.5], [ 0.5, 0.5,  0.5], [ 0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]],
        [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],
        [0.0, 1.0, 0.0], top_col, &mut verts, &mut idxs,
    );

    // ---- Bottom face (-Y), vertices CCW when viewed from -Y (outside) ----
    push_quad(
        [[-0.5,-0.5, -0.5], [ 0.5,-0.5, -0.5], [ 0.5,-0.5,  0.5], [-0.5,-0.5,  0.5]],
        [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],
        [0.0,-1.0, 0.0], bot_col, &mut verts, &mut idxs,
    );

    // ---- North face (-Z) ----
    push_quad(
        [[ 0.5,-0.5,-0.5], [-0.5,-0.5,-0.5], [-0.5, 0.5,-0.5], [ 0.5, 0.5,-0.5]],
        [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]],
        [0.0, 0.0,-1.0], side_col, &mut verts, &mut idxs,
    );

    // ---- South face (+Z) ----
    push_quad(
        [[-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5], [ 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
        [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]],
        [0.0, 0.0, 1.0], side_col, &mut verts, &mut idxs,
    );

    // ---- East face (+X) ----
    push_quad(
        [[ 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5], [ 0.5, 0.5,-0.5], [ 0.5, 0.5, 0.5]],
        [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]],
        [1.0, 0.0, 0.0], side_col, &mut verts, &mut idxs,
    );

    // ---- West face (-X) ----
    push_quad(
        [[-0.5,-0.5,-0.5], [-0.5,-0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5,-0.5]],
        [[0.0,1.0],[1.0,1.0],[1.0,0.0],[0.0,0.0]],
        [-1.0, 0.0, 0.0], side_col, &mut verts, &mut idxs,
    );

    (verts, idxs)
}

// =============================================================================
// wgpu helpers
// =============================================================================

fn create_rgba_texture(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    w: u32, h: u32,
    data: &[u8],
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("preview_face_tex"),
        size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Rgba8UnormSrgb,
        usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats:    &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture:   &tex,
            mip_level: 0,
            origin:    wgpu::Origin3d::ZERO,
            aspect:    wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset:         0,
            bytes_per_row:  Some(4 * w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    tex
}

fn create_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("preview_depth"),
        size:            wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Depth32Float,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats:    &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_tex_bg(
    device:  &wgpu::Device,
    layout:  &wgpu::BindGroupLayout,
    view:    &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:   Some("preview_tex_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler)  },
        ],
    })
}

// Convert glam Mat4 to column-major [[f32;4];4] for WGSL
fn mat4_cols(m: Mat4) -> [[f32; 4]; 4] {
    let c = m.to_cols_array_2d();
    c
}
