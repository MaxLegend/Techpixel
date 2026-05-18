// =============================================================================
// QubePixel — EntityPreviewRenderer
//
// Lightweight wgpu renderer for the entity light editor preview panel.
// Renders a static (T-pose) skeletal model using Lambert + ambient shading.
// No VCT/GI required — editor-only.
//
// Orbit camera controlled by yaw / pitch (set by EntityEditorScreen each frame).
// Viewport + scissor constrain rendering to the preview panel rect.
// =============================================================================

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::core::entity_definition::EntityDefinition;
use crate::core::player_model::PlayerModel;
use crate::debug_log;

// ---------------------------------------------------------------------------
// Shader
// ---------------------------------------------------------------------------

const ENTITY_PREVIEW_SHADER: &str = r#"
struct Uniforms {
    mvp:       mat4x4<f32>,   // 0..64
    model_mat: mat4x4<f32>,   // 64..128
    light_dir: vec4<f32>,     // 128..144  xyz = key-light direction
    tint:      vec4<f32>,     // 144..160  rgb = color multiplier, a = ambient level
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var t_skin: texture_2d<f32>;
@group(1) @binding(1) var s_skin: sampler;

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) uv:       vec2<f32>,
    @location(2) normal:   vec3<f32>,
}

struct VertOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv:         vec2<f32>,
    @location(1) world_n:    vec3<f32>,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    out.clip    = u.mvp * vec4<f32>(in.position, 1.0);
    out.world_n = normalize((u.model_mat * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv      = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let tex   = textureSample(t_skin, s_skin, in.uv);
    if tex.a < 0.05 { discard; }
    let L     = normalize(u.light_dir.xyz);
    let ndotl = max(dot(in.world_n, L), 0.0);
    let amb   = u.tint.a;
    let diff  = amb + ndotl * (1.0 - amb);
    let col   = tex.rgb * u.tint.rgb * diff;
    return vec4<f32>(col, tex.a);
}
"#;

// ---------------------------------------------------------------------------
// Uniform buffer — must byte-match the WGSL struct above
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Default)]
struct EntityPreviewUniforms {
    mvp:       [[f32; 4]; 4],  // 64
    model_mat: [[f32; 4]; 4],  // 64
    light_dir: [f32; 4],       // 16
    tint:      [f32; 4],       // 16
}
const _: () = assert!(std::mem::size_of::<EntityPreviewUniforms>() == 160);

// ---------------------------------------------------------------------------
// Merged model GPU buffers (all bones baked into a single buffer for T-pose)
// ---------------------------------------------------------------------------

struct EntityModelBuffers {
    vertex_buf:  wgpu::Buffer,
    index_buf:   wgpu::Buffer,
    index_count: u32,
}

// ---------------------------------------------------------------------------
// EntityPreviewRenderer
// ---------------------------------------------------------------------------

pub struct EntityPreviewRenderer {
    // Pipeline & global GPU resources
    pipeline:       wgpu::RenderPipeline,
    uniform_buf:    wgpu::Buffer,
    uniform_bg:     wgpu::BindGroup,
    skin_bg_layout: wgpu::BindGroupLayout,
    white_bg:       wgpu::BindGroup,
    sampler:        wgpu::Sampler,
    depth_view:     wgpu::TextureView,
    depth_size:     (u32, u32),

    // Orbit camera (set by EntityEditorScreen before each render call)
    pub yaw:   f32,
    pub pitch: f32,

    // Per-entity model state
    model_buffers: Option<EntityModelBuffers>,
    skin_bg:       Option<wgpu::BindGroup>,
    loaded_model:  String,
    loaded_skin:   String,
}

impl EntityPreviewRenderer {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        debug_log!("EntityPreviewRenderer", "new", "Initializing");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("entity_preview_shader"),
            source: wgpu::ShaderSource::Wgsl(ENTITY_PREVIEW_SHADER.into()),
        });

        // ---- Bind group layouts ----
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("entity_preview_uniform_bgl"),
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

        let skin_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("entity_preview_skin_bgl"),
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
            label:              Some("entity_preview_pipeline_layout"),
            bind_group_layouts: &[Some(&uniform_bgl), Some(&skin_bg_layout)],
            immediate_size:     0,
        });

        // PlayerVertex: position(3) @ 0, uv(2) @ 12, normal(3) @ 20 = 32 bytes
        let vb_layout = wgpu::VertexBufferLayout {
            array_stride: 32,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset:  0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 20, shader_location: 2 },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("entity_preview_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[vb_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Cw,
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
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // ---- Uniform buffer ----
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("entity_preview_uniforms"),
            size:               std::mem::size_of::<EntityPreviewUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("entity_preview_uniform_bg"),
            layout:  &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ---- White/grey fallback texture ----
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("entity_preview_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter:     wgpu::FilterMode::Nearest,
            min_filter:     wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let white_data: [u8; 4] = [200, 200, 210, 255];
        let white_tex  = create_rgba_texture(device, queue, 1, 1, &white_data);
        let white_view = white_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let white_bg   = make_tex_bg(device, &skin_bg_layout, &white_view, &sampler);

        // ---- Depth placeholder ----
        let (_, depth_view) = create_depth_texture(device, 1, 1);

        Self {
            pipeline,
            uniform_buf,
            uniform_bg,
            skin_bg_layout,
            white_bg,
            sampler,
            depth_view,
            depth_size: (1, 1),
            yaw:   0.4,
            pitch: 0.2,
            model_buffers: None,
            skin_bg:       None,
            loaded_model:  String::new(),
            loaded_skin:   String::new(),
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
        def:              &EntityDefinition,
        preview_rect:     egui::Rect,
        pixels_per_point: f32,
        fb_width:         u32,
        fb_height:        u32,
    ) {
        // Resize depth texture on framebuffer resize
        if self.depth_size != (fb_width, fb_height) {
            let (_, dv) = create_depth_texture(device, fb_width, fb_height);
            self.depth_view = dv;
            self.depth_size = (fb_width, fb_height);
            debug_log!("EntityPreviewRenderer", "render",
                "Depth texture resized to {}×{}", fb_width, fb_height);
        }

        // Reload model if path changed
        if def.model_path != self.loaded_model {
            let path = def.model_path.clone();
            self.load_model(device, &path);
            self.loaded_model = def.model_path.clone();
        }

        // Reload skin if path changed
        let skin_path = def.skin_path.as_deref().unwrap_or("").to_string();
        if skin_path != self.loaded_skin {
            let sp = skin_path.clone();
            self.load_skin(device, queue, &sp);
            self.loaded_skin = skin_path;
        }

        // Nothing to render without geometry
        let Some(buffers) = &self.model_buffers else { return; };
        if buffers.index_count == 0 { return; }

        // Update uniforms
        let uniforms = self.compute_uniforms(preview_rect.width(), preview_rect.height());
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Physical-pixel viewport
        let px = (preview_rect.min.x * pixels_per_point).round() as u32;
        let py = (preview_rect.min.y * pixels_per_point).round() as u32;
        let pw = (preview_rect.width()  * pixels_per_point).round() as u32;
        let ph = (preview_rect.height() * pixels_per_point).round() as u32;
        if pw == 0 || ph == 0 { return; }
        let px = px.min(fb_width.saturating_sub(1));
        let py = py.min(fb_height.saturating_sub(1));
        let pw = pw.min(fb_width.saturating_sub(px)).max(1);
        let ph = ph.min(fb_height.saturating_sub(py)).max(1);

        let skin_bg = self.skin_bg.as_ref().unwrap_or(&self.white_bg);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("entity_preview_pass"),
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
                view: &self.depth_view,
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
        rpass.set_viewport(px as f32, py as f32, pw as f32, ph as f32, 0.0, 1.0);
        rpass.set_scissor_rect(px, py, pw, ph);
        rpass.set_bind_group(0, &self.uniform_bg, &[]);
        rpass.set_bind_group(1, skin_bg, &[]);
        rpass.set_vertex_buffer(0, buffers.vertex_buf.slice(..));
        rpass.set_index_buffer(buffers.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..buffers.index_count, 0, 0..1);
    }

    // -----------------------------------------------------------------------
    // Model loading (merges all bones into a flat T-pose buffer)
    // -----------------------------------------------------------------------

    fn load_model(&mut self, device: &wgpu::Device, path: &str) {
        self.model_buffers = None;
        if path.is_empty() {
            debug_log!("EntityPreviewRenderer", "load_model", "Empty path — clearing model");
            return;
        }

        let json = match std::fs::read_to_string(path) {
            Ok(s)  => s,
            Err(e) => {
                debug_log!("EntityPreviewRenderer", "load_model", "Cannot read '{}': {}", path, e);
                return;
            }
        };

        let model = match PlayerModel::from_json(&json) {
            Ok(m)  => m,
            Err(e) => {
                debug_log!("EntityPreviewRenderer", "load_model", "Parse error for '{}': {}", path, e);
                return;
            }
        };

        // Merge all bone meshes into a single flat buffer (static T-pose)
        let mut all_verts: Vec<crate::core::player_model::PlayerVertex> = Vec::new();
        let mut all_idxs:  Vec<u32> = Vec::new();

        let mut bone_names: Vec<String> = model.bone_meshes.keys().cloned().collect();
        bone_names.sort(); // deterministic draw order

        for name in &bone_names {
            if let Some(bone) = model.bone_meshes.get(name) {
                let base = all_verts.len() as u32;
                all_verts.extend_from_slice(&bone.vertices);
                all_idxs.extend(bone.indices.iter().map(|&i| i + base));
            }
        }

        if all_verts.is_empty() {
            debug_log!("EntityPreviewRenderer", "load_model", "No geometry in '{}'", path);
            return;
        }

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("entity_preview_vbuf"),
            contents: bytemuck::cast_slice(&all_verts),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("entity_preview_ibuf"),
            contents: bytemuck::cast_slice(&all_idxs),
            usage:    wgpu::BufferUsages::INDEX,
        });

        let index_count = all_idxs.len() as u32;
        self.model_buffers = Some(EntityModelBuffers { vertex_buf, index_buf, index_count });

        debug_log!("EntityPreviewRenderer", "load_model",
            "Loaded '{}': {} verts, {} tris", path, all_verts.len(), all_idxs.len() / 3);
    }

    // -----------------------------------------------------------------------
    // Skin texture loading
    // -----------------------------------------------------------------------

    fn load_skin(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, path: &str) {
        self.skin_bg = None;
        if path.is_empty() {
            debug_log!("EntityPreviewRenderer", "load_skin", "No skin path — using grey fallback");
            return;
        }

        let img = match image::open(path) {
            Ok(i)  => i.to_rgba8(),
            Err(e) => {
                debug_log!("EntityPreviewRenderer", "load_skin", "Cannot load '{}': {}", path, e);
                return;
            }
        };

        let (w, h) = img.dimensions();
        let tex    = create_rgba_texture(device, queue, w, h, img.as_raw());
        let view   = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bg     = make_tex_bg(device, &self.skin_bg_layout, &view, &self.sampler);
        self.skin_bg = Some(bg);

        debug_log!("EntityPreviewRenderer", "load_skin", "Loaded skin '{}' {}×{}", path, w, h);
    }

    // -----------------------------------------------------------------------
    // Uniform computation
    // -----------------------------------------------------------------------

    fn compute_uniforms(&self, w_logical: f32, h_logical: f32) -> EntityPreviewUniforms {
        let aspect = (w_logical / h_logical).max(0.01);
        let proj   = Mat4::perspective_rh(50.0_f32.to_radians(), aspect, 0.05, 100.0);
        let eye    = Vec3::new(0.0, 0.0, 3.0);
        let view   = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        // Center entity at origin (player ~1.8 blocks tall, mid at ~Y=0.9)
        let model  = Mat4::from_rotation_y(self.yaw)
            * Mat4::from_rotation_x(self.pitch)
            * Mat4::from_translation(Vec3::new(0.0, -0.9, 0.0));
        let mvp    = proj * view * model;
        let light  = Vec3::new(0.45, 0.80, 0.55).normalize();

        EntityPreviewUniforms {
            mvp:       mvp.to_cols_array_2d(),
            model_mat: model.to_cols_array_2d(),
            light_dir: [light.x, light.y, light.z, 0.0],
            tint:      [1.0, 1.0, 1.0, 0.35],
        }
    }
}

// ---------------------------------------------------------------------------
// wgpu helpers
// ---------------------------------------------------------------------------

fn create_rgba_texture(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, data: &[u8]) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("entity_preview_tex"),
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
        label:           Some("entity_preview_depth"),
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
        label:   Some("entity_preview_skin_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view)  },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler)   },
        ],
    })
}
