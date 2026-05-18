// =============================================================================
// QubePixel — InventoryModelRenderer
// =============================================================================
// Renders block models as isometric previews in inventory slots.
//
// Design:
//   - Own render pipeline with simplified Lambert + Blinn-Phong shader.
//   - Uses the same PlayerVertex format as BlockModelRenderer.
//   - Renders directly into the swapchain after egui using viewport/scissor.
//   - Per-model GPU resources: vertex/index buffers + bind group (shared
//     texture atlas view/sampler from BlockModelRenderer).
//   - Single large uniform buffer with dynamic offsets for per-slot data.
// =============================================================================

use std::collections::HashMap;
use std::mem;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::core::player_model::PlayerVertex;

const PREVIEW_SHADER: &str = r#"
struct Uniforms {
    view_proj:     mat4x4<f32>,
    model:         mat4x4<f32>,
    light_dir:     vec4<f32>,
    light_color:   vec4<f32>,
    ambient_color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var t_skin: texture_2d<f32>;
@group(0) @binding(2) var s_skin: sampler;

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) uv:       vec2<f32>,
    @location(2) normal:   vec3<f32>,
}

struct VertOut {
    @builtin(position) clip:    vec4<f32>,
    @location(0)         uv:    vec2<f32>,
    @location(1)         world_n: vec3<f32>,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    let world_pos = u.model * vec4<f32>(in.position, 1.0);
    out.clip    = u.view_proj * world_pos;
    out.world_n = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv      = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let tex = textureSample(t_skin, s_skin, in.uv);
    let L   = normalize(u.light_dir.xyz);
    let N   = normalize(in.world_n);
    let ndotl = max(dot(N, L), 0.0);

    let V     = vec3<f32>(0.0, 0.0, 1.0);
    let H     = normalize(L + V);
    let ndoth = max(dot(N, H), 0.0);
    let spec  = pow(ndoth, 32.0) * 0.25;

    let ambient = u.ambient_color.rgb * 0.45;
    let diffuse = u.light_color.rgb * ndotl * 0.55;
    let color   = tex.rgb * (ambient + diffuse) + vec3<f32>(spec);
    return vec4<f32>(color, tex.a);
}
"#;

const MAX_SLOTS: u32 = 256;
const DYNAMIC_ALIGN: u64 = 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PreviewUniforms {
    view_proj:     [f32; 16],
    model:         [f32; 16],
    light_dir:     [f32;  4],
    light_color:   [f32;  4],
    ambient_color: [f32;  4],
}
const _: () = assert!(mem::size_of::<PreviewUniforms>() == 176);

fn aligned_uniform_size() -> u64 {
    (mem::size_of::<PreviewUniforms>() as u64).next_multiple_of(DYNAMIC_ALIGN)
}

struct InventoryModelGpu {
    vertex_buf:  wgpu::Buffer,
    index_buf:   wgpu::Buffer,
    index_count: u32,
    bind_group:  wgpu::BindGroup,
}

pub struct ModelPreviewSlot {
    pub model_id: String,
    pub rect:     egui::Rect,
}

pub struct InventoryModelRenderer {
    pipeline:    wgpu::RenderPipeline,
    bgl:         wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    depth_view:  wgpu::TextureView,
    depth_size:  (u32, u32),
    models:      HashMap<String, InventoryModelGpu>,
}

impl InventoryModelRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("InventoryModel Shader"),
            source: wgpu::ShaderSource::Wgsl(PREVIEW_SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("InventoryModel BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size:   Some(
                            std::num::NonZeroU64::new(mem::size_of::<PreviewUniforms>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled:   false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("InventoryModel Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let vb_layout = wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<PlayerVertex>() as u64,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format:          wgpu::VertexFormat::Float32x3,
                    offset:          0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format:          wgpu::VertexFormat::Float32x2,
                    offset:          mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format:          wgpu::VertexFormat::Float32x3,
                    offset:          (mem::size_of::<[f32; 3]>() + mem::size_of::<[f32; 2]>()) as u64,
                    shader_location: 2,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("InventoryModel Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                compilation_options: Default::default(),
                buffers:             &[vb_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        let aus = aligned_uniform_size();
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:  Some("InventoryModel Uniforms"),
            size:   MAX_SLOTS as u64 * aus,
            usage:  wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (_, depth_view) = create_depth_texture(device, 1, 1);

        Self {
            pipeline,
            bgl,
            uniform_buf,
            depth_view,
            depth_size: (1, 1),
            models: HashMap::new(),
        }
    }

    pub fn register_model(
        &mut self,
        device:   &wgpu::Device,
        model_id: &str,
        vertices: &[PlayerVertex],
        indices:  &[u32],
        tex_view: &wgpu::TextureView,
        sampler:  &wgpu::Sampler,
    ) {
        if self.models.contains_key(model_id) {
            return;
        }

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some(&format!("InvModel VB: {}", model_id)),
            contents: bytemuck::cast_slice(vertices),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some(&format!("InvModel IB: {}", model_id)),
            contents: bytemuck::cast_slice(indices),
            usage:    wgpu::BufferUsages::INDEX,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some(&format!("InvModel BG: {}", model_id)),
            layout:  &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buf,
                        offset: 0,
                        size:   Some(std::num::NonZeroU64::new(mem::size_of::<PreviewUniforms>() as u64).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::TextureView(tex_view),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        self.models.insert(model_id.to_string(), InventoryModelGpu {
            vertex_buf:  vb,
            index_buf:   ib,
            index_count: indices.len() as u32,
            bind_group,
        });
    }

    pub fn is_registered(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    pub fn render_previews(
        &mut self,
        encoder:          &mut wgpu::CommandEncoder,
        surface_view:     &wgpu::TextureView,
        device:           &wgpu::Device,
        queue:            &wgpu::Queue,
        slots:            &[ModelPreviewSlot],
        fb_width:         u32,
        fb_height:        u32,
        pixels_per_point: f32,
    ) {
        if slots.is_empty() { return }

        if self.depth_size != (fb_width, fb_height) {
            let (_, dv) = create_depth_texture(device, fb_width, fb_height);
            self.depth_view = dv;
            self.depth_size = (fb_width, fb_height);
        }

        let light_dir   = Vec3::new(0.45, 0.80, 0.55).normalize();
        let light_color = [1.0f32, 0.95, 0.85, 1.0];
        let ambient     = [1.0f32, 0.98, 0.95, 1.0];

        let rot_angle = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f32()
            * 0.5;
        let base_model_mat = Mat4::from_rotation_y(rot_angle);

        let aus = aligned_uniform_size();

        let mut valid_slots: Vec<(usize, u32, u32, u32, u32, &InventoryModelGpu)> = Vec::new();

        for (i, slot) in slots.iter().enumerate() {
            if i as u32 >= MAX_SLOTS { break }
            let Some(model) = self.models.get(&slot.model_id) else { continue };

            let x = (slot.rect.min.x * pixels_per_point).round() as u32;
            let y = (slot.rect.min.y * pixels_per_point).round() as u32;
            let w = (slot.rect.width()  * pixels_per_point).round() as u32;
            let h = (slot.rect.height() * pixels_per_point).round() as u32;
            if w == 0 || h == 0 { continue }
            let x = x.min(fb_width.saturating_sub(1));
            let y = y.min(fb_height.saturating_sub(1));
            let w = w.min(fb_width.saturating_sub(x)).max(1);
            let h = h.min(fb_height.saturating_sub(y)).max(1);

            let aspect = (w as f32 / h as f32).max(0.01);
            let proj = Mat4::perspective_rh(40.0f32.to_radians(), aspect, 0.05, 100.0);
            let eye  = Vec3::new(1.2, 1.0, 1.2);
            let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 0.35, 0.0), Vec3::Y);
            let view_proj = (proj * view).to_cols_array();

            let uniforms = PreviewUniforms {
                view_proj,
                model:         base_model_mat.to_cols_array(),
                light_dir:     [light_dir.x, light_dir.y, light_dir.z, 0.0],
                light_color,
                ambient_color: ambient,
            };

            let offset = i as u64 * aus;
            queue.write_buffer(&self.uniform_buf, offset, bytemuck::bytes_of(&uniforms));

            valid_slots.push((i, x, y, w, h, model));
        }

        if valid_slots.is_empty() { return }

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("InventoryModel Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           surface_view,
                resolve_target: None,
                depth_slice:    None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
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

        for (i, x, y, w, h, model) in &valid_slots {
            let dynamic_offset = (*i as u64 * aus) as u32;
            rpass.set_viewport(*x as f32, *y as f32, *w as f32, *h as f32, 0.0, 1.0);
            rpass.set_scissor_rect(*x, *y, *w, *h);
            rpass.set_bind_group(0, &model.bind_group, &[dynamic_offset]);
            rpass.set_vertex_buffer(0, model.vertex_buf.slice(..));
            rpass.set_index_buffer(model.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..model.index_count, 0, 0..1);
        }
    }
}

fn create_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("InvModel Depth"),
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
