// =============================================================================
// QubePixel — LightEmitterRenderer
//
// Renders a bright camera-facing billboard glow at every active point or spot
// light position. Both types get identical treatment — just a glowing disc.
//
// Additive blending + depth test (no depth write) so glows occlude behind
// walls but never mask scene colour.
// =============================================================================

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::debug_log;

// ---------------------------------------------------------------------------
// EmitterLight — billboard-only view of a light source.
//
// LightEmitterRenderer only needs position, colour, intensity and a per-light
// radius override. Built fresh each frame by GameScreen so the caller can
// freely filter (e.g. skip the player's own lights in first-person view).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct EmitterLight {
    pub pos:             Vec3,
    pub color:           [f32; 3],
    pub intensity:       f32,
    /// Visual sphere radius (blocks). 0.0 = derive from intensity.
    pub radius_override: f32,
}

const SHADER: &str = include_str!("../shaders/light_emitter.wgsl");

// ---------------------------------------------------------------------------
// Vertex layout — position, colour (rgba; a = intensity), uv
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EmitterVertex {
    pub pos:   [f32; 3],
    pub color: [f32; 4],
    pub uv:    [f32; 2],
}

// ---------------------------------------------------------------------------
// LightEmitterRenderer
// ---------------------------------------------------------------------------

pub struct LightEmitterRenderer {
    pipeline: wgpu::RenderPipeline,
    ub:       wgpu::Buffer,
    bg:       wgpu::BindGroup,
    vb:       Option<wgpu::Buffer>,
    vb_cap:   usize,
}

impl LightEmitterRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        debug_log!("LightEmitterRenderer", "new", "Initializing light emitter renderer");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("light_emitter_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("light_emitter_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("light_emitter_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let ub = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("light_emitter_ub"),
            size:               64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("light_emitter_bg"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: ub.as_entire_binding(),
            }],
        });

        let attrs = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x4,
            2 => Float32x2,
        ];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("light_emitter_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<EmitterVertex>() as u64,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &attrs,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation:  wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare:       Some(wgpu::CompareFunction::Less),
                stencil:             Default::default(),
                bias:                Default::default(),
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        Self { pipeline, ub, bg, vb: None, vb_cap: 0 }
    }

    // -----------------------------------------------------------------------
    // Main render — builds billboard geometry then submits one draw call
    // -----------------------------------------------------------------------

    pub fn render(
        &mut self,
        encoder:    &mut wgpu::CommandEncoder,
        view:       &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        device:     &wgpu::Device,
        queue:      &wgpu::Queue,
        vp:         glam::Mat4,
        cam_right:  Vec3,
        cam_up:     Vec3,
        emitters:   &[EmitterLight],
    ) {
        let mut verts: Vec<EmitterVertex> = Vec::new();

        for em in emitters {
            let size = if em.radius_override > 0.0 {
                em.radius_override
            } else {
                billboard_size(em.intensity)
            };
            push_billboard(&mut verts, em.pos, cam_right, cam_up, size, em.color, em.intensity);
        }

        if verts.is_empty() { return; }

        queue.write_buffer(&self.ub, 0, bytemuck::cast_slice(&vp.to_cols_array()));

        if self.vb_cap < verts.len() {
            self.vb = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("light_emitter_vb"),
                size:               (verts.len() * std::mem::size_of::<EmitterVertex>()) as u64,
                usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vb_cap = verts.len();
        }

        let vb       = self.vb.as_ref().unwrap();
        let byte_len = (verts.len() * std::mem::size_of::<EmitterVertex>()) as u64;
        queue.write_buffer(vb, 0, bytemuck::cast_slice(&verts));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("light_emitter_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.set_vertex_buffer(0, vb.slice(..byte_len));
        pass.draw(0..verts.len() as u32, 0..1);


    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn billboard_size(intensity: f32) -> f32 {
    (0.25 + 0.20 * intensity.sqrt()).min(1.2)
}

fn push_billboard(
    out:       &mut Vec<EmitterVertex>,
    center:    Vec3,
    cam_right: Vec3,
    cam_up:    Vec3,
    size:      f32,
    color:     [f32; 3],
    intensity: f32,
) {
    let r  = cam_right * size * 0.5;
    let u  = cam_up    * size * 0.5;
    let tl = center - r + u;
    let tr = center + r + u;
    let bl = center - r - u;
    let br = center + r - u;
    let c  = [color[0], color[1], color[2], intensity.clamp(0.1, 8.0)];
    let v  = |p: Vec3, uv: [f32; 2]| EmitterVertex { pos: p.into(), color: c, uv };
    out.push(v(tl, [0.0, 1.0]));
    out.push(v(tr, [1.0, 1.0]));
    out.push(v(br, [1.0, 0.0]));
    out.push(v(tl, [0.0, 1.0]));
    out.push(v(br, [1.0, 0.0]));
    out.push(v(bl, [0.0, 0.0]));
}
