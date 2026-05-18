// =============================================================================
// QubePixel — BlockModelRenderer
// =============================================================================
// Renderer for static blocks with custom 3D models (crafting stations, etc.).
//
// Design:
//   - Dedicated renderer separate from EntityRenderer (which handles skeletal
//     entities with bone animation and is CW-winding).
//   - Uses FrontFace::Ccw to match the main chunk pipeline winding convention.
//   - Supports per-model texture atlases (packed by ModelMessenger).
//   - Each block model type is registered once; each world-position instance
//     gets its own uniform buffer (position transform + lighting).
//   - Shares the entity_pbr.wgsl shader (same PBR + VCT GI + sun/moon lighting).
//
// Usage per frame:
//   1. `register_model(device, model_id, vertices, indices, tex, view)`.
//   2. `spawn(device, pos, model_id)` / `despawn(pos)` as blocks appear/vanish.
//   3. `render(encoder, color_view, depth_view, queue, vct_bg, view_proj, ...)`.

use std::collections::HashMap;
use std::mem;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::core::player_model::PlayerVertex;

const SHADER_SRC: &str = include_str!("../shaders/entity_pbr.wgsl");

// ---------------------------------------------------------------------------
// Uniform layout (mirrors EntityUniforms in entity_pbr.wgsl)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockUniforms {
    view_proj:      [f32; 16],  // 64 bytes
    model:          [f32; 16],  // 64 bytes
    sun_direction:  [f32;  4],  // 16 bytes
    sun_color:      [f32;  4],  // 16 bytes
    moon_direction: [f32;  4],  // 16 bytes
    moon_color:     [f32;  4],  // 16 bytes
    ambient_color:  [f32;  4],  // 16 bytes
    camera_pos:     [f32;  4],  // 16 bytes
    shadow_params:  [f32;  4],  // 16 bytes  x=shadow_on, y=normal_offset
}
const _: () = assert!(mem::size_of::<BlockUniforms>() == 240);

const UNIFORM_SIZE: u64 = mem::size_of::<BlockUniforms>() as u64;

// ---------------------------------------------------------------------------
// Per-model GPU geometry
// ---------------------------------------------------------------------------

struct BlockModelGpu {
    vertex_buf:  wgpu::Buffer,
    index_buf:   wgpu::Buffer,
    index_count: u32,
    _skin_tex:   wgpu::Texture,
    skin_view:   wgpu::TextureView,
    skin_samp:   wgpu::Sampler,
}

// ---------------------------------------------------------------------------
// Per-instance GPU resources (one per world block position)
// ---------------------------------------------------------------------------

struct BlockInstanceGpu {
    model_id:    String,
    rotation:    u8,
    uniform_buf: wgpu::Buffer,
    bind_group:  wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Lighting parameters passed each frame
// ---------------------------------------------------------------------------

pub struct BlockModelLighting {
    pub view_proj:          Mat4,
    pub sun_dir:            Vec3,
    pub sun_color:          Vec3,
    pub sun_intensity:      f32,
    pub moon_dir:           Vec3,
    pub moon_color:         Vec3,
    pub moon_intensity:     f32,
    pub ambient:            Vec3,
    pub ambient_min:        f32,
    pub camera_pos:         Vec3,
    pub shadow_sun_enabled: bool,
    pub shadow_offset:      f32,
}

// ---------------------------------------------------------------------------
// BlockModelRenderer
// ---------------------------------------------------------------------------

pub struct BlockModelRenderer {
    pipeline:  wgpu::RenderPipeline,
    bgl_0:     wgpu::BindGroupLayout,
    models:    HashMap<String, BlockModelGpu>,
    instances: HashMap<(i32, i32, i32), BlockInstanceGpu>,
}

impl BlockModelRenderer {
    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /// `vct_frag_bgl` is `VCTSystem::frag_bgl` — shared Group 1 layout.
    pub fn new(
        device:       &wgpu::Device,
        format:       wgpu::TextureFormat,
        vct_frag_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("BlockModel PBR Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SRC)),
        });

        // Group 0: per-block uniform + atlas texture + sampler
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("BlockModel BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
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
            label:              Some("BlockModel Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl_0), Some(vct_frag_bgl)],
            immediate_size:     0,
        });

        // Vertex layout: PlayerVertex (position f32x3, uv f32x2, normal f32x3)
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
                    offset:          12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format:          wgpu::VertexFormat::Float32x3,
                    offset:          20,
                    shader_location: 2,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("BlockModel Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                compilation_options: Default::default(),
                buffers:             &[vb_layout.clone()],
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
                // CCW — matches the chunk pipeline winding convention.
                // Block model geometry (Java and Bedrock after winding fix) is CCW.
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:  Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Option::from(true),
                depth_compare:       Some(wgpu::CompareFunction::Less),
                stencil:             Default::default(),
                // Negative bias pulls model geometry slightly toward the camera so
                // the model's bottom face wins over the coplanar terrain top face
                // (Z-fighting would otherwise ripple at block floor level).
    bias: wgpu::DepthBiasState {
                    constant:    -4,
                    slope_scale: -1.0,
                    clamp:       0.0,
                }
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        Self {
            pipeline,
            bgl_0,
            models:    HashMap::new(),
            instances: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Model registration
    // -----------------------------------------------------------------------

    /// Register a block model type with a pre-built GPU texture (per-model atlas).
    /// Call once per model type when ModelMessenger responds with loaded geometry.
    pub fn register_model(
        &mut self,
        device:   &wgpu::Device,
        model_id: &str,
        vertices: &[PlayerVertex],
        indices:  &[u32],
        tex:      wgpu::Texture,
        view:     wgpu::TextureView,
    ) {
        if self.models.contains_key(model_id) {
            return;
        }

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some(&format!("BlockModel VB: {}", model_id)),
            contents: bytemuck::cast_slice(vertices),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some(&format!("BlockModel IB: {}", model_id)),
            contents: bytemuck::cast_slice(indices),
            usage:    wgpu::BufferUsages::INDEX,
        });

        let skin_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some(&format!("BlockModel Sampler: {}", model_id)),
            mag_filter:     wgpu::FilterMode::Nearest,
            min_filter:     wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        self.models.insert(model_id.to_string(), BlockModelGpu {
            vertex_buf:  vb,
            index_buf:   ib,
            index_count: indices.len() as u32,
            _skin_tex:   tex,
            skin_view:   view,
            skin_samp,
        });
    }

    /// Register a model with a fallback checkerboard texture (no atlas available).
    pub fn register_model_fallback(
        &mut self,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        model_id: &str,
        vertices: &[PlayerVertex],
        indices:  &[u32],
    ) {
        let (tex, view) = Self::make_checkerboard(device, queue);
        self.register_model(device, model_id, vertices, indices, tex, view);
    }

    /// Returns true if a model is already registered.
    pub fn is_registered(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    /// Returns the texture atlas view for a registered model.
    pub fn model_texture_view(&self, model_id: &str) -> Option<&wgpu::TextureView> {
        self.models.get(model_id).map(|m| &m.skin_view)
    }

    /// Returns the sampler for a registered model.
    pub fn model_sampler(&self, model_id: &str) -> Option<&wgpu::Sampler> {
        self.models.get(model_id).map(|m| &m.skin_samp)
    }

    // -----------------------------------------------------------------------
    // Instance management
    // -----------------------------------------------------------------------

    /// Spawn a block model instance at the given world block position.
    pub fn spawn(
        &mut self,
        device:   &wgpu::Device,
        pos:      (i32, i32, i32),
        model_id: &str,
        rotation: u8,
    ) {
        if self.instances.contains_key(&pos) {
            return;
        }
        let Some(model) = self.models.get(model_id) else {
            eprintln!("[BlockModelRenderer] spawn: model '{}' not registered", model_id);
            return;
        };

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some(&format!("BlockModel UB {:?}", pos)),
            size:               UNIFORM_SIZE,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some(&format!("BlockModel BG0 {:?}", pos)),
            layout:  &self.bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::TextureView(&model.skin_view),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: wgpu::BindingResource::Sampler(&model.skin_samp),
                },
            ],
        });

        self.instances.insert(pos, BlockInstanceGpu {
            model_id:    model_id.to_string(),
            rotation,
            uniform_buf,
            bind_group,
        });
    }

    /// Despawn the block model instance at the given world position.
    pub fn despawn(&mut self, pos: (i32, i32, i32)) {
        self.instances.remove(&pos);
    }

    /// Despawn all instances whose world positions fall within any of the given
    /// chunk keys (cx, cy, cz).  Call when chunks are evicted or remeshed.
    pub fn despawn_in_chunks(&mut self, chunk_keys: &[(i32, i32, i32)]) {
        if chunk_keys.is_empty() || self.instances.is_empty() {
            return;
        }
        use crate::core::config;
        let sx = config::chunk_size_x() as i32;
        let sy = config::chunk_size_y() as i32;
        let sz = config::chunk_size_z() as i32;

        let to_remove: Vec<(i32, i32, i32)> = self.instances
            .keys()
            .filter(|&&(wx, wy, wz)| {
                let ck = (wx.div_euclid(sx), wy.div_euclid(sy), wz.div_euclid(sz));
                chunk_keys.contains(&ck)
            })
            .copied()
            .collect();

        for pos in to_remove {
            self.instances.remove(&pos);
        }
    }

    /// Returns `true` if a block instance is already tracked at this position.
    pub fn has_instance(&self, pos: (i32, i32, i32)) -> bool {
        self.instances.contains_key(&pos)
    }

    /// Number of live instances.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Returns the stored rotation (0-15) for the model instance at `pos`, or 0 if absent.
    pub fn get_instance_rotation(&self, pos: (i32, i32, i32)) -> u8 {
        self.instances.get(&pos).map(|i| i.rotation).unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    /// Render all block model instances for this frame.
    pub fn render(
        &self,
        encoder:    &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        queue:      &wgpu::Queue,
        vct_bg:     &wgpu::BindGroup,
        lighting:   &BlockModelLighting,
    ) {
        if self.instances.is_empty() { return; }

        let vp = lighting.view_proj.to_cols_array();

        // Pre-write all uniform buffers before the render pass opens
        for (&(wx, wy, wz), inst) in &self.instances {
            // X and Z: center of block cell (+0.5).
            // Y: block floor — Bedrock model Y=0 is the bottom face of the block,
            // so place origin at the block's base (wy), not its centre (wy+0.5).
            let world_pos = Vec3::new(wx as f32 + 0.5, wy as f32, wz as f32 + 0.5);
            let model_mat = (Mat4::from_translation(world_pos)
                * block_rotation_matrix(inst.rotation)).to_cols_array();
let uniforms = BlockUniforms {
                view_proj:      vp,
                model:          model_mat,
                sun_direction:  [lighting.sun_dir.x,   lighting.sun_dir.y,   lighting.sun_dir.z,   lighting.sun_intensity],
                sun_color:      [lighting.sun_color.x,  lighting.sun_color.y,  lighting.sun_color.z,  0.0],
                moon_direction: [lighting.moon_dir.x,  lighting.moon_dir.y,  lighting.moon_dir.z,  lighting.moon_intensity],
                moon_color:     [lighting.moon_color.x, lighting.moon_color.y, lighting.moon_color.z, 0.0],
                ambient_color:  [lighting.ambient.x,   lighting.ambient.y,   lighting.ambient.z,   lighting.ambient_min],
                camera_pos:     [lighting.camera_pos.x, lighting.camera_pos.y, lighting.camera_pos.z, 0.0],
                shadow_params:  [
                    if lighting.shadow_sun_enabled { 1.0 } else { 0.0 },
                    lighting.shadow_offset,
                    0.0,
                    0.0,
                ],
            };
            queue.write_buffer(&inst.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        }

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("BlockModel Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                depth_slice:    None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            multiview_mask: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(1, vct_bg, &[]);

        for inst in self.instances.values() {
            let Some(model) = self.models.get(&inst.model_id) else { continue };
            rpass.set_bind_group(0, &inst.bind_group, &[]);
            rpass.set_vertex_buffer(0, model.vertex_buf.slice(..));
            rpass.set_index_buffer(model.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..model.index_count, 0, 0..1);
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn make_checkerboard(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let mut data = vec![0u8; 64 * 64 * 4];
        for y in 0..64u32 {
            for x in 0..64u32 {
                let i   = ((y * 64 + x) * 4) as usize;
                let odd = (x / 8 + y / 8) % 2 == 0;
                data[i]   = if odd { 255 } else { 80 };
                data[i+1] = 0;
                data[i+2] = if odd { 255 } else { 80 };
                data[i+3] = 255;
            }
        }
        Self::upload_rgba(device, queue, &data, 64, 64)
    }

    fn upload_rgba(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        data:   &[u8],
        w: u32, h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("BlockModel Atlas"),
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
                bytes_per_row:  Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&Default::default());
        (tex, view)
    }
}

fn block_rotation_matrix(rot: u8) -> Mat4 {
    use std::f32::consts::{PI, FRAC_PI_2};
    match rot {
        0 | 2  => Mat4::IDENTITY,
        1      => Mat4::from_rotation_y(PI),
        3      => Mat4::from_rotation_y(FRAC_PI_2),
        4      => Mat4::from_rotation_y(-FRAC_PI_2),
        5 | 11 => Mat4::from_rotation_x(PI),
        6      => Mat4::from_rotation_x(-FRAC_PI_2),   // wall north
        7      => Mat4::from_rotation_x(FRAC_PI_2),    // wall south
        8      => Mat4::from_rotation_z(-FRAC_PI_2),   // wall east
        9      => Mat4::from_rotation_z(FRAC_PI_2),    // wall west
        10     => Mat4::from_rotation_z(PI),
        12     => Mat4::from_rotation_x(PI) * Mat4::from_rotation_y(FRAC_PI_2),
        13     => Mat4::from_rotation_x(PI) * Mat4::from_rotation_y(-FRAC_PI_2),
        14     => Mat4::from_rotation_z(-FRAC_PI_2),   // axis X (log E-W)
        15     => Mat4::from_rotation_x(FRAC_PI_2),    // axis Z (log N-S)
        _      => Mat4::IDENTITY,
    }
}
