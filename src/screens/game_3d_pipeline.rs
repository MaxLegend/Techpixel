// =============================================================================
// QubePixel — Game3DPipeline  (camera, depth buffer, world chunk meshes)
// =============================================================================
// Simplified: no Radiance Cascades GI, no voxel shadows, no volumetric lights.
// Only static directional (sun/moon) PBR lighting with ambient.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// VRAM budget for chunk meshes.  When exceeded, the farthest chunks are
/// evicted from GPU and re-queued in the world worker for future re-meshing.
/// Tune this to match available VRAM (default: 256 MB).
const CHUNK_MESH_VRAM_BUDGET: u64 = 4096 * 1024 * 1024;
use crate::{debug_log, ext_debug_log, flow_debug_log};
use glam::{Mat4, Vec3};

// ---------------------------------------------------------------------------
// FrustumPlanes
// ---------------------------------------------------------------------------
pub struct FrustumPlanes {
    planes: [[f32; 4]; 6],
}

impl FrustumPlanes {
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let c   = vp.to_cols_array_2d();
        let row = |i: usize| -> [f32; 4] { [c[0][i], c[1][i], c[2][i], c[3][i]] };
        let r0  = row(0);
        let r1  = row(1);
        let r2  = row(2);
        let r3  = row(3);

        let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]
        };
        let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]]
        };

        Self {
            planes: [
                add(r3, r0), sub(r3, r0),
                add(r3, r1), sub(r3, r1),
                add(r3, r2), sub(r3, r2),
            ],
        }
    }

    pub fn intersects_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            let [a, b, c, d] = *plane;
            let px = if a >= 0.0 { max[0] } else { min[0] };
            let py = if b >= 0.0 { max[1] } else { min[1] };
            let pz = if c >= 0.0 { max[2] } else { min[2] };
            if a * px + b * py + c * pz + d < 0.0 {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
pub struct Camera {
    pub position: Vec3,
    yaw:          f32,
    pitch:        f32,
    aspect:       f32,
    fov:          f32,
    pub(crate) near:         f32,
    far:          f32,
    pub speed:    f32,
    sensitivity:  f32,
}

impl Camera {
    pub fn new(width: u32, height: u32) -> Self {
        debug_log!("Camera", "new", "Creating camera {}x{}", width, height);
        Self {
            position:    Vec3::new(8.0, 20.0, 30.0),
            yaw:         -std::f32::consts::FRAC_PI_2,
            pitch:       -0.30,
            aspect:      width as f32 / height.max(1) as f32,
            fov:         70.0_f32.to_radians(),
            near:        0.1,
            far:         500.0,
            speed:       10.0,
            sensitivity: 0.003,
        }


    }
    /// Синхронизирует дальнюю плоскость отсечения с радиусом прогрузки чанков
    pub fn update_render_distance(&mut self, chunk_size: f32, render_distance_chunks: u32) {
        // Запас 20%, чтобы AABB чанков на границе не обрезались впритык
        let margin = 1.2_f32;
        let new_far = render_distance_chunks as f32 * chunk_size * margin;
        // far обязан быть существенно > near, иначе проекционная матрица даст NaN/Inf
        self.far = new_far.max(self.near * 10.0);
    }
    pub fn update_aspect(&mut self, width: u32, height: u32) {
        if height > 0 { self.aspect = width as f32 / height as f32; }
    }
    pub fn fov(&self) -> f32 { self.fov }
    pub fn aspect(&self) -> f32 { self.aspect }
    pub fn near(&self) -> f32 { self.near }
    pub fn far(&self) -> f32 { self.far }
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp).normalize()
    }

    pub fn right(&self) -> Vec3 { self.forward().cross(Vec3::Y).normalize() }
    pub fn yaw(&self)   -> f32  { self.yaw }
    pub fn pitch(&self) -> f32  { self.pitch }

    pub fn move_forward(&mut self, amount: f32) { self.position += self.forward() * amount; }
    pub fn move_right(&mut self, amount: f32)   { self.position += self.right()   * amount; }
    pub fn move_up(&mut self, amount: f32)       { self.position.y += amount; }

    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.yaw  += dx as f32 * self.sensitivity;
        self.pitch = (self.pitch - dy as f32 * self.sensitivity)
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    pub fn rotate_smooth(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw  += delta_yaw;
        self.pitch = (self.pitch - delta_pitch)
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    pub fn set_sensitivity(&mut self, sens: f32) {
        self.sensitivity = sens;
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
}

// ---------------------------------------------------------------------------
// Vertex3D encoding helpers
// ---------------------------------------------------------------------------

/// Pack f32 ∈ [−1, 1] → i8 (Snorm).  Used for normals and tangents.
#[inline(always)]
pub fn snorm8(v: f32) -> i8 { (v.clamp(-1.0, 1.0) * 127.0).round() as i8 }

/// Pack f32 ∈ [0, 1] → u8 (Unorm).  Used for color, AO, roughness, metalness.
#[inline(always)]
pub fn unorm8(v: f32) -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 }

// ---------------------------------------------------------------------------
// Vertex3D — packed, 52 bytes
// ---------------------------------------------------------------------------

/// Packed vertex — 52 bytes (was 96).  All fields preserved; compressed representation.
///
/// Layout (offsets):
///   [  0] position:  Float32x3 — world XYZ
///   [ 12] normal:    Snorm8x4  — xyz = normal [-1..1],  w = unused
///   [ 16] color_ao:  Unorm8x4  — rgb = albedo [0..1],   a = AO [0..1]
///   [ 20] texcoord:  Float32x2 — UV coordinates
///   [ 28] material:  Unorm8x4  — r = roughness, g = metalness, ba = unused
///   [ 32] emission:  Float32x4 — [r, g, b, intensity]  (kept f32; can exceed 1.0)
///   [ 48] tangent:   Snorm8x4  — xyz = tangent [-1..1], w = unused
///   [ 52] — end
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3D {
    pub position:  [f32; 3],  // offset  0 | 12b | Float32x3
    pub normal:    [i8;  4],  // offset 12 |  4b | Snorm8x4
    pub color_ao:  [u8;  4],  // offset 16 |  4b | Unorm8x4
    pub texcoord:  [f32; 2],  // offset 20 |  8b | Float32x2
    pub material:  [u8;  4],  // offset 28 |  4b | Unorm8x4
    pub emission:  [f32; 4],  // offset 32 | 16b | Float32x4
    pub tangent:   [i8;  4],  // offset 48 |  4b | Snorm8x4
}                             // total: 52 bytes

impl Default for Vertex3D {
    fn default() -> Self {
        Self {
            position:  [0.0; 3],
            normal:    [0, 127, 0, 0],           // (0, 1, 0) up
            color_ao:  [255, 255, 255, 255],     // white, ao = 1.0
            texcoord:  [0.0; 2],
            material:  [unorm8(0.9), 0, 0, 0],  // roughness=0.9, metalness=0
            emission:  [0.0; 4],
            tangent:   [127, 0, 0, 0],           // (1, 0, 0)
        }
    }
}

// ---------------------------------------------------------------------------
// DebugVertex — packed vertex for coloured debug line rendering, 28 bytes.
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugVertex {
    pub pos:   [f32; 3],  // offset  0 | 12b | Float32x3
    pub color: [f32; 4],  // offset 12 | 16b | Float32x4
}                          // total: 28 bytes

// ---------------------------------------------------------------------------
// WGSL shaders — simplified PBR with day/night, emission
// ---------------------------------------------------------------------------
const SHADER_SOURCE: &str = include_str!("../shaders/pbr_vct.wgsl");
const OUTLINE_SHADER_SOURCE: &str = include_str!("../shaders/outline.wgsl");
const WATER_SHADER_SOURCE: &str = include_str!("../shaders/water.wgsl");
const DEBUG_LINES_SHADER: &str = include_str!("../shaders/debug_lines.wgsl");
const WIREFRAME_SHADER:   &str = include_str!("../shaders/wireframe.wgsl");

const OUTLINE_UNIFORM_SIZE: u64 = 176; // mat4x4(vp) + mat4x4(rotation) + vec4(block_pos) + vec4(aabb_min) + vec4(aabb_max)

// ---------------------------------------------------------------------------
// ChunkMesh
// ---------------------------------------------------------------------------

/// Raw DrawIndexedIndirect command (20 bytes, matches GPU indirect buffer layout).
/// Stored per chunk so draw loops can use `draw_indexed_indirect` instead of
/// `draw_indexed`, preparing for future VBO-merge multi-draw-indirect batching.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

pub struct ChunkMesh {
    pub vertex_buffer:  wgpu::Buffer,
    pub index_buffer:   wgpu::Buffer,
    pub index_count:    u32,
    pub vram_bytes:     u64,
    pub aabb_min:       [f32; 3],
    pub aabb_max:       [f32; 3],
    /// Pre-built indirect draw command buffer — 20 bytes, usage = INDIRECT.
    pub indirect_buffer: wgpu::Buffer,
}

/// Create a `ChunkMesh`, including a pre-built `DrawIndexedIndirect` buffer.
/// The indirect buffer is populated once at creation (mapped_at_creation) and
/// never mutated — the draw parameters are fixed for the life of the mesh.
fn make_chunk_mesh(
    device: &wgpu::Device,
    vertex_buffer: wgpu::Buffer,
    index_buffer:  wgpu::Buffer,
    index_count:   u32,
    vram_bytes:    u64,
    aabb_min:      [f32; 3],
    aabb_max:      [f32; 3],
) -> ChunkMesh {
    let cmd = DrawIndexedIndirect {
        index_count,
        instance_count: 1,
        first_index:    0,
        base_vertex:    0,
        first_instance: 0,
    };
    let indirect_buffer = {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Chunk Indirect"),
            size:               std::mem::size_of::<DrawIndexedIndirect>() as u64,
            usage:              wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: true,
        });
        buf.slice(..).get_mapped_range_mut()
            .copy_from_slice(bytemuck::bytes_of(&cmd));
        buf.unmap();
        buf
    };
    ChunkMesh { vertex_buffer, index_buffer, index_count, vram_bytes, aabb_min, aabb_max, indirect_buffer }
}

// ---------------------------------------------------------------------------
// Game3DPipeline
// ---------------------------------------------------------------------------
pub struct Game3DPipeline {
    pipeline:          wgpu::RenderPipeline,
    /// Alpha-blended variant of `pipeline` used for transparent solid blocks
    /// (glass). Reuses the same pbr_vct shader and layouts; the only state
    /// changes are blend mode and depth-write disabled. Created on first use.
    transparent_pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer:    wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    depth_texture:     Option<wgpu::Texture>,
    depth_view:        Option<wgpu::TextureView>,
    depth_size:        (u32, u32),
    chunk_meshes:      HashMap<(i32, i32, i32), ChunkMesh>,
    /// Water meshes — separate storage for transparent rendering pass.
    water_meshes:      HashMap<(i32, i32, i32), ChunkMesh>,
    /// Glass / transparent-solid meshes — alpha-blended chunk geometry pass.
    transparent_meshes: HashMap<(i32, i32, i32), ChunkMesh>,
    vram_usage:        u64,
    /// LRU order: front = oldest inserted, back = most recently inserted.
    /// Used to evict farthest chunks when VRAM budget is exceeded.
    lru_order:         VecDeque<(i32, i32, i32)>,
    pub culled_last_frame:        u32,
    pub last_draw_calls:          u32,
    pub last_shadow_draw_calls:   u32,
    pub last_visible_triangles:   u32,
    pub last_visible_vram_bytes:  u64,
    atlas_sampler:     Option<wgpu::Sampler>,
    outline_pipeline:           Option<wgpu::RenderPipeline>,
    outline_bind_group_layout:  Option<wgpu::BindGroupLayout>,
    outline_bind_group:         Option<wgpu::BindGroup>,
    outline_uniform_buffer:     Option<wgpu::Buffer>,
    outline_vb:                 Option<wgpu::Buffer>,
    /// Per-model exact wireframe edge buffers (model_name → (buffer, vertex_count)).
    /// Each vertex is a line-segment endpoint in block-local [0..1] space.
    outline_model_vbs:          HashMap<String, (wgpu::Buffer, u32)>,
    /// Water pipeline (alpha-blended, no culling).
    water_pipeline:             Option<wgpu::RenderPipeline>,
    surface_format:             wgpu::TextureFormat,
    /// Still-water sprite sheet  (16 × 512, 32 frames of 16×16 stacked vertically).
    water_anim_texture:         Option<wgpu::Texture>,
    water_anim_view:            Option<wgpu::TextureView>,
    water_anim_sampler:         Option<wgpu::Sampler>,
    /// Flowing-water sprite sheet (same format, used on side faces).
    water_flow_texture:         Option<wgpu::Texture>,
    water_flow_view:            Option<wgpu::TextureView>,
    water_flow_sampler:         Option<wgpu::Sampler>,
    // -- Debug line pipeline (coloured 3-D segments, no depth test) ----------
    debug_lines_pipeline:       Option<wgpu::RenderPipeline>,
    debug_lines_bgl:            Option<wgpu::BindGroupLayout>,
    debug_lines_ub:             Option<wgpu::Buffer>,
    debug_lines_bg:             Option<wgpu::BindGroup>,
    debug_lines_vb:             Option<wgpu::Buffer>,
    debug_lines_vb_cap:         usize,
    // -- Wireframe pipeline (PolygonMode::Line over chunk geometry) ----------
    wireframe_pipeline:         Option<wgpu::RenderPipeline>,
    wireframe_bgl:              Option<wgpu::BindGroupLayout>,
    wireframe_ub:               Option<wgpu::Buffer>,
    wireframe_bg:               Option<wgpu::BindGroup>,
    /// True when the device was created with Features::POLYGON_MODE_LINE.
    pub wireframe_supported:    bool,
    /// Cached main-pass bind group — rebuilt only when atlas views change.
    cached_bg:               Option<wgpu::BindGroup>,
    cached_atlas_ptr:        usize,
    cached_normal_atlas_ptr: usize,
    /// Cached water-pass bind group — same key as main (atlas ptr).
    cached_water_bg:         Option<wgpu::BindGroup>,
}
/// Uniform buffer size — 272 bytes (68 floats)
const UNIFORM_SIZE: u64 = 272;

impl Game3DPipeline {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        vct_frag_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        debug_log!("Game3DPipeline", "new", "Creating 3D pipeline, format={:?}", format);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Game3D Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SOURCE)),
        });

        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("Game3D BGL"),
                entries: &[
                    // @binding(0) uniform buffer (272 bytes)
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
                    // @binding(1) atlas texture
                    wgpu::BindGroupLayoutEntry {
                        binding:    1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled:   false,
                        },
                        count: None,
                    },
                    // @binding(2) atlas sampler
                    wgpu::BindGroupLayoutEntry {
                        binding:    2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // @binding(3) normal atlas texture
                    wgpu::BindGroupLayoutEntry {
                        binding:    3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled:   false,
                        },
                        count: None,
                    },
                    // @binding(4) normal atlas sampler
                    wgpu::BindGroupLayoutEntry {
                        binding:    4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:              Some("Game3D Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout), Some(vct_frag_bgl)],
                immediate_size: 0,
            },
        );

        // Store attributes in a named array so we can borrow it for multiple pipeline descriptors.
        // wgpu::VertexBufferLayout is not Copy, so we re-create the layout each time from the attrs.
        let vertex_attrs = wgpu::vertex_attr_array![
            0 => Float32x3,  // position  (offset  0, 12b)
            1 => Snorm8x4,   // normal    (offset 12,  4b)
            2 => Unorm8x4,   // color_ao  (offset 16,  4b)
            3 => Float32x2,  // texcoord  (offset 20,  8b)
            4 => Unorm8x4,   // material  (offset 28,  4b)
            5 => Float32x4,  // emission  (offset 32, 16b)
            6 => Snorm8x4,   // tangent   (offset 48,  4b)
        ];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Game3D Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &vertex_attrs,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                cull_mode:  Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Option::from(true),
                depth_compare: Option::from(wgpu::CompareFunction::Less),
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Game3D Uniform Buffer"),
            size:               UNIFORM_SIZE,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        debug_log!("Game3DPipeline", "new", "Pipeline created (simplified PBR)");

        // -- Wireframe pipeline (created eagerly if POLYGON_MODE_LINE is supported) --
        let wireframe_supported = device.features()
            .contains(wgpu::Features::POLYGON_MODE_LINE);
        let (wireframe_pipeline, wireframe_bgl, wireframe_ub, wireframe_bg) =
            if wireframe_supported {
                let wf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label:  Some("Wireframe Shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        std::borrow::Cow::Borrowed(WIREFRAME_SHADER)
                    ),
                });
                let wf_bgl = device.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
                        label:   Some("Wireframe BGL"),
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
                    },
                );
                let wf_pl = device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label:              Some("Wireframe Layout"),
                        bind_group_layouts: &[Some(&wf_bgl)],
                        immediate_size:     0,
                    },
                );
                let wf_ub = device.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("Wireframe UB"),
                    size:               64,
                    usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let wf_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label:   Some("Wireframe BG"),
                    layout:  &wf_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding:  0,
                        resource: wf_ub.as_entire_binding(),
                    }],
                });
                // Vertex stride matches Vertex3D (52 b) but shader only reads location(0).
                let wf_vertex_layout = wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &wgpu::vertex_attr_array![0 => Float32x3],
                };
                let wf_pipeline = device.create_render_pipeline(
                    &wgpu::RenderPipelineDescriptor {
                        label:  Some("Wireframe Pipeline"),
                        layout: Some(&wf_pl),
                        vertex: wgpu::VertexState {
                            module:              &wf_shader,
                            entry_point:         Some("vs_main"),
                            buffers:             &[wf_vertex_layout],
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module:              &wf_shader,
                            entry_point:         Some("fs_main"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology:     wgpu::PrimitiveTopology::TriangleList,
                            cull_mode:    None,
                            front_face:   wgpu::FrontFace::Ccw,
                            polygon_mode: wgpu::PolygonMode::Line,
                            ..Default::default()
                        },
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format:              wgpu::TextureFormat::Depth32Float,
                            depth_write_enabled: Option::from(false),
                            depth_compare:       Option::from(wgpu::CompareFunction::LessEqual),
                            stencil:             wgpu::StencilState::default(),
                            bias:                wgpu::DepthBiasState::default(),
                        }),
                        multisample: wgpu::MultisampleState::default(),
                        cache: None,
                        multiview_mask: None,
                    },
                );
                debug_log!("Game3DPipeline", "new", "Wireframe pipeline created");
                (Some(wf_pipeline), Some(wf_bgl), Some(wf_ub), Some(wf_bg))
            } else {
                debug_log!("Game3DPipeline", "new",
                    "POLYGON_MODE_LINE not supported — wireframe disabled");
                (None, None, None, None)
            };

        Self {
            pipeline,
            transparent_pipeline: None,
            uniform_buffer,
            bind_group_layout,
            depth_texture: None,
            depth_view:    None,
            depth_size:    (0, 0),
            chunk_meshes:  HashMap::new(),
            water_meshes:  HashMap::new(),
            transparent_meshes: HashMap::new(),
            vram_usage:    UNIFORM_SIZE,
            lru_order:               VecDeque::new(),
            culled_last_frame:       0,
            last_draw_calls:         0,
            last_shadow_draw_calls:  0,
            last_visible_triangles:  0,
            last_visible_vram_bytes: 0,
            atlas_sampler: None,
            outline_pipeline:          None,
            outline_bind_group_layout: None,
            outline_bind_group:        None,
            outline_uniform_buffer:    None,
            outline_vb:                None,
            outline_model_vbs:         HashMap::new(),
            water_pipeline:            None,
            surface_format:            format,
            water_anim_texture:        None,
            water_anim_view:           None,
            water_anim_sampler:        None,
            water_flow_texture:        None,
            water_flow_view:           None,
            water_flow_sampler:        None,
            debug_lines_pipeline:      None,
            debug_lines_bgl:           None,
            debug_lines_ub:            None,
            debug_lines_bg:            None,
            debug_lines_vb:            None,
            debug_lines_vb_cap:        0,
            wireframe_pipeline,
            wireframe_bgl,
            wireframe_ub,
            wireframe_bg,
            wireframe_supported,
            cached_bg:               None,
            cached_atlas_ptr:        0,
            cached_normal_atlas_ptr: 0,
            cached_water_bg:         None,
        }
    }

    // -----------------------------------------------------------------------
    // Incremental chunk mesh management
    // -----------------------------------------------------------------------

    pub fn update_chunk_meshes(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        meshes: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
    ) -> u128 {
        let t0    = Instant::now();
        let count = meshes.len();
        let mut new_vram: u64 = 0;

        for (key, vertices, indices, aabb_min, aabb_max) in meshes {
            if let Some(old) = self.chunk_meshes.remove(&key) {
                self.vram_usage = self.vram_usage.saturating_sub(old.vram_bytes);
            }
            if vertices.is_empty() || indices.is_empty() { continue; }

            let vert_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    vertices.as_ptr() as *const u8,
                    vertices.len() * std::mem::size_of::<Vertex3D>(),
                )
            };
            let idx_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    indices.as_ptr() as *const u8,
                    indices.len() * std::mem::size_of::<u32>(),
                )
            };

            let vb = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Chunk VB"), size: vert_bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&vb, 0, vert_bytes);

            let ib = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Chunk IB"), size: idx_bytes.len() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ib, 0, idx_bytes);

            let bytes = vert_bytes.len() as u64 + idx_bytes.len() as u64;
            new_vram += bytes;

            self.chunk_meshes.insert(key, make_chunk_mesh(
                device, vb, ib,
                indices.len() as u32,
                bytes, aabb_min, aabb_max,
            ));
        }

        self.vram_usage += new_vram;

        let upload_us = t0.elapsed().as_micros();
        ext_debug_log!(
            "Game3DPipeline", "update_chunk_meshes",
            "[PERF] upload={:.2}ms chunks={} total={} VRAM={:.2}MB",
            upload_us as f64 / 1000.0, count,
            self.chunk_meshes.len(),
            self.vram_usage as f64 / (1024.0 * 1024.0)
        );
        upload_us
    }

    pub fn insert_chunk_mesh(
        &mut self,
        device:        &wgpu::Device,
        key:           (i32, i32, i32),
        vertex_buffer: wgpu::Buffer,
        index_buffer:  wgpu::Buffer,
        index_count:   u32,
        vram_bytes:    u64,
        aabb_min:      [f32; 3],
        aabb_max:      [f32; 3],
    ) -> Vec<(i32, i32, i32)> {
        if let Some(old) = self.chunk_meshes.remove(&key) {
            self.vram_usage = self.vram_usage.saturating_sub(old.vram_bytes);
            if let Some(pos) = self.lru_order.iter().position(|k| k == &key) {
                self.lru_order.remove(pos);
            }
        }

        self.vram_usage += vram_bytes;
        self.chunk_meshes.insert(key, make_chunk_mesh(
            device, vertex_buffer, index_buffer,
            index_count, vram_bytes, aabb_min, aabb_max,
        ));
        self.lru_order.push_back(key);

        let mut evicted = Vec::new();
        while self.vram_usage > CHUNK_MESH_VRAM_BUDGET {
            if let Some(oldest) = self.lru_order.pop_front() {
                if oldest == key { break; }
                if let Some(mesh) = self.chunk_meshes.remove(&oldest) {
                    self.vram_usage = self.vram_usage.saturating_sub(mesh.vram_bytes);
                    evicted.push(oldest);
                }
            } else {
                break;
            }
        }

        if !evicted.is_empty() {
            debug_log!(
                "Game3DPipeline", "insert_chunk_mesh",
                "LRU evicted {} chunks to stay within {:.0}MB VRAM budget",
                evicted.len(),
                CHUNK_MESH_VRAM_BUDGET as f64 / (1024.0 * 1024.0),
            );
        }
        evicted
    }

    pub fn remove_chunk_meshes(&mut self, keys: &[(i32, i32, i32)]) {
        let mut freed = 0u64;
        let mut removed = 0u32;
        for key in keys {
            if let Some(mesh) = self.chunk_meshes.remove(key) {
                freed   += mesh.vram_bytes;
                removed += 1;
                if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
                    self.lru_order.remove(pos);
                }
            }
            // Also remove water meshes for evicted chunks
            if let Some(mesh) = self.water_meshes.remove(key) {
                freed   += mesh.vram_bytes;
                removed += 1;
            }
            // ...and transparent (glass) meshes
            if let Some(mesh) = self.transparent_meshes.remove(key) {
                freed   += mesh.vram_bytes;
                removed += 1;
            }
        }
        if removed > 0 {
            self.vram_usage = self.vram_usage.saturating_sub(freed);
            debug_log!(
                "Game3DPipeline", "remove_chunk_meshes",
                "Freed {} chunks, recovered {:.2}MB",
                removed, freed as f64 / (1024.0 * 1024.0)
            );
        }
    }

    /// Remove only the transparent (glass) mesh for each key.
    /// Called when a chunk is rebuilt so a stale glass mesh does not persist
    /// after all glass blocks in that chunk are broken.
    pub fn evict_transparent_for_keys(&mut self, keys: &[(i32, i32, i32)]) {
        let mut freed = 0u64;
        for key in keys {
            if let Some(mesh) = self.transparent_meshes.remove(key) {
                freed += mesh.vram_bytes;
            }
        }
        if freed > 0 {
            self.vram_usage = self.vram_usage.saturating_sub(freed);
        }
    }

    // -----------------------------------------------------------------------
    // Water mesh management
    // -----------------------------------------------------------------------

    /// Insert a water mesh for a chunk. Replaces any existing water mesh.
    pub fn insert_water_mesh(
        &mut self,
        device:        &wgpu::Device,
        key:           (i32, i32, i32),
        vertex_buffer: wgpu::Buffer,
        index_buffer:  wgpu::Buffer,
        index_count:   u32,
        vram_bytes:    u64,
        aabb_min:      [f32; 3],
        aabb_max:      [f32; 3],
    ) {
        if let Some(old) = self.water_meshes.remove(&key) {
            self.vram_usage = self.vram_usage.saturating_sub(old.vram_bytes);
        }
        self.vram_usage += vram_bytes;
        self.water_meshes.insert(key, make_chunk_mesh(
            device, vertex_buffer, index_buffer,
            index_count, vram_bytes, aabb_min, aabb_max,
        ));
    }

    // -----------------------------------------------------------------------
    // Transparent (glass) mesh management
    // -----------------------------------------------------------------------

    /// Insert or replace a transparent (glass) mesh for a chunk.
    pub fn insert_transparent_mesh(
        &mut self,
        device:        &wgpu::Device,
        key:           (i32, i32, i32),
        vertex_buffer: wgpu::Buffer,
        index_buffer:  wgpu::Buffer,
        index_count:   u32,
        vram_bytes:    u64,
        aabb_min:      [f32; 3],
        aabb_max:      [f32; 3],
    ) {
        if let Some(old) = self.transparent_meshes.remove(&key) {
            self.vram_usage = self.vram_usage.saturating_sub(old.vram_bytes);
        }
        self.vram_usage += vram_bytes;
        self.transparent_meshes.insert(key, make_chunk_mesh(
            device, vertex_buffer, index_buffer,
            index_count, vram_bytes, aabb_min, aabb_max,
        ));
    }

    // -----------------------------------------------------------------------
    // Depth texture
    // -----------------------------------------------------------------------

    fn ensure_depth(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.depth_size == (width, height) { return; }

        debug_log!(
            "Game3DPipeline", "ensure_depth",
            "Recreating depth texture {}x{}", width, height
        );

        if self.depth_texture.is_some() {
            let old = (self.depth_size.0 as u64) * (self.depth_size.1 as u64) * 4;
            self.vram_usage = self.vram_usage.saturating_sub(old);
        }

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Game3D Depth"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Depth32Float,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:    &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        self.vram_usage   += (width as u64) * (height as u64) * 4;
        self.depth_texture = Some(tex);
        self.depth_view    = Some(view);
        self.depth_size    = (width, height);
    }

    // -----------------------------------------------------------------------
    // Render — frustum culling
    // -----------------------------------------------------------------------

    pub fn render(
        &mut self,
        encoder:        &mut wgpu::CommandEncoder,
        color_view:     &wgpu::TextureView,
        device:         &wgpu::Device,
        queue:          &wgpu::Queue,
        camera:         &Camera,
        width:          u32,
        height:         u32,
        atlas_view:     &wgpu::TextureView,
        normal_atlas_view: &wgpu::TextureView,
        lighting_data:  &[f32; 68],
        vct_bind_group: &wgpu::BindGroup,
    ) -> u128 {
        let t0 = Instant::now();
        self.ensure_depth(device, width, height);

        if self.atlas_sampler.is_none() {
            self.atlas_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
                label:         Some("Atlas Mipmap Sampler"),
                mag_filter:    wgpu::FilterMode::Nearest,
                min_filter:    wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }));
            debug_log!("Game3DPipeline", "render", "Atlas sampler created");
        }
        
        let vp              = camera.view_projection_matrix();
        let frustum         = FrustumPlanes::from_view_projection(&vp);

        // Upload uniforms
        let uniform_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                lighting_data.as_ptr() as *const u8,
                std::mem::size_of::<[f32; 68]>()
            )
        };
        queue.write_buffer(&self.uniform_buffer, 0, uniform_bytes);

        // Rebuild bind group only when atlas views change; uniform data is
        // written directly into the buffer via write_buffer, so the descriptor
        // itself never needs rebuilding for uniform updates.
        let atlas_ptr        = atlas_view        as *const _ as usize;
        let normal_atlas_ptr = normal_atlas_view as *const _ as usize;
        if self.cached_bg.is_none()
            || atlas_ptr        != self.cached_atlas_ptr
            || normal_atlas_ptr != self.cached_normal_atlas_ptr
        {
            self.cached_bg = Some(device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label:   Some("Game3D BG"),
                    layout:  &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(atlas_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                self.atlas_sampler.as_ref().unwrap()
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(normal_atlas_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(
                                self.atlas_sampler.as_ref().unwrap()
                            ),
                        },
                    ],
                },
            ));
            self.cached_atlas_ptr        = atlas_ptr;
            self.cached_normal_atlas_ptr = normal_atlas_ptr;
            debug_log!("Game3DPipeline", "render", "Bind group rebuilt (atlas changed)");
        }
        let bind_group = self.cached_bg.as_ref().unwrap();

        self.last_shadow_draw_calls = 0;

        // Frustum culling counts
        let mut visible = 0u32;
        let mut culled  = 0u32;
        for mesh in self.chunk_meshes.values() {
            if frustum.intersects_aabb(mesh.aabb_min, mesh.aabb_max) { visible += 1; }
            else { culled += 1; }
        }
        self.culled_last_frame = culled;

        if visible == 0 { return 0; }

        // Depth clear
        {
            let _p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:             Some("Depth Clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view:      self.depth_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes:    None,
                occlusion_query_set: None,
                multiview_mask:      None,
            });
        }

        // Main 3D pass
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Game3D Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:      self.depth_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:     None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_bind_group(1, vct_bind_group, &[]);

        let mut draw_calls       = 0u32;
        let mut visible_tris     = 0u32;
        let mut visible_vram     = 0u64;
        for mesh in self.chunk_meshes.values() {
            if !frustum.intersects_aabb(mesh.aabb_min, mesh.aabb_max) { continue; }
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed_indirect(&mesh.indirect_buffer, 0);
            draw_calls   += 1;
            visible_tris += mesh.index_count / 3;
            visible_vram += mesh.vram_bytes;
        }
        self.last_draw_calls         = draw_calls;
        self.last_visible_triangles  = visible_tris;
        self.last_visible_vram_bytes = visible_vram;

        let render_us = t0.elapsed().as_micros();
        flow_debug_log!(
            "Game3DPipeline", "render",
            "[PERF] render={:.2}ms visible={} culled={} gpu_chunks={}",
            render_us as f64 / 1000.0, visible, culled, self.chunk_meshes.len()
        );
        render_us
    }

    pub fn vram_usage(&self) -> u64 { self.vram_usage }
    pub fn gpu_chunk_count(&self) -> usize { self.chunk_meshes.len() }

    // -----------------------------------------------------------------------
    // Water rendering (transparent, alpha-blended pass)
    // -----------------------------------------------------------------------

    /// Fallback still-water sprite sheet used when `assets/textures/common/water_anim.png`
    /// is absent.  32 frames of 16×16 stacked vertically → 16×512 RGBA image.
    /// Each frame is near-white (so the block albedo dominates after multiplication)
    /// with a subtle horizontal highlight band that scrolls across frames,
    /// mimicking a slow surface ripple.
    fn generate_water_anim_pixels() -> Vec<u8> {
        const FRAMES: usize = 32;
        const TILE:   usize = 16;
        const W: usize = TILE;
        const H: usize = TILE * FRAMES;
        let mut data = vec![0u8; W * H * 4];

        for frame in 0..FRAMES {
            // Horizontal highlight band scrolls down across frames
            let band_center = (frame as f32 / FRAMES as f32) * TILE as f32;
            for ty in 0..TILE {
                for tx in 0..TILE {
                    let dist = ((ty as f32 - band_center).abs()).min(
                        (ty as f32 + TILE as f32 - band_center).abs(),
                    );
                    // Bright near the band, slightly dimmer elsewhere (0.82..1.0)
                    let band_v = 1.0 - (dist / TILE as f32).min(1.0) * 0.18;
                    let v = (band_v * 255.0).round() as u8;
                    let pi = ((frame * TILE + ty) * W + tx) * 4;
                    data[pi]     = v;
                    data[pi + 1] = v;
                    data[pi + 2] = v;
                    data[pi + 3] = 255;
                }
            }
        }
        data
    }

    /// Fallback flowing-water sprite sheet used when
    /// `assets/textures/common/water_flow_anim.png` is absent.
    /// Same format (16×512, 32 frames of 16×16).
    /// A vertical highlight band scrolls downward across frames, suggesting flow.
    fn generate_water_flow_pixels() -> Vec<u8> {
        const FRAMES: usize = 32;
        const TILE:   usize = 16;
        const W: usize = TILE;
        const H: usize = TILE * FRAMES;
        let mut data = vec![0u8; W * H * 4];

        for frame in 0..FRAMES {
            // Vertical highlight band shifts right across frames
            let band_center = (frame as f32 / FRAMES as f32) * TILE as f32;
            for ty in 0..TILE {
                for tx in 0..TILE {
                    let dist = ((tx as f32 - band_center).abs()).min(
                        (tx as f32 + TILE as f32 - band_center).abs(),
                    );
                    let band_v = 1.0 - (dist / TILE as f32).min(1.0) * 0.18;
                    let v = (band_v * 255.0).round() as u8;
                    let pi = ((frame * TILE + ty) * W + tx) * 4;
                    data[pi]     = v;
                    data[pi + 1] = v;
                    data[pi + 2] = v;
                    data[pi + 3] = 255;
                }
            }
        }
        data
    }

    fn init_water_pipeline(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        debug_log!("Game3DPipeline", "init_water_pipeline", "Creating water pipeline");
        let vertex_attrs = wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Snorm8x4, 2 => Unorm8x4,
            3 => Float32x2, 4 => Unorm8x4, 5 => Float32x4, 6 => Snorm8x4,
        ];

        // Helper: upload an RGBA8 sprite sheet as a GPU texture.
        // Accepts actual pixel dimensions so files with different sizes (e.g. 32×1024
        // flowing water) are uploaded correctly instead of being corrupted by a
        // hardcoded 16×512 stride.
        let upload_tex = |pixels: Vec<u8>, w: u32, h: u32, label: &'static str| -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label:           Some(label),
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
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset:         0,
                    bytes_per_row:  Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            // Nearest-neighbor: pixel-art water tiles, no blurring
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label:            Some(label),
                address_mode_u:   wgpu::AddressMode::Repeat,
                address_mode_v:   wgpu::AddressMode::ClampToEdge, // frames are clamped vertically
                address_mode_w:   wgpu::AddressMode::Repeat,
                mag_filter:       wgpu::FilterMode::Nearest,
                min_filter:       wgpu::FilterMode::Nearest,
                mipmap_filter:    wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });
            (tex, view, sampler)
        };

        // -- Still water (top face) --
        let still_pixels = {
            let path = "assets/textures/simpleblocks/common/water_anim.png";
            if let Ok(img) = image::open(path) {
                let rgba = img.to_rgba8();
                let (w, h) = (rgba.width(), rgba.height());
                debug_log!("Game3DPipeline", "init_water_pipeline", "Loaded water_anim {}x{}", w, h);
                (rgba.into_raw(), w, h)
            } else {
                debug_log!("Game3DPipeline", "init_water_pipeline", "Using fallback water_anim texture");
                (Self::generate_water_anim_pixels(), 16u32, 512u32)
            }
        };
        let (anim_tex, anim_view, anim_sampler) = upload_tex(still_pixels.0, still_pixels.1, still_pixels.2, "Water Still Texture");
        self.water_anim_texture = Some(anim_tex);
        self.water_anim_view    = Some(anim_view);
        self.water_anim_sampler = Some(anim_sampler);

        // -- Flowing water (side faces) --
        let flow_pixels = {
            let path = "assets/textures/simpleblocks/common/water_flow_anim.png";
            if let Ok(img) = image::open(path) {
                let rgba = img.to_rgba8();
                let (w, h) = (rgba.width(), rgba.height());
                debug_log!("Game3DPipeline", "init_water_pipeline", "Loaded water_flow_anim {}x{}", w, h);
                (rgba.into_raw(), w, h)
            } else {
                debug_log!("Game3DPipeline", "init_water_pipeline", "Using fallback water_flow_anim texture");
                (Self::generate_water_flow_pixels(), 16u32, 512u32)
            }
        };
        let (flow_tex, flow_view, flow_sampler) = upload_tex(flow_pixels.0, flow_pixels.1, flow_pixels.2, "Water Flow Texture");
        self.water_flow_texture = Some(flow_tex);
        self.water_flow_view    = Some(flow_view);
        self.water_flow_sampler = Some(flow_sampler);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Water Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WATER_SHADER_SOURCE)),
        });

        let water_bgl = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("Water BGL"),
                entries: &[
                    // @binding(0) uniform buffer (same 272 bytes)
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
                    // @binding(1) atlas texture (kept for layout compat, unused in fragment)
                    wgpu::BindGroupLayoutEntry {
                        binding:    1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled:   false,
                        },
                        count: None,
                    },
                    // @binding(2) atlas sampler
                    wgpu::BindGroupLayoutEntry {
                        binding:    2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // @binding(3) water animation texture
                    wgpu::BindGroupLayoutEntry {
                        binding:    3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled:   false,
                        },
                        count: None,
                    },
                    // @binding(4) still water animation sampler
                    wgpu::BindGroupLayoutEntry {
                        binding:    4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // @binding(5) flowing water texture (side faces)
                    wgpu::BindGroupLayoutEntry {
                        binding:    5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled:   false,
                        },
                        count: None,
                    },
                    // @binding(6) flowing water sampler
                    wgpu::BindGroupLayoutEntry {
                        binding:    6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:              Some("Water Pipeline Layout"),
                bind_group_layouts: &[Some(&water_bgl)],
                immediate_size:     0,
            },
        );

        let _vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &wgpu::vertex_attr_array![
                0 => Float32x3,  // position
                1 => Snorm8x4,   // normal
                2 => Unorm8x4,   // color_ao (alpha = water transparency)
                3 => Float32x2,  // texcoord
                4 => Unorm8x4,   // material
                5 => Float32x4,  // emission
                6 => Snorm8x4,   // tangent
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Water Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &vertex_attrs,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                cull_mode:  None, // No culling for water (visible from both sides)
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Option::from(false), // Don't write depth for transparent
                depth_compare: Option::from(wgpu::CompareFunction::LessEqual),
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        self.water_pipeline = Some(pipeline);
        debug_log!("Game3DPipeline", "init_water_pipeline", "Water pipeline ready");
    }

    /// Render water meshes (transparent pass — call AFTER main opaque render).
    pub fn render_water(
        &mut self,
        encoder:        &mut wgpu::CommandEncoder,
        color_view:     &wgpu::TextureView,
        device:         &wgpu::Device,
        queue:          &wgpu::Queue,
        camera:         &Camera,
        atlas_view:     &wgpu::TextureView,
        lighting_data:  &[f32; 68],
        elapsed_secs:   f32,
    ) {
        if self.water_meshes.is_empty() { return; }
        if self.water_pipeline.is_none() { self.init_water_pipeline(device, queue); }
        if self.depth_view.is_none() { return; }

        let water_pipeline = self.water_pipeline.as_ref().unwrap();

        // Upload uniforms, injecting real elapsed time into time_params.z
        let mut water_lighting = *lighting_data;
        water_lighting[50] = elapsed_secs; // time_params.z — used for water animation
        let uniform_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                water_lighting.as_ptr() as *const u8,
                std::mem::size_of::<[f32; 68]>(),
            )
        };
        queue.write_buffer(&self.uniform_buffer, 0, uniform_bytes);

        let anim_view     = self.water_anim_view.as_ref().unwrap();
        let anim_sampler  = self.water_anim_sampler.as_ref().unwrap();
        let flow_view     = self.water_flow_view.as_ref().unwrap();
        let flow_sampler  = self.water_flow_sampler.as_ref().unwrap();

        // Water bind group: cache and rebuild only when atlas view changes.
        let atlas_ptr = atlas_view as *const _ as usize;
        if self.cached_water_bg.is_none() || atlas_ptr != self.cached_atlas_ptr {
            self.cached_water_bg = Some(device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label:   Some("Water BG"),
                    layout:  &water_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(atlas_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                self.atlas_sampler.as_ref().unwrap()
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(anim_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(anim_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(flow_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::Sampler(flow_sampler),
                        },
                    ],
                },
            ));
        }
        let water_bg = self.cached_water_bg.as_ref().unwrap();

        let vp = camera.view_projection_matrix();
        let frustum = FrustumPlanes::from_view_projection(&vp);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Water Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:      self.depth_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });

        pass.set_pipeline(water_pipeline);
        pass.set_bind_group(0, water_bg, &[]);

        for mesh in self.water_meshes.values() {
            if !frustum.intersects_aabb(mesh.aabb_min, mesh.aabb_max) { continue; }
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed_indirect(&mesh.indirect_buffer, 0);
        }
    }

    /// Depth buffer view — used by subsequent passes (e.g. outline).
    pub fn depth_view(&self) -> Option<&wgpu::TextureView> {
        self.depth_view.as_ref()
    }

    // -----------------------------------------------------------------------
    // Transparent (glass) rendering — alpha-blended pbr_vct pass
    // -----------------------------------------------------------------------

    /// Build the alpha-blended chunk pipeline used for glass blocks.
    /// Reuses the same `pbr_vct.wgsl` shader and bind group layouts as the
    /// opaque chunk pipeline so we get full PBR + GI + tinted shadows.
    fn init_transparent_pipeline(
        &mut self,
        device: &wgpu::Device,
        vct_frag_bgl: &wgpu::BindGroupLayout,
    ) {
        debug_log!("Game3DPipeline", "init_transparent_pipeline",
            "Creating alpha-blended chunk pipeline (glass)");
        let vertex_attrs = wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Snorm8x4, 2 => Unorm8x4,
            3 => Float32x2, 4 => Unorm8x4, 5 => Float32x4, 6 => Snorm8x4,
        ];

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Game3D Transparent Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SOURCE)),
        });

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:              Some("Game3D Transparent Layout"),
                bind_group_layouts: &[Some(&self.bind_group_layout), Some(vct_frag_bgl)],
                immediate_size:     0,
            },
        );

        let _vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &wgpu::vertex_attr_array![
                0 => Float32x3, 1 => Snorm8x4, 2 => Unorm8x4,
                3 => Float32x2, 4 => Unorm8x4, 5 => Float32x4, 6 => Snorm8x4,
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Game3D Transparent Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &vertex_attrs,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     self.surface_format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                // Don't cull — glass is two-sided to keep both faces visible
                // when looking from inside (e.g. greenhouse interior).
                cull_mode:  None,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                // Write depth so glass properly occludes water and other transparent
                // objects rendered after it in the same frame.
                depth_write_enabled: Option::from(true),
                depth_compare:       Option::from(wgpu::CompareFunction::Less),
                stencil:             Default::default(),
                bias:                Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        self.transparent_pipeline = Some(pipeline);
        debug_log!("Game3DPipeline", "init_transparent_pipeline", "Transparent pipeline ready");
    }

    /// Render glass / transparent solid meshes. Call AFTER opaque chunks but
    /// BEFORE water to keep glass behind water surfaces working correctly.
    pub fn render_transparent(
        &mut self,
        encoder:        &mut wgpu::CommandEncoder,
        color_view:     &wgpu::TextureView,
        device:         &wgpu::Device,
        queue:          &wgpu::Queue,
        camera:         &Camera,
        atlas_view:     &wgpu::TextureView,
        normal_atlas_view: &wgpu::TextureView,
        lighting_data:  &[f32; 68],
        vct_bind_group: &wgpu::BindGroup,
        vct_frag_bgl:   &wgpu::BindGroupLayout,
    ) {
        if self.transparent_meshes.is_empty() { return; }
        if self.depth_view.is_none() { return; }
        if self.transparent_pipeline.is_none() {
            self.init_transparent_pipeline(device, vct_frag_bgl);
        }
        let pipeline = self.transparent_pipeline.as_ref().unwrap();

        // Re-upload lighting uniforms (the water pass clobbers time_params.z).
        let uniform_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                lighting_data.as_ptr() as *const u8,
                std::mem::size_of::<[f32; 68]>(),
            )
        };
        queue.write_buffer(&self.uniform_buffer, 0, uniform_bytes);

        // Reuse the cached main-pass bind group (same layout + same atlas views).
        // render() is always called before render_transparent(), so cached_bg is valid.
        let atlas_ptr        = atlas_view        as *const _ as usize;
        let normal_atlas_ptr = normal_atlas_view as *const _ as usize;
        if self.cached_bg.is_none()
            || atlas_ptr        != self.cached_atlas_ptr
            || normal_atlas_ptr != self.cached_normal_atlas_ptr
        {
            self.cached_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("Game3D Transparent BG"),
                layout:  &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(atlas_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(self.atlas_sampler.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(normal_atlas_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(self.atlas_sampler.as_ref().unwrap()) },
                ],
            }));
            self.cached_atlas_ptr        = atlas_ptr;
            self.cached_normal_atlas_ptr = normal_atlas_ptr;
        }
        let bg = self.cached_bg.as_ref().unwrap();

        let vp = camera.view_projection_matrix();
        let frustum = FrustumPlanes::from_view_projection(&vp);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Transparent (Glass) Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:      self.depth_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.set_bind_group(1, vct_bind_group, &[]);

        for mesh in self.transparent_meshes.values() {
            if !frustum.intersects_aabb(mesh.aabb_min, mesh.aabb_max) { continue; }
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed_indirect(&mesh.indirect_buffer, 0);
        }
    }

    // -----------------------------------------------------------------------
    // Outline pipeline
    // -----------------------------------------------------------------------

    /// Upload a per-model exact-geometry wireframe edge buffer.
    ///
    /// `edge_verts` is a flat list of line-segment endpoint pairs produced by
    /// `extract_quad_edges`, with the render offset (+0.5 on X and Z) already
    /// applied so positions are in block-local [0..1] space.
    pub fn register_model_outline(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        name:   &str,
        edge_verts: &[[f32; 3]],
    ) {
        if edge_verts.is_empty() { return; }
        let byte_data: &[u8] = unsafe {
            std::slice::from_raw_parts(
                edge_verts.as_ptr() as *const u8,
                edge_verts.len() * std::mem::size_of::<[f32; 3]>(),
            )
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some(&format!("Outline Model VB {}", name)),
            size:               byte_data.len() as u64,
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, byte_data);
        self.vram_usage += byte_data.len() as u64;
        self.outline_model_vbs.insert(name.to_string(), (buf, edge_verts.len() as u32));
        debug_log!("Game3DPipeline", "register_model_outline",
            "Registered outline for '{}' ({} line verts)", name, edge_verts.len());
    }

    fn init_outline(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        debug_log!("Game3DPipeline", "init_outline", "Creating outline pipeline");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Outline Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(OUTLINE_SHADER_SOURCE)),
        });

        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("Outline BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   std::num::NonZeroU64::new(OUTLINE_UNIFORM_SIZE),
                        },
                        count: None,
                    },
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:              Some("Outline Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size:     0,
            },
        );

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Outline Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:      self.surface_format,
                    blend:       None,
                    write_mask:  wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Option::from(false),
                depth_compare: Option::from(wgpu::CompareFunction::LessEqual),
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Outline Uniform Buffer"),
            size:               OUTLINE_UNIFORM_SIZE,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        const E: f32 = 0.005;
        let vertices: [[f32; 3]; 24] = [
            [-E, -E, -E], [ 1.0+E, -E, -E],
            [ 1.0+E, -E, -E], [ 1.0+E, -E,  1.0+E],
            [ 1.0+E, -E,  1.0+E], [-E, -E,  1.0+E],
            [-E, -E,  1.0+E], [-E, -E, -E],
            [-E,  1.0+E, -E], [ 1.0+E,  1.0+E, -E],
            [ 1.0+E,  1.0+E, -E], [ 1.0+E,  1.0+E,  1.0+E],
            [ 1.0+E,  1.0+E,  1.0+E], [-E,  1.0+E,  1.0+E],
            [-E,  1.0+E,  1.0+E], [-E,  1.0+E, -E],
            [-E, -E, -E], [-E,  1.0+E, -E],
            [ 1.0+E, -E, -E], [ 1.0+E,  1.0+E, -E],
            [ 1.0+E, -E,  1.0+E], [ 1.0+E,  1.0+E,  1.0+E],
            [-E, -E,  1.0+E], [-E,  1.0+E,  1.0+E],
        ];
        let vb_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * std::mem::size_of::<[f32; 3]>(),
            )
        };
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Outline VB"), size: vb_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vb, 0, vb_bytes);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Outline BG"),
            layout:  &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        self.vram_usage += vb_bytes.len() as u64 + OUTLINE_UNIFORM_SIZE;

        self.outline_pipeline          = Some(pipeline);
        self.outline_bind_group_layout = Some(bind_group_layout);
        self.outline_bind_group        = Some(bind_group);
        self.outline_uniform_buffer    = Some(uniform_buffer);
        self.outline_vb                = Some(vb);

        debug_log!("Game3DPipeline", "init_outline", "Outline pipeline ready");
    }

    /// Render a block selection outline.
    ///
    /// * `block_pos`  — world voxel position of the targeted block.
    /// * `model_aabb` — fallback AABB for model blocks if no exact edge VB exists.
    /// * `model_name` — when `Some` and a per-model edge VB has been registered via
    ///   [`register_model_outline`], the exact quad-edge wireframe is drawn instead of
    ///   an AABB box, exactly matching the model's geometry.
    pub fn render_outline(
        &mut self,
        encoder:         &mut wgpu::CommandEncoder,
        color_view:      &wgpu::TextureView,
        device:          &wgpu::Device,
        queue:           &wgpu::Queue,
        camera:          &Camera,
        width:           u32,
        height:          u32,
        block_pos:       Option<glam::IVec3>,
        model_aabb:      Option<(glam::Vec3, glam::Vec3)>,
        model_name:      Option<&str>,
        model_rotation:  u8,
    ) {
        let block_pos = match block_pos { Some(p) => p, None => return };
        if self.depth_view.is_none() { return; }
        if self.outline_pipeline.is_none() { self.init_outline(device, queue); }

        self.ensure_depth(device, width, height);

        // Check if we have an exact per-model edge VB.
        let has_model_vb = model_name
            .and_then(|n| self.outline_model_vbs.get(n))
            .is_some();

        // AABB uniforms:
        //   - Model VB: identity (edge verts already in block-local [0..1] space,
        //     mix(0, 1, pos) == pos so the shader passes them through unchanged).
        //   - Standard blocks: small epsilon expansion around the unit cube.
        const E: f32 = 0.005;
        let (aabb_min, aabb_max) = if has_model_vb {
            (glam::Vec3::ZERO, glam::Vec3::ONE)
        } else {
            model_aabb.unwrap_or_else(|| {
                (glam::Vec3::new(-E, -E, -E), glam::Vec3::new(1.0 + E, 1.0 + E, 1.0 + E))
            })
        };

        let vertex_count: u32 = if has_model_vb {
            self.outline_model_vbs.get(model_name.unwrap()).map_or(24, |&(_, c)| c)
        } else {
            24
        };

        let vp  = camera.view_projection_matrix();
        let rot = outline_rotation_matrix(model_rotation);
        // Layout: [0..64] view_proj, [64..128] rotation, [128..144] block_pos,
        //         [144..160] aabb_min, [160..176] aabb_max
        let mut data = [0.0f32; 44];
        data[0..16].copy_from_slice(&vp.to_cols_array());
        data[16..32].copy_from_slice(&rot.to_cols_array());
        data[32] = block_pos.x as f32;
        data[33] = block_pos.y as f32;
        data[34] = block_pos.z as f32;
        data[35] = 0.0;
        data[36] = aabb_min.x;
        data[37] = aabb_min.y;
        data[38] = aabb_min.z;
        data[39] = 0.0;
        data[40] = aabb_max.x;
        data[41] = aabb_max.y;
        data[42] = aabb_max.z;
        data[43] = 0.0;

        let uniform_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, OUTLINE_UNIFORM_SIZE as usize)
        };
        queue.write_buffer(self.outline_uniform_buffer.as_ref().unwrap(), 0, uniform_bytes);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Outline Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:      self.depth_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });

        pass.set_pipeline(self.outline_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.outline_bind_group.as_ref().unwrap(), &[]);

        if has_model_vb {
            if let Some((vb, _)) = self.outline_model_vbs.get(model_name.unwrap()) {
                pass.set_vertex_buffer(0, vb.slice(..));
            }
        } else {
            pass.set_vertex_buffer(0, self.outline_vb.as_ref().unwrap().slice(..));
        }

        pass.draw(0..vertex_count, 0..1);

        flow_debug_log!(
            "Game3DPipeline", "render_outline",
            "Drawing outline at ({}, {}, {}) model_vb={} verts={}",
            block_pos.x, block_pos.y, block_pos.z, has_model_vb, vertex_count,
        );
    }

    // -----------------------------------------------------------------------
    // Expose chunk AABBs for debug rendering (world-space min/max)
    // -----------------------------------------------------------------------

    pub fn chunk_aabbs(&self) -> Vec<([f32; 3], [f32; 3])> {
        self.chunk_meshes.values()
            .map(|m| (m.aabb_min, m.aabb_max))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Debug line pipeline — lazy init
    // -----------------------------------------------------------------------

    fn init_debug_lines(&mut self, device: &wgpu::Device) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Debug Lines Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::borrow::Cow::Borrowed(DEBUG_LINES_SHADER)
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Debug Lines BGL"),
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
            label:              Some("Debug Lines Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });
        let ub = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Debug Lines UB"),
            size:               64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Debug Lines BG"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: ub.as_entire_binding(),
            }],
        });

        let debug_vertex_attrs = wgpu::vertex_attr_array![
            0 => Float32x3,  // pos   (offset  0, 12b)
            1 => Float32x4,  // color (offset 12, 16b)
        ];
        let fmt = self.surface_format;
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Debug Lines Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<DebugVertex>() as wgpu::BufferAddress,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &debug_vertex_attrs,
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     fmt,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None, // always on top — no depth test
            multisample:   wgpu::MultisampleState::default(),
            cache:         None,
            multiview_mask: None,
        });

        self.debug_lines_pipeline = Some(pipeline);
        self.debug_lines_bgl      = Some(bgl);
        self.debug_lines_ub       = Some(ub);
        self.debug_lines_bg       = Some(bg);
        debug_log!("Game3DPipeline", "init_debug_lines", "Debug lines pipeline ready");
    }

    // -----------------------------------------------------------------------
    // Render coloured debug line segments (2 DebugVertex per segment).
    // No depth test — always visible on top of the scene.
    // -----------------------------------------------------------------------

    pub fn render_debug_lines(
        &mut self,
        encoder:  &mut wgpu::CommandEncoder,
        view:     &wgpu::TextureView,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        camera:   &Camera,
        vertices: &[DebugVertex],
    ) {
        if vertices.is_empty() { return; }
        if self.debug_lines_pipeline.is_none() {
            self.init_debug_lines(device);
        }

        // Upload view-projection uniform
        let vp = camera.view_projection_matrix();
        queue.write_buffer(
            self.debug_lines_ub.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&vp.to_cols_array()),
        );

        // (Re-)create vertex buffer if it needs to grow
        let needed = vertices.len();
        if self.debug_lines_vb_cap < needed {
            self.debug_lines_vb = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Debug Lines VB"),
                size:               (needed * std::mem::size_of::<DebugVertex>()) as u64,
                usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.debug_lines_vb_cap = needed;
        }
        queue.write_buffer(
            self.debug_lines_vb.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(vertices),
        );

        let byte_len = (vertices.len() * std::mem::size_of::<DebugVertex>()) as u64;

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Debug Lines Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });
        pass.set_pipeline(self.debug_lines_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.debug_lines_bg.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, self.debug_lines_vb.as_ref().unwrap().slice(..byte_len));
        pass.draw(0..vertices.len() as u32, 0..1);
    }

    // -----------------------------------------------------------------------
    // Render wireframe overlay — draws all loaded chunk meshes as line edges.
    // Requires wireframe_supported == true.
    // -----------------------------------------------------------------------

    pub fn render_wireframe(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view:    &wgpu::TextureView,
        queue:   &wgpu::Queue,
        camera:  &Camera,
    ) {
        if !self.wireframe_supported { return; }
        if self.wireframe_pipeline.is_none()
            || self.wireframe_ub.is_none()
            || self.wireframe_bg.is_none()
            || self.depth_view.is_none()
        {
            return;
        }

        // Upload view-projection
        let vp = camera.view_projection_matrix();
        queue.write_buffer(
            self.wireframe_ub.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&vp.to_cols_array()),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Wireframe Pass"),
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
                view:        self.depth_view.as_ref().unwrap(),
                depth_ops:   Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });

        pass.set_pipeline(self.wireframe_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.wireframe_bg.as_ref().unwrap(), &[]);
        for mesh in self.chunk_meshes.values() {
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }
}

/// Returns the rotation matrix for block outline rendering.
/// Mirrors `block_rotation_matrix` in block_model_renderer.rs — must stay in sync.
fn outline_rotation_matrix(rot: u8) -> Mat4 {
    use std::f32::consts::{PI, FRAC_PI_2};
    match rot {
        0 | 2  => Mat4::IDENTITY,
        1      => Mat4::from_rotation_y(PI),
        3      => Mat4::from_rotation_y(FRAC_PI_2),
        4      => Mat4::from_rotation_y(-FRAC_PI_2),
        5 | 11 => Mat4::from_rotation_x(PI),
        6      => Mat4::from_rotation_x(-FRAC_PI_2),
        7      => Mat4::from_rotation_x(FRAC_PI_2),
        8      => Mat4::from_rotation_z(-FRAC_PI_2),
        9      => Mat4::from_rotation_z(FRAC_PI_2),
        10     => Mat4::from_rotation_z(PI),
        12     => Mat4::from_rotation_x(PI) * Mat4::from_rotation_y(FRAC_PI_2),
        13     => Mat4::from_rotation_x(PI) * Mat4::from_rotation_y(-FRAC_PI_2),
        14     => Mat4::from_rotation_z(-FRAC_PI_2),
        15     => Mat4::from_rotation_x(FRAC_PI_2),
        _      => Mat4::IDENTITY,
    }
}
