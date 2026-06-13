// =============================================================================
// QubePixel — VCTSystem: GPU resources for Voxel Global Illumination
// =============================================================================
//
// Owns:
//   - 3D textures: voxel_data (RGBA8), voxel_emission (RGBA8), radiance A/B (RGBA16F)
//   - Compute pipelines: inject, propagate (ping-pong)
//   - Bind group for the fragment shader (group 1)
//   - Point/spot light storage buffers
//
// Called each frame from GameScreen:
//   1. upload_volume()   — writes voxel_data + voxel_emission from CPU snapshot
//   2. dispatch_gi()     — runs inject + N propagation iterations
//   3. bind_group()      — returns the group(1) bind group for the render pass

use crate::core::vct::voxel_volume::{VOLUME_SIZE, VoxelSnapshot, pack_volume, vol_idx, build_volume_luts};
use crate::core::vct::dynamic_lights::{PointLightGPU, SpotLightGPU, EntityShadowAABB};
use crate::core::gameobjects::block::BlockRegistry;
use crate::{debug_log, flow_debug_log};

/// Maximum number of entity AABBs cast into the shadow ray test (e.g. player, mobs).
const MAX_ENTITY_AABBS: usize = 32;

/// Propagation step counts per GI mode (indexed by GI_MODE value).
/// All counts are even so (steps-1) is odd and the last ping-pong write lands in radiance_a
/// (which is what the pre-built frag_bg binds at binding 2).
///   Mode 0 = Off    — unused (dispatch skipped)
///   Mode 1 = Mono16 — 16 steps  (~16 block radius, monochromatic)
///   Mode 2 = Mono32 — 32 steps  (~32 block radius, monochromatic)
///   Mode 3 = RGB32  — 32 steps  (~32 block radius, full colour)
///   Mode 4 = Full   — 64 steps  (~64 block radius, full colour)
const PROPAGATION_STEPS_BY_MODE: [u32; 5] = [0, 16, 32, 32, 64];

/// Propagation steps dispatched per frame (must be EVEN so the A↔B ping-pong
/// always ends on `radiance_a`, the texture the fragment shader samples).
/// Amortises the up-to-64-step convergence across several frames so that a single
/// voxel edit / volume re-upload no longer stalls one frame with ~65 compute passes.
const PROPAGATION_STEPS_PER_FRAME: u32 = 8;

/// Light decay per propagation step (0..1). Higher = light travels farther.
/// With max-based propagation: 0.97 → after 64 steps = 14% initial brightness.
const PROPAGATION_DECAY: f32 = 0.97;

/// Max voxel shadow ray-march steps (sent to fragment shader).
const MAX_SHADOW_STEPS: f32 = 64.0;

/// Default GI intensity multiplier.
/// Controls how bright propagated block light appears on surfaces.
const GI_INTENSITY: f32 = 2.0;

// ---------------------------------------------------------------------------
// GPU uniform structs — must match WGSL byte-for-byte
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InjectParams {
    volume_size: [u32; 4], // x = size, yzw = 0
    _pad: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PropagateParams {
    volume_size: [u32; 4],
    config: [f32; 4], // x = decay, y = iteration, zw = 0
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VCTFragParams {
    pub volume_origin: [f32; 4], // xyz = world origin, w = volume_size (float)
    pub inv_size: [f32; 4],      // xyz = 1/size, w = gi_intensity
    pub gi_config: [f32; 4],     // x = max_shadow_steps, y = decay, z = ambient_min, w = 0
    /// x = point lights, y = spot lights, z = entity shadow AABBs, w = 0
    pub light_counts: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<VCTFragParams>() == 64);

// ---------------------------------------------------------------------------
// VCTSystem
// ---------------------------------------------------------------------------

pub struct VCTSystem {
    // 3D textures
    voxel_data_tex:     wgpu::Texture,
    voxel_data_view:    wgpu::TextureView,
    voxel_emission_tex: wgpu::Texture,
    voxel_emission_view: wgpu::TextureView,
    /// Per-voxel tint/opacity for glass blocks (RGBA8: rgb=tint, a=opacity).
    /// All zero for non-glass voxels.
    voxel_tint_tex:     wgpu::Texture,
    voxel_tint_view:    wgpu::TextureView,
    radiance_a_tex:     wgpu::Texture,
    radiance_a_view:    wgpu::TextureView,
    radiance_b_tex:     wgpu::Texture,
    radiance_b_view:    wgpu::TextureView,

    // Sampler for 3D textures (linear filtering for smooth GI)
    voxel_sampler: wgpu::Sampler,

    // Compute pipelines
    inject_pipeline:    wgpu::ComputePipeline,
    propagate_pipeline: wgpu::ComputePipeline,

    // Compute bind group layouts
    inject_bgl:    wgpu::BindGroupLayout,
    propagate_bgl: wgpu::BindGroupLayout,

    // Compute uniform buffers
    inject_ub:    wgpu::Buffer,
    propagate_ub: wgpu::Buffer,

    // Fragment bind group layout + live bind group
    pub frag_bgl:    wgpu::BindGroupLayout,
    frag_ub:         wgpu::Buffer,
    point_light_buf: wgpu::Buffer,
    spot_light_buf:  wgpu::Buffer,
    // Per-cube model shadow data (bindings 6, 7)
    // header: 256 × vec2<u32> → (start_index, cube_count) indexed by block_id
    // cubes:  flat f32 array, 6 floats per cube AABB [xmin,xmax,ymin,ymax,zmin,zmax]
    model_shadow_header_buf: wgpu::Buffer,
    model_shadow_cubes_buf:  wgpu::Buffer,
    /// Entity AABBs that cast shadows (e.g. player) — binding 9.
    entity_aabb_buf:  wgpu::Buffer,

    // Cached compute bind groups — built once, reused every frame.
    inject_bg:       wgpu::BindGroup,   // voxel_emission → radiance_a
    propagate_bg_ab: wgpu::BindGroup,   // step%2==0: read A, write B
    propagate_bg_ba: wgpu::BindGroup,   // step%2==1: read B, write A
    /// Pre-built fragment bind group — content updated via frag_ub each frame.
    pub frag_bg:     wgpu::BindGroup,

    // Current volume state
    volume_origin: [f32; 3],
    volume_uploaded: bool,
    /// Tracks last applied GI mode so uniform buffers are only rewritten on change.
    last_gi_mode: u32,
    /// Set when the voxel volume (or GI mode) changed since the last propagation.
    /// Propagation re-runs from inject each time, so when inputs are unchanged the
    /// converged radiance in `radiance_a` is already exact — dispatch is skipped.
    radiance_dirty: bool,
    /// Propagation steps already dispatched for the current radiance solve.
    /// Reset to 0 on each re-inject, then advanced by PROPAGATION_STEPS_PER_FRAME
    /// until it reaches the active GI mode's total step count (amortised solve).
    propagation_cursor: u32,
    /// World-space origin of the volume currently resident in the GPU textures.
    /// A reported dirty-region can be patched in place only while this matches the
    /// new snapshot's origin; otherwise the camera moved and a full upload is needed.
    last_uploaded_origin: [i32; 3],
    /// Hash of the last uploaded model-shadow data; skips redundant rebuilds.
    model_shadow_version: u64,
}

impl VCTSystem {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        debug_log!("VCTSystem", "new", "Creating VCT system, volume={}^3", VOLUME_SIZE);

        // ===== 3D textures =====
        let vol_extent = wgpu::Extent3d {
            width: VOLUME_SIZE,
            height: VOLUME_SIZE,
            depth_or_array_layers: VOLUME_SIZE,
        };

        let voxel_data_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VCT voxel_data"),
            size: vol_extent,
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let voxel_emission_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VCT voxel_emission"),
            size: vol_extent,
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let voxel_tint_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VCT voxel_tint"),
            size: vol_extent,
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let radiance_a_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VCT radiance_A"),
            size: vol_extent,
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let radiance_b_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VCT radiance_B"),
            size: vol_extent,
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let default_view = wgpu::TextureViewDescriptor::default();
        let voxel_data_view     = voxel_data_tex.create_view(&default_view);
        let voxel_emission_view = voxel_emission_tex.create_view(&default_view);
        let voxel_tint_view     = voxel_tint_tex.create_view(&default_view);
        let radiance_a_view     = radiance_a_tex.create_view(&default_view);
        let radiance_b_view     = radiance_b_tex.create_view(&default_view);

        let voxel_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("VCT 3D sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ===== Inject compute pipeline =====
        let inject_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VCT Inject Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::borrow::Cow::Borrowed(include_str!("../../shaders/vct_inject.wgsl")),
            ),
        });

        let inject_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VCT Inject BGL"),
            entries: &[
                // @binding(0) uniform InjectParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1) voxel_emission texture_3d
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2) radiance_out storage_3d write
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let inject_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VCT Inject PL"),
            bind_group_layouts: &[Some(&inject_bgl)],
            immediate_size: 0,
        });

        let inject_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VCT Inject Pipeline"),
            layout: Some(&inject_pl),
            module: &inject_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ===== Propagate compute pipeline =====
        let propagate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VCT Propagate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                std::borrow::Cow::Borrowed(include_str!("../../shaders/vct_propagate.wgsl")),
            ),
        });

        let propagate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VCT Propagate BGL"),
            entries: &[
                // @binding(0) uniform PropagateParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1) voxel_data texture_3d
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2) voxel_emission texture_3d
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(3) radiance_in texture_3d (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(4) radiance_out storage_3d (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // @binding(5) voxel_tint texture_3d (read) — glass colouring
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let propagate_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VCT Propagate PL"),
            bind_group_layouts: &[Some(&propagate_bgl)],
            immediate_size: 0,
        });

        let propagate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("VCT Propagate Pipeline"),
                layout: Some(&propagate_pl),
                module: &propagate_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ===== Uniform buffers for compute =====
        let inject_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VCT Inject UB"),
            size: std::mem::size_of::<InjectParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&inject_ub, 0, bytemuck::bytes_of(&InjectParams {
            volume_size: [VOLUME_SIZE, 0, 0, 0],
            _pad: [0.0; 4],
        }));

        let propagate_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VCT Propagate UB"),
            size: std::mem::size_of::<PropagateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Shader only reads config.x (decay); config.y (step) is unused.
        queue.write_buffer(&propagate_ub, 0, bytemuck::bytes_of(&PropagateParams {
            volume_size: [VOLUME_SIZE, 0, 0, 0],
            config: [PROPAGATION_DECAY, 0.0, 0.0, 0.0],
        }));

        // ===== Fragment bind group layout (group 1) =====
        let frag_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VCT Fragment BGL"),
            entries: &[
                // @binding(0) VCTParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1) voxel_data texture_3d
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2) voxel_radiance texture_3d
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(3) voxel_sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // @binding(4) point_lights storage
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(5) spot_lights storage
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // @binding(6) model_shadow_header — (start, count) per block_id
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // @binding(7) model_shadow_cubes — flat f32 cube AABBs
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // @binding(8) voxel_tint texture_3d — glass tint/opacity per voxel
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(9) entity_shadow_aabbs storage — non-voxel shadow casters
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let frag_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VCT Fragment UB"),
            size: std::mem::size_of::<VCTFragParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Light buffers — pre-allocate for up to 64 lights each
        let point_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VCT PointLights"),
            size: (std::mem::size_of::<PointLightGPU>() * 64).max(32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spot_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VCT SpotLights"),
            size: (std::mem::size_of::<SpotLightGPU>() * 64).max(64) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Model shadow buffers — pre-allocate fixed size.
        // header: 256 block IDs × 2 u32 (start, count) = 2 KB
        let model_shadow_header_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VCT ModelShadowHeader"),
            size: 256 * 2 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // cubes: 256 block types × 16 cubes max × 6 f32 = 96 KB
        let model_shadow_cubes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VCT ModelShadowCubes"),
            size: (256 * 16 * 6 * 4).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Entity shadow AABB storage. Layout matches `EntityShadowAABB`:
        // [min.xyz, _, max.xyz, opacity] — 32 bytes per entry.
        // We always allocate the maximum; the shader reads `light_counts.z`
        // entries from this buffer (set in `prepare_fragment`).
        let entity_aabb_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VCT EntityShadowAABBs"),
            size: (std::mem::size_of::<EntityShadowAABB>() * MAX_ENTITY_AABBS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ===== Pre-built compute bind groups =====
        // Built once here; reused every frame to avoid 65 allocations per frame.

        let inject_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("VCT Inject BG (cached)"),
            layout:  &inject_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: inject_ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&voxel_emission_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&radiance_a_view) },
            ],
        });

        let propagate_bg_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("VCT Propagate A→B (cached)"),
            layout:  &propagate_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: propagate_ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&voxel_data_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&voxel_emission_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&radiance_a_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&radiance_b_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&voxel_tint_view) },
            ],
        });

        let propagate_bg_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("VCT Propagate B→A (cached)"),
            layout:  &propagate_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: propagate_ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&voxel_data_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&voxel_emission_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&radiance_b_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&radiance_a_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&voxel_tint_view) },
            ],
        });

        // Fragment bind group — (PROPAGATION_STEPS-1)%2 == 1 → last step wrote radiance_a.
        let frag_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("VCT Fragment BG (cached)"),
            layout:  &frag_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frag_ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&voxel_data_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&radiance_a_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&voxel_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: point_light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: spot_light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: model_shadow_header_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: model_shadow_cubes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&voxel_tint_view) },
                wgpu::BindGroupEntry { binding: 9, resource: entity_aabb_buf.as_entire_binding() },
            ],
        });

        debug_log!("VCTSystem", "new", "VCT system initialised");

        Self {
            voxel_data_tex,
            voxel_data_view,
            voxel_emission_tex,
            voxel_emission_view,
            voxel_tint_tex,
            voxel_tint_view,
            radiance_a_tex,
            radiance_a_view,
            radiance_b_tex,
            radiance_b_view,
            voxel_sampler,
            inject_pipeline,
            propagate_pipeline,
            inject_bgl,
            propagate_bgl,
            inject_ub,
            propagate_ub,
            inject_bg,
            propagate_bg_ab,
            propagate_bg_ba,
            frag_bgl,
            frag_ub,
            frag_bg,
            point_light_buf,
            spot_light_buf,
            model_shadow_header_buf,
            model_shadow_cubes_buf,
            entity_aabb_buf,
            volume_origin: [0.0; 3],
            volume_uploaded: false,
            last_gi_mode: u32::MAX, // force UB write on first dispatch
            radiance_dirty: false,
            propagation_cursor: 0,
            last_uploaded_origin: [i32::MAX; 3], // force first upload through the full path
            model_shadow_version: u64::MAX, // force first upload
        }
    }

    // -----------------------------------------------------------------------
    // Upload voxel data from CPU snapshot
    // -----------------------------------------------------------------------

    pub fn upload_volume(
        &mut self,
        queue: &wgpu::Queue,
        snapshot: &VoxelSnapshot,
        registry: &BlockRegistry,
    ) {
        // Incremental fast path: when the only change since the resident upload was a
        // small player edit (a reported world-space dirty box) and the volume origin
        // is unchanged, patch just that sub-box instead of re-streaming the whole
        // 128³ (3×8 MiB) volume + re-packing 2M voxels.
        if let Some((wmin, wmax)) = snapshot.dirty_region {
            if snapshot.origin == self.last_uploaded_origin {
                self.upload_region(queue, snapshot, registry, wmin, wmax);
                return;
            }
        }

        let (data_pixels, emission_pixels, tint_pixels) = pack_volume(snapshot, registry);

        self.volume_origin = [
            snapshot.origin[0] as f32,
            snapshot.origin[1] as f32,
            snapshot.origin[2] as f32,
        ];

        let size = VOLUME_SIZE;
        let extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        };

        // Upload voxel_data (RGBA8)
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.voxel_data_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data_pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * size),
                rows_per_image: Some(size),
            },
            extent,
        );

        // Upload voxel_emission (RGBA8)
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.voxel_emission_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&emission_pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * size),
                rows_per_image: Some(size),
            },
            extent,
        );

        // Upload voxel_tint (RGBA8) — only non-zero for glass voxels.
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.voxel_tint_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&tint_pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * size),
                rows_per_image: Some(size),
            },
            extent,
        );

        self.volume_uploaded = true;
        self.radiance_dirty  = true;
        self.last_uploaded_origin = snapshot.origin;
    }

    // -----------------------------------------------------------------------
    // Incremental sub-box upload (patch only the voxels a player edit touched)
    // -----------------------------------------------------------------------

    /// Patch only a sub-box of the resident voxel volume. Caller guarantees the
    /// snapshot origin matches `last_uploaded_origin`, so volume-local coordinates
    /// line up with the GPU textures. `wmin`/`wmax` are an inclusive world-space AABB.
    fn upload_region(
        &mut self,
        queue: &wgpu::Queue,
        snapshot: &VoxelSnapshot,
        registry: &BlockRegistry,
        wmin: [i32; 3],
        wmax: [i32; 3],
    ) {
        let origin = snapshot.origin;
        let sz = VOLUME_SIZE as i32;

        // Reject boxes that fall entirely outside the volume (nothing to patch).
        if wmax[0] < origin[0] || wmin[0] >= origin[0] + sz
            || wmax[1] < origin[1] || wmin[1] >= origin[1] + sz
            || wmax[2] < origin[2] || wmin[2] >= origin[2] + sz
        {
            return;
        }

        // World AABB → volume-local, clamped to [0, VOLUME_SIZE).
        let lx0 = (wmin[0] - origin[0]).clamp(0, sz - 1);
        let ly0 = (wmin[1] - origin[1]).clamp(0, sz - 1);
        let lz0 = (wmin[2] - origin[2]).clamp(0, sz - 1);
        let lx1 = (wmax[0] - origin[0]).clamp(0, sz - 1);
        let ly1 = (wmax[1] - origin[1]).clamp(0, sz - 1);
        let lz1 = (wmax[2] - origin[2]).clamp(0, sz - 1);

        let w = (lx1 - lx0 + 1) as u32;
        let h = (ly1 - ly0 + 1) as u32;
        let d = (lz1 - lz0 + 1) as u32;
        let count = (w * h * d) as usize;

        // Pack just the sub-box densely (row-major X→Y→Z, matching write_texture).
        let (lut_data, lut_emission, lut_tint) = build_volume_luts(registry);
        let mut data     = vec![[0u8; 4]; count];
        let mut emission = vec![[0u8; 4]; count];
        let mut tint     = vec![[0u8; 4]; count];

        let mut di = 0usize;
        for lz in 0..d {
            for ly in 0..h {
                for lx in 0..w {
                    let vx = lx0 as u32 + lx;
                    let vy = ly0 as u32 + ly;
                    let vz = lz0 as u32 + lz;
                    let bid = snapshot.blocks[vol_idx(vx, vy, vz)] as usize;
                    data[di]     = lut_data[bid];
                    emission[di] = lut_emission[bid];
                    tint[di]     = lut_tint[bid];
                    di += 1;
                }
            }
        }

        let dst_origin = wgpu::Origin3d { x: lx0 as u32, y: ly0 as u32, z: lz0 as u32 };
        let extent = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d };

        for (tex, pixels) in [
            (&self.voxel_data_tex, &data),
            (&self.voxel_emission_tex, &emission),
            (&self.voxel_tint_tex, &tint),
        ] {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: dst_origin,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(pixels.as_slice()),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(h),
                },
                extent,
            );
        }

        self.volume_uploaded = true;
        self.radiance_dirty  = true;

        flow_debug_log!("VCTSystem", "upload_region",
            "patched [{},{},{}]..[{},{},{}] ({} voxels)",
            lx0, ly0, lz0, lx1, ly1, lz1, count);
    }

    // -----------------------------------------------------------------------
    // Dispatch GI compute passes (inject + propagate)
    // -----------------------------------------------------------------------

    pub fn dispatch_gi(&mut self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue) {
        if !self.volume_uploaded {
            return;
        }
        let gi_mode = crate::core::config::gi_mode();
        if gi_mode == 0 {
            return;
        }

        // Rewrite compute uniform buffers only when the mode changes (typically once per session).
        if gi_mode != self.last_gi_mode {
            let mono = if gi_mode == 1 || gi_mode == 2 { 1u32 } else { 0u32 };
            queue.write_buffer(&self.inject_ub, 0, bytemuck::bytes_of(&InjectParams {
                volume_size: [VOLUME_SIZE, mono, 0, 0],
                _pad: [0.0; 4],
            }));
            queue.write_buffer(&self.propagate_ub, 0, bytemuck::bytes_of(&PropagateParams {
                volume_size: [VOLUME_SIZE, 0, 0, 0],
                config: [PROPAGATION_DECAY, mono as f32, 0.0, 0.0],
            }));
            self.last_gi_mode = gi_mode;
            // Mode change alters propagation parameters — radiance must be rebuilt.
            self.radiance_dirty = true;
            debug_log!("VCTSystem", "dispatch_gi",
                "GI mode changed to {} (mono={}, steps={})",
                gi_mode, mono, PROPAGATION_STEPS_BY_MODE[gi_mode as usize]);
        }

        let total_steps = PROPAGATION_STEPS_BY_MODE[gi_mode as usize];

        // workgroup_size(8, 8, 4): XY groups = ceil(128/8)=16, Z groups = ceil(128/4)=32
        let groups_xy = (VOLUME_SIZE + 7) / 8;
        let groups_z  = (VOLUME_SIZE + 3) / 4;

        // A changed volume (or GI mode) restarts the solve: re-inject emission into
        // radiance_a and reset the per-frame step cursor. Propagation is then spread
        // across the next few frames instead of all-at-once, removing the spike.
        if self.radiance_dirty {
            self.radiance_dirty = false;
            self.propagation_cursor = 0;

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VCT Inject"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inject_pipeline);
            pass.set_bind_group(0, &self.inject_bg, &[]);
            pass.dispatch_workgroups(groups_xy, groups_xy, groups_z);
        }

        // Converged already — radiance_a holds the exact result, skip the chain.
        if self.propagation_cursor >= total_steps {
            return;
        }

        // Dispatch a bounded, even-sized chunk of propagation steps this frame.
        // Starting from an even cursor and advancing by an even count guarantees the
        // final write lands back in radiance_a (the texture frag_bg samples).
        let start = self.propagation_cursor;
        let end = (start + PROPAGATION_STEPS_PER_FRAME).min(total_steps);
        for step in start..end {
            let bg = if step % 2 == 0 { &self.propagate_bg_ab } else { &self.propagate_bg_ba };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VCT Propagate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.propagate_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups_xy, groups_xy, groups_z);
        }
        self.propagation_cursor = end;

        flow_debug_log!("VCTSystem", "dispatch_gi",
            "propagated steps {}..{} of {}", start, end, total_steps);
    }

    // -----------------------------------------------------------------------
    // Update dynamic lights
    // -----------------------------------------------------------------------

    pub fn update_lights(
        &self,
        queue: &wgpu::Queue,
        point_lights: &[PointLightGPU],
        spot_lights: &[SpotLightGPU],
    ) {
        if !point_lights.is_empty() {
            queue.write_buffer(
                &self.point_light_buf,
                0,
                bytemuck::cast_slice(point_lights),
            );
        }
        if !spot_lights.is_empty() {
            queue.write_buffer(
                &self.spot_light_buf,
                0,
                bytemuck::cast_slice(spot_lights),
            );
        }
    }

    // -----------------------------------------------------------------------
    // Build fragment bind group (group 1) + update uniforms
    // -----------------------------------------------------------------------

    /// Update per-frame fragment uniforms (camera origin, light counts).
    /// The bind group itself is pre-built at init — no allocations here.
    pub fn prepare_fragment(
        &self,
        queue: &wgpu::Queue,
        point_count: u32,
        spot_count: u32,
        entity_aabb_count: u32,
    ) {
        let size_f = VOLUME_SIZE as f32;
        let inv = 1.0 / size_f;
        let params = VCTFragParams {
            volume_origin: [
                self.volume_origin[0],
                self.volume_origin[1],
                self.volume_origin[2],
                size_f,
            ],
            inv_size: [inv, inv, inv, GI_INTENSITY],
            gi_config: [crate::core::config::shadow_quality() as f32, PROPAGATION_DECAY, 0.08, 0.0],
            light_counts: [point_count, spot_count, entity_aabb_count, 0],
        };
        queue.write_buffer(&self.frag_ub, 0, bytemuck::bytes_of(&params));
    }

    /// Upload an array of entity shadow AABBs (player + future mobs).
    /// At most `MAX_ENTITY_AABBS` entries are written; surplus is ignored.
    /// The shader reads `light_counts.z` entries — pass the same value to
    /// `prepare_fragment`.
    pub fn update_entity_aabbs(&self, queue: &wgpu::Queue, aabbs: &[EntityShadowAABB]) {
        if aabbs.is_empty() { return; }
        let n = aabbs.len().min(MAX_ENTITY_AABBS);
        queue.write_buffer(&self.entity_aabb_buf, 0, bytemuck::cast_slice(&aabbs[..n]));
    }

    /// Rebuild and upload the per-cube model shadow buffers from the block registry.
    /// Called each frame, but rebuilds/uploads only when the registry's
    /// model-shadow version changed (typically a handful of times at startup).
    pub fn update_model_shadows(&mut self, queue: &wgpu::Queue, registry: &BlockRegistry) {
        if registry.model_shadow_version() == self.model_shadow_version {
            return;
        }
        self.model_shadow_version = registry.model_shadow_version();
        debug_log!("VCTSystem", "update_model_shadows",
            "Rebuilding model shadow buffers (version {})", self.model_shadow_version);

        let mut header = vec![0u32; 512]; // 256 × (start, count)
        let mut cubes: Vec<f32> = Vec::new();

        for block_id in 1u8..=255u8 {
            if let Some(cube_list) = registry.model_shadow_cubes(block_id) {
                if cube_list.is_empty() { continue; }
                let start = cubes.len() as u32;
                let count = cube_list.len() as u32;
                header[(block_id as usize) * 2    ] = start;
                header[(block_id as usize) * 2 + 1] = count;
                for c in cube_list {
                    cubes.extend_from_slice(c);
                }
            }
        }

        queue.write_buffer(&self.model_shadow_header_buf, 0, bytemuck::cast_slice(&header));
        if !cubes.is_empty() {
            queue.write_buffer(&self.model_shadow_cubes_buf, 0, bytemuck::cast_slice(&cubes));
        }
    }

    pub fn is_ready(&self) -> bool {
        self.volume_uploaded
    }
}
