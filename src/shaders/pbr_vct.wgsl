// =============================================================================
// QubePixel вЂ” PBR + Voxel Global Illumination
// =============================================================================
// Full PBR (Cook-Torrance) with:
//   - Voxel ray-marched sun/moon shadows
//   - Coloured block light from propagation volume
//   - Analytical point + spot lights with voxel shadow occlusion
//   - Normal mapping, emission, ambient occlusion, tone mapping

// ===== Group 0: Standard PBR bindings (unchanged) ==========================

struct Uniforms {
    view_proj:        mat4x4<f32>,   // [0..16]
    camera_pos:       vec4<f32>,     // [16..20]
    sun_direction:    vec4<f32>,     // [20..24]  xyz=dir, w=intensity
    sun_color:        vec4<f32>,     // [24..28]
    moon_direction:   vec4<f32>,     // [28..32]  xyz=dir, w=intensity
    moon_color:       vec4<f32>,     // [32..36]
    ambient_color:    vec4<f32>,     // [36..40]  w=min_level
    shadow_params:    vec4<f32>,     // [40..44]
    ssao_params:      vec4<f32>,     // [44..48]
    time_params:      vec4<f32>,     // [48..52]
    shadow_view_proj: mat4x4<f32>,   // [52..68]
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var atlas_tex:      texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler:  sampler;
@group(0) @binding(3) var normal_tex:     texture_2d<f32>;
@group(0) @binding(4) var normal_sampler: sampler;

// ===== Group 1: VCT bindings ===============================================

struct VCTParams {
    volume_origin: vec4<f32>,    // xyz = world origin, w = volume_size (float)
    inv_size:      vec4<f32>,    // xyz = 1.0 / volume_size, w = gi_intensity
    gi_config:     vec4<f32>,    // x = max_shadow_steps, y = decay, z = ambient_min, w = 0
    light_counts:  vec4<u32>,    // x = point_count, y = spot_count
};

struct PointLight {
    pos_range:        vec4<f32>, // xyz = position, w = range
    color_intensity:  vec4<f32>, // rgb = colour, w = intensity
    source_voxel:     vec4<i32>, // xyz = world block coords of emitter; w = 1 if valid, 0 for entity lights
};

struct SpotLight {
    pos_range:        vec4<f32>, // xyz = position, w = range
    dir_inner:        vec4<f32>, // xyz = direction, w = cos(inner_angle)
    color_intensity:  vec4<f32>, // rgb = colour, w = intensity
    outer_pad:        vec4<f32>, // x = cos(outer_angle), yzw = 0
};

struct EntityShadowAABB {
    min_pad:     vec4<f32>,   // xyz = min, w = unused
    max_opacity: vec4<f32>,   // xyz = max, w = opacity (1 = fully opaque)
};

@group(1) @binding(0) var<uniform>       vct: VCTParams;
@group(1) @binding(1) var                voxel_data:     texture_3d<f32>;
@group(1) @binding(2) var                voxel_radiance: texture_3d<f32>;
@group(1) @binding(3) var                voxel_sampler:  sampler;
@group(1) @binding(4) var<storage, read> point_lights:   array<PointLight>;
@group(1) @binding(5) var<storage, read> spot_lights:    array<SpotLight>;
@group(1) @binding(6) var<storage, read> model_shadow_header: array<vec2<u32>>;
@group(1) @binding(7) var<storage, read> model_shadow_cubes:  array<f32>;
@group(1) @binding(8) var                voxel_tint:     texture_3d<f32>;
@group(1) @binding(9) var<storage, read> entity_shadow_aabbs: array<EntityShadowAABB>;

// Voxel alpha bands — must match VOXEL_ALPHA_* in voxel_volume.rs.
const FLUID_ALPHA_MAX:  f32 = 0.20;  // 0 < a < 0.20 → fluid (water/lava)
const GLASS_ALPHA_MIN:  f32 = 0.55;
const GLASS_ALPHA_MAX:  f32 = 0.85;
// Per-voxel sun/moon light absorption when the ray passes through fluid.
// 0.65 means each water block dims the light to 65% — gradual darkening
// proportional to depth, no hard shadow edge.
const WATER_ABSORPTION: f32 = 0.65;

/// Test ray vs entity AABBs (player + other shadow casters not in voxel grid).
/// Returns 0.0 if the ray hits any entity within `max_t` units of `origin`,
/// 1.0 otherwise. Used as a pre-pass before voxel DDA so entities cast
/// vector shadows just like blocks.
fn entity_shadow_blocked(origin: vec3<f32>, dir: vec3<f32>, max_t: f32) -> f32 {
    let n = vct.light_counts.z;
    for (var i = 0u; i < n; i++) {
        let ab = entity_shadow_aabbs[i];
        let bmin = ab.min_pad.xyz;
        let bmax = ab.max_opacity.xyz;
        if (ray_seg_hits_aabb(origin, dir, max_t, bmin, bmax)) {
            // Opacity controls how much light is blocked. For a fully opaque
            // entity we return its "shadow factor" (1 - opacity) directly so a
            // semi-transparent entity casts a softer shadow.
            return 1.0 - clamp(ab.max_opacity.w, 0.0, 1.0);
        }
    }
    return 1.0;
}

// ===== Vertex I/O ==========================================================

struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec4<f32>,
    @location(2) color_ao:  vec4<f32>,
    @location(3) texcoord:  vec2<f32>,
    @location(4) material:  vec4<f32>,
    @location(5) emission:  vec4<f32>,
    @location(6) tangent:   vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       v_normal:     vec3<f32>,
    @location(1)       v_color:      vec3<f32>,
    @location(2)       v_world_pos:  vec3<f32>,
    @location(3)       v_texcoord:   vec2<f32>,
    @location(4)       v_ao:         f32,
    @location(5)       v_material:   vec2<f32>,
    @location(6)       v_emission:   vec4<f32>,
    @location(7)       v_tangent:    vec3<f32>,
    // Bitangent precomputed in vertex shader to avoid cross() per fragment.
    // Valid for flat (voxel block) faces; interpolation is exact on planar geometry.
    @location(8)       v_bitangent:  vec3<f32>,
    // Alpha mode flag carried in tangent.w: 0 = opaque (alpha forced to 1, no
    // discard), 1 = transparent (glass) → opacity comes from texture.a × v_ao.
    @location(9)       v_alpha_mode: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos     = u.view_proj * vec4<f32>(input.position, 1.0);
    out.v_normal     = normalize(input.normal.xyz);
    out.v_color      = input.color_ao.xyz;
    out.v_world_pos  = input.position;
    out.v_texcoord   = input.texcoord;
    out.v_ao         = input.color_ao.w;
    out.v_material   = input.material.xy;
    out.v_emission   = input.emission;
    out.v_tangent    = normalize(input.tangent.xyz);
    // Precompute bitangent once per vertex instead of once per fragment.
    // On flat voxel faces all vertices share the same N and T, so the
    // interpolated bitangent is identical to recomputing it per-fragment.
    out.v_bitangent  = cross(out.v_normal, out.v_tangent);
    // tangent.w marks transparent (glass) vertices; xyz is the real tangent.
    out.v_alpha_mode = input.tangent.w;
    return out;
}

// ===== Model-block shadow helpers ==========================================

/// Decode two 4-bit nibbles packed in a normalised float channel.
/// Returns (lo, hi) each in [0,1], matching the Rust pack_nibble_pair encoding.
fn decode_nibble_pair(ch: f32) -> vec2<f32> {
    let byte = u32(round(ch * 255.0));
    return vec2<f32>(f32(byte >> 4u) / 15.0, f32(byte & 0xFu) / 15.0);
}

/// Slab-based ray-AABB intersection test for a finite segment.
/// ray_orig: ray origin relative to the voxel cell corner (in [0,1]^3).
/// ray_dir:  unnormalised ray direction.
/// t_seg:    length of the segment (time from entry to exit of voxel cell).
/// Returns true iff the segment intersects [box_min, box_max].
fn ray_seg_hits_aabb(
    ray_orig: vec3<f32>, ray_dir: vec3<f32>, t_seg: f32,
    box_min: vec3<f32>, box_max: vec3<f32>,
) -> bool {
    var t_near = 0.0;
    var t_far  = t_seg;

    if (abs(ray_dir.x) > 0.0001) {
        let t1 = (box_min.x - ray_orig.x) / ray_dir.x;
        let t2 = (box_max.x - ray_orig.x) / ray_dir.x;
        t_near = max(t_near, min(t1, t2));
        t_far  = min(t_far,  max(t1, t2));
        if (t_near > t_far) { return false; }
    } else if (ray_orig.x < box_min.x || ray_orig.x > box_max.x) { return false; }

    if (abs(ray_dir.y) > 0.0001) {
        let t1 = (box_min.y - ray_orig.y) / ray_dir.y;
        let t2 = (box_max.y - ray_orig.y) / ray_dir.y;
        t_near = max(t_near, min(t1, t2));
        t_far  = min(t_far,  max(t1, t2));
        if (t_near > t_far) { return false; }
    } else if (ray_orig.y < box_min.y || ray_orig.y > box_max.y) { return false; }

    if (abs(ray_dir.z) > 0.0001) {
        let t1 = (box_min.z - ray_orig.z) / ray_dir.z;
        let t2 = (box_max.z - ray_orig.z) / ray_dir.z;
        t_near = max(t_near, min(t1, t2));
        t_far  = min(t_far,  max(t1, t2));
        if (t_near > t_far) { return false; }
    } else if (ray_orig.z < box_min.z || ray_orig.z > box_max.z) { return false; }

    return true;
}

// ===== PBR core (Cook-Torrance) ============================================

const PI: f32 = 3.14159265;

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness)
         * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn compute_pbr_light(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
    light_color: vec3<f32>,
    albedo: vec3<f32>, f0: vec3<f32>,
    roughness: f32, metalness: f32,
) -> vec3<f32> {
    let H = normalize(V + L);
    let n_dot_l = max(dot(N, L), 0.0);
    let n_dot_v = max(dot(N, V), 0.001);
    let n_dot_h = max(dot(N, H), 0.0);
    let h_dot_v = max(dot(H, V), 0.0);

    let D = distribution_ggx(n_dot_h, roughness);
    let G = geometry_smith(n_dot_v, n_dot_l, roughness);
    let F = fresnel_schlick(h_dot_v, f0);

    let specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l + 0.0001);
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metalness);
    let diffuse = kD * albedo / PI;

    return (diffuse + specular) * light_color * n_dot_l;
}

// ===== VCT helpers =========================================================

fn world_to_voxel_uv(world_pos: vec3<f32>) -> vec3<f32> {
    return (world_pos - vct.volume_origin.xyz) / vct.volume_origin.w;
}

fn is_in_volume(uv: vec3<f32>) -> bool {
    return all(uv >= vec3<f32>(0.0)) && all(uv < vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// Voxel Ray-Traced Shadow: DDA march through voxel texture (directional)
// ---------------------------------------------------------------------------
// Uses DDA (Amanatides-Woo) for efficient voxel grid traversal.
// Returns 0.0 = fully in shadow, 1.0 = fully lit.
// Emissive voxels are treated as transparent (light sources don't block light).

fn voxel_shadow_directional(world_pos: vec3<f32>, light_dir: vec3<f32>) -> vec3<f32> {
    // Per-channel transmission accumulator. Starts fully lit (white), reduced
    // by partial tints (glass) along the way and clamped to zero when the ray
    // hits an opaque surface or entity.
    var transmission = vec3<f32>(1.0);

    // Entity (player, mobs) shadow test — runs before voxel DDA so entities
    // cast shadows even when their AABB is outside the voxel grid.
    let dir_norm = normalize(light_dir);
    let entity_factor = entity_shadow_blocked(world_pos + dir_norm * 0.05, dir_norm, 256.0);
    if (entity_factor < 0.001) { return vec3<f32>(0.0); }
    transmission = transmission * entity_factor;

    let voxel_origin = vct.volume_origin.xyz;
    let vsize = vct.volume_origin.w;
    let local_pos = world_pos - voxel_origin;

    // If fragment is outside voxel texture bounds, only entity shadow applies
    if (local_pos.x < 0.0 || local_pos.y < 0.0 || local_pos.z < 0.0 ||
        local_pos.x >= vsize || local_pos.y >= vsize || local_pos.z >= vsize) {
        return transmission;
    }

    var voxel = vec3<i32>(
        i32(floor(local_pos.x)),
        i32(floor(local_pos.y)),
        i32(floor(local_pos.z))
    );

    let dir = dir_norm;
    let sz = i32(vsize);

    let step_x: i32 = select(1, -1, dir.x < 0.0);
    let step_y: i32 = select(1, -1, dir.y < 0.0);
    let step_z: i32 = select(1, -1, dir.z < 0.0);

    let t_delta = vec3<f32>(
        select(1.0e10, abs(1.0 / dir.x), abs(dir.x) > 0.0001),
        select(1.0e10, abs(1.0 / dir.y), abs(dir.y) > 0.0001),
        select(1.0e10, abs(1.0 / dir.z), abs(dir.z) > 0.0001)
    );

    var t_max = vec3<f32>(0.0, 0.0, 0.0);
    if (abs(dir.x) > 0.0001) {
        t_max.x = select(
            f32(voxel.x + 1) - local_pos.x,
            f32(voxel.x) - local_pos.x,
            dir.x < 0.0
        ) / dir.x;
    } else { t_max.x = 1.0e10; }
    if (abs(dir.y) > 0.0001) {
        t_max.y = select(
            f32(voxel.y + 1) - local_pos.y,
            f32(voxel.y) - local_pos.y,
            dir.y < 0.0
        ) / dir.y;
    } else { t_max.y = 1.0e10; }
    if (abs(dir.z) > 0.0001) {
        t_max.z = select(
            f32(voxel.z + 1) - local_pos.z,
            f32(voxel.z) - local_pos.z,
            dir.z < 0.0
        ) / dir.z;
    } else { t_max.z = 1.0e10; }
    t_max = max(t_max, vec3<f32>(0.0));

    // Pre-skip: starting voxel may be a model block (surface bias landed inside it).
    // Check its AABB at t_prev = 0 before unconditionally skipping.
    {
        let sv = textureLoad(voxel_data, voxel, 0);
        if (sv.a > 0.2 && sv.a < 0.5) {
            let block_lut = u32(round(sv.r * 255.0));
            let hdr       = model_shadow_header[block_lut];
            let c_start   = hdr.x;
            let c_count   = hdr.y;
            let t_seg     = min(t_max.x, min(t_max.y, t_max.z));
            let vf        = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
            for (var ci = 0u; ci < c_count; ci++) {
                let b    = c_start + ci * 6u;
                let bmin = vec3<f32>(model_shadow_cubes[b],     model_shadow_cubes[b+2u], model_shadow_cubes[b+4u]);
                let bmax = vec3<f32>(model_shadow_cubes[b+1u],  model_shadow_cubes[b+3u], model_shadow_cubes[b+5u]);
                if (ray_seg_hits_aabb(local_pos - vf, dir, t_seg, bmin, bmax)) {
                    return vec3<f32>(0.0);
                }
            }
        }
    }

    var t_prev = 0.0;
    // Skip the first voxel (contains the surface) to avoid self-shadowing
    if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
        voxel.x = voxel.x + step_x;
        t_prev = t_max.x;
        t_max.x = t_max.x + t_delta.x;
    } else if (t_max.y <= t_max.z) {
        voxel.y = voxel.y + step_y;
        t_prev = t_max.y;
        t_max.y = t_max.y + t_delta.y;
    } else {
        voxel.z = voxel.z + step_z;
        t_prev = t_max.z;
        t_max.z = t_max.z + t_delta.z;
    }

    let max_steps = i32(vct.gi_config.x);
    for (var i: i32 = 0; i < max_steps; i = i + 1) {
        if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
            voxel.x >= sz || voxel.y >= sz || voxel.z >= sz) {
            return transmission; // Exited texture — lit by sun (with any accumulated glass tint)
        }

        let data = textureLoad(voxel_data, voxel, 0);

        // Fluid voxel (water/lava): absorb light gradually, no hard shadow.
        if (data.a > 0.0 && data.a < FLUID_ALPHA_MAX) {
            transmission = transmission * WATER_ABSORPTION;
            if (dot(transmission, transmission) < 0.001) {
                return vec3<f32>(0.0);
            }
        // Glass voxel: accumulate coloured transmission and continue.
        } else if (data.a >= GLASS_ALPHA_MIN && data.a <= GLASS_ALPHA_MAX) {
            let tint = textureLoad(voxel_tint, voxel, 0);
            let op   = tint.a;
            // Per-channel transmission: clear glass passes everything, fully
            // opaque glass would block — but we cap at small minimum so glass
            // never goes fully black (use mix to keep it physical).
            let mult = mix(vec3<f32>(1.0), tint.rgb, op) * (1.0 - 0.85 * op);
            transmission = transmission * mult;
            // Early-out when transmission is essentially zero
            if (dot(transmission, transmission) < 0.0005) {
                return vec3<f32>(0.0);
            }
        } else if (data.a > 0.85) {
            // Fully opaque solid block.
            let rad = textureLoad(voxel_radiance, voxel, 0);
            let is_emissive = rad.a > 0.5 && dot(rad.rgb, rad.rgb) > 0.1;
            if (!is_emissive) {
                return vec3<f32>(0.0); // Hit opaque non-emissive block — in shadow
            }
        } else if (data.a > 0.2 && data.a < 0.5) {
            // Model block: look up per-cube AABBs from storage buffers.
            let block_lut = u32(round(data.r * 255.0));
            let hdr       = model_shadow_header[block_lut];
            let c_start   = hdr.x;
            let c_count   = hdr.y;
            let t_next    = min(t_max.x, min(t_max.y, t_max.z));
            let voxel_f   = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
            let ray_entry = local_pos + dir * t_prev - voxel_f;
            for (var ci = 0u; ci < c_count; ci++) {
                let b    = c_start + ci * 6u;
                let bmin = vec3<f32>(model_shadow_cubes[b],     model_shadow_cubes[b+2u], model_shadow_cubes[b+4u]);
                let bmax = vec3<f32>(model_shadow_cubes[b+1u],  model_shadow_cubes[b+3u], model_shadow_cubes[b+5u]);
                if (ray_seg_hits_aabb(ray_entry, dir, t_next - t_prev, bmin, bmax)) {
                    return vec3<f32>(0.0);
                }
            }
        }

        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            voxel.x = voxel.x + step_x;
            t_prev = t_max.x;
            t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y <= t_max.z) {
            voxel.y = voxel.y + step_y;
            t_prev = t_max.y;
            t_max.y = t_max.y + t_delta.y;
        } else {
            voxel.z = voxel.z + step_z;
            t_prev = t_max.z;
            t_max.z = t_max.z + t_delta.z;
        }
    }

    return transmission; // Didn't hit anything — lit (possibly tinted by glass)
}

// ---------------------------------------------------------------------------
// Voxel shadow for point/spot lights (DDA, shorter range)
// ---------------------------------------------------------------------------

// source_vox: xyz = world block coords of emitting block; w = 1 if valid (0 for entity/spot lights).
// When w == 1 the DDA skips that voxel to prevent the source block from self-shadowing.
fn voxel_shadow_to_point(world_pos: vec3<f32>, light_pos: vec3<f32>, source_vox: vec4<i32>) -> vec3<f32> {
    let diff = light_pos - world_pos;
    let dist = length(diff);
    if (dist < 1.0) { return vec3<f32>(1.0); }
    let light_dir = diff / dist;

    var transmission = vec3<f32>(1.0);

    // Entity AABB pre-test — entities cast shadows even outside voxel volume
    let entity_factor = entity_shadow_blocked(world_pos + light_dir * 0.05, light_dir, dist);
    if (entity_factor < 0.001) { return vec3<f32>(0.0); }
    transmission = transmission * entity_factor;

    let voxel_origin = vct.volume_origin.xyz;
    let vsize = vct.volume_origin.w;
    let local_pos = world_pos - voxel_origin;

    // Precompute source block in VCT-local voxel coords for self-shadow skip.
    let vox_origin_i     = vec3<i32>(i32(voxel_origin.x), i32(voxel_origin.y), i32(voxel_origin.z));
    let source_vox_local = source_vox.xyz - vox_origin_i;

    if (local_pos.x < 0.0 || local_pos.y < 0.0 || local_pos.z < 0.0 ||
        local_pos.x >= vsize || local_pos.y >= vsize || local_pos.z >= vsize) {
        return transmission;
    }

    var voxel = vec3<i32>(
        i32(floor(local_pos.x)),
        i32(floor(local_pos.y)),
        i32(floor(local_pos.z))
    );

    let dir = light_dir;
    let sz = i32(vsize);

    let step_x: i32 = select(1, -1, dir.x < 0.0);
    let step_y: i32 = select(1, -1, dir.y < 0.0);
    let step_z: i32 = select(1, -1, dir.z < 0.0);

    let t_delta = vec3<f32>(
        select(1.0e10, abs(1.0 / dir.x), abs(dir.x) > 0.0001),
        select(1.0e10, abs(1.0 / dir.y), abs(dir.y) > 0.0001),
        select(1.0e10, abs(1.0 / dir.z), abs(dir.z) > 0.0001)
    );

    var t_max = vec3<f32>(0.0, 0.0, 0.0);
    if (abs(dir.x) > 0.0001) {
        t_max.x = select(f32(voxel.x + 1) - local_pos.x, f32(voxel.x) - local_pos.x, dir.x < 0.0) / dir.x;
    } else { t_max.x = 1.0e10; }
    if (abs(dir.y) > 0.0001) {
        t_max.y = select(f32(voxel.y + 1) - local_pos.y, f32(voxel.y) - local_pos.y, dir.y < 0.0) / dir.y;
    } else { t_max.y = 1.0e10; }
    if (abs(dir.z) > 0.0001) {
        t_max.z = select(f32(voxel.z + 1) - local_pos.z, f32(voxel.z) - local_pos.z, dir.z < 0.0) / dir.z;
    } else { t_max.z = 1.0e10; }
    t_max = max(t_max, vec3<f32>(0.0));

    // Pre-skip: starting voxel may be a model block.
    {
        let sv = textureLoad(voxel_data, voxel, 0);
        if (sv.a > 0.2 && sv.a < 0.5) {
            let block_lut = u32(round(sv.r * 255.0));
            let hdr       = model_shadow_header[block_lut];
            let c_start   = hdr.x;
            let c_count   = hdr.y;
            let t_seg     = min(t_max.x, min(t_max.y, t_max.z));
            let vf        = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
            for (var ci = 0u; ci < c_count; ci++) {
                let b    = c_start + ci * 6u;
                let bmin = vec3<f32>(model_shadow_cubes[b],     model_shadow_cubes[b+2u], model_shadow_cubes[b+4u]);
                let bmax = vec3<f32>(model_shadow_cubes[b+1u],  model_shadow_cubes[b+3u], model_shadow_cubes[b+5u]);
                if (ray_seg_hits_aabb(local_pos - vf, dir, t_seg, bmin, bmax)) {
                    return vec3<f32>(0.0);
                }
            }
        }
    }

    var t_prev = 0.0;
    // Skip first voxel (surface)
    if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
        voxel.x = voxel.x + step_x;
        t_prev = t_max.x;
        t_max.x = t_max.x + t_delta.x;
    } else if (t_max.y <= t_max.z) {
        voxel.y = voxel.y + step_y;
        t_prev = t_max.y;
        t_max.y = t_max.y + t_delta.y;
    } else {
        voxel.z = voxel.z + step_z;
        t_prev = t_max.z;
        t_max.z = t_max.z + t_delta.z;
    }

    let max_steps = min(i32(dist), 64);
    for (var i: i32 = 0; i < max_steps; i = i + 1) {
        if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
            voxel.x >= sz || voxel.y >= sz || voxel.z >= sz) {
            return transmission;
        }

        // Skip the emitting block itself — prevents self-shadowing from the source block.
        if (source_vox.w != 0 && all(voxel == source_vox_local)) {
            if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
                voxel.x = voxel.x + step_x; t_prev = t_max.x; t_max.x = t_max.x + t_delta.x;
            } else if (t_max.y <= t_max.z) {
                voxel.y = voxel.y + step_y; t_prev = t_max.y; t_max.y = t_max.y + t_delta.y;
            } else {
                voxel.z = voxel.z + step_z; t_prev = t_max.z; t_max.z = t_max.z + t_delta.z;
            }
            continue;
        }

        // Air fast-path: a single voxel_data fetch decides everything.
        // The emissive-skip below needs radiance only for non-air voxels
        // (air radiance is written with alpha 0, so it can never pass the
        // rad.a > 0.5 test) — skipping the radiance fetch in air is exact.
        let data = textureLoad(voxel_data, voxel, 0);
        let opacity = data.a;

        if (opacity > 0.0) {
        // Emissive voxels don't block light
        let rad = textureLoad(voxel_radiance, voxel, 0);
        if (rad.a > 0.5 && dot(rad.rgb, rad.rgb) > 2.0) {
            if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
                voxel.x = voxel.x + step_x;
                t_prev = t_max.x;
                t_max.x = t_max.x + t_delta.x;
            } else if (t_max.y <= t_max.z) {
                voxel.y = voxel.y + step_y;
                t_prev = t_max.y;
                t_max.y = t_max.y + t_delta.y;
            } else {
                voxel.z = voxel.z + step_z;
                t_prev = t_max.z;
                t_max.z = t_max.z + t_delta.z;
            }
            continue;
        }

        // Fluid voxel: gradual absorption, no hard shadow.
        if (opacity < FLUID_ALPHA_MAX) {
            transmission = transmission * WATER_ABSORPTION;
            if (dot(transmission, transmission) < 0.001) {
                return vec3<f32>(0.0);
            }
        // Glass voxel: accumulate coloured transmission and continue.
        } else if (opacity >= GLASS_ALPHA_MIN && opacity <= GLASS_ALPHA_MAX) {
            let tint = textureLoad(voxel_tint, voxel, 0);
            let op   = tint.a;
            let mult = mix(vec3<f32>(1.0), tint.rgb, op) * (1.0 - 0.85 * op);
            transmission = transmission * mult;
            if (dot(transmission, transmission) < 0.0005) {
                return vec3<f32>(0.0);
            }
        } else if (opacity > 0.85) {
            return vec3<f32>(0.0);
        } else if (opacity > 0.2 && opacity < 0.5 && t_prev > 0.05) {
            // Model block: per-cube AABB list lookup (matches directional path).
            let block_lut = u32(round(data.r * 255.0));
            let hdr       = model_shadow_header[block_lut];
            let c_start   = hdr.x;
            let c_count   = hdr.y;
            let t_next    = min(t_max.x, min(t_max.y, t_max.z));
            let voxel_f   = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
            let ray_entry = local_pos + dir * t_prev - voxel_f;
            for (var ci = 0u; ci < c_count; ci++) {
                let b    = c_start + ci * 6u;
                let bmin = vec3<f32>(model_shadow_cubes[b],     model_shadow_cubes[b+2u], model_shadow_cubes[b+4u]);
                let bmax = vec3<f32>(model_shadow_cubes[b+1u],  model_shadow_cubes[b+3u], model_shadow_cubes[b+5u]);
                if (ray_seg_hits_aabb(ray_entry, dir, t_next - t_prev, bmin, bmax)) {
                    return vec3<f32>(0.0);
                }
            }
        }
        } // end non-air branch

        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            voxel.x = voxel.x + step_x;
            t_prev = t_max.x;
            t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y <= t_max.z) {
            voxel.y = voxel.y + step_y;
            t_prev = t_max.y;
            t_max.y = t_max.y + t_delta.y;
        } else {
            voxel.z = voxel.z + step_z;
            t_prev = t_max.z;
            t_max.z = t_max.z + t_delta.z;
        }
    }
    return transmission;
}

// ---------------------------------------------------------------------------
// Sample propagated block light volume
// ---------------------------------------------------------------------------

fn sample_block_light(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // Offset slightly along normal to sample the air voxel in front of the surface
    let sample_pos = world_pos + normal * 0.6;
    let uv = world_to_voxel_uv(sample_pos);
    if (!is_in_volume(uv)) {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(voxel_radiance, voxel_sampler, uv, 0.0).rgb;
}

// ---------------------------------------------------------------------------
// Point light (with PBR and voxel shadow)
// ---------------------------------------------------------------------------

fn eval_point_light(
    N: vec3<f32>, V: vec3<f32>, world_pos: vec3<f32>,
    light: PointLight,
    albedo: vec3<f32>, f0: vec3<f32>,
    roughness: f32, metalness: f32,
) -> vec3<f32> {
    let to_light = light.pos_range.xyz - world_pos;
    let dist = length(to_light);
    let range = light.pos_range.w;
    if (dist > range) { return vec3<f32>(0.0); }

    let L = to_light / max(dist, 0.001);
    // Surface faces away from the light — PBR term is exactly zero,
    // skip attenuation + voxel shadow march entirely.
    if (dot(N, L) <= 0.0) { return vec3<f32>(0.0); }

    // Smooth quadratic attenuation
    let t = clamp(dist / range, 0.0, 1.0);
    let attenuation = (1.0 - t) * (1.0 - t);

    let radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;

    // Voxel shadow toward the light source (respects block shadow toggle).
    // Returns per-channel transmission so coloured glass tints point lights.
    var shadow = vec3<f32>(1.0);
    if (u.shadow_params.y > 0.5) {
        shadow = voxel_shadow_to_point(world_pos, light.pos_range.xyz, light.source_voxel);
    }

    return compute_pbr_light(N, V, L, radiance * shadow, albedo, f0, roughness, metalness);
}

// ---------------------------------------------------------------------------
// Spot light (cone + PBR + voxel shadow)
// ---------------------------------------------------------------------------

fn eval_spot_light(
    N: vec3<f32>, V: vec3<f32>, world_pos: vec3<f32>,
    light: SpotLight,
    albedo: vec3<f32>, f0: vec3<f32>,
    roughness: f32, metalness: f32,
) -> vec3<f32> {
    let to_light = light.pos_range.xyz - world_pos;
    let dist = length(to_light);
    let range = light.pos_range.w;
    if (dist > range) { return vec3<f32>(0.0); }

    let L = to_light / max(dist, 0.001);
    // Surface faces away from the light — PBR term is exactly zero.
    if (dot(N, L) <= 0.0) { return vec3<f32>(0.0); }
    let spot_dir = normalize(light.dir_inner.xyz);
    let cos_angle = dot(-L, spot_dir);

    let inner_cos = light.dir_inner.w;
    let outer_cos = light.outer_pad.x;

    // Smooth cone falloff
    let spot_factor = clamp(
        (cos_angle - outer_cos) / max(inner_cos - outer_cos, 0.001),
        0.0, 1.0
    );
    if (spot_factor <= 0.0) { return vec3<f32>(0.0); }

    let t = clamp(dist / range, 0.0, 1.0);
    let attenuation = (1.0 - t) * (1.0 - t) * spot_factor;

    let radiance = light.color_intensity.rgb * light.color_intensity.w * attenuation;

    // Voxel shadow toward the light source (respects block shadow toggle).
    var shadow = vec3<f32>(1.0);
    if (u.shadow_params.y > 0.5) {
        shadow = voxel_shadow_to_point(world_pos, light.pos_range.xyz, vec4<i32>(0, 0, 0, 0));
    }

    return compute_pbr_light(N, V, L, radiance * shadow, albedo, f0, roughness, metalness);
}

// ===== Fragment Shader =====================================================

// =============================================================================
// Tunable display parameters вЂ” adjust these to control the final look
// =============================================================================

/// Overall exposure multiplier. Lower = darker, higher = brighter.
/// Range: 0.3 (very dark) .. 1.5 (very bright). Default: 0.8
const EXPOSURE: f32 = 0.7;

/// Saturation multiplier. 0.0 = grayscale, 1.0 = normal, 1.5 = vivid.
const SATURATION: f32 = 1.2;

/// Contrast curve power. 1.0 = linear (no change), >1.0 = more contrast.
const CONTRAST_POWER: f32 = 1.15;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // ----- Texture + albedo -----
    // Sample full RGBA: .rgb feeds albedo, .a drives per-texel opacity for glass.
    let tex    = textureSample(atlas_tex, atlas_sampler, input.v_texcoord);
    let albedo = input.v_color * tex.rgb;

    // V must be computed first — used to orient the normal for two-sided rendering.
    let V = normalize(u.camera_pos.xyz - input.v_world_pos);

    // ----- Normal mapping (TBN) -----
    // N_raw: stored outward normal from vertex data (used for VCT sampling offset).
    // N_geom: flipped to face the viewer when the surface is seen from the back side
    //         (transparent/glass blocks rendered with cull_mode: None).
    let N_raw      = normalize(input.v_normal);
    let back_face  = dot(N_raw, V) < 0.0;
    let N_geom     = select(N_raw, -N_raw, back_face);
    let T          = normalize(input.v_tangent);
    // Bitangent precomputed in vs_main; flip for back-face transparent geometry.
    var B          = input.v_bitangent;
    if (back_face) { B = -B; }
    let nm_sample  = textureSample(normal_tex, normal_sampler, input.v_texcoord).rgb;
    let N_t        = normalize(nm_sample * 2.0 - 1.0);
    let N          = normalize(T * N_t.x + B * N_t.y + N_geom * N_t.z);

    let roughness  = input.v_material.x;
    let metalness  = input.v_material.y;
    let vertex_ao  = input.v_ao;

    let f0 = mix(vec3<f32>(0.04), albedo, metalness);

    var lo = vec3<f32>(0.0);

    // shadow_params.x = sun/moon shadow toggle (1.0 = on, 0.0 = off)
    // shadow_params.y = block light shadow toggle (1.0 = on, 0.0 = off)
    let sun_shadow_on   = u.shadow_params.x > 0.5;
    let block_shadow_on = u.shadow_params.y > 0.5;

    // Bias the shadow origin along the outward face normal (N_raw, not viewer-dependent N_geom).
    // Without this, world_pos lands exactly on a voxel boundary — sometimes mapping to the
    // glass voxel itself (skipped), sometimes to the adjacent air voxel (first DDA step hits
    // glass), causing z-fighting. The bias makes the starting voxel consistently the air cell
    // just outside the face.
    let shadow_origin = input.v_world_pos + N_raw * 0.01;

    // ===========================================
    // 1. Sun light + voxel shadow
    // ===========================================
    let sun_intensity = u.sun_direction.w;
    if (sun_intensity > 0.001) {
        let sun_L = normalize(u.sun_direction.xyz);
        // N·L ≤ 0 → compute_pbr_light is exactly zero; skip the shadow ray march.
        if (dot(N, sun_L) > 0.0) {
            let sun_radiance = u.sun_color.rgb * sun_intensity;
            var sun_shadow = vec3<f32>(1.0);
            if (sun_shadow_on) { sun_shadow = voxel_shadow_directional(shadow_origin, sun_L); }
            lo += compute_pbr_light(N, V, sun_L, sun_radiance * sun_shadow, albedo, f0, roughness, metalness);
        }
    }

    // ===========================================
    // 2. Moon light + voxel shadow
    // ===========================================
    let moon_intensity = u.moon_direction.w;
    if (moon_intensity > 0.001) {
        let moon_L = normalize(u.moon_direction.xyz);
        // N·L ≤ 0 → contribution is exactly zero; skip the shadow ray march.
        if (dot(N, moon_L) > 0.0) {
            let moon_radiance = u.moon_color.rgb * moon_intensity;
            var moon_shadow = vec3<f32>(1.0);
            if (sun_shadow_on) { moon_shadow = voxel_shadow_directional(shadow_origin, moon_L); }
            lo += compute_pbr_light(N, V, moon_L, moon_radiance * moon_shadow, albedo, f0, roughness, metalness);
        }
    }

    // ===========================================
    // 3. Block light from propagation volume
    // ===========================================
    let gi_intensity = vct.inv_size.w; // stored in w component
    let block_light  = sample_block_light(input.v_world_pos, N_raw);
    // Diffuse indirect: metals have no diffuse component (kD в‰€ 0)
    let kD_gi = (1.0 - metalness);
    lo += block_light * albedo * kD_gi * gi_intensity;

    // ===========================================
    // 4. Dynamic point lights
    // ===========================================
    let n_point = vct.light_counts.x;
    for (var i = 0u; i < n_point; i++) {
        lo += eval_point_light(N, V, shadow_origin, point_lights[i],
                               albedo, f0, roughness, metalness);
    }

    // ===========================================
    // 5. Dynamic spot lights
    // ===========================================
    let n_spot = vct.light_counts.y;
    for (var i = 0u; i < n_spot; i++) {
        lo += eval_spot_light(N, V, shadow_origin, spot_lights[i],
                              albedo, f0, roughness, metalness);
    }

    // ===========================================
    // 6. Self-emission (HDR — ACES handles range)
    // ===========================================
    // Multiplied by albedo so texture detail is preserved — bright texture spots
    // glow brighter, dark crevices stay dark instead of a uniform colour overlay.
    let raw_intensity = input.v_emission.w;
    let emission = input.v_emission.rgb * albedo * abs(raw_intensity);

    // ===========================================
    // 7. Ambient (minimum base light everywhere)
    // ===========================================
    let ambient = u.ambient_color.rgb * albedo * vertex_ao * 0.35;

    // ===========================================
    // Final compositing
    // ===========================================
    var color = lo + emission + ambient;

    // --- Exposure ---
    color = color * EXPOSURE;

    // --- ACES filmic tone mapping ---
    let aces_a = vec3<f32>(2.51);
    let aces_b = vec3<f32>(0.03);
    let aces_c = vec3<f32>(2.43);
    let aces_d = vec3<f32>(0.59);
    let aces_e = vec3<f32>(0.14);
    color = clamp(
        (color * (aces_a * color + aces_b)) / (color * (aces_c * color + aces_d) + aces_e),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    // --- Saturation ---
    let lum = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    color = mix(vec3<f32>(lum), color, SATURATION);

    // --- Contrast (power curve around mid-grey) ---
    color = pow(color, vec3<f32>(CONTRAST_POWER));

    // ----- Output alpha -----
    // Opaque geometry (v_alpha_mode ≈ 0) is written fully opaque. Glass geometry
    // (v_alpha_mode ≈ 1) takes opacity from the texture's alpha channel × the
    // per-block opacity (carried in v_ao): opaque texels stay solid, partial texels
    // stay partial, and fully-clear texels are discarded (→ no colour blend and no
    // depth write), so a textured glass block is no longer uniformly see-through.
    var out_alpha = 1.0;
    if (input.v_alpha_mode > 0.5) {
        out_alpha = tex.a * vertex_ao;
        if (out_alpha < 0.02) { discard; }
    }
    return vec4<f32>(color, out_alpha);
}
