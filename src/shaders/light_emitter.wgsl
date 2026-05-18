// =============================================================================
// QubePixel — light_emitter.wgsl
//
// Renders bright billboard glows at each active light source position.
// Additive blend: src_alpha*rgb + dst — natural HDR glow with no masking.
// =============================================================================

struct Uniforms {
    vp: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertIn {
    @location(0) pos:       vec3<f32>,
    @location(1) color:     vec4<f32>,  // rgb = light color, a = intensity scale
    @location(2) uv:        vec2<f32>,  // 0..1 quad UV, center = (0.5, 0.5)
}

struct VertOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv:    vec2<f32>,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;
    out.clip  = u.vp * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    out.uv    = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    if d >= 1.0 { discard; }

    let intensity = in.color.a;
    let col = in.color.rgb * intensity;

    // Sharp saturated inner core
    let core = pow(max(0.0, 1.0 - d * 0.5), 5.0) * 3.0;
    // Medium glow ring
    let mid  = pow(1.0 - d, 2.5) * 1.5;
    // Soft outer halo
    let halo = pow(1.0 - d, 1.2) * 0.6;

    let glow  = core + mid + halo;
    let alpha = clamp(glow * 0.6, 0.0, 1.0);

    return vec4<f32>(col * glow, alpha);
}
