// Field background: sample the current front field texture and paint a heat-map.

@group(0) @binding(0) var field_tex : texture_2d<f32>;
@group(0) @binding(1) var samp      : sampler;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv        : vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
  // Fullscreen triangle using vertex index to generate positions
  var out: VSOut;
  
  // Convert vertex index to position using bit manipulation
  let x = f32(vid & 1u) * 4.0 - 1.0;
  let y = f32((vid & 2u) >> 1u) * 4.0 - 1.0;
  
  out.pos = vec4<f32>(x, y, 0.0, 1.0);
  
  // Generate UV coordinates that cover the full screen
  let u = f32(vid & 1u) * 2.0;
  let v = f32((vid & 2u) >> 1u) * 2.0;
  out.uv = vec2<f32>(u, v);
  
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  // sample field; assume R=resource, G=0, B=waste (or your mapping)
  let v = textureSample(field_tex, samp, in.uv);

  // tweak this palette however you like:
  // red = resource (R), cyan-ish = waste (B)
  let r = clamp(v.r, 0.0, 1.0);
  let w = clamp(v.b, 0.0, 1.0);
  let color = vec3<f32>(r, 1.0 - w, w);

  return vec4<f32>(color, 1.0);
}
