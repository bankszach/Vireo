// Diffusion of a 4-channel field (RGBA). We primarily use channel 0 as "food/scent".
struct SimParams {
  dt: f32,
  world_w: f32,
  world_h: f32,
  _pad0: f32,
  grid_w: u32,
  grid_h: u32,
  _reserved: u32,  // Was group_size, now reserved for future use
  paused: u32,
  time: f32,
  diffusion: f32,
  decay: f32,
  _pad1: f32,
  // Camera parameters
  camera_pos_x: f32,
  camera_pos_y: f32,
  camera_zoom: f32,
  _pad2: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var dstTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: SimParams;

fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
  return max(lo, min(hi, v));
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gx = i32(gid.x);
  let gy = i32(gid.y);
  if (gx >= i32(params.grid_w) || gy >= i32(params.grid_h)) { return; }

  // 5-point Laplacian on channel 0..3
  let w = i32(params.grid_w);
  let h = i32(params.grid_h);

  let cx = clamp_i(gx, 0, w-1);
  let cy = clamp_i(gy, 0, h-1);

  let c  = textureLoad(srcTex, vec2<i32>(cx, cy), 0);
  let l  = textureLoad(srcTex, vec2<i32>(clamp_i(cx-1,0,w-1), cy), 0);
  let r  = textureLoad(srcTex, vec2<i32>(clamp_i(cx+1,0,w-1), cy), 0);
  let u  = textureLoad(srcTex, vec2<i32>(cx, clamp_i(cy-1,0,h-1)), 0);
  let d  = textureLoad(srcTex, vec2<i32>(cx, clamp_i(cy+1,0,h-1)), 0);

  let lap = (l + r + u + d - 4.0 * c);
  let diff = c + params.diffusion * lap - params.decay * c;
  // Simple clamp to keep things sane
  let outv = clamp(diff, vec4<f32>(0.0), vec4<f32>(1000.0));
  textureStore(dstTex, vec2<i32>(cx, cy), outv);
}
