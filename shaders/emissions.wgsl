struct SimParams {
  dt: f32,
  world_w: f32,
  world_h: f32,
  _pad0: f32,
  grid_w: u32,
  grid_h: u32,
  _reserved: u32,
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
  // Emissions toggle
  emissions_enabled: u32,
  _pad3: f32,
};

struct Particle {
  pos: vec2<f32>,
  vel: vec2<f32>,
  energy: f32,
  kind: u32,
  age: f32,
  reproduction_cooldown: f32,
  state_flags: u32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var fieldTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) { return; }
  
  let p = particles[i];
  
  // Convert world position to texture coordinates
  let tex_x = u32(clamp(p.pos.x, 0.0, params.world_w - 1.0));
  let tex_y = u32(clamp(p.pos.y, 0.0, params.world_h - 1.0));
  
  // Only emit if emissions are enabled
  if (params.emissions_enabled == 1u) {
    // Calculate emission amount based on particle type and state
    var emission_amount: f32 = 0.0;
    
    if (p.kind == 0u) {
      // Plants emit slowly and steadily
      emission_amount = 0.001;
    } else if (p.kind == 1u) {
      // Herbivores emit more when moving fast or feeding
      let speed = length(p.vel);
      let is_feeding = (p.state_flags & 2u) != 0u;
      emission_amount = 0.002 + speed * 0.0005;
      if (is_feeding) {
        emission_amount += 0.001;
      }
    } else if (p.kind == 2u) {
      // Predators emit more when hunting or attacking
      let speed = length(p.vel);
      let is_attacking = (p.state_flags & 4u) != 0u;
      emission_amount = 0.003 + speed * 0.0008;
      if (is_attacking) {
        emission_amount += 0.002;
      }
    }
    
    // Write emission to channel 0 (food/scent) - diffusion pass will handle accumulation
    textureStore(fieldTex, vec2<i32>(i32(tex_x), i32(tex_y)), 
                 vec4<f32>(emission_amount, 0.0, 0.0, 0.0));
  }
}
