struct SimParams {
  dt: f32,
  world_w: f32,
  world_h: f32,
  _pad0: f32,
  grid_w: u32,
  grid_h: u32,
  group_size: u32,
  paused: u32,
  time: f32,
  diffusion: f32,
  decay: f32,
  _pad1: f32,
};

struct Particle {
  pos: vec2<f32>,
  vel: vec2<f32>,
  energy: f32,
  kind: u32,
  age: f32,
  reproduction_cooldown: f32,
  state_flags: u32,  // Visual state indicators
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
};

fn ndc_from_world(p: vec2<f32>) -> vec2<f32> {
  let x = (p.x / params.world_w) * 2.0 - 1.0;
  let y = 1.0 - (p.y / params.world_h) * 2.0;
  return vec2<f32>(x, y);
}

@vertex
fn vs_main(@builtin(instance_index) inst: u32,
           @builtin(vertex_index) vid: u32) -> VSOut {
  let P = particles[inst];
  
  // Debug: Force specific colors for testing
  // This will help us verify if the issue is in the shader logic or data
  var debug_color: vec3<f32>;
  if (P.kind == 0u) {
    debug_color = vec3<f32>(0.2, 0.8, 0.3);  // Green for plants
  } else if (P.kind == 1u) {
    debug_color = vec3<f32>(0.2, 0.6, 1.0);  // Blue for herbivores
  } else if (P.kind == 2u) {
    debug_color = vec3<f32>(0.8, 0.2, 0.2);  // Red for predators
  } else {
    debug_color = vec3<f32>(1.0, 1.0, 0.0);  // Yellow for unknown (debug)
  }
  
  // Add visual effects based on state flags
  var final_color = debug_color;
  var alpha = 0.85;
  
  // State flag bits: 0=reproducing, 1=feeding, 2=attacking, 3=herding
  let is_reproducing = (P.state_flags & 1u) != 0u;
  let is_feeding = (P.state_flags & 2u) != 0u;
  let is_attacking = (P.state_flags & 4u) != 0u;
  let is_herding = (P.state_flags & 8u) != 0u;
  
  // Reproduction indicator: bright white flash
  if (is_reproducing) {
    let flash_intensity = sin(params.time * 20.0) * 0.5 + 0.5;
    final_color = mix(final_color, vec3<f32>(1.0, 1.0, 1.0), flash_intensity * 0.8);
    alpha = 1.0;
  }
  
  // Feeding indicator: bright yellow/orange glow
  if (is_feeding) {
    let feed_intensity = sin(params.time * 15.0) * 0.5 + 0.5;
    final_color = mix(final_color, vec3<f32>(1.0, 0.8, 0.2), feed_intensity * 0.3);  // Reduced from 0.6
    alpha = 0.95;
  }
  
  // Attack indicator: bright red flash
  if (is_attacking) {
    let attack_intensity = sin(params.time * 25.0) * 0.5 + 0.5;
    final_color = mix(final_color, vec3<f32>(1.0, 0.3, 0.3), attack_intensity * 0.7);
    alpha = 1.0;
  }
  
  // Herding indicator: cyan glow
  if (is_herding) {
    let herd_intensity = sin(params.time * 10.0) * 0.5 + 0.5;
    final_color = mix(final_color, vec3<f32>(0.2, 0.8, 1.0), herd_intensity * 0.4);
    alpha = 0.9;
  }
  
  // Add trail effect for moving particles
  let velocity_magnitude = length(P.vel);
  if (velocity_magnitude > 0.5) {
    let trail_intensity = clamp(velocity_magnitude / 5.0, 0.0, 0.3);
    let trail_color = select(
      select(vec3<f32>(0.8, 1.0, 0.8), vec3<f32>(0.8, 0.9, 1.0), P.kind == 1u),
      vec3<f32>(1.0, 0.8, 0.8),
      P.kind == 2u
    );
    final_color = mix(final_color, trail_color, trail_intensity);
  }
  
  // Add energy glow for high-energy particles
  if (P.energy > 8.0) {
    let glow_intensity = (P.energy - 8.0) / 2.0;
    let glow_color = vec3<f32>(1.0, 1.0, 0.8); // Soft yellow glow
    final_color = mix(final_color, glow_color, glow_intensity * 0.2);  // Reduced from 0.4
  }
  
  // Add stress indicator for low-energy particles
  if (P.energy < 3.0) {
    let stress_intensity = (3.0 - P.energy) / 3.0;
    let stress_color = vec3<f32>(1.0, 0.4, 0.4); // Red stress indicator
    final_color = mix(final_color, stress_color, stress_intensity * 0.3);  // Reduced from 0.6
    
    // Make stressed particles flicker
    let flicker = sin(params.time * 30.0 + f32(inst) * 0.1) * 0.5 + 0.5;
    alpha = mix(0.6, 0.9, flicker);
  }
  
  // Energy-based size variation
  let energy_factor = clamp(P.energy / 10.0, 0.3, 1.5);
  let base_size = select(
    select(3.0, 2.5, P.kind == 1u),  // Plants: 3.0, Herbivores: 2.5
    4.0,  // Predators: 4.0
    P.kind == 2u
  );
  
  // Add pulse effect for high-energy particles
  var pulse_factor: f32 = 1.0;
  if (P.energy > 7.0) {
    let pulse = sin(params.time * 8.0 + f32(inst) * 0.1) * 0.2 + 1.0;
    pulse_factor = pulse;
  }
  
  let size = base_size * energy_factor * pulse_factor;

  // quad verts (two triangles): ( -1,-1 ), ( 1,-1 ), ( 1,1 ), ( -1,-1 ), ( 1,1 ), ( -1,1 )
  var quad: array<vec2<f32>, 6>;
  quad[0] = vec2<f32>(-1.0, -1.0);
  quad[1] = vec2<f32>( 1.0, -1.0);
  quad[2] = vec2<f32>( 1.0,  1.0);
  quad[3] = vec2<f32>(-1.0, -1.0);
  quad[4] = vec2<f32>( 1.0,  1.0);
  quad[5] = vec2<f32>(-1.0,  1.0);

  let pos_ndc = ndc_from_world(P.pos);
  let pix = vec2<f32>(size / params.world_w * 2.0, size / params.world_h * 2.0);
  let v = quad[vid] * pix + pos_ndc;

  var out: VSOut;
  out.pos = vec4<f32>(v, 0.0, 1.0);
  
  // Use debug color instead of complex select logic
  out.color = vec4<f32>(final_color, alpha);
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  return in.color;
}
