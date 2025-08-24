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
  state_flags: u32,  // New field for visual state indicators
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var fieldTex: texture_2d<f32>;
@group(0) @binding(2) var fieldSamp: sampler;
@group(0) @binding(3) var<uniform> params: SimParams;

fn sample_field01(p: vec2<f32>) -> f32 {
  // p in world coords; convert to normalized [0,1]
  let uv = vec2<f32>(p.x / params.world_w, p.y / params.world_h);
  let v = textureSampleLevel(fieldTex, fieldSamp, uv, 0.0);
  return v.x; // channel 0
}

fn gradient(p: vec2<f32>) -> vec2<f32> {
  let eps = vec2<f32>(1.5, 1.5);
  let fpx = sample_field01(p + vec2<f32>(eps.x, 0.0));
  let fmx = sample_field01(p - vec2<f32>(eps.x, 0.0));
  let fpy = sample_field01(p + vec2<f32>(0.0, eps.y));
  let fmy = sample_field01(p - vec2<f32>(0.0, eps.y));
  let gx = (fpx - fmx) / (2.0*eps.x);
  let gy = (fpy - fmy) / (2.0*eps.y);
  return vec2<f32>(gx, gy);
}

@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) { return; }

  var p = particles[i];
  
  // COMPLETELY REMOVED: Ring spring system that was causing edge clustering
  
  var force = vec2<f32>(0.0, 0.0);

  // Add natural random movement for all particles
  let random_seed = f32(i) * 0.1 + params.time * 0.01;
  let random_angle = sin(random_seed) * 6.28;
  let random_strength = 0.3;
  let random_force = vec2<f32>(cos(random_angle), sin(random_angle)) * random_strength;
  force += random_force;

  // Behaviors based on particle type
  if (p.kind == 1u) {
    // HERBIVORE: Seek food, avoid predators, herd with others
    let g = gradient(p.pos);
    let food_force = normalize(g + vec2<f32>(1e-5, 0.0)) * 1.5;
    force += food_force;
    
    // Avoid predators (strong avoidance)
    var predator_avoidance = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < 50u; j++) {
      let check_idx = (i + j * 400u) % arrayLength(&particles);
      let other = particles[check_idx];
      if (other.kind == 2u) {  // predator
        let dist = length(other.pos - p.pos);
        if (dist < 100.0) {
          let avoid_dir = normalize(p.pos - other.pos);
          let avoid_strength = (100.0 - dist) / 100.0 * 4.0;
          predator_avoidance += avoid_dir * avoid_strength;
        }
      }
    }
    force += predator_avoidance;
    
    // Herd with other herbivores (gentle attraction)
    var herd_center = vec2<f32>(0.0, 0.0);
    var herd_count = 0u;
    for (var j = 0u; j < 30u; j++) {
      let check_idx = (i + j * 300u) % arrayLength(&particles);
      let other = particles[check_idx];
      if (other.kind == 1u && length(other.pos - p.pos) < 80.0) {
        herd_center += other.pos;
        herd_count += 1u;
      }
    }
    if (herd_count > 2u) {
      herd_center = herd_center / f32(herd_count);
      let to_herd = normalize(herd_center - p.pos);
      force += to_herd * 0.8;
    }
    
    // Energy consumption
    let movement_cost = length(p.vel) * 0.0001;
    p.energy = clamp(p.energy - movement_cost - 0.00005, 0.0, 10.0);
    
  } else if (p.kind == 2u) {
    // PREDATOR: Hunt herbivores, avoid other predators, maintain territory
    var closest_herbivore = vec2<f32>(0.0, 0.0);
    var closest_dist = 1000.0;
    
    for (var j = 0u; j < 80u; j++) {
      let check_idx = (i + j * 250u) % arrayLength(&particles);
      let other = particles[check_idx];
      if (other.kind == 1u) {  // herbivore
        let dist = length(other.pos - p.pos);
        if (dist < closest_dist && dist < 150.0) {
          closest_dist = dist;
          closest_herbivore = other.pos;
        }
      }
    }
    
    if (closest_dist < 1000.0) {
      let chase_dir = normalize(closest_herbivore - p.pos);
      let chase_force = 2.5;
      force += chase_dir * chase_force;
    }
    
    // Avoid other predators (territorial behavior)
    for (var j = 0u; j < 40u; j++) {
      let check_idx = (i + j * 200u) % arrayLength(&particles);
      let other = particles[check_idx];
      if (other.kind == 2u && check_idx != i) {
        let dist = length(other.pos - p.pos);
        if (dist < 80.0) {
          let repel_dir = normalize(p.pos - other.pos);
          let repel_force = (80.0 - dist) / 80.0 * 3.0;
          force += repel_dir * repel_force;
        }
      }
    }
    
    // Energy consumption
    let movement_cost = length(p.vel) * 0.0002;
    var hunting_cost: f32 = 0.0;
    if (closest_dist < 1000.0) {
      hunting_cost = 0.0001;
    }
    p.energy = clamp(p.energy - movement_cost - hunting_cost - 0.0001, 0.0, 10.0);
    
  } else {
    // PLANT: Minimal movement, grow in food-rich areas
    let jitter = vec2<f32>(sin(params.time*1.7 + f32(i)*0.01), cos(params.time*1.3 + f32(i)*0.02)) * 0.01;
    force += jitter;
    
    // Plants gain energy from being in food-rich areas
    let local_food = sample_field01(p.pos);
    let food_growth = local_food * 0.0001;
    p.energy = clamp(p.energy + food_growth + 0.00003, 0.0, 10.0);
  }

  // Update age and reproduction cooldown
  p.age += params.dt;
  p.reproduction_cooldown = max(p.reproduction_cooldown - params.dt, 0.0);
  
  // Reproduction logic
  if (p.reproduction_cooldown <= 0.0 && p.energy > 6.0 && p.age > 3.0) {
    let reproduction_chance = select(
      select(0.0001, 0.0002, p.kind == 1u),
      0.00005,
      p.kind == 2u
    );
    
    let hash = f32(i) * 0.1 + params.time * 0.01;
    if (fract(hash) < reproduction_chance) {
      p.energy *= 0.6;
      p.reproduction_cooldown = select(
        select(15.0, 20.0, p.kind == 1u),
        30.0,
        p.kind == 2u
      );
      p.state_flags |= 1u;
    }
  }
  
  // Clear and set state flags
  p.state_flags = 0u;
  
  let local_food = sample_field01(p.pos);
  if (local_food > 0.5) {
    p.state_flags |= 2u;  // Feeding flag
  }
  
  // Set herding flag for herbivores
  if (p.kind == 1u) {
    var nearby_herbivores = 0u;
    for (var j = 0u; j < 30u; j++) {
      let check_idx = (i + j * 300u) % arrayLength(&particles);
      let other = particles[check_idx];
      if (other.kind == 1u && length(other.pos - p.pos) < 80.0) {
        nearby_herbivores += 1u;
      }
    }
    if (nearby_herbivores > 3u) {
      p.state_flags |= 8u;  // Herding flag
    }
  }
  
  // Set attack flag for predators
  if (p.kind == 2u) {
    var closest_prey_dist = 1000.0;
    for (var j = 0u; j < 40u; j++) {
      let check_idx = (i + j * 200u) % arrayLength(&particles);
      let other = particles[check_idx];
      if (other.kind == 1u) {
        let dist = length(other.pos - p.pos);
        if (dist < closest_prey_dist) {
          closest_prey_dist = dist;
        }
      }
    }
    if (closest_prey_dist < 40.0) {
      p.state_flags |= 4u;  // Attack flag
    }
  }
  
  // GENTLE center guidance (much weaker than before)
  let center_x = params.world_w * 0.5;
  let center_y = params.world_h * 0.5;
  let to_center = vec2<f32>(center_x - p.pos.x, center_y - p.pos.y);
  let dist_to_center = length(to_center);
  
  if (dist_to_center > 0.0) {
    let max_force = 1.0;  // Very weak center guidance
    let force_radius = params.world_w * 0.45;
    let force_strength = clamp(dist_to_center / force_radius, 0.0, 1.0) * max_force;
    force += normalize(to_center) * force_strength;
  }
  
  // SOFT boundary handling - no strong edge forces
  let edge_margin = 40.0;
  let max_edge_force = 1.5;  // Very soft edge handling
  
  if (p.pos.x < edge_margin) {
    let edge_factor = (edge_margin - p.pos.x) / edge_margin;
    let edge_force = edge_factor * edge_factor * max_edge_force;
    force += vec2<f32>(edge_force, 0.0);
  }
  if (p.pos.x > params.world_w - edge_margin) {
    let edge_factor = (p.pos.x - (params.world_w - edge_margin)) / edge_margin;
    let edge_force = edge_factor * edge_factor * max_edge_force;
    force += vec2<f32>(-edge_force, 0.0);
  }
  if (p.pos.y < edge_margin) {
    let edge_factor = (edge_margin - p.pos.y) / edge_margin;
    let edge_force = edge_factor * edge_factor * max_edge_force;
    force += vec2<f32>(0.0, edge_force);
  }
  if (p.pos.y > params.world_h - edge_margin) {
    let edge_factor = (p.pos.y - (params.world_h - edge_margin)) / edge_margin;
    let edge_force = edge_factor * edge_factor * max_edge_force;
    force += vec2<f32>(0.0, -edge_force);
  }
  
  // Add exploration behavior - particles occasionally move away from center
  if (fract(random_seed * 0.5) < 0.15) {  // 15% chance
    let away_from_center = normalize(p.pos - vec2<f32>(center_x, center_y));
    force += away_from_center * 1.5;
  }
  
  // Integrate movement
  p.vel = (p.vel + force * params.dt) * 0.95;  // Slightly more damping
  p.pos += p.vel * params.dt;

  // BOUNDARY HANDLING: Bounce instead of wrap to prevent edge clustering
  let bounce_damping = 0.7;
  if (p.pos.x < 0.0) { 
    p.pos.x = 0.0; 
    p.vel.x = -p.vel.x * bounce_damping;
  }
  if (p.pos.y < 0.0) { 
    p.pos.y = 0.0; 
    p.vel.y = -p.vel.y * bounce_damping;
  }
  if (p.pos.x > params.world_w) { 
    p.pos.x = params.world_w; 
    p.vel.x = -p.vel.x * bounce_damping;
  }
  if (p.pos.y > params.world_h) { 
    p.pos.y = params.world_h; 
    p.vel.y = -p.vel.y * bounce_damping;
  }

  particles[i] = p;
}
