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

fn idx_left(i: u32, gsz: u32) -> u32 {
  let g = i / gsz;
  let l = i % gsz;
  let l2 = (l + gsz - 1u) % gsz;
  return g*gsz + l2;
}
fn idx_right(i: u32, gsz: u32) -> u32 {
  let g = i / gsz;
  let l = i % gsz;
  let r2 = (l + 1u) % gsz;
  return g*gsz + r2;
}

@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) { return; }

  var p = particles[i];
  // Spring forces to ring neighbors
  let gsz = params.group_size;
  let il = idx_left(i, gsz);
  let ir = idx_right(i, gsz);
  let pl = particles[il];
  let pr = particles[ir];

  var force = vec2<f32>(0.0, 0.0);

  // Hooke springs
  let k = 0.8;  // Further reduced from 1.5
  let rest = 12.0;  // Further increased from 8.0
  let dir_l = p.pos - pl.pos;
  let dist_l = max(length(dir_l), 0.0001);
  force += (1.0 - rest/dist_l) * dir_l * k;

  let dir_r = p.pos - pr.pos;
  let dist_r = max(length(dir_r), 0.0001);
  force += (1.0 - rest/dist_r) * dir_r * k;

  // Behaviors
  if (p.kind == 1u) {
    // herbivore: climb gradient of field (channel 0) and seek mates
    let g = gradient(p.pos);
    let food_force = normalize(g + vec2<f32>(1e-5, 0.0)) * 1.2; // Increased food seeking
    force += food_force;
    
    // Herding behavior - seek other herbivores for safety
    var herd_center = vec2<f32>(0.0, 0.0);
    var herd_count = 0u;
    
    for (var j = 0u; j < 100u; j++) {
      let check_idx = (i + j * 200u) % arrayLength(&particles);
      let other = particles[check_idx];
      if (other.kind == 1u && length(other.pos - p.pos) < 120.0) {
        herd_center += other.pos;
        herd_count += 1u;
      }
    }
    
    if (herd_count > 1u) {
      herd_center = herd_center / f32(herd_count);
      let to_herd = normalize(herd_center - p.pos);
      force += to_herd * 0.6; // Increased herding force
    }
    
    // Seek mates when ready to reproduce
    if (p.reproduction_cooldown <= 0.0 && p.energy > 5.0) {
      var closest_mate = vec2<f32>(0.0, 0.0);
      var closest_dist = 1000.0;
      
      for (var j = 0u; j < 50u; j++) {
        let check_idx = (i + j * 400u) % arrayLength(&particles);
        let other = particles[check_idx];
        if (other.kind == 1u && other.reproduction_cooldown <= 0.0 && other.energy > 5.0) {
          let dist = length(other.pos - p.pos);
          if (dist < closest_dist && dist < 80.0) {
            closest_dist = dist;
            closest_mate = other.pos;
          }
        }
      }
      
      if (closest_dist < 1000.0) {
        let mate_dir = normalize(closest_mate - p.pos);
        force += mate_dir * 0.8; // Increased mating attraction
      }
    }
    
    // Energy consumption based on movement
    let movement_cost = length(p.vel) * 0.0001;
    p.energy = clamp(p.energy - movement_cost - 0.00005, 0.0, 10.0);
  } else if (p.kind == 2u) {
    // predator: chase nearby herbivores
    var closest_herbivore = vec2<f32>(0.0, 0.0);
    var closest_dist = 1000.0;
    
    for (var j = 0u; j < 100u; j++) {
      let check_idx = (i + j * 200u) % arrayLength(&particles);
      let other = particles[check_idx];
      if (other.kind == 1u) {  // herbivore
        let dist = length(other.pos - p.pos);
        if (dist < closest_dist && dist < 120.0) { // Increased detection range
          closest_dist = dist;
          closest_herbivore = other.pos;
        }
      }
    }
    
    if (closest_dist < 1000.0) {
      let chase_dir = normalize(closest_herbivore - p.pos);
      let chase_force = 2.0; // Increased chase force
      force += chase_dir * chase_force;
      
      // If very close to prey, increase speed
      if (closest_dist < 30.0) {
        force += chase_dir * 1.5; // Extra burst of speed
      }
    }
    
    // Energy consumption based on movement and hunting
    let movement_cost = length(p.vel) * 0.0002;
    var hunting_cost: f32 = 0.0;
    if (closest_dist < 1000.0) {
      hunting_cost = 0.0001;
    }
    p.energy = clamp(p.energy - movement_cost - hunting_cost - 0.0001, 0.0, 10.0);
  } else {
    // plant: very light jitter keeps things visually alive
    let jitter = vec2<f32>(sin(params.time*1.7 + f32(i)*0.01), cos(params.time*1.3 + f32(i)*0.02)) * 0.02;
    force += jitter;
    
    // Plants grow slowly over time and are more stable
    let growth_factor = clamp(p.age / 10.0, 0.0, 1.0);
    let stability_force = -p.vel * 2.0; // Dampen movement for stability
    force += stability_force;
    
    // Plants gain energy from being in food-rich areas
    let local_food = sample_field01(p.pos);
    let food_growth = local_food * 0.0001;
    p.energy = clamp(p.energy + food_growth + 0.00003, 0.0, 10.0);
  }

  // Integrate
  let dt = params.dt;
  
  // Update age and reproduction cooldown
  p.age += dt;
  p.reproduction_cooldown = max(p.reproduction_cooldown - dt, 0.0);
  
  // Simple reproduction logic (GPU-based)
  if (p.reproduction_cooldown <= 0.0 && p.energy > 6.0 && p.age > 3.0) {
    // Small chance to reproduce based on particle type
    let reproduction_chance = select(
      select(0.0001, 0.0002, p.kind == 1u),  // Plants: 0.0001, Herbivores: 0.0002
      0.00005,  // Predators: 0.00005
      p.kind == 2u
    );
    
    // Use a simple hash of position and time for randomness
    let hash = f32(i) * 0.1 + params.time * 0.01;
    if (fract(hash) < reproduction_chance) {
      // Reproduction costs energy
      p.energy *= 0.6;
      p.reproduction_cooldown = select(
        select(15.0, 20.0, p.kind == 1u),  // Plants: 15s, Herbivores: 20s
        30.0,  // Predators: 30s
        p.kind == 2u
      );
      
      // Set reproduction flag for visual indicator
      p.state_flags |= 1u;
    }
  }
  
  // Clear old state flags and set new ones based on current behavior
  p.state_flags = 0u;
  
  // Set feeding flag if particle is in food-rich area
  let local_food = sample_field01(p.pos);
  if (local_food > 0.3) {
    p.state_flags |= 2u;  // Feeding flag
  }
  
  // Set herding flag for herbivores that are near other herbivores
  if (p.kind == 1u) {
    var nearby_herbivores = 0u;
    for (var j = 0u; j < 50u; j++) {
      let check_idx = (i + j * 400u) % arrayLength(&particles);
      let other = particles[check_idx];
      if (other.kind == 1u && length(other.pos - p.pos) < 80.0) {
        nearby_herbivores += 1u;
      }
    }
    if (nearby_herbivores > 3u) {
      p.state_flags |= 8u;  // Herding flag
    }
  }
  
  // Set attack flag for predators that are close to prey
  if (p.kind == 2u) {
    var closest_prey_dist = 1000.0;
    for (var j = 0u; j < 50u; j++) {
      let check_idx = (i + j * 400u) % arrayLength(&particles);
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
  
  // Improved center clustering with smooth force field
  let center_x = params.world_w * 0.5;
  let center_y = params.world_h * 0.5;
  let to_center = vec2<f32>(center_x - p.pos.x, center_y - p.pos.y);
  let dist_to_center = length(to_center);
  
  // Smooth center attraction that increases with distance
  if (dist_to_center > 0.0) {
    // Use a smoother force curve: stronger attraction when far, gentle when close
    let max_force = 12.0;
    let force_radius = params.world_w * 0.3; // Force acts within 30% of world size
    let force_strength = clamp(dist_to_center / force_radius, 0.0, 1.0) * max_force;
    force += normalize(to_center) * force_strength;
  }
  
  // Smooth edge repulsion that prevents particles from getting stuck at borders
  let edge_margin = 80.0;
  let max_edge_force = 8.0;
  
  // Left edge
  if (p.pos.x < edge_margin) {
    let edge_factor = (edge_margin - p.pos.x) / edge_margin;
    let edge_force = edge_factor * edge_factor * max_edge_force; // Quadratic falloff
    force += vec2<f32>(edge_force, 0.0);
  }
  // Right edge
  if (p.pos.x > params.world_w - edge_margin) {
    let edge_factor = (p.pos.x - (params.world_w - edge_margin)) / edge_margin;
    let edge_force = edge_factor * edge_factor * max_edge_force;
    force += vec2<f32>(-edge_force, 0.0);
  }
  // Top edge
  if (p.pos.y < edge_margin) {
    let edge_factor = (edge_margin - p.pos.y) / edge_margin;
    let edge_force = edge_factor * edge_factor * max_edge_force;
    force += vec2<f32>(0.0, edge_force);
  }
  // Bottom edge
  if (p.pos.y > params.world_h - edge_margin) {
    let edge_factor = (p.pos.y - (params.world_h - edge_margin)) / edge_margin;
    let edge_force = edge_factor * edge_factor * max_edge_force;
    force += vec2<f32>(0.0, -edge_force);
  }
  
  p.vel = (p.vel + force * dt) * 0.97;  // Slightly less damping for better center movement
  p.pos += p.vel * dt;

  // Wrap around boundaries instead of bouncing
  if (p.pos.x < 0.0) { p.pos.x = params.world_w; }
  if (p.pos.y < 0.0) { p.pos.y = params.world_h; }
  if (p.pos.x > params.world_w) { p.pos.x = 0.0; }
  if (p.pos.y > params.world_h) { p.pos.y = 0.0; }

  particles[i] = p;
}
