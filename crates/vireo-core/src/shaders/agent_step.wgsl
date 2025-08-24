struct AgentParams {
    chi_R: f32,    // Resource attraction strength
    chi_W: f32,    // Waste repulsion strength
    kappa: f32,    // Gradient saturation parameter
    gamma: f32,    // Velocity damping
    v_max: f32,    // Maximum velocity
    eps0: f32,     // Basal energy drain rate
    eta_R: f32,    // Energy gain from resource
    dt: f32,       // Time step
    size: vec2<f32>, // World size
    _pad: vec2<f32>, // Padding for alignment
}

struct Agent {
    pos: vec2<f32>,     // Position (x, y)
    vel: vec2<f32>,     // Velocity (vx, vy)
    energy: f32,        // Current energy
    alive: u32,         // Alive flag (1 = alive, 0 = dead)
}

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var fieldTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: AgentParams;
@group(0) @binding(3) var<storage, read_write> herbOcc: array<u32>; // Herbivore occupancy

fn sample_field(p: vec2<f32>) -> vec2<f32> {
    // Convert world coordinates to texture coordinates
    let uv = vec2<f32>(p.x / params.size.x, p.y / params.size.y);
    let v = textureLoad(fieldTex, vec2<i32>(uv * params.size), 0);
    return vec2<f32>(v.r, v.g); // R, W channels
}

fn gradient(p: vec2<f32>) -> vec2<f32> {
    let eps = vec2<f32>(1.0, 1.0);
    
    let fpx = sample_field(p + vec2<f32>(eps.x, 0.0));
    let fmx = sample_field(p - vec2<f32>(eps.x, 0.0));
    let fpy = sample_field(p + vec2<f32>(0.0, eps.y));
    let fmy = sample_field(p - vec2<f32>(0.0, eps.y));
    
    let gx = (fpx.r - fmx.r) / (2.0 * eps.x); // Resource gradient
    let gy = (fpy.r - fmy.r) / (2.0 * eps.y);
    
    return vec2<f32>(gx, gy);
}

fn gradient_waste(p: vec2<f32>) -> vec2<f32> {
    let eps = vec2<f32>(1.0, 1.0);
    
    let fpx = sample_field(p + vec2<f32>(eps.x, 0.0));
    let fmx = sample_field(p - vec2<f32>(eps.x, 0.0));
    let fpy = sample_field(p + vec2<f32>(0.0, eps.y));
    let fmy = sample_field(p - vec2<f32>(0.0, eps.y));
    
    let gx = (fpx.g - fmx.g) / (2.0 * eps.x); // Waste gradient
    let gy = (fpy.g - fmy.g) / (2.0 * eps.y);
    
    return vec2<f32>(gx, gy);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&agents)) { return; }

    var a = agents[i];
    if (a.alive == 0u) { return; }

    // Sample gradients
    let gR = gradient(a.pos);
    let gW = gradient_waste(a.pos);

    // Chemotactic forces with saturation
    let fR = gR / (1.0 + params.kappa * length(gR));
    let fW = gW / (1.0 + params.kappa * length(gW));

    // Update velocity with chemotaxis
    var v = a.vel;
    v += (params.chi_R * fR - params.chi_W * fW) * params.dt;
    
    // Apply damping
    v *= (1.0 - params.gamma);
    
    // Clamp to maximum velocity
    if (length(v) > params.v_max) {
        v = normalize(v) * params.v_max;
    }

    // Update position
    var x = a.pos + v * params.dt;
    
    // Bounce at boundaries
    let bounce_damping = 0.7;
    if (x.x < 0.0) { 
        x.x = 0.0; 
        v.x = -v.x * bounce_damping;
    }
    if (x.y < 0.0) { 
        x.y = 0.0; 
        v.y = -v.y * bounce_damping;
    }
    if (x.x >= params.size.x) { 
        x.x = params.size.x - 0.1; 
        v.x = -v.x * bounce_damping;
    }
    if (x.y >= params.size.y) { 
        x.y = params.size.y - 0.1; 
        v.y = -v.y * bounce_damping;
    }

    // Energy management
    let local_field = sample_field(x);
    let R = local_field.r;
    let energy_gain = params.eta_R * R * params.dt;
    let energy_drain = params.eps0 * params.dt;
    
    a.energy += energy_gain - energy_drain;
    
    // Death on zero energy
    if (a.energy <= 0.0) {
        a.alive = 0u;
    }

    // Update agent
    a.pos = x;
    a.vel = v;
    agents[i] = a;

    // Write occupancy to grid (simple increment for now)
    if (a.alive == 1u) {
        let cell_x = u32(clamp(floor(x.x), 0.0, params.size.x - 1.0));
        let cell_y = u32(clamp(floor(x.y), 0.0, params.size.y - 1.0));
        let cell_idx = cell_y * u32(params.size.x) + cell_x;
        
        // Simple increment (in a real implementation, use atomic operations)
        // For now, this works for single-threaded agent updates
        herbOcc[cell_idx] += 1u;
    }
}
