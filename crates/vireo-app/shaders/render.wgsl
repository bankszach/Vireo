struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct SimParams {
    world_size: vec2<f32>,
    time: f32,
    zoom: f32,
    camera: vec2<f32>,
    _pad0: vec2<f32>, // pad so struct size is multiple of 16
}

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    energy: f32,
    alive: u32,
    kind: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) inst: u32,
) -> VertexOutput {
    let P = particles[inst];
    
    // Skip dead particles
    if (P.alive == 0u) {
        // Return a degenerate triangle for dead particles
        var output: VertexOutput;
        output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return output;
    }
    
    // Create a small square for each particle
    let quad_size = 0.05 * sim_params.zoom; // Much larger for better visibility
    
    // Simple square generation - map vertex index to square corners
    // This should create perfect squares, not triangles
    let x = select(-quad_size, quad_size, vid >= 1u && vid <= 2u);
    let y = select(-quad_size, quad_size, vid == 2u || vid == 4u || vid == 5u);
    
    let vertex_pos = vec2<f32>(x, y);
    let world_pos = P.pos + vertex_pos;
    
    // Convert to clip space
    let clip_pos = (world_pos - sim_params.camera) * sim_params.zoom;
    
    // Color based on particle kind - completely different colors
    let kind = P.kind;
    
    // Very different colors for each agent type
    let color = select(
        vec4<f32>(1.0, 0.0, 0.0, 1.0),  // Bright red for plants (kind 0)
        select(
            vec4<f32>(0.0, 1.0, 0.0, 1.0),  // Bright green for herbivores (kind 1) 
            vec4<f32>(0.0, 0.0, 1.0, 1.0),  // Bright blue for predators (kind 2+)
            kind == 1u
        ),
        kind == 0u
    );
    
    var output: VertexOutput;
    output.position = vec4<f32>(clip_pos, 0.0, 1.0);
    output.color = color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
