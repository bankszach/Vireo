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
    kind: u32,
    flags: u32,
    energy: f32,
    _pad1: f32, // pad so each element is 32B (16-aligned)
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) inst: u32,
) -> VertexOutput {
    let P = particles[inst];
    
    // Create a small quad for each particle
    let quad_size = 0.02 * sim_params.zoom;
    
    // Simple quad generation using basic math
    let x = select(-quad_size, quad_size, vid >= 1u && vid <= 2u);
    let y = select(-quad_size, quad_size, vid == 2u || vid == 4u || vid == 5u);
    
    let vertex_pos = vec2<f32>(x, y);
    let world_pos = P.pos + vertex_pos;
    
    // Convert to clip space
    let clip_pos = (world_pos - sim_params.camera) * sim_params.zoom;
    
    // Color based on particle kind
    let kind = P.kind;
    let color = select(
        vec4<f32>(0.0, 1.0, 0.0, 0.8),  // Green for plants (kind 0)
        select(
            vec4<f32>(0.0, 0.0, 1.0, 0.8),  // Blue for herbivores (kind 1)
            vec4<f32>(1.0, 0.0, 0.0, 0.8),  // Red for predators (kind 2+)
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
