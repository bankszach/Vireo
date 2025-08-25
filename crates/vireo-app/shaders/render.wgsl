struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@group(0) @binding(0) var sim_params: SimParams;
@group(0) @binding(1) var particles: array<Particle>;

struct SimParams {
    camera_pos: vec2<f32>,
    camera_zoom: f32,
    time: f32,
    _pad: f32,
}

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    energy: f32,
    kind: u32,
    alive: u32,
    _pad: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) inst: u32,
) -> VertexOutput {
    let P = particles[inst];
    
    // Create a small quad for each particle
    let quad_size = 0.02 * sim_params.camera_zoom;
    let quad_vertices = array<vec2<f32>, 6>(
        vec2<f32>(-quad_size, -quad_size),  // bottom-left
        vec2<f32>( quad_size, -quad_size),  // bottom-right
        vec2<f32>( quad_size,  quad_size),  // top-right
        vec2<f32>(-quad_size, -quad_size),  // bottom-left
        vec2<f32>( quad_size,  quad_size),  // top-right
        vec2<f32>(-quad_size,  quad_size)   // top-left
    );
    
    let vertex_pos = quad_vertices[vid];
    let world_pos = P.pos + vertex_pos;
    
    // Convert to clip space
    let clip_pos = (world_pos - sim_params.camera_pos) * sim_params.camera_zoom;
    
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
