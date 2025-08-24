struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.tex_coords = input.tex_coords;
    return output;
}

@group(0) @binding(0) var field_texture: texture_2d<f32>;
@group(0) @binding(1) var field_sampler: sampler;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(field_texture, field_sampler, input.tex_coords);
    
    // Map R (resource) to red channel, W (waste) to green channel
    let r = tex_color.r;
    let w = tex_color.g;
    
    // Create a heatmap visualization
    // Resource: red to yellow (high = bright)
    // Waste: blue to cyan (high = bright)
    let resource_color = vec3<f32>(r, r * 0.5, 0.0);
    let waste_color = vec3<f32>(0.0, w * 0.5, w);
    
    // Blend resource and waste colors
    let final_color = resource_color + waste_color;
    
    return vec4<f32>(final_color, 1.0);
}
