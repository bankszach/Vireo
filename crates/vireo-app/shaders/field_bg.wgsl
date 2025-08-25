// Field background: sample the current front field texture and paint a heat-map.

@group(0) @binding(0) var fieldTex: texture_2d<f32>;
@group(0) @binding(1) var fieldSamp: sampler;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var out: VSOut;
    
    // Fullscreen triangle using vertex index
    // p = (0,0), (2,0), (0,2) for vid=0,1,2
    let p = vec2<f32>(
        f32((vid << 1u) & 2u),
        f32( vid        & 2u)
    );
    
    // NDC: (-1,-1), (3,-1), (-1,3)
    let ndc = p * 2.0 - 1.0;
    
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    // UVs that match the screen (flip Y to taste)
    out.uv = vec2<f32>(p.x * 0.5, 1.0 - p.y * 0.5);
    return out;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Sample the field texture
    let field_value = textureSample(fieldTex, fieldSamp, uv);
    
    // Use black background instead of green
    let background = vec3<f32>(0.0, 0.0, 0.0); // Black background
    
    // Keep some subtle field visualization but make it very dark
    let field_intensity = field_value.x * 0.1; // Very subtle field visualization
    let field_color = vec3<f32>(0.0, field_intensity, 0.0); // Very dark green
    
    // Blend background with field
    let final_color = mix(background, field_color, 0.3); // Keep field very subtle
    
    return vec4<f32>(final_color, 1.0);
}
