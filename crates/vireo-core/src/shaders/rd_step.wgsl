struct RDParams {
    D_R: f32,      // Resource diffusion coefficient
    D_W: f32,      // Waste diffusion coefficient
    sigma_R: f32,  // Resource replenishment rate
    alpha_H: f32,  // Herbivore resource uptake rate
    beta_H: f32,   // Herbivore waste emission rate
    lambda_R: f32, // Resource decay rate
    lambda_W: f32, // Waste decay rate
    dt: f32,       // Time step
    size: vec2<u32>, // Grid size
    H_SCALE: f32,  // Herbivore density scale factor
    _pad: u32,     // Padding for alignment
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var dstTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: RDParams;
@group(0) @binding(3) var<storage, read> herbDensity: array<u32>; // Herbivore occupancy

fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
    return max(lo, min(hi, v));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let xy = vec2<i32>(gid.xy);
    if (xy.x >= i32(params.size.x) || xy.y >= i32(params.size.y)) { return; }

    let w = i32(params.size.x);
    let h = i32(params.size.y);

    // Load current R, W values
    let v = textureLoad(srcTex, xy, 0);
    var R = v.r; // Resource channel
    var W = v.g; // Waste channel

    // 5-point Laplacian with Neumann boundary conditions
    let cx = clamp_i(xy.x, 0, w-1);
    let cy = clamp_i(xy.y, 0, h-1);
    
    let c = textureLoad(srcTex, vec2<i32>(cx, cy), 0);
    let l = textureLoad(srcTex, vec2<i32>(clamp_i(cx-1, 0, w-1), cy), 0);
    let r = textureLoad(srcTex, vec2<i32>(clamp_i(cx+1, 0, w-1), cy), 0);
    let u = textureLoad(srcTex, vec2<i32>(cx, clamp_i(cy-1, 0, h-1)), 0);
    let d = textureLoad(srcTex, vec2<i32>(cx, clamp_i(cy+1, 0, h-1)), 0);

    let lapR = (l.r + r.r + u.r + d.r - 4.0 * c.r);
    let lapW = (l.g + r.g + u.g + d.g - 4.0 * c.g);

    // Get herbivore density at this cell
    let cell_idx = u32(cy * w + cx);
    let H = min(f32(herbDensity[cell_idx]) * params.H_SCALE, 1.0); // Normalize occupancy with scale

    // Reaction-diffusion equations
    let dR = params.D_R * lapR + params.sigma_R - params.alpha_H * H * R - params.lambda_R * R;
    let dW = params.D_W * lapW + params.beta_H * H - params.lambda_W * W;

    // Update with explicit Euler, clamp to non-negative
    R = max(0.0, R + params.dt * dR);
    W = max(0.0, W + params.dt * dW);

    // Store result
    textureStore(dstTex, xy, vec4<f32>(R, W, 0.0, 0.0));
}
