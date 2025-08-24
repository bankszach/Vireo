@group(0) @binding(0) var<storage, read_write> occ : array<u32>;
@group(0) @binding(1) var<uniform> dims : vec2<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    let total = dims.x * dims.y;
    if (idx < total) { 
        occ[idx] = 0u; 
    }
}
