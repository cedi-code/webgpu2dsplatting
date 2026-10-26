@group(0) @binding(0) var<storage, read_write> splats: array<Params>;
@group(0) @binding(7) var<uniform> uniforms : Uniform;

@compute @workgroup_size(GROUPSIZE, 1, 1)
fn cs(
    @builtin(global_invocation_id) global_invocation_id : vec3u,
    ) {

    let splat_id = global_invocation_id.x;
    splats[splat_id].alpha *= uniforms.opacityReset;
}