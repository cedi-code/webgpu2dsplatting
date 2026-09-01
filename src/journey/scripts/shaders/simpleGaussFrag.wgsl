
struct MyUniforms {
    canvasDim : vec2f
}

@group(0) @binding(0) var<uniform> uniforms : MyUniforms;

@fragment fn fs(
    @builtin(position) p : vec4f
    ) -> @location(0) vec4f {
    return vec4f(p.x * uniforms.canvasDim[0],
                 p.x * uniforms.canvasDim[0],
                 p.x * uniforms.canvasDim[0],
                  1.0);
}