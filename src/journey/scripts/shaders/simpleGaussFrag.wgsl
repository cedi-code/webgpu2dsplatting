
struct MyUniforms {
    canvasDim : vec2f,
    diagMat : mat2x2f,
    mean : vec2f,
}

struct vsOut {
    @builtin(position) p: vec4f,
    @location(0) texCoord: vec2f,
};

@group(0) @binding(0) var<uniform> uniforms : MyUniforms;

@fragment fn fs(
    in : vsOut
    ) -> @location(0) vec4f {
    
    const PI = 355.0/113.0;
    let V_inv = uniforms.diagMat;

    let pNorm = in.p.xy * uniforms.canvasDim;
    let d = pNorm - uniforms.mean;
    let D2 = dot(d * V_inv, d);

    let gauss = exp(-0.5 * D2);

    return vec4f(gauss, gauss, gauss, 1.0);
}