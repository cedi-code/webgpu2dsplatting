struct SimpleVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) grid: vec2f,
    @location(1) color: vec3f,
};

@fragment fn fs(
    fsIn : SimpleVertexShaderOutput
    ) -> @location(0) vec4f {

    let gauss = exp(-6.0 * dot(fsIn.grid , fsIn.grid));
    let alpha : f32 = gauss;
    let c = fsIn.color;

    return vec4f(vec3f(c) * gauss, gauss);
}