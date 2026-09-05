struct SimpleVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) grid: vec2f,
    @location(1) color: vec4f,
};

@fragment fn fs(
    fsIn : SimpleVertexShaderOutput
    ) -> @location(0) vec4f {

    let gauss = exp(-6.0 * dot(fsIn.grid , fsIn.grid));
    let c = fsIn.color;
    let alpha : f32 = gauss * c.a;


    return vec4f(vec3f(c.rgb) * alpha, gauss * alpha);
}