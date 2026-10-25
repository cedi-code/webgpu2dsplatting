struct SimpleVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) grid: vec2f,
    @location(1) color: vec4f,
};


override withFade : bool = true;

@fragment fn fs(
    fsIn : SimpleVertexShaderOutput
    ) -> @location(0) vec4f {

    let gauss = exp(-0.5*TILE_SCALE*TILE_SCALE * dot(fsIn.grid , fsIn.grid));
    let c = fsIn.color;
    var alpha : f32;
    if(withFade) {
        alpha = gauss * c.a;
    }
    else {
        if(gauss < OPACITY_PERCENTAGE) {
            discard;
        }
        alpha = c.a;
    }


    return vec4f(vec3f(c.rgb) * alpha,  alpha);
}