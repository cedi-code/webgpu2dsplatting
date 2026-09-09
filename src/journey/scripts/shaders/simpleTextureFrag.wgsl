

struct vsOut {
    @builtin(position) p: vec4f,
    @location(0) texCoord: vec2f,
};

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;

@fragment fn fs(
    in : vsOut
    ) -> @location(0) vec4f {
    
    let V_inv = mat2x2f(10.0, 0.0, 0.0, 10.0);

    let pNorm = in.p.xy * vec2(1.0/256.0);

    let d = pNorm - vec2f(0.5) ;
    let D2 = dot(d * V_inv, d);

    let gauss = exp(-0.5 * D2);

    //return vec4f(in.texCoord.y, 1.0-gauss, in.texCoord.x, 1.0);
    return vec4f(gauss, gauss, gauss, 1.0) * textureSample(ourTexture, ourSampler, in.texCoord);
}