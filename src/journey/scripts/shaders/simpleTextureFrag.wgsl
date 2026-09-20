

struct vsOut {
    @builtin(position) p: vec4f,
    @location(0) texCoord: vec2f,
};

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> output: GaussParams;

@fragment fn fs(
    in : vsOut
    ) -> @location(0) vec4f {
    
    let pNorm = in.p.xy * vec2f(1.0/256.0);
    
    var resultColor = 1.0 * textureSample(ourTexture, ourSampler, in.texCoord);;

    let gauss = g(output, pNorm);
    let colorGauss = gauss * vec4f(1.0, 0.0, 1.0, 1.0);

    return colorGauss + (1.0 - gauss) * resultColor;;

}