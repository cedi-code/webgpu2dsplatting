

struct vsOut {
    @builtin(position) p: vec4f,
    @location(0) texCoord: vec2f,
};

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> output: GaussParams;
@group(0) @binding(3) var<storage, read> outputNoActivation: GaussParams;

fn gCorrected(p : GaussParams, x : vec2f) -> f32 {
    var pp = p;
    pp.pos *= vec2f(1.0, -1.0);
    pp.pos += vec2f(0.0, 1.0);

    return g(pp, x);
}

@fragment fn fs(
    in : vsOut
    ) -> @location(0) vec4f {
    
    let pNorm = in.p.xy * vec2f(1.0/256.0);
    
    var resultColor = 1.0 * textureSample(ourTexture, ourSampler, in.texCoord);

    var p = output;

    let gauss = gCorrected(p, pNorm);
    let colorGauss =  select(vec4f(0.0), vec4f(1.0, 0.0, 0.0, 1.0), gauss > 0.4);

    var p2 = outputNoActivation;

    let gauss2 = gCorrected(p2, pNorm);
    let colorGauss2 =  select(vec4f(0.0), vec4f(0.0, 0.0, 1.0, 1.0), gauss2 > 0.4);

    return colorGauss2 +  colorGauss + (1.0 - gauss2) * (1.0 - gauss) * resultColor;

}