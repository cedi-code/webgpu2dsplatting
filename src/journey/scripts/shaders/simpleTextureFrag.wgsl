

struct vsOut {
    @builtin(position) p: vec4f,
    @location(0) texCoord: vec2f,
};

struct Params {
    gauss : GaussParams,
    color : vec3f,
    alpha: f32,
};


@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<Params>;

@fragment fn fs(
    in : vsOut
    ) -> @location(0) vec4f {
    

    let pNorm = in.p.xy * vec2f(1.0/256.0);
    
    var resultColor = vec4f(vec3f(0.0), 1.0); //1.0 * textureSample(ourTexture, ourSampler, in.texCoord);;

    for(var  i = 0; i < 2; i++) {
        let gauss = g(output[i].gauss, pNorm);
        let color = vecSigmoid(output[i].color) * gauss;

        //return vec4f(in.texCoord.y, 1.0-gauss, in.texCoord.x, 1.0);
        let center = select(vec3f(0.0),vec3f(1.0)-color, gauss > 0.9999);

        let alpha = sigmoid(output[i].alpha, 4.0) * gauss;

        // this is sloppy but just for testing
        resultColor = alpha * vec4f(color+center, 1.0) + (1.0 - alpha) * resultColor;
    }
    return resultColor;

}