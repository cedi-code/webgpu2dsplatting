

struct vsOut {
    @builtin(position) p: vec4f,
    @location(0) texCoord: vec2f,
};

struct Uniform {
    pos: vec2f,
    scale: vec2f,
    rot: f32,
    color: vec3f,
};

struct Params {
    pos : vec2f,
    scale : vec2f,
    rot : f32
};

fn rotMat(r: f32) -> mat2x2f {
    return mat2x2f(
        cos(r), sin(r), // column 0
        -sin(r), cos(r) // column 1
    ); 
}

fn sigmoid(x : f32) -> f32 {
    return 1.0 / (1.0 + exp(-x + 4));
}

fn vecSigmoid(x : vec3f) -> vec3f {
    return vec3f(
        sigmoid(x.r),
        sigmoid(x.g),
        sigmoid(x.b),
    );
}

fn g(p : Params, x : vec2f) -> f32 {

    let d = x - p.pos;
    let a = exp(p.scale) * (transpose(rotMat(p.rot)) * d);
    let D2 = dot(a, a);

    return exp(-0.5 * D2);
}

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uni : Uniform;

@fragment fn fs(
    in : vsOut
    ) -> @location(0) vec4f {
    

    let pNorm = in.p.xy * vec2f(1.0/256.0);

    let gauss = g(Params(uni.pos, uni.scale, uni.rot), pNorm);
    let color = vecSigmoid(uni.color) * gauss;

    //return vec4f(in.texCoord.y, 1.0-gauss, in.texCoord.x, 1.0);
    let red = select(0.0, gauss, gauss > 0.9);
    return vec4f(color, 1.0) * textureSample(ourTexture, ourSampler, in.texCoord);
}