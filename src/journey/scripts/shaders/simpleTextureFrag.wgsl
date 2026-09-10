

struct vsOut {
    @builtin(position) p: vec4f,
    @location(0) texCoord: vec2f,
};

struct Uniform {
    pos: vec2f,
    scale: vec2f,
};

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var ourTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniform;

@fragment fn fs(
    in : vsOut
    ) -> @location(0) vec4f {
    

    let pNorm = in.p.xy * vec2f(1.0/256.0);

    let d = pNorm - uniforms.pos;
    let D2 = dot(d * uniforms.scale, d);

    let gauss = exp(-0.5 * D2);

    //return vec4f(in.texCoord.y, 1.0-gauss, in.texCoord.x, 1.0);
    let red = select(0.0, gauss, gauss > 0.9);
    return vec4f(red, gauss - red, 0.0, 1.0) * textureSample(ourTexture, ourSampler, in.texCoord);
}