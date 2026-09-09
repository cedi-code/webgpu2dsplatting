struct Uniform {
    stepSize : f32,
};

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var goalTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms : Uniform;


fn g(q: vec2f, x : vec2f, s : vec2f) -> f32 {

    let pNorm = x; // currently a constant but once we switch to vertex gauss tile approach not needed

    let d = pNorm - q;
    let D2 = dot(d * s, d);

    return exp(-0.5 * D2);
}

fn Loss(q: vec2f, s : vec2f, x: vec2f, imgC: vec4f) -> f32 {

    let gColor = vec4f(vec3f(g(q,x,s)), 1.0);

    return (gColor.r - imgC.r)*(gColor.r - imgC.r);
}

fn GradLoss_Q_S(q: vec2f, s: vec2f, x: vec2f, imgC: vec4f) -> vec4f {

    let gColor = vec4f(vec3f(g(q,x,s)), 1.0);
    let diff = (gColor.r - imgC.r);
    let dist = (x - q);

    
    let gradQ = 2.0 * diff * gColor.r * s * dist;
    let gradS = -1.0 * diff * gColor.r * dist * dist;

    return vec4f(gradQ, gradS);
}


@compute @workgroup_size(1) fn computeGD() {

    let size = textureDimensions(goalTexture, 0);
    let initalQ = vec2f(output[1], output[2]);
    let initalS = vec2f(output[3], output[4]);
    let n = f32(size.y * size.x);
    let sizeTex = vec2f(vec2u(size.x, size.y));
    
    var currLoss = 0.0;
    var paramsPartial = vec4(0.0);

    currLoss = 0.0;
    for (var y = 0u; y < size.y; y++) {
        for (var x = 0u; x < size.x; x++) {
            let position = vec2u(x, y);
            let color = textureLoad(goalTexture, position, 0);
            let posNorm = vec2f(position) / sizeTex;
            // loss
            currLoss += Loss(initalQ, initalS, posNorm, color);

            // params
            paramsPartial += GradLoss_Q_S(initalQ, initalS, posNorm, color);            
        }
    }
    currLoss /= n;
    paramsPartial /= n;


    let step = paramsPartial;
    
    let newQ = initalQ - uniforms.stepSize * step.xy;
    let newS = initalS - uniforms.stepSize * step.zw;

    output[0] = currLoss;
    output[1] = newQ.x;
    output[2] = newQ.y;
    output[3] = newS.x;
    output[4] = newS.y;

    output[7] = sizeTex.x; 
    output[8] = sizeTex.y;
    output[9] = uniforms.stepSize * length(step);
    
}