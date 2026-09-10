struct Uniform {
    stepSize : f32,
};

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var ourSampler: sampler;
@group(0) @binding(2) var goalTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> uniforms : Uniform;


struct Params {
    pos : vec2f,
    scale : vec2f,
    rot : f32
};

struct Grad {  
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

fn g(p : Params, x : vec2f) -> f32 {

    let d = x - p.pos;
    let a = exp(p.scale) * (transpose(rotMat(p.rot)) * d);
    let D2 = dot(a, a);

    return exp(-0.5 * D2);
}

fn Loss(p : Params, x: vec2f, imgC: vec4f) -> f32 {

    let gColor = vec4f(vec3f(g(p,x)), 1.0);

    return (gColor.r - imgC.r)*(gColor.r - imgC.r);
}

fn GradLoss_Q_S(p : Params, x: vec2f, imgC: vec4f) -> Grad {

    let gColor = vec4f(vec3f(g(p,x)), 1.0);
    let diff = (gColor.r - imgC.r);
    let dist = (x - p.pos);

    //  v = R^T*(x-q)
    let R = rotMat(p.rot);
    let v = transpose(R)*dist;

    let dLossGauss2 = -2.0 * diff * gColor.r;

    // \Sigma = R * S^2 * R^T
    let sigma = R * exp(p.scale) * transpose(R);

    // 2 * (g(x) - I) * -0.5 * g(x) * \Sigma * * -2 * (x-q)
    let gradQ = -1.0 * dLossGauss2 * sigma * dist;

    // 2 * (g(x) - I) * -0.5 * g(x) * 2 * v^t * s * v 
    let gradS = 10.0 * dLossGauss2 * v * v * exp(p.scale);
    
    //d/d\theta (x-u)^T * RSS^TR^T * (x-u) <=>  v^T * (L*S2 - S2*L) * v => dR/d\theta = L*R  => 
    // 2 * (g(x) - I) * -0.5 * g(x) * 2*v_x*v_y*(s_x^2 - s_y^2)
    let gradR = 40.0 * dLossGauss2 * v.x * v.y * (exp(p.scale.x) - exp(p.scale.y));

    return Grad(gradQ, gradS, gradR);
}


@compute @workgroup_size(1) fn computeGD() {

    let size = vec2u(128, 128); // static choosen size
    let initalQ = vec2f(output[1], output[2]);
    let initalS = vec2f(output[3], output[4]);
    let initalR = output[5];
    let initalParams = Params(initalQ, initalS, initalR);
    let n = f32(size.y * size.x);
    let sizeSample = vec2f(size);
    
    var currLoss = 0.0;
    var paramsPartial = Params(vec2f(0.0), vec2f(0.0), 0.0);

    currLoss = 0.0;
    for (var y = 0u; y < size.y; y++) {
        for (var x = 0u; x < size.x; x++) {

            let uv = vec2f(vec2u(x, y)) / sizeSample;
            let color = textureSampleLevel(goalTexture, ourSampler, uv, 0.0);
            // loss
            currLoss += Loss(initalParams, uv, color);

            // params

            let grad = GradLoss_Q_S(initalParams, uv, color);  
            paramsPartial.pos += grad.pos;
            paramsPartial.scale += grad.scale;
            paramsPartial.rot += grad.rot;

        }
    }
    currLoss /= n;


    let stepQ = paramsPartial.pos / n;
    let stepS = paramsPartial.scale / n;
    let stepR = paramsPartial.rot / n;


    let newQ = initalQ - uniforms.stepSize * stepQ;
    let newS = initalS - uniforms.stepSize * stepS;
    let newR = initalR - uniforms.stepSize * stepR;

    output[0] = currLoss;
    output[1] = newQ.x;
    output[2] = newQ.y;
    output[3] = newS.x;
    output[4] = newS.y;
    output[5] = newR;
    
}