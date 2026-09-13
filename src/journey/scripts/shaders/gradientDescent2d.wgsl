struct Uniform {
    stepSize : f32,
};

struct Params {
    pos : vec2f,
    scale : vec2f,
    rot : f32,
    color : vec3f,
    alpha: f32,
};

@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(1) var ourSampler: sampler;
@group(0) @binding(2) var goalTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> uniforms : Uniform;

struct Grad {  
    pos : vec2f,
    scale : vec2f,
    rot : f32,
    color : vec3f,
    alpha: f32,
};

fn rotMat(r: f32) -> mat2x2f {
    return mat2x2f(
        cos(r), sin(r), // column 0
        -sin(r), cos(r) // column 1
    ); 
}

fn dSigmoid(x : f32) -> f32 {
    return exp(-x + 4.0) / ((1.0 + exp(-x + 4.0))*(1.0 + exp(-x + 4.0)));
}

fn sigmoid(x : f32) -> f32 {
    return 1.0 / (1.0 + exp(-x + 4.0));
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

fn Loss(gColor: vec4f, imgC: vec4f) -> f32 {
    return (
        (gColor.r - imgC.r) + 
        (gColor.g - imgC.g) + 
        (gColor.b - imgC.b)
    ) / 3.0;;
}

fn GradLoss_Q_S(p : Params, x: vec2f, imgC: vec4f, gColor: vec4f, background : f32, oneMinusAlpha : f32) -> Grad {

    let gauss = g(p,x);
    // let gColor = vec4f(vecSigmoid(p.color)*p.alpha*gauss, p.alpha);
    let diff = ((gColor.r - imgC.r) + (gColor.g - imgC.g) + (gColor.b - imgC.b)) / 3.0;
    let dist = (x - p.pos);

    //  v = R^T*(x-q)
    let R = rotMat(p.rot);
    let v = transpose(R)*dist;

    let dLossGauss2 = -2.0 * diff * gauss;

    // \Sigma = R * S^2 * R^T
    let sigma = R * exp(p.scale) * transpose(R);

    // 2 * (g(x) - I) * -0.5 * g(x) * \Sigma * * -2 * (x-q)
    let gradQ = -1.0 * dLossGauss2 * sigma * dist;

    // 2 * (g(x) - I) * -0.5 * g(x) * 2 * v^t * s * v 
    let gradS = 10.0 * dLossGauss2 * v * v * exp(p.scale);
    
    //d/d\theta (x-u)^T * RSS^TR^T * (x-u) <=>  v^T * (L*S2 - S2*L) * v => dR/d\theta = L*R  => 
    // 2 * (g(x) - I) * -0.5 * g(x) * 2*v_x*v_y*(s_x^2 - s_y^2)
    let gradR = 40.0 * dLossGauss2 * v.x * v.y * (exp(p.scale.x) - exp(p.scale.y));

    let gradC = vec3f(
        50.0 * (gColor.r - imgC.r) * dSigmoid(p.color.r) * sigmoid(p.alpha) * gauss,
        50.0 * (gColor.g - imgC.g) * dSigmoid(p.color.g) * sigmoid(p.alpha) * gauss,
        50.0 * (gColor.b - imgC.b) * dSigmoid(p.color.b) * sigmoid(p.alpha) * gauss,
    );

    // alpha gradient
    // 
    let gradA = 50.0 * 2.0 * diff * (gauss - background) * oneMinusAlpha * dSigmoid(p.alpha);    

    return Grad(gradQ, gradS, gradR, gradC, gradA);
}


@compute @workgroup_size(1) fn computeGD() {

    let size = vec2u(128, 128); // static choosen size
    let n = f32(size.y * size.x);
    let sizeSample = vec2f(size);
    
    const nGauss = 2;

    var gradients = array<Grad, nGauss>();

        var currLoss = 0.0;
        var paramsPartial = Params(vec2f(0.0), vec2f(0.0), 0.0, vec3f(0.0), 0.0);

        currLoss = 0.0;
        for (var y = 0u; y < size.y; y++) {
            for (var x = 0u; x < size.x; x++) {

                let uv = vec2f(vec2u(x, y)) / sizeSample;
                let color = textureSampleLevel(goalTexture, ourSampler, uv, 0.0);
                let colorPreMult = vec4f(color.rgb * color.a, color.a);
                
                var gColor = vec4f(vec3f(0.0), 1.0);

                var backgrounds = array<f32, nGauss>();

                for(var i = 0; i < nGauss; i++) {
                    let p = output[i];
                    
                    let gauss = g(p,uv);
                    let alpha = gauss * sigmoid(p.alpha);

                    backgrounds[i] = (gColor.r + gColor.g + gColor.b) / 3.0;
                    
                    gColor = alpha * vec4f(vecSigmoid(p.color), 1.0) + (1.0 - alpha) * gColor;
                }

                // loss
                currLoss += Loss(gColor, color);
                
                var oneMinusAlpha = 1.0;
                for(var i = nGauss-1; i >= 0; i--) {
                    let p = output[i];
                    
                    // params
                    let grad = GradLoss_Q_S(p, uv, colorPreMult, gColor, backgrounds[i], oneMinusAlpha); 

                    oneMinusAlpha *= (1.0 - g(p, uv) * sigmoid(p.alpha));
                    
                    gradients[i].pos += grad.pos;
                    gradients[i].scale += grad.scale;
                    gradients[i].rot += grad.rot;
                    gradients[i].color += grad.color;
                    gradients[i].alpha += grad.alpha;
                
                }
                }
        }
        currLoss /= n;

        for(var i = 0; i < nGauss; i++) {
            let initalParams = output[i];
            
            let stepQ = gradients[i].pos / n;
            let stepS = gradients[i].scale / n;
            let stepR = gradients[i].rot / n;
            let stepC = gradients[i].color / n;
            let stepA = gradients[i].alpha / n;

            // this is just a vector, when optimizing, no need to convert it
            let newQ = initalParams.pos - uniforms.stepSize * stepQ;
            let newS = initalParams.scale - uniforms.stepSize * stepS;
            let newR = initalParams.rot - uniforms.stepSize * stepR;
            let newC = initalParams.color - uniforms.stepSize * stepC;
            let newA = initalParams.alpha - uniforms.stepSize * stepA;

            output[i] = Params(newQ, newS, newR, newC, newA);
        }
    }