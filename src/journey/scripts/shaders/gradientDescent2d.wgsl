struct Uniform {
    stepSize : f32,
};

struct Params {
    gauss : GaussParams,
    color : vec3f,
    alpha: f32,
};

@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(1) var ourSampler: sampler;
@group(0) @binding(2) var goalTexture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> uniforms : Uniform;
@group(0) @binding(4) var<storage, read_write> lossOutput : array<f32>;

struct Grad {  
    gauss: GradGauss,
    color : vec3f,
    alpha: f32,
};

fn Loss(gColor: vec4f, imgC: vec4f) -> f32 {
    return (
        (gColor.r - imgC.r) * (gColor.r - imgC.r) + 
        (gColor.g - imgC.g) * (gColor.g - imgC.g) + 
        (gColor.b - imgC.b) * (gColor.b - imgC.b)
    ) / 3.0;;
}


fn GradLoss_Q_S(p : Params, x: vec2f, imgC: vec4f, gColor: vec4f, background : f32, oneMinusAlpha : f32) -> Grad {


    let gauss = g(p.gauss,x);
    // let gColor = vec4f(vecSigmoid(p.color)*p.alpha*gauss, p.alpha);
    let diff = ((gColor.r - imgC.r) + (gColor.g - imgC.g) + (gColor.b - imgC.b)) / 3.0;

    let gradGauss = EvalGradGauss(p.gauss, x);    
    let dLossGauss2 = 2.0 * diff;

    let gradC = vec3f(
        50.0 * (gColor.r - imgC.r) * dSigmoid(p.color.r, 4.0) * sigmoid(p.alpha, 4.0) * gauss,
        50.0 * (gColor.g - imgC.g) * dSigmoid(p.color.g, 4.0) * sigmoid(p.alpha, 4.0) * gauss,
        50.0 * (gColor.b - imgC.b) * dSigmoid(p.color.b, 4.0) * sigmoid(p.alpha, 4.0) * gauss,
    );

    // alpha gradient    
    let gradA = 50.0 * 2.0 * diff * (gauss - background) * oneMinusAlpha * dSigmoid(p.alpha, 4.0);    

    let gradLossQ = dLossGauss2 * gradGauss.pos;
    let gradLossS = 5.0 * dLossGauss2 * gradGauss.scale;
    let gradLossR = 20.0 * dLossGauss2 * gradGauss.rot;

    return Grad(GradGauss(gradLossQ, gradLossS, gradLossR), gradC, gradA);
}


@compute @workgroup_size(1) fn computeGD() {

    let size = vec2u(128, 128); // static choosen size
    let n = f32(size.y * size.x);
    let sizeSample = vec2f(size);
    
    const nGauss = 2;

    var gradients = array<Grad, nGauss>();

        var currLoss = 0.0;
        var paramsPartial = Params(GaussParams(vec2f(0.0), vec2f(0.0), 0.0), vec3f(0.0), 0.0);

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
                    
                    let gauss = g(p.gauss,uv);
                    let alpha = gauss * sigmoid(p.alpha, 4.0);

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

                    oneMinusAlpha *= (1.0 - g(p.gauss, uv) * sigmoid(p.alpha, 4.0));
                    
                    gradients[i].gauss.pos      += grad.gauss.pos;
                    gradients[i].gauss.scale    += grad.gauss.scale;
                    gradients[i].gauss.rot      += grad.gauss.rot;
                    gradients[i].color          += grad.color;
                    gradients[i].alpha          += grad.alpha;
                
                }
                }
        }
        currLoss /= n;
        lossOutput[0] = currLoss;

        for(var i = 0; i < nGauss; i++) {
            let initalParams = output[i];
            
            let stepQ = gradients[i].gauss.pos / n;
            let stepS = gradients[i].gauss.scale / n;
            let stepR = gradients[i].gauss.rot / n;
            let stepC = gradients[i].color / n;
            let stepA = gradients[i].alpha / n;

            // this is just a vector, when optimizing, no need to convert it
            let newQ = initalParams.gauss.pos - uniforms.stepSize * stepQ;
            let newS = initalParams.gauss.scale - uniforms.stepSize * stepS;
            let newR = initalParams.gauss.rot - uniforms.stepSize * stepR;
            let newC = initalParams.color - uniforms.stepSize * stepC;
            let newA = initalParams.alpha - uniforms.stepSize * stepA;

            output[i] = Params(GaussParams(newQ, newS, newR), newC, newA);
        }
    }