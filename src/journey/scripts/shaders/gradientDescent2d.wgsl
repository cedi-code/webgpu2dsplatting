struct Uniform {
    adamP : AdamParams,
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
@group(0) @binding(4) var<storage, read_write> lossOutput : array<f32>;
@group(0) @binding(5) var<storage, read_write> adamMemory : AdamMemory;


struct Grad {  
    pos: vec2f,
    scale : vec2f,
    rot : f32,
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


fn GradLoss(
    param : ptr<storage, array<Params>, read_write>, // should not be read_write!
    i : i32, 
    x: vec2f, 
    imgC: vec4f,
    gColor: vec4f,
    background : f32,
    oneMinusAlpha : f32,
    grad : ptr<function, array<Grad, 2>>,
) {
    let p = (*param)[i]; // kinda defeats the purpose, but ok for now
    let gaussP = GaussParams(p.pos, p.scale, p.rot);
    let gauss = g(gaussP,x);

    let diff = ((gColor.r - imgC.r) + (gColor.g - imgC.g) + (gColor.b - imgC.b)) / 3.0;

    let gradGauss = EvalGradGauss(gaussP, x);    
    let dLossGauss2 = 2.0 * diff;

    (*grad)[i].pos += dLossGauss2 * gradGauss.pos;
    (*grad)[i].scale += dLossGauss2 * gradGauss.scale;
    (*grad)[i].rot += dLossGauss2 * gradGauss.rot;

    (*grad)[i].color += vec3f(
        (gColor.r - imgC.r) * dSigmoid(p.color.r, 4.0) * sigmoid(p.alpha, 4.0) * gauss,
        (gColor.g - imgC.g) * dSigmoid(p.color.g, 4.0) * sigmoid(p.alpha, 4.0) * gauss,
        (gColor.b - imgC.b) * dSigmoid(p.color.b, 4.0) * sigmoid(p.alpha, 4.0) * gauss,
    );

    // alpha gradient    
    (*grad)[i].alpha += 2.0 * diff * (gauss - background) * oneMinusAlpha * dSigmoid(p.alpha, 4.0);
}


@compute @workgroup_size(1) fn computeGD() {

    let size = vec2u(128, 128); // static choosen size
    let n = f32(size.y * size.x);
    let sizeSample = vec2f(size);
    
    const nGauss = 2;

    var gradients = array<Grad, nGauss>();

        var currLoss = 0.0;

        currLoss = 0.0;
        for (var y = 0u; y < size.y; y++) {
            for (var x = 0u; x < size.x; x++) {

                let uv = vec2f(vec2u(x, y)) / sizeSample;
                let color = textureSampleLevel(goalTexture, ourSampler, uv, 0.0);
                let colorPreMult = vec4f(color.rgb * color.a, color.a);
                
                var gColor = vec4f(vec3f(0.0), 1.0);

                var backgrounds = array<f32, nGauss>();

                // isnt this basically a forward pass?
                for(var i = 0; i < nGauss; i++) {
                    let p = output[i];
                    let gaussP = GaussParams(p.pos, p.scale, p.rot);
                    
                    let gauss = g(gaussP,uv);
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
                    GradLoss(&output, i, uv, colorPreMult, gColor, backgrounds[i], oneMinusAlpha, &gradients); 

                    let gaussP = GaussParams(p.pos, p.scale, p.rot);
                    oneMinusAlpha *= (1.0 - g(gaussP, uv) * sigmoid(p.alpha, 4.0));
                }
                }
        }
        currLoss /= n;
        lossOutput[0] = currLoss;

        for(var i = 0; i < nGauss; i++) {
            // super ugly but for now i guess
            gradients[i].pos    /= n;
            gradients[i].scale  /= n;
            gradients[i].rot    /= n;
            gradients[i].color  /= n;
            gradients[i].alpha  /= n;
        }
        adamMemory.t += 1u;
        var moment = adamMemory.m;
        var varian = adamMemory.v;

        adamStepGrad(uniforms.adamP, f32(adamMemory.t), &moment, &varian, &gradients);
        adamMemory.m = moment;
        adamMemory.v = varian;

        for(var i = 0; i < nGauss; i++) {

            let initalParams = output[i];

            let newQ = initalParams.pos - gradients[i].pos;
            let newS = initalParams.scale - gradients[i].scale;
            let newR = initalParams.rot - gradients[i].rot;
            let newC = initalParams.color - gradients[i].color;
            let newA = initalParams.alpha - gradients[i].alpha;

            output[i] = Params(newQ, newS, newR, newC, newA);
        }
    }