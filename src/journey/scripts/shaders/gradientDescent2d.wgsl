// constants
const NUM_GAUSS = 2;
const xRAY = false;


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
@group(0) @binding(6) var forwardTexture : texture_2d<f32>; 

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

fn Luminance(color : vec3f) -> f32 {
    return (color.r + color.g + color.b) / 3.0;
}

fn GradLoss(
    param : ptr<storage, array<Params>, read_write>, // should not be read_write!
    i : i32, 
    x: vec2f, 
    colorDiff : vec3f,
    background : ptr<function, vec3f>,
    oneMinusAlpha : ptr<function, f32>,
    grad : ptr<function, array<Grad, 2>>,
) {
    let p = (*param)[i]; // kinda defeats the purpose, but ok for now
    let gaussP = GaussParams(p.pos, p.scale, p.rot);
    let gauss = g(gaussP,x);


    // let diff = ((gColor.r - imgC.r) + (gColor.g - imgC.g) + (gColor.b - imgC.b)) / 3.0;
    let dLoss = Luminance(colorDiff);
    let alpha = sigmoid(p.alpha, 4.0);

    (*background) -= alpha * gauss * vecSigmoid(p.color);
    (*background) /= (1.0 - alpha * gauss + uniforms.adamP.eps);  

    let dAlphaBlend = alpha * Luminance(vecSigmoid(p.color) - (*background));

    let gradGauss = EvalGradGauss(gaussP, x);    

    (*grad)[i].pos   += dLoss * gradGauss.pos   * select((dAlphaBlend), 1.0, xRAY);
    (*grad)[i].scale += dLoss * gradGauss.scale * select((dAlphaBlend), 1.0, xRAY);
    (*grad)[i].rot   += dLoss * gradGauss.rot   * select((dAlphaBlend), 1.0, xRAY);

    // color gradient

    (*grad)[i].color += alpha * gauss * select((*oneMinusAlpha), 1.0, xRAY) * vec3f(
        colorDiff.r * dSigmoid(p.color.r, 4.0),
        colorDiff.g * dSigmoid(p.color.g, 4.0),
        colorDiff.b * dSigmoid(p.color.b, 4.0),
    );

    // alpha gradient  
    (*grad)[i].alpha += dLoss * (gauss - Luminance((*background))) * (*oneMinusAlpha) * dSigmoid(p.alpha, 4.0);

    (*oneMinusAlpha) *= (1.0 - alpha * gauss);
}


@compute @workgroup_size(1) fn computeGD() {

    let size = vec2u(128, 128); // static choosen size
    let n = f32(size.y * size.x);
    let sizeSample = vec2f(size);
    
    const nGauss = 2;

    var gradients = array<Grad, nGauss>();

    // loss + backwards pass
    var currLoss = 0.0;
    var gradsLoss = vec3f(0.0);
    for (var y = 0u; y < size.y; y++) {
        for (var x = 0u; x < size.x; x++) {
            let uv = vec2f(vec2u(x, y)) / sizeSample;
            let color = textureSampleLevel(goalTexture, ourSampler, uv, 0.0);
            let colorPreMult = vec4f(color.rgb * color.a, color.a);
            let gColorRaw = textureSampleLevel(forwardTexture, ourSampler, uv, 0.0);
            let gColor = vec4f(gColorRaw.rgb * gColorRaw.a, gColorRaw.a);

            currLoss += Loss(gColor, color);
   
            let colorGrad = 2.0 * vec3f(
                (gColor.r - colorPreMult.r),
                (gColor.g - colorPreMult.g),
                (gColor.b - colorPreMult.b),
            );

            var oneMinusAlpha = 1.0;
            var background = gColor.rgb;
            for(var i = nGauss-1; i >= 0; i--) {
                // params
                GradLoss(&output, i, uv, colorGrad, &background, &oneMinusAlpha, &gradients);                     
            }

        }
    }
    currLoss /= n;
    lossOutput[0] = currLoss;

    for(var i = 0; i < nGauss; i++) {
        
        gradients[i].pos    /= n;
        gradients[i].scale  /= n;
        gradients[i].rot    /= n;
        gradients[i].color  /= n;
        gradients[i].alpha  /= n;

    }

    // adam step
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