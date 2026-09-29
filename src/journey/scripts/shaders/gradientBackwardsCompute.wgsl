

@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(1) var ourSampler: sampler;
@group(0) @binding(2) var goalTexture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> lossOutput : array<f32>;
@group(0) @binding(4) var forwardTexture : texture_2d<f32>; 
@group(0) @binding(5) var gradients : array<AtomicGrad, nGauss>; 


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
    colorDiff : vec3f,
    background : ptr<function, vec3f>,
    oneMinusAlpha : ptr<function, f32>,
    grad : ptr<storage, array<Grad, nGauss>>,
) {
    let p = (*param)[i]; // kinda defeats the purpose, but ok for now
    let gaussP = GaussParams(p.pos, p.scale, p.rot);
    let gauss = g(gaussP,x);

    let colorDiffDot = dot(colorDiff, vecSigmoid(p.color) - (*background));
    let alpha = sigmoid(p.alpha, 4.0);

    (*background) -= alpha * gauss * vecSigmoid(p.color);
    (*background) /= (1.0 - alpha * gauss + uniforms.adamP.eps);  

    let dAlphaBlend = alpha * (*oneMinusAlpha) * colorDiffDot;

    let gradGauss = EvalGradGauss(gaussP, x);    

    atomicAdd(&(*grad)[i].pos,    gradGauss.pos   * select((dAlphaBlend), 1.0, xRAY));
    atomicAdd(&(*grad)[i].scale,  gradGauss.scale * select((dAlphaBlend), 1.0, xRAY));
    atomicAdd(&(*grad)[i].rot,    gradGauss.rot   * select((dAlphaBlend), 1.0, xRAY));

    // color gradient

    atomicAdd(&(*grad)[i].color, alpha * gauss * select((*oneMinusAlpha), 1.0, xRAY) * vec3f(
        colorDiff.r * dSigmoid(p.color.r, 4.0),
        colorDiff.g * dSigmoid(p.color.g, 4.0),
        colorDiff.b * dSigmoid(p.color.b, 4.0),
    ));

    // alpha gradient  
    atomicAdd(&(*grad)[i].alpha, colorDiffDot * gauss * (*oneMinusAlpha) * dSigmoid(p.alpha, 4.0));

    (*oneMinusAlpha) *= (1.0 - alpha * gauss);
}


@compute @workgroup_size(1,1,1) 
fn computeGD(@builtin(global_invocation_id) global_invocation_id : vec3u) {

    // loss + backwards pass
    let sizeSample = vec2f(sampleDim);
    let uv = global_invocation_id.xy / sizeSample;
    let color = textureSampleLevel(goalTexture, ourSampler, uv, 0.0);
    let colorPreMult = vec4f(color.rgb * color.a, color.a);
    let gColorRaw = textureSampleLevel(forwardTexture, ourSampler, uv, 0.0);
    let gColor = vec4f(gColorRaw.rgb * gColorRaw.a, gColorRaw.a);

    lossOutput[0] += Loss(gColor, color);

    let colorGrad = vec3f(
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