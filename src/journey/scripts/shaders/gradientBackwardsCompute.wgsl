

@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(1) var ourSampler: sampler;
@group(0) @binding(2) var goalTexture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> lossOutput : array<atomic<u32>>;
@group(0) @binding(4) var forwardTexture : texture_2d<f32>; 
@group(0) @binding(5) var<storage, read_write> gradientChunks : array<array<Grad, nGauss>>; 


var<workgroup> grad: array<AtomicGrad, nGauss>;

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
    ) {
    let p = param[i]; 
    let gaussP = GaussParams(p.pos, p.scale, p.rot);
    let gauss = g(gaussP,x);

    let colorDiffDot = dot(colorDiff, vecSigmoid(p.color) - (*background));
    let alpha = sigmoid(p.alpha, 4.0);

    (*background) -= alpha * gauss * vecSigmoid(p.color);
    (*background) /= (1.0 - alpha * gauss + EPSILON);  

    let dAlphaBlend = alpha * (*oneMinusAlpha) * colorDiffDot;

    let gradGauss = EvalGradGauss(gaussP, x);    

    atomicAddvec2f_workgroup(&grad[i].pos,    gradGauss.pos   * select((dAlphaBlend), 1.0, xRAY));
    atomicAddvec2f_workgroup(&grad[i].scale,  gradGauss.scale * select((dAlphaBlend), 1.0, xRAY));
    atomicAddf32_workgroup(&grad[i].rot,    gradGauss.rot   * select((dAlphaBlend), 1.0, xRAY));

    // color gradient
    atomicAddvec3f_workgroup(&grad[i].color, alpha * gauss * select((*oneMinusAlpha), 1.0, xRAY) * vec3f(
        colorDiff.r * dSigmoid(p.color.r, 4.0),
        colorDiff.g * dSigmoid(p.color.g, 4.0),
        colorDiff.b * dSigmoid(p.color.b, 4.0),
    ));

    // alpha gradient  
    atomicAddf32_workgroup(&grad[i].alpha, colorDiffDot * gauss * (*oneMinusAlpha) * dSigmoid(p.alpha, 4.0));

    (*oneMinusAlpha) *= (1.0 - alpha * gauss);
}

fn loadGrad(i : i32) -> Grad {
    var g: Grad;
    g.pos   = atomicLoadvec2f_workgroup(&(grad[i].pos));
    g.scale = atomicLoadvec2f_workgroup(&(grad[i].scale));
    g.rot   = atomicLoadf32_workgroup(&(grad[i].rot));
    g.color = atomicLoadvec3f_workgroup(&(grad[i].color));
    g.alpha = atomicLoadf32_workgroup(&(grad[i].alpha));
    return g;
}


@compute @workgroup_size(chunkWidth, chunkHeight, 1) 
fn computeGD(
    @builtin(global_invocation_id) global_invocation_id : vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {

    let n = f32(sampleDim * sampleDim);
    // loss + backwards pass
    let sizeSample = vec2f(sampleDim);
    let uv = vec2f(global_invocation_id.xy) / sizeSample;
    let color = textureSampleLevel(goalTexture, ourSampler, uv, 0.0);
    let colorPreMult = vec4f(color.rgb * color.a, color.a);
    let gColorRaw = textureSampleLevel(forwardTexture, ourSampler, uv, 0.0);
    let gColor = vec4f(gColorRaw.rgb * gColorRaw.a, gColorRaw.a);

    atomicAddf32_storage_u(&lossOutput[0], (1.0/n * Loss(gColor, color)));

    let colorGrad = vec3f(
        (gColor.r - colorPreMult.r),
        (gColor.g - colorPreMult.g),
        (gColor.b - colorPreMult.b),
    );

    var oneMinusAlpha = 1.0;
    var background = gColor.rgb;
    for(var i = nGauss-1; i >= 0; i--) {
        // params
        GradLoss(&output, i, uv, colorGrad, &background, &oneMinusAlpha);                     
    }

    workgroupBarrier();

    // save atomic grads into storage buffer.
    let chunkAcross = (u32(sampleDim) + chunkWidth - 1) / chunkWidth; // (sampleDim is suqare, and + chunkWidht-1, rounds it up)
    let chunk = workgroup_id.y * chunkAcross + workgroup_id.x;

    for(var i = nGauss-1; i >= 0; i--) {
        gradientChunks[chunk][i] = loadGrad(i);  
    }
}