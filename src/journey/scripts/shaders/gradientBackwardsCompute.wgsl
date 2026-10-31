@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(1) var ourSampler: sampler;
@group(0) @binding(2) var goalTexture: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> lossOutput : array<array<f32, nSteps>, NUM_CHUNKS>;
@group(0) @binding(4) var forwardTexture : texture_2d<f32>; 
@group(0) @binding(5) var<storage, read_write> gradientStorage : array<Grad, MAX_SLOTS>; 
@group(0) @binding(6) var<storage, read_write> adamMemory : AdamMemory;
@group(0) @binding(7) var<uniform> uniforms : Uniform;
@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;


var<workgroup> grad: array<AtomicGrad, MAX_SPLATS_CHUNK>;
var<workgroup> loss: atomic<u32>;

fn readSplatId(index : u32) -> u32 {
    if(NUM_SORTS % 2 == 0) 
    {
        return radixMem.data[index * RADIX_ELEMS + R_SPLAT_INDEX];
    }
    else 
    {
        return radixMem.data2[index * RADIX_ELEMS + R_SPLAT_INDEX];
    }
}
fn readInvId(index : u32) -> u32 {
    if(NUM_SORTS % 2 == 0) 
    {
        return radixMem.data[index * RADIX_ELEMS + R_TRANSPOSE_INDEX];
    }
    else 
    {
        return radixMem.data2[index * RADIX_ELEMS + R_TRANSPOSE_INDEX];
    }
}

fn Loss(gColor: vec4f, imgC: vec4f) -> f32 {
    let l2 = (
            (gColor.r - imgC.r) * (gColor.r - imgC.r) + 
            (gColor.g - imgC.g) * (gColor.g - imgC.g) + 
            (gColor.b - imgC.b) * (gColor.b - imgC.b)
        ) / 3.0;
    
    let l1 = (
            (gColor.r - imgC.r) + 
            (gColor.g - imgC.g) + 
            (gColor.b - imgC.b)
        ) / 3.0;
    
    return (1.0 - uniforms.lossWeight) * l2 + (uniforms.lossWeight) * l1;
}

fn GradLoss(
    p : Params,
    i : u32,
    x: vec2f, 
    colorDiff : vec3f,
    background : ptr<function, vec3f>,
    oneMinusAlpha : ptr<function, f32>,
    ) {    
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

fn loadGrad(i : u32) -> Grad {
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
    @builtin(local_invocation_id) local_invocation_id : vec3u,
) {
    let chunkAcross = u32(( (sampleDim) + chunkWidth - 1) / chunkWidth); // (sampleDim is suqare, and + chunkWidht-1, rounds it up)
    let chunk = workgroup_id.y * chunkAcross + workgroup_id.x;

    let offData = radixMem.offsets[chunk];
    var prevOff = 0u;
    if(chunk > 0u) {
        prevOff = radixMem.offsets[chunk-1u];
    }
    let numSplats = min((offData - prevOff), MAX_SPLATS_CHUNK); // if for some reason there are more, we just have to ignore them? :(

    if(numSplats == 0u) {
        // do nothing
    }
    else {


        let n = f32(sampleDim * sampleDim);
        // backwards pass
        let sizeSample = vec2f(sampleDim);
        let uv = vec2f(global_invocation_id.xy) / sizeSample;
        let color = textureSampleLevel(goalTexture, ourSampler, uv, 0.0);
        let colorPreMult = vec4f(color.rgb * color.a, color.a);
        let gColorRaw = textureSampleLevel(forwardTexture, ourSampler, uv, 0.0);
        let gColor = vec4f(gColorRaw.rgb * gColorRaw.a, gColorRaw.a);

        atomicAddf32_workgroup_u(&loss, (Loss(gColor, color)));

        let colorGradL1 : vec3f = sign(vec3f(
                (gColor.r - colorPreMult.r),
                (gColor.g - colorPreMult.g),
                (gColor.b - colorPreMult.b),
            ));
        let colorGradL2 : vec3f = (vec3f(
                (gColor.r - colorPreMult.r),
                (gColor.g - colorPreMult.g),
                (gColor.b - colorPreMult.b),
            ));
        let colorGrad = (1.0 - uniforms.lossWeight) * colorGradL2 + (uniforms.lossWeight) * colorGradL1;
    
        var oneMinusAlpha = 1.0;
        var background = gColor.rgb;
        for(var i = numSplats; i > 0u; i--) {
            // params
            let off = (prevOff + i - 1u);
            let splatId = readSplatId(off);
            GradLoss(output[splatId], i-1u, uv, colorGrad, &background, &oneMinusAlpha);                     
        }
    }
    workgroupBarrier();
    let local_id = local_invocation_id.y * chunkWidth + local_invocation_id.x;
    if(numSplats == 0u || local_id > 0u) {
        return;
    }

    // save atomic grads into storage buffer.
    for(var i = numSplats; i > 0; i--) {
        let off = (prevOff + i - 1u);
        let data1StorageIndex = readInvId(off);
        gradientStorage[data1StorageIndex] = loadGrad(i-1u);  
    }

    // save loss
    let lossI = adamMemory.t % nSteps;
    lossOutput[chunk][lossI] = atomicLoadf32_workgroup_u(&loss) / (chunkWidth * chunkHeight);

}