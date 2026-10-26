struct MomentumPair {
    m: Grad,
    v: Grad,
};

@group(0) @binding(0) var<storage, read_write> splats: array<Params>;
@group(1) @binding(0) var<storage, read_write> splatsPingPong: array<Params>;
@group(1) @binding(1) var<storage, read_write> momentumPingPong: array<MomentumPair>;
@group(0) @binding(5) var<storage, read_write> gradients: array<Grad, MAX_SLOTS>; 
@group(0) @binding(6) var<storage, read_write> adamMemory : AdamMemory;
@group(0) @binding(7) var<uniform> uniforms : Uniform;
@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;
@group(0) @binding(9) var<storage, read_write> drawBuffer : array<u32>;


override first_pass: bool;
var<workgroup> groupOffset : atomic<u32>;
var<workgroup> sums: array<u32, GROUPSIZE>;

const NUM_256 = (NUM_DISPATCHES + GROUPSIZE - 1u) / GROUPSIZE;

fn copySplat(splat : Params) -> Params {
    return Params(splat.pos, splat.scale, splat.rot, splat.color, splat.alpha);
}

fn eigenValues(varMat : mat2x2f) -> vec2f {
    // varMat is symetric
    // first eigen value is det(A - \lambda * I) = 0
    // A = ( a, c // c, b )
    // det(A - lambda * I) = (a - \lambda) * (b - lambda) - c^2 = 0
    // (ab - (a + b)\lambda - lambda^2 - c^2 = 0
    // lambda^2 - (a + b)\lambda + (ab - c^2) = 0
    // mitternachts formel:
    // lambda_1,2 = 0.5 * ((a+b) +- sqrt((a+b)^2 - 4*(ab - c^2)))
    // we want the bigger eigen value and sqrt > 0 so we take +?
    // lambda_1 = 0.5 * (a+b + sqrt((a-b)^2 + 4c^2))
    
    let a = varMat[0].x;
    let b = varMat[1].y;
    let c = varMat[0].y;

    let D = max(0.0, (a-b)*(a-b) + 4.0 * c * c);
    let Dsqrt = sqrt(D);
    let lambda1 = 0.5 * ((a + b) + Dsqrt);
    let lambda2 = 0.5 * ((a + b) - Dsqrt);

    return vec2f(lambda1, lambda2);
}

fn firstEigenVec(varMat : mat2x2f, lambda1 : f32, lambda2 : f32) -> vec2f {

    // v1 = [ (\lambda_1 - b), c ]

    let a = varMat[0].x;
    let b = varMat[1].y;
    let c = varMat[0].y;

    let v1 = vec2f(lambda1 - b , c);
    let v2 = vec2f(c , lambda1 - a);

    let v1Len2 = dot(v1, v1);

    let v3 = select(v1, v2, v1Len2 < 1.0e-10);
    
    let v3Len2 = dot(v3, v3);

    let scale = sqrt(max(0.0, lambda1));

    return normalize(select(v3, vec2f(1,0), v3Len2 < 1e-10)) * scale;
}

@compute @workgroup_size(GROUPSIZE, 1, 1)
fn cs(
    @builtin(global_invocation_id) global_invocation_id : vec3u,
    @builtin(local_invocation_id) local_invocation_id : vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
    ) {

    let splat_id = global_invocation_id.x;
    let local_id = local_invocation_id.x;
    let gradientOff = radixMem.splatOffsetsTotal[splat_id] / RADIX_ELEMS;
    let gradient = gradients[gradientOff];
    
    var splatP: Params;
    var mGrad : Grad;
    var vGrad : Grad;
    if(first_pass) {
        splatP = splats[splat_id];
        mGrad = adamMemory.m[splat_id];
        vGrad = adamMemory.v[splat_id];
    } else {
        splatP = splatsPingPong[splat_id];
        mGrad = momentumPingPong[splat_id].m;
        vGrad = momentumPingPong[splat_id].v;
    }

    let n = f32(sampleDim * sampleDim);

    let sMat2 = mat2x2f(exp(-2.0 * splatP.scale).x, 0.0, 0.0, exp(-2.0 * splatP.scale).y);
    let rMat = rotMat(splatP.rot);
    let varMat = rMat * sMat2 * transpose(rMat);
    let eigenVals = eigenValues(varMat);

    // == cutoff checks ==

    let maxAlpha = uniforms.alphaCutOff;

    let screenMin = 1.0 - OFF_CANVAS_TRHESHOLD; // (1.0 - 1.5) = -0.5
    let screenMax = OFF_CANVAS_TRHESHOLD;

    let offScreen =
        any(splatP.pos < vec2f(screenMin)) ||
        any(splatP.pos > vec2f(screenMax));

    let lowAlpha = splatP.alpha < maxAlpha;

    let aspectMaxCheck = eigenVals.x > uniforms.eigenVAspectThreshMax;


    let removeFlag = offScreen || lowAlpha || aspectMaxCheck;

    // == Gradient check ==

    let posGrad = gradient.pos / n;
    let scaleGrad = gradient.scale / n;

    let scaleGradFlag = any(abs(scaleGrad) >= vec2f(uniforms.scaleGradThresh));
    let posGradFlag = any(abs(posGrad) >= vec2f(uniforms.posGradThresh));
    let refineFlag = posGradFlag || 
                     scaleGradFlag;
    // == Flag assignment ==
    sums[local_id] = 0;
    if(!removeFlag) {
        sums[local_id] = select(1u, 2u, refineFlag);
    }

    workgroupBarrier();

    // prefix sum with 8 barriers
    for(var i = 0u; i < BARRIERS; i++) {

        let chunk_size = (2u << i);
        let chunk_id = local_id % chunk_size;
        let half_chunk = chunk_size >> 1;

        if(chunk_id >= half_chunk) {

            let chunk_start = (local_id - chunk_id) ;
            let chunk_read_id = chunk_start + half_chunk - 1u;
            sums[local_id] += sums[chunk_read_id];
        }
        workgroupBarrier();
    }

    if(first_pass) {
        splatsPingPong[splat_id] = splatP;
        momentumPingPong[splat_id] = MomentumPair(mGrad, vGrad);

        // store total sum in a storage buffer
        if(local_id == GROUPSIZE - 1) { // last thread does it
            radixMem.splatOffsets[workgroup_id.x] = sums[local_id];
        }

        return; // stops here
    }

    // another dispatch recalculates the prefix sum and corrects the offsets by total sums in the offsets
    for(var i = 0u; i < NUM_256; i++) {
        let dispatchId = local_id + GROUPSIZE * i;
        if(dispatchId >= workgroup_id.x) {
            break;
        }
        atomicAdd(&groupOffset, radixMem.splatOffsets[dispatchId]);
    }
    workgroupBarrier();

    // last thread
    if(workgroup_id.x == NUM_DISPATCHES-1u && local_id == GROUPSIZE-1u) {
        let newNumSplats = atomicLoad(&groupOffset)  + sums[local_id];
        drawBuffer[1] = newNumSplats;
        drawBuffer[4] = newNumSplats;

    }


    let currOff = atomicLoad(&groupOffset) + sums[local_id] - select(1u, 2u, refineFlag);

    if(removeFlag || currOff >= NUM_SPLATS-2u) {
        return; 
    }

    if(refineFlag) {
        // densification!
        var splat2 = copySplat(splatP);


        if(scaleGradFlag) {
            // overconstruction
            let eigenVec = firstEigenVec(varMat, eigenVals.x, eigenVals.y);
            
            splatP.pos -= eigenVec;
            splat2.pos += eigenVec;

            let DOWNSCALE_BY = log(uniforms.scaleDivideConst);

            splatP.scale += DOWNSCALE_BY; // /= SCALE_DIVIDE_CONST;
            splat2.scale += DOWNSCALE_BY; // /= SCALE_DIVIDE_CONST;

        }
        else {
            // underconstruction
            let alpha : f32 = 1.0 - sqrt(1.0 - sigmoid(splatP.alpha, 4.0));
            splat2.alpha = sigmoidInv(alpha, 4.0);
            splatP.alpha = sigmoidInv(alpha, 4.0);
            // splat2.pos -= gradDir * stdGradDir * 0.5;
        }
        splats[currOff + 1u] = splat2;
    }

    splats[currOff] = splatP;
    adamMemory.m[currOff] = mGrad;
    adamMemory.v[currOff] = vGrad;
}