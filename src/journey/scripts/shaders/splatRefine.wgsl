@group(0) @binding(0) var<storage, read_write> splats: array<Params>;
@group(1) @binding(0) var<storage, read_write> splatsPingPong: array<Params>;
@group(0) @binding(5) var<storage, read_write> gradients: array<Grad, MAX_SLOTS>; 
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
    if(first_pass) {
        splatP = splats[splat_id];
    } else {
        splatP = splatsPingPong[splat_id];
    }

    let n = f32(sampleDim * sampleDim);

    // == cutoff checks ==

    let maxAlpha = uniforms.posGradThresh;

    let screenMin = 1.0 - OFF_CANVAS_TRHESHOLD; // (1.0 - 1.5) = -0.5
    let screenMax = OFF_CANVAS_TRHESHOLD;

    let offScreen =
        any(splatP.pos < vec2f(screenMin)) ||
        any(splatP.pos > vec2f(screenMax));

    let lowAlpha = splatP.alpha < maxAlpha;

    let removeFlag = offScreen || lowAlpha;

    // == Gradient check ==

    let posGrad = gradient.pos / n;
    let scaleGrad = gradient.scale / n;

    let refineFlag =
        any(abs(posGrad) >= vec2f(uniforms.posGradThresh));

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

    if(removeFlag) {
        return; 
    }

    let currOff = atomicLoad(&groupOffset) + sums[local_id] - select(1u, 2u, refineFlag);


    if(refineFlag) {
        // densification!
        var splat2 = copySplat(splatP);
        if(any(abs(scaleGrad) >= vec2f(uniforms.scaleGradThresh))) {
            // overconstruction
            splatP.pos -= posGrad * 0.5;
            splat2.pos += posGrad * 0.5;
            splatP.scale /= SCALE_DIVIDE_CONST;
            splat2.scale /= SCALE_DIVIDE_CONST;

        }
        else {
            // underconstruction
            splatP.pos -= posGrad * 1;
            splat2.pos += posGrad * 1;
            splat2.color *= 2;
        }
        splats[currOff + 1u] = splat2;
    }

    splats[currOff] = splatP;

}