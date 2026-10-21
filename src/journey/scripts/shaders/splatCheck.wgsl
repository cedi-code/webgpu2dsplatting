@group(0) @binding(0) var<storage, read_write> splats: array<Params>;
@group(1) @binding(0) var<storage, read_write> splatsPingPong: array<Params>;
@group(0) @binding(5) var<storage, read_write> gradients: array<Grad, MAX_SLOTS>; 
@group(0) @binding(7) var<uniform> uniforms : Uniform;
@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

override first_pass: bool;
var<workgroup> groupOffset : atomic<u32>;
var<workgroup> sums: array<u32, GROUPSIZE>;

const NUM_256 = (NUM_DISPATCHES + GROUPSIZE - 1u) / GROUPSIZE;


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

    let maxScreenOff = uniforms.posGradThresh;
    let maxAlpha = uniforms.posGradThresh;

    let screenMin = 1.0 - maxScreenOff;
    let screenMax = maxScreenOff;

    let offScreen =
        any(splatP.pos < vec2f(screenMin)) ||
        any(splatP.pos > vec2f(screenMax));

    let lowAlpha = splatP.alpha < maxAlpha;

    let removeFlag = offScreen || lowAlpha;

    // == Gradient check ==

    let posGrad = gradient.pos / n;

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

    if(removeFlag) {
        return; 
    }

    let currOff = atomicLoad(&groupOffset) + sums[local_id] - select(1u, 2u, refineFlag);

    if(refineFlag) {
        // refine i will do that 
    }

    splats[currOff] = splatP;

}