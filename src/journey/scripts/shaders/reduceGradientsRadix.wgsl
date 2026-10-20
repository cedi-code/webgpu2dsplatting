@group(0) @binding(3) var<storage, read_write> lossOutput : array<array<f32, nSteps>>;
@group(0) @binding(5) var<storage, read_write> gradientStorage : array<Grad, MAX_SLOTS>; 
@group(0) @binding(6) var<storage, read_write> adamMemory : AdamMemory;
@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

var<workgroup> gradSums: array<Grad, GROUPSIZE>;

fn addGradStorToWork(work_i : u32, storage_j : u32) {
    gradSums[work_i].pos   += gradientStorage[storage_j].pos;
    gradSums[work_i].scale += gradientStorage[storage_j].scale;
    gradSums[work_i].rot   += gradientStorage[storage_j].rot;
    gradSums[work_i].color += gradientStorage[storage_j].color;
    gradSums[work_i].alpha += gradientStorage[storage_j].alpha; 
}

fn addGradWorkToWork(work_i : u32, work_j : u32) {
    gradSums[work_i].pos   += gradSums[work_j].pos;
    gradSums[work_i].scale += gradSums[work_j].scale;
    gradSums[work_i].rot   += gradSums[work_j].rot;
    gradSums[work_i].color += gradSums[work_j].color;
    gradSums[work_i].alpha += gradSums[work_j].alpha; 
}

@compute @workgroup_size(GROUPSIZE, 1, 1)
fn cs(
  @builtin(local_invocation_id) local_invocation_id: vec3u,
  @builtin(workgroup_id) workgroup_id : vec3u
  ) {

    let local_id = local_invocation_id.x;
    let splat_id = workgroup_id.x;

    let totalBufferSize =  radixMem.splatOffsets[NUM_DISPATCHES-1u] / RADIX_ELEMS;

    let offset = radixMem.splatOffsetsTotal[splat_id] / RADIX_ELEMS;
    var nextOffset = totalBufferSize;
    if(splat_id < NUM_SPLATS-1u) {
        nextOffset = radixMem.splatOffsetsTotal[splat_id+1] / RADIX_ELEMS;
    }    
    let splatStorageSize = nextOffset - offset;

    let chunkSize = (splatStorageSize + GROUPSIZE - 1u) / GROUPSIZE;
    let threadOffset = offset + min(splatStorageSize, chunkSize * local_id);
    let nextThreadOffset = offset + min(splatStorageSize, chunkSize * (local_id+1u));
    
    for(var i = threadOffset; i < nextThreadOffset; i++) {
        // sum up the gradients
        addGradStorToWork(local_id, i);
    }

    
    workgroupBarrier();

    // do some sort of prefix sum but only need to update the last element? 

    // prefix sum with 8 barriers
    for(var i = 0u; i < BARRIERS; i++) {

        let chunk_size = (2u << i);
        let chunk_id = local_id % chunk_size;
        let half_chunk = chunk_size >> 1;
        let last_elem_chunk = chunk_size - 1u; 

        if(chunk_id >= last_elem_chunk) {

            let chunk_start = (local_id - chunk_id) ;
            let chunk_read_id = chunk_start + half_chunk - 1u;

            addGradWorkToWork(local_id, chunk_read_id);            
        }
        workgroupBarrier();
    }

    if(local_id > 0) {
        return;
    }
    gradientStorage[offset] = gradSums[GROUPSIZE-1u];
}