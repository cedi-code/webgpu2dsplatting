struct RadixMemory {
    splatOffsets : array<u32, NUM_DISPATCHES>,
    offsets : array<u32, numChunksW * numChunksH>,
    data2 : array<u32, MAX_SLOTS>,
    data : array<u32, MAX_SLOTS>,
    splatOffsetsTotal: array<u32, NUM_SPLATS>,
};

@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

// max workgroup memory is 16384
const GROUPSIZE_RADIX = 16;
var<workgroup> groupBuckets: array<array<u32, GROUPSIZE>, GROUPSIZE_RADIX>;


@compute @workgroup_size(GROUPSIZE, 1, 1)
fn radixSort(
    @builtin(local_invocation_id) local_invocation_id : vec3u,
) {
    let local_id = local_invocation_id.x;
    // write the number in base 2^8 = 256

    let dataSize = radixMem.splatOffsets[NUM_DISPATCHES-1] / RADIX_ELEMS;
    let chunk_size_d = (dataSize + GROUPSIZE_RADIX - 1u) / GROUPSIZE_RADIX;
    let chunk_offset_d = local_id * chunk_size_d;

    let maxChunk = min(chunk_offset_d + chunk_size_d, dataSize);

    // 1. fill up your local buckets (thread)
    if(local_id < GROUPSIZE_RADIX) {
        for(var i = chunk_offset_d; i < maxChunk; i++) {
            groupBuckets[local_id][radixMem.data[i * RADIX_ELEMS + R_CHUNK_INDEX] % GROUPSIZE]++;
        }
    }

    workgroupBarrier();

    // 2. prefix sum all buckets of same number from different threads
    for(var i = 1u; i < GROUPSIZE_RADIX; i++) {
        groupBuckets[i][local_id] += groupBuckets[i-1u][local_id];
    }
    
    workgroupBarrier();

    // 3.1 do prefix sum on last thread elements
    // prefix sum with 8 barriers
    for(var i = 0u; i < BARRIERS; i++) {
    
        let chunk_size = (2u << i);
        let chunk_id = local_id % chunk_size;
        let half_chunk = chunk_size >> 1;
        
        if(chunk_id >= half_chunk) {
            let chunk_start = (local_id - chunk_id) ;
            let chunk_read_id = chunk_start + half_chunk - 1u;
            groupBuckets[GROUPSIZE_RADIX-1][local_id] += groupBuckets[GROUPSIZE_RADIX-1][chunk_read_id];
        }
        workgroupBarrier();
    } 

    // bit of a hacky way, and will stop working once chunks are >> 256
    if(local_id < numChunksW * numChunksH) {
        radixMem.offsets[local_id] = groupBuckets[GROUPSIZE_RADIX-1][local_id];
    }

    // 3.2 correct the other buckets 
    if(local_id > 0u) {
        for(var i = 0u; i < GROUPSIZE_RADIX-1u; i++) {
            groupBuckets[i][local_id] += groupBuckets[GROUPSIZE_RADIX-1u][local_id-1u];
        }
    }

    workgroupBarrier();
    
    // 4. writing buckets i for each thread in data2 (can do better? dont know)
    if(local_id < GROUPSIZE_RADIX) {
        for(var i = maxChunk; i > chunk_offset_d; i--) {
            let idx = i - 1u;
            let bucket = radixMem.data[idx * RADIX_ELEMS] % GROUPSIZE;
            let newI = &groupBuckets[local_id][bucket]; // first subtract then read
            *newI = *newI - 1u;
            let newIv = *newI;
            radixMem.data2[newIv * RADIX_ELEMS + R_CHUNK_INDEX]     = radixMem.data[idx * RADIX_ELEMS]; 
            radixMem.data2[newIv * RADIX_ELEMS + R_SPLAT_INDEX]     = radixMem.data[idx * RADIX_ELEMS + 1u]; 
            radixMem.data2[newIv * RADIX_ELEMS + R_TRANSPOSE_INDEX] = idx;
            // radixMem.data[idx * RADIX_ELEMS + 2u] = newIv; // not needed for now?
        }
    }
    
}