struct RadixMemory {
    splatOffsets : array<u32, NUM_DISPATCHES>,
    data : array<u32, MAX_SLOTS>,
    data2 : array<u32, MAX_SLOTS>,
    offsets : array<u32, numChunksW * numChunksH>
};

@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

// maybe workgroup memory? as buckets? no clue yet
// atomic counts?
var<workgroup> countsAtomic: array<atomic<u32>, GROUPSIZE>;


@compute @workgroup_size(GROUPSIZE, 1, 1)
fn radixSort(
    @builtin(local_invocation_id) local_invocation_id : vec3u,
) {
    let local_id = local_invocation_id.x;
    // write the number in base 2^8 = 256

    let dataSize = radixMem.splatOffsets[NUM_DISPATCHES-1] << 1;
    let chunk_size_d = (dataSize + GROUPSIZE - 1u) / GROUPSIZE;
    let chunk_offset_d = local_id * chunk_size_d;

    // count how many elements there are per bucket
    let chunkMax = min(chunk_offset_d + chunk_size_d, dataSize);
    for(var i = chunk_offset_d; i < chunkMax; i++) {
        atomicAdd(&countsAtomic[radixMem.data[i] % GROUPSIZE], 1u);
    }

    workgroupBarrier();
    
    // do prefix sum over counts to know the offsets of the bucket
    // -> same as in setup lol
    for(var i = 0u; i < BARRIERS; i++) {
        let chunk_size = (2u << i);
        let chunk_id = local_id % chunk_size;
        let half_chunk = chunk_size >> 1;
        if(chunk_id >= half_chunk) {
            let chunk_start = (local_id - chunk_id) ;
            let chunk_read_id = chunk_start + half_chunk - 1u;
            // can be done without atomics but it gets a bit dumb
            atomicAdd(&countsAtomic[local_id], atomicLoad(&countsAtomic[chunk_read_id]));
        }
        workgroupBarrier();
    }

    // store data into the result buffer using count bucket (ping pong buffer)
    for(var i = chunkMax - 1u; i > chunk_offset_d; i--) {
        let newI = atomicSub(&countsAtomic[radixMem.data[i] % GROUPSIZE], 1u) - 1u;
        radixMem.data2[newI * 2u] = radixMem.data[i * 2u];
        radixMem.data2[newI * 2u + 1u] = radixMem.data[i * 2u + 1u];
    }
    
}