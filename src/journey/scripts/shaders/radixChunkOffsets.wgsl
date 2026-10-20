@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

const numChunks = numChunksW * numChunksH;
 
fn readChunkId(index : u32) -> u32 {
    if(NUM_SORTS % 2 == 0) 
    {
        return radixMem.data[index * RADIX_ELEMS + R_CHUNK_INDEX];
    }
    else 
    {
        return radixMem.data2[index * RADIX_ELEMS + R_CHUNK_INDEX];
    }
}


fn binarySearch(chunk_id : u32, max_i : u32) -> u32 {

    var lo = 0u;
    var hi = max_i;

    while(lo < hi) {
        let mid = lo + ((hi - lo) / 2u);

        if(readChunkId(mid) < chunk_id) {
            lo = mid + 1u;
        }
        else {  
            hi = mid;
        }
    }
    return lo;
    
}
// is called #chunks-times

// this fills the radixMem.offsets of the chunks inside the data array
@compute @workgroup_size(GROUPSIZE, 1, 1)
fn radixSort(
    @builtin(global_invocation_id) global_invocation_id : vec3u,
) {
    let chunk_id = global_invocation_id.x;

    if(chunk_id >= numChunks) {
        return;
    }
    let dataSize = radixMem.splatOffsets[NUM_DISPATCHES-1u] / RADIX_ELEMS;

    radixMem.offsets[chunk_id] = binarySearch(chunk_id, dataSize);
}