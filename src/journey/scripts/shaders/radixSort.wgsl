struct RadixMemory {
    currI : atomic<u32>,
    data : array<u32, MAX_SLOTS>,
    offsets : array<u32, numChunksW * numChunksH>
};

@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

// maybe workgroup memory? as buckets? no clue yet

@compute @workgroup_size(256, 1, 1)
fn radixSort() {

    // write the number in base 2^8 = 256

    // count how many elements there are per bucket

    // do prefix sum to know the offsets of the bucket

    // store data into the result buffer using count bucket
}