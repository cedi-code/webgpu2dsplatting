struct RadixMemory {
    splatOffsets : array<u32, NUM_DISPATCHES>,
    data : array<u32, MAX_SLOTS>,
    offsets : array<u32, numChunksW * numChunksH>
};

@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

// maybe workgroup memory? as buckets? no clue yet
// atomic counts?

@compute @workgroup_size(256, 1, 1)
fn radixSort() {

    // write the number in base 2^8 = 256

    // count how many elements there are per bucket
    // for ??
    // counts[data[i?] % 256]++ (atomic)

    // do prefix sum over counts to know the offsets of the bucket
    // -> same as in setup lol

    // store data into the result buffer using count bucket
    // -> seperate storage buffer? idk yet
}