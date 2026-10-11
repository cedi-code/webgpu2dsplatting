struct RadixMemory {
    currI : atomic<u32>,
    data : array<u32, MAX_SLOTS>,
    offsets : array<u32, numChunksW * numChunksH>
};

@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

// maybe workgroup memory? as buckets? no clue yet

@compute @workgroup_size(256, 1, 1)
fn radixSort() {

    // do special bucket sort!
}