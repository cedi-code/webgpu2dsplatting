struct RadixMemory {
    splatOffsets : array<u32, NUM_DISPATCHES>,
    offsets : array<u32, NUM_CHUNKS>,
    data2 : array<u32, MAX_SLOTS>,
    data : array<u32, MAX_SLOTS>,
    splatOffsetsTotal: array<u32, NUM_SPLATS>,
};