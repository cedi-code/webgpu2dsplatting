@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

// pass numbers should be index in reverse
override pass_nr: u32;

const GROUPSIZE_NR_BITS = u32(log2(GROUPSIZE));
const GROUPSIZE_BITS = GROUPSIZE - 1u; // should make binary number all 11111...


// max workgroup memory is 16384
const GROUPSIZE_RADIX = 16;
var<workgroup> groupBuckets: array<array<u32, GROUPSIZE>, GROUPSIZE_RADIX>;

// RADIX_ELEMS means how many elements there are in a single entry of radixMem. for example 3, (splatId, rectangleId, oldId)

// === context ===
// there are 16 groups of the same 256 (GROUPSIZE ) buckets, this is stored in workgroup memory groupBuckets.
// groupBuckets[group][bucket] group = i, bucket = j
// each bucket represents a chunk-id
// 1. we count how many chunk-id per bucket we found foreach subgroup from the 16 groups.
// 
// 2. reduce on the group axis, so group 16 (GROUPSIZE_RADIX-1) has the real bucket size for chunk-j
// example in group 16:
// bucket 0 has found 30 splats for chunk 0
// bucket 1 has found 10 splats for chunk 1
// 
// 3. now we do the prefix sum over all final buckets (so we get the offsets) (this is only for group 16)
// example group 16:
// bucket 0 has offset 30 (starts at 0 ends at 30)
// bucket 1 has offset 40 (starts at 30 ends at 40)

// 3.2 add the final offset of the final bucket size (from group 16) and adds it to the relative offset within the group

// 4. 16 threads rearrange the data array using the offsets

fn readRadixData(index : u32) -> u32 {
    if(pass_nr % 2 == 0) {
        return radixMem.data[index];
    }
    else {
        return radixMem.data2[index];
    }
}

fn swapRadixData(oldI : u32, newI : u32) {
    if(pass_nr % 2 == 0) {
        radixMem.data2[newI * RADIX_ELEMS + R_CHUNK_INDEX]     = radixMem.data[oldI * RADIX_ELEMS + R_CHUNK_INDEX];
        radixMem.data2[newI * RADIX_ELEMS + R_SPLAT_INDEX]     = radixMem.data[oldI * RADIX_ELEMS + R_SPLAT_INDEX];
        radixMem.data2[newI * RADIX_ELEMS + R_TRANSPOSE_INDEX] = radixMem.data[oldI * RADIX_ELEMS + R_TRANSPOSE_INDEX];
    }
    else {
        radixMem.data[newI * RADIX_ELEMS + R_CHUNK_INDEX]     = radixMem.data2[oldI * RADIX_ELEMS + R_CHUNK_INDEX];
        radixMem.data[newI * RADIX_ELEMS + R_SPLAT_INDEX]     = radixMem.data2[oldI * RADIX_ELEMS + R_SPLAT_INDEX];
        radixMem.data[newI * RADIX_ELEMS + R_TRANSPOSE_INDEX] = radixMem.data2[oldI * RADIX_ELEMS + R_TRANSPOSE_INDEX];
    }

    // if its the first pass, we store the idx (whic his acutally the original order where its orderd by splat_id)
    if(pass_nr == 0) { 
        radixMem.data2[newI * RADIX_ELEMS + R_TRANSPOSE_INDEX] = oldI;
    }
}

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

    // when chunks > GROUPSIZE, we shift our chunk id by GROUPSIZE bits
    let shift_amount = pass_nr * GROUPSIZE_NR_BITS;

    // 1. fill up your local buckets (thread)
    if(local_id < GROUPSIZE_RADIX) {
        for(var i = chunk_offset_d; i < maxChunk; i++) {
            let data_id = i * RADIX_ELEMS + R_CHUNK_INDEX;
            let bucket_id = (readRadixData(data_id) >> shift_amount) & GROUPSIZE_BITS;
            groupBuckets[local_id][bucket_id]++;
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

    // 3.2 correct the other buckets 
    if(local_id > 0u) {
        for(var i = 0u; i < GROUPSIZE_RADIX-1u; i++) {
            groupBuckets[i][local_id] += groupBuckets[GROUPSIZE_RADIX-1u][local_id-1u];
        }
    }

    workgroupBarrier();
    
    // 4. writing buckets i for each thread in data2 (can do better? only 16 threads are doing that!)
    if(local_id < GROUPSIZE_RADIX) {
        for(var i = maxChunk; i > chunk_offset_d; i--) {
            let idx = i - 1u;

            let bucket = (readRadixData(idx * RADIX_ELEMS + R_CHUNK_INDEX) >> shift_amount) & GROUPSIZE_BITS;
            let newI = &groupBuckets[local_id][bucket]; // first subtract then read
            *newI = *newI - 1u;
            let newIv = *newI;

            swapRadixData(idx, newIv);
        }
    }

    
}