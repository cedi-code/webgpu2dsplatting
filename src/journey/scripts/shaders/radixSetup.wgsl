const tile = array<vec2f, 4>(
    // triangle 1
    vec2f( -1.0,  1.0),  // left top
    vec2f( -1.0,  -1.0), // left bottom
    vec2f( 1.0,  -1.0),  // right bottom
    vec2f( 1.0,  1.0),   // right top
);

// preferably power of 2!
const NUM_256 = (NUM_DISPATCHES + GROUPSIZE - 1u) / GROUPSIZE;

fn ndcToBCords(pos : vec2f) -> vec2u {
    // let posNorm = (pos * vec2f(2.0, -2.0) - vec2f(1.0, -1.0)); 
    return vec2u(
        clamp(u32(pos.x * f32(numChunksW)), 0u, numChunksW-1u),
        clamp(u32(pos.y * f32(numChunksH)), 0u, numChunksH-1u)
    );
}

struct SplatInfo {
    allocate: u32,
    bounds: vec4<u32>,
}

fn getSplatInfo(global_id: u32) -> SplatInfo {
    if (global_id >= NUM_SPLATS) {
        return SplatInfo(0u, vec4(0)); // out-of-bounds threads return 0
    }

    let splat = splats[global_id];

    // get the left, right, top, bottom most points of gaussian
    let R : mat2x2f = rotMat(splat.rot);
    let s : vec2f = TILE_SCALE * exp(-splat.scale); 
    let t : vec2f = splat.pos; 

    let lt = ndcToBCords(R * (s * tile[0]) + t);  // left top
    let lb = ndcToBCords(R * (s * tile[1]) + t);  // left bottom
    let rb = ndcToBCords(R * (s * tile[2]) + t);  // right bottom
    let rt = ndcToBCords(R * (s * tile[3]) + t);  // right top

    let minX = min(min(lt.x, lb.x), min(rt.x, rb.x));
    let maxX = max(max(lt.x, lb.x), max(rt.x, rb.x));
    let minY = min(min(lt.y, rt.y), min(lb.y, rb.y));
    let maxY = max(max(lt.y, rt.y), max(lb.y, rb.y));

    // grab the current index and increase it by bbox
    let allocate = RADIX_ELEMS * (maxX - minX + 1) * (maxY - minY + 1);

    return SplatInfo(allocate, vec4u(minX, maxX, minY, maxY));
}


@group(0) @binding(0) var<storage, read_write> splats : array<Params, NUM_SPLATS>;
@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

override first_pass: bool;
var<workgroup> sums: array<u32, GROUPSIZE>;
var<workgroup> groupOffset : atomic<u32>;

// NUM_SPLATS / 256 invocations
@compute @workgroup_size(GROUPSIZE, 1, 1)
fn registerSplats(
        @builtin(global_invocation_id) global_invocation_id : vec3u,
        @builtin(local_invocation_id) local_invocation_id : vec3u,
        @builtin(workgroup_id) workgroup_id: vec3u,
) {
    
    let local_id = local_invocation_id.x;
    let splatInfo = getSplatInfo(global_invocation_id.x);

    // store allocate into  workgroup memory and wait
    sums[local_id] = splatInfo.allocate;
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

    // then we write into radix sort memory
    var currOff = atomicLoad(&groupOffset) + sums[local_id] - splatInfo.allocate;

    if (global_invocation_id.x >= NUM_SPLATS) { // no writing when thread is not splat anymore!
        return;
    }
    radixMem.splatOffsetsTotal[global_invocation_id.x] = currOff;

    // write down the key value pairs
    let minX = splatInfo.bounds[0];
    let maxX = splatInfo.bounds[1];
    let minY = splatInfo.bounds[2];
    let maxY = splatInfo.bounds[3];

    for(var j = minY; j < maxY + 1; j++) {
        for(var k = minX; k < maxX + 1; k++) {
            let boxI = (j * numChunksW) + k;
            // box id
            radixMem.data[currOff + R_CHUNK_INDEX] = boxI;
            // splat id
            radixMem.data[currOff + R_SPLAT_INDEX] = global_invocation_id.x;

            radixMem.data[currOff + R_TRANSPOSE_INDEX] = 0u; // we dont know the other order yet!
            currOff += RADIX_ELEMS;
        }
    }
}