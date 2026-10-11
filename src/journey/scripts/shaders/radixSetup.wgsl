const tile = array<vec2f, 6>(
    // triangle 1
    vec2f( -1.0,  1.0),  // left top
    vec2f( -1.0,  -1.0), // left bottom
    vec2f( 1.0,  -1.0),  // right bottom
    vec2f( 1.0,  1.0),   // right top
);

const numBoxesW = u32(( (sampleDim) + chunkWidth - 1) / chunkWidth); // (sampleDim is suqare, and + chunkWidht-1, rounds it up)
const numBoxesH = u32(( (sampleDim) + chunkHeight - 1) / chunkHeight); // (sampleDim is suqare, and + chunkWidht-1, rounds it up)

struct RadixMemory {
    currI : atomic<u32>,
    data : array<u32, MAX_SLOTS>,
};

@group(0) @binding(0) var<storage, read_write> splats : array<Params, NUM_SPLATS>;
@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

fn ndcToBCords(pos : vec2f) -> vec2u {
    let posNorm = (pos * vec2f(2.0, -2.0) - vec2f(1.0, -1.0)); 
    return vec2u(
        clamp(u32(posNorm.x * f32(numBoxesW)), 0u, numBoxesW-1u),
        clamp(u32(posNorm.y * f32(numBoxesH)), 0u, numBoxesH-1u)
    );
}

// NUM_SPLATS % 256 invocations
@workgroup(256,1,1)
fn registerSplats(
        @builtin(global_invocation_id) global_invocation_id : vec3u,
) {

    let splat = splats[global_invocation_id.x];

    // get the left, right, top, bottom most points of gaussian
    let R : mat2x2f = rotMat(splat.rot);
    let s : vec2f = exp(-splat.scale); 
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
    let allocate = 2u * (minX - maxX + 1) * (minY - maxY + 1);
    var currOff = atomicAdd(&radixMem.currI, allocate);

    // write down the key value pairs
    for(var j = minY; j < maxY+1; j++) {
        for(var k = minX; l < maxX+1; k++) {
            let boxI = (j * numBoxesW) + k;

            // splat id
            radixMem.data[currOff] = global_invocation_id.x; 
            // box id
            radixMem.data[currOff + 1] = boxI;
            currOff += 2;
        }
    }
}