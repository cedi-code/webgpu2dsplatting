@group(0) @binding(3) var<storage, read_write> lossOutput : array<array<f32, nSteps>, numChunksW * numChunksH>;
@group(0) @binding(6) var<storage, read_write> adamMemory : AdamMemory;

var<workgroup> shared_loss : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn reduceLoss(@builtin(local_invocation_id) local_id: vec3u) {
    let lossI = adamMemory.t % nSteps;
    let numChunks = u32(numChunksW * numChunksH);
    
    var sum = 0.0;
    for(var i = local_id.x; i < numChunks; i += 256u) {
        sum += lossOutput[i][lossI];
    }
    
    shared_loss[local_id.x] = sum;
    workgroupBarrier();

    let local = local_id.x;
    
    // prefix sums again
    for(var i = 0u; i < BARRIERS; i++) {
        let chunkS = 2u << i;
        let halfChunk = chunkS >> 1u;
        let readChunk = halfChunk - 1u;
        let chunkId = local % chunkS;
        if(chunkId >= halfChunk) {
            let read_value = shared_loss[local - chunkId + readChunk];
            shared_loss[local] += read_value; 
        }
        workgroupBarrier();
    }

    if (local == 0u) {
        lossOutput[0][lossI] = shared_loss[255] / 256; 
    }
}