@group(0) @binding(3) var<storage, read_write> lossOutput : array<array<f32, nSteps>>;
@group(0) @binding(6) var<storage, read_write> adamMemory : AdamMemory;


@compute @workgroup_size(256, 1, 1)
fn reduceLoss(@builtin(local_invocation_id) local_id: vec3u) {
    let lossI = adamMemory.t % nSteps;
    let numChunks = arrayLength(&lossOutput);
    
    var sum = 0.0;
    for(var i = local_id.x; i < numChunks; i += 256u) {
        sum += lossOutput[i][lossI];
    }
    
    shared_loss[local_id.x] = sum;
    workgroupBarrier();
    
    // todo prefix sums again

    if (local_id.x == 0u) {
        lossOutput[0][lossI] = shared_loss[0]; 
    }
}