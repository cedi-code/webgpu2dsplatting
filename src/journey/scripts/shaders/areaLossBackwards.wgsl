@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(3) var<storage, read_write> lossOutput: array<array<f32, nSteps>, NUM_CHUNKS>;
@group(0) @binding(5) var<storage, read_write> gradientStorage: array<Grad, MAX_SLOTS>;
@group(0) @binding(6) var<storage, read_write> adamMemory: AdamMemory;
@group(0) @binding(7) var<uniform> uniforms : Uniform;
@group(0) @binding(8) var<storage, read_write> radixMem: RadixMemory;
@group(0) @binding(9) var<storage, read_write> drawBuffer: array<u32>;

var<workgroup> areaLosses: array<f32, GROUPSIZE>;

@compute @workgroup_size(GROUPSIZE, 1, 1)
fn cs(
    @builtin(global_invocation_id) global_invocation_id: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let localId = local_invocation_id.x;
    let splatId = global_invocation_id.x;
    var areaLoss = 0.0;
    let n = f32(sampleDim * sampleDim);


    if (splatId < drawBuffer[1]) {
        let p = output[splatId];
        let rotMat = rotMat(p.rot);
        let scaleMat2 = mat2x2f(
            exp(-2.0 * p.scale.x), 0.0,
            0.0, exp(-2.0 * p.scale.y),
        );
        let cov = rotMat * scaleMat2 * transpose(rotMat);
        let eigenV = eigenValues(cov);
        areaLoss = sqrt(max(0.0, eigenV.x * eigenV.y));
        
        // d/d_s A = sqrt(lambda1 * lambda2) =? sqrt(det(\sigma)) = (rotation ignored) sqrt(e^(-2s_x + -2s_y)) = e^(s_x + s_y)
        let gradientArea = select(vec2f(0.0), -vec2f(areaLoss) * uniforms.areaLossWeight * n, areaLoss > uniforms.eigenVAspectThreshMax);
        let gradientOffset = radixMem.splatOffsetsTotal[splatId] / RADIX_ELEMS;

        gradientStorage[gradientOffset].scale += gradientArea;
    }

    areaLosses[localId] = 0.0;//areaLoss;
    workgroupBarrier();

    for (var i = 0u; i < BARRIERS; i++) {
        let stride = u32(GROUPSIZE >> (i + 1u));
        if (localId < stride) {
            areaLosses[localId] += areaLosses[localId + stride];
        }
        workgroupBarrier();
    }

    if (localId == 0u) {
        let lossIndex = adamMemory.t % nSteps;
        lossOutput[workgroup_id.x][lossIndex] += areaLosses[0];
    }
}
