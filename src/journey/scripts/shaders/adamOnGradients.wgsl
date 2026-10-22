
@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(5) var<storage, read_write> gradients: array<Grad, MAX_SLOTS>; 
@group(0) @binding(6) var<storage, read_write> adamMemory : AdamMemory;
@group(0) @binding(7) var<uniform> uniforms : Uniform;
@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;
@group(0) @binding(9) var<storage, read_write> drawBuffer : array<u32>;


@compute @workgroup_size(GROUPSIZE, 1, 1)
fn cs(
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(global_invocation_id) global_invocation_id : vec3u,
    ) {

    let i = global_invocation_id.x;
    if(i == 0) {
        adamMemory.t += 1u;
    }
    workgroupBarrier();

    if(i >= drawBuffer[1]) { // drawBuffer index one is # instances
        return;
    }

    let n = f32(sampleDim * sampleDim);
    // a bit dumb but works

    var moment = adamMemory.m[i];
    var varian = adamMemory.v[i];

    let gradientOff = radixMem.splatOffsetsTotal[i] / RADIX_ELEMS;
    var gradient = gradients[gradientOff];

    gradient.pos    /= n;
    gradient.scale  /= n;
    gradient.rot    /= n;
    gradient.color  /= n;
    gradient.alpha  /= n;

    adamStepGrad(uniforms.adamP, f32(adamMemory.t), &moment, &varian, &(gradient));
    adamMemory.m[i] = moment;
    adamMemory.v[i] = varian;

    let initalParams = output[i];

    let newQ = initalParams.pos - gradient.pos;
    let newS = initalParams.scale - gradient.scale;
    let newR = initalParams.rot - gradient.rot;
    let newC = initalParams.color - gradient.color;
    let newA = initalParams.alpha - gradient.alpha;

    output[i] = Params(newQ, newS, newR, newC, newA);
}