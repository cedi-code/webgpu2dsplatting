
struct Uniform {
    adamP : AdamParams,
};


@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(1) var<storage, read_write> gradients: array<array<Grad, nGauss>>; 
@group(0) @binding(2) var<storage, read_write> adamMemory : AdamMemory;
@group(0) @binding(3) var<uniform> uniforms : Uniform;

@compute @workgroup_size(nGauss, 1, 1)
fn cs(@builtin(local_invocation_id) local_invocation_id: vec3u) {

    let n = f32(sampleDim * sampleDim);
    
    // this is a issue / will not work! todo needs to be updated outside of the shader
    let i = local_invocation_id.x;

    if(i == 0) {
        adamMemory.t += 1u;
    }
    workgroupBarrier();
        
    var moment = adamMemory.m[i];
    var varian = adamMemory.v[i];

    var gradient = gradients[0][i];

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