
struct Uniform {
    adamP : AdamParams,
};


@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(1) var<storage, read> gradients : array<AtomicGrad, nGauss>; 
@group(0) @binding(2) var<storage, read_write> adamMemory : AdamMemory;
@group(0) @binding(3) var<uniform> uniforms : Uniform;


@compute @workgroup_size(1,1,1) 
fn computeGD() {
    // assume gradients is done (but idk if i also assume that they are normalized?)

    // let n = f32(sampleDim * sampleDim);
    // for(var i = 0; i < nGauss; i++) {
        
    //     gradients[i].pos    /= n;
    //     gradients[i].scale  /= n;
    //     gradients[i].rot    /= n;
    //     gradients[i].color  /= n;
    //     gradients[i].alpha  /= n;

    // }

    // adam step
    adamMemory.t += 1u;
    var moment = adamMemory.m;
    var varian = adamMemory.v;
    var gradientsNonAtomic = array<Grad, nGauss>();

    for(var i = 0; i < nGauss; i++) {
        gradientsNonAtomic[i].pos = atomicLoad(&(gradients[i].pos));
        gradientsNonAtomic[i].scale = atomicLoad(&(gradients[i].scale));
        gradientsNonAtomic[i].rot = atomicLoad(&(gradients[i].rot));
        gradientsNonAtomic[i].color = atomicLoad(&(gradients[i].color));
        gradientsNonAtomic[i].alpha = atomicLoad(&(gradients[i].alpha));
    }

    adamStepGrad(uniforms.adamP, f32(adamMemory.t), &moment, &varian, &gradientsNonAtomic);
    adamMemory.m = moment;
    adamMemory.v = varian;

    for(var i = 0; i < nGauss; i++) {

        let initalParams = output[i];

        let newQ = initalParams.pos - gradientsNonAtomic[i].pos;
        let newS = initalParams.scale - gradientsNonAtomic[i].scale;
        let newR = initalParams.rot - gradientsNonAtomic[i].rot;
        let newC = initalParams.color - gradientsNonAtomic[i].color;
        let newA = initalParams.alpha - gradientsNonAtomic[i].alpha;

        output[i] = Params(newQ, newS, newR, newC, newA);
    }
}