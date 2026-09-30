
struct Uniform {
    adamP : AdamParams,
};


@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(1) var<storage, read_write> gradients : array<AtomicGrad, nGauss>; 
@group(0) @binding(2) var<storage, read_write> adamMemory : AdamMemory;
@group(0) @binding(3) var<uniform> uniforms : Uniform;

fn loadGrad(p: ptr<storage, AtomicGrad, read_write>) -> Grad {
    var g: Grad;
    g.pos   = loadVec2f(&((*p).pos));
    g.scale = loadVec2f(&((*p).scale));
    g.rot   = loadf32(&((*p).rot));
    g.color = loadVec3f(&((*p).color));
    g.alpha = loadf32(&((*p).alpha));
    return g;
}

@compute @workgroup_size(1,1,1) 
fn computeGD() {
    // assume gradients is done (but idk if i also assume that they are normalized?)

    let n = f32(sampleDim * sampleDim);
    // for(var i = 0; i < nGauss; i++) {
        
    //     gradients[i].pos    /= n;

    // }

    // adam step
    adamMemory.t += 1u;
    var moment = adamMemory.m;
    var varian = adamMemory.v;
    var gradientsNonAtomic = array<Grad, nGauss>();

    for(var i = 0; i < nGauss; i++) {
        gradientsNonAtomic[i] = loadGrad(&gradients[i]);
        gradientsNonAtomic[i].pos /= n;
        gradientsNonAtomic[i].scale /= n;
        gradientsNonAtomic[i].rot /= n;
        gradientsNonAtomic[i].color /= n;
        gradientsNonAtomic[i].alpha /= n;

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