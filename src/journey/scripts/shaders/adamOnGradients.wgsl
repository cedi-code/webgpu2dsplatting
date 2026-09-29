// constants
const NUM_GAUSS = 2;
const xRAY = false;
const nGauss = 2;
const sampleDim = 128;


struct Uniform {
    adamP : AdamParams,
};

struct Params {
    pos : vec2f,
    scale : vec2f,
    rot : f32,
    color : vec3f,
    alpha: f32,
};

struct Grad {  
    pos: vec2f,
    scale : vec2f,
    rot : f32,
    color : vec3f,
    alpha: f32,
};

@group(0) @binding(0) var<storage, read_write> output: array<Params>;
@group(0) @binding(1) var<storage, read> gradients : array<Grad, nGauss>; 
@group(0) @binding(1) var<storage, read_write> adamMemory : AdamMemory;
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

    adamStepGrad(uniforms.adamP, f32(adamMemory.t), &moment, &varian, &gradients);
    adamMemory.m = moment;
    adamMemory.v = varian;

    for(var i = 0; i < nGauss; i++) {

        let initalParams = output[i];

        let newQ = initalParams.pos - gradients[i].pos;
        let newS = initalParams.scale - gradients[i].scale;
        let newR = initalParams.rot - gradients[i].rot;
        let newC = initalParams.color - gradients[i].color;
        let newA = initalParams.alpha - gradients[i].alpha;

        output[i] = Params(newQ, newS, newR, newC, newA);
    }
}