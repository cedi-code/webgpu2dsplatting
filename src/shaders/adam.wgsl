

const DIM_GRAD = 18;

struct AdamParams {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
};

struct AdamMemory {
    t : u32,
    m : array<f32, DIM_GRAD>,
    v : array<f32, DIM_GRAD>,
}

fn adamStep(
    p : AdamParams, 
    mem : ptr<storage, AdamMemory, read_write>, 
    g   : ptr<function, array<f32,DIM_GRAD>>, // by copy for now
    ) 
{
    (*mem).t += 1u;
    for(var i = 0u; i < DIM_GRAD; i++) {
        let gi = (*g)[i];
        (*mem).m[i] = p.b1 * (*mem).m[i] + (1.0 - p.b1) * gi;     // raw first momentum estimate
        (*mem).v[i] = p.b2 * (*mem).v[i] + (1.0 - p.b2) * gi * gi;// raw second momentum estimate
        let mHat = (*mem).m[i] / (1.0 - pow(p.b1,f32((*mem).t)));        // bias corrected first moment
        let vHat = (*mem).v[i] / (1.0 - pow(p.b2,f32((*mem).t)));        // bias corrected second moment
        (*g)[i] = p.lr * (mHat / (sqrt(vHat) + p.eps));           // step size
    }
}