
const NUM_GAUSS = 2;

struct AdamParams {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
};

// todo, adam-memory should NOT EXIST when done correctly / should be allocated with adamInit
struct AdamMemory {
    t : u32,
    m : array<Grad, NUM_GAUSS>,
    v : array<Grad, NUM_GAUSS>,
}

fn adamStepGrad(
    p : AdamParams, 
    mem : ptr<storage, AdamMemory, read_write>, 
    grad   : ptr<function, array<Grad, NUM_GAUSS>>, // by copy for now
    ) 
{
    (*mem).t += 1u;
    let alpha_t = p.lr * sqrt((1.0 - pow(p.b2, f32((*mem).t))))/(1-p.b1);

    for(var i = 0; i < NUM_GAUSS; i++) {

        let mPos = &((*mem).m[i].pos);
        let vPos = &((*mem).v[i].pos);
        let gPos =  &(*grad)[i].pos;
        adamStepVec2(p, mPos, vPos, gPos, alpha_t);

        let mSca = &(*mem).m[i].scale;
        let vSca = &(*mem).v[i].scale;
        let gSca =  &(*grad)[i].scale;
        adamStepVec2(p, mSca, vSca, gSca, alpha_t);

        let mRot = &(*mem).m[i].rot;
        let vRot = &(*mem).v[i].rot;
        let gRot =  &(*grad)[i].rot;
        adamStepScalar(p,mRot, vRot, gRot, alpha_t);

        let mCol = &(*mem).m[i].color;
        let vCol = &(*mem).v[i].color;
        let gCol =  &(*grad)[i].color;
        adamStepVec3(p,mCol, vCol, gCol, alpha_t);

        let mAlp = &(*mem).m[i].alpha;
        let vAlp = &(*mem).v[i].alpha;
        let gAlp =  &(*grad)[i].alpha;
        adamStepScalar(p,mAlp, vAlp, gAlp, alpha_t);

    }

    
}


fn adamStepScalar(
    p : AdamParams, 
    m : ptr<storage, f32, read_write>,
    v : ptr<storage, f32, read_write>,
    grad  : ptr<function, f32>,
    alpha_t : f32
) {
    let gi = *grad;
    (*m) = p.b1 * (*m) + (1.0 - p.b1) * gi;     // raw first momentum estimate
    (*v) = p.b2 * (*v) + (1.0 - p.b2) * gi * gi;// raw second momentum estimate
    (*grad) = alpha_t * (*m) / (sqrt((*v)) + p.eps);
}


fn adamStepVec2(
    p : AdamParams, 
    m : ptr<storage, vec2f, read_write>,
    v : ptr<storage, vec2f, read_write>,
    grad  : ptr<function, vec2f>,
    alpha_t : f32
) {
    let gi = *grad;
    (*m) = p.b1 * (*m) + (1.0 - p.b1) * gi;     // raw first momentum estimate
    (*v) = p.b2 * (*v) + (1.0 - p.b2) * gi * gi;// raw second momentum estimate
    (*grad) = alpha_t * (*m) / (sqrt((*v)) + p.eps);
}

fn adamStepVec3(
    p : AdamParams, 
    m : ptr<storage, vec3f, read_write>,
    v : ptr<storage, vec3f, read_write>,
    grad  : ptr<function, vec3f>,
    alpha_t : f32
) {
    let gi = *grad;
    (*m) = p.b1 * (*m) + (1.0 - p.b1) * gi;     // raw first momentum estimate
    (*v) = p.b2 * (*v) + (1.0 - p.b2) * gi * gi;// raw second momentum estimate
    (*grad) = alpha_t * (*m) / (sqrt((*v)) + p.eps);
}