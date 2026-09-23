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

fn adamStepGradGauss(
    p : AdamParams, 
    t : f32,
    momentum : ptr<function, array<GradGauss, NUM_GAUSS>>,
    variance : ptr<function, array<GradGauss, NUM_GAUSS>>,
    grad   : ptr<function, array<GradGauss, NUM_GAUSS>>, // by copy for now
    ) 
{
    let alpha_t = p.lr * sqrt((1.0 - pow(p.b2, t)))/(1.0-pow(p.b1, t));

    for(var i = 0; i < NUM_GAUSS; i++) {

        let mPos = &((*momentum)[i].pos);
        let vPos = &((*variance)[i].pos);
        let gPos =  &(*grad)[i].pos;
        adamStepVec2(p, mPos, vPos, gPos, alpha_t);

        let mSca = &(*momentum)[i].scale;
        let vSca = &(*variance)[i].scale;
        let gSca =  &(*grad)[i].scale;
        adamStepVec2(p, mSca, vSca, gSca, alpha_t);

        let mRot = &(*momentum)[i].rot;
        let vRot = &(*variance)[i].rot;
        let gRot =  &(*grad)[i].rot;
        adamStepScalar(p,mRot, vRot, gRot, alpha_t);
    }   
}

fn adamStepGrad(
    p : AdamParams, 
    t : f32,
    momentum : ptr<function, array<Grad, NUM_GAUSS>>,
    variance : ptr<function, array<Grad, NUM_GAUSS>>,
    grad   : ptr<function, array<Grad, NUM_GAUSS>>, // by copy for now
    ) 
{
    let alpha_t = p.lr * sqrt((1.0 - pow(p.b2, t)))/(1.0-pow(p.b1, t));

    for(var i = 0; i < NUM_GAUSS; i++) {

        let mPos = &((*momentum)[i].pos);
        let vPos = &((*variance)[i].pos);
        let gPos =  &(*grad)[i].pos;
        adamStepVec2(p, mPos, vPos, gPos, alpha_t);

        let mSca = &(*momentum)[i].scale;
        let vSca = &(*variance)[i].scale;
        let gSca =  &(*grad)[i].scale;
        adamStepVec2(p, mSca, vSca, gSca, alpha_t);

        let mRot = &(*momentum)[i].rot;
        let vRot = &(*variance)[i].rot;
        let gRot =  &(*grad)[i].rot;
        adamStepScalar(p,mRot, vRot, gRot, alpha_t);

        let mCol = &(*momentum)[i].color;
        let vCol = &(*variance)[i].color;
        let gCol =  &(*grad)[i].color;
        adamStepVec3(p,mCol, vCol, gCol, alpha_t);

        let mAlp = &(*momentum)[i].alpha;
        let vAlp = &(*variance)[i].alpha;
        let gAlp =  &(*grad)[i].alpha;
        adamStepScalar(p,mAlp, vAlp, gAlp, alpha_t);

    }

    
}


fn adamStepScalar(
    p : AdamParams, 
    m : ptr<function, f32>,
    v : ptr<function, f32>,
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
    m : ptr<function, vec2f>,
    v : ptr<function, vec2f>,
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
    m : ptr<function, vec3f>,
    v : ptr<function, vec3f>,
    grad  : ptr<function, vec3f>,
    alpha_t : f32
) {
    let gi = *grad;
    (*m) = p.b1 * (*m) + (1.0 - p.b1) * gi;     // raw first momentum estimate
    (*v) = p.b2 * (*v) + (1.0 - p.b2) * gi * gi;// raw second momentum estimate
    (*grad) = alpha_t * (*m) / (sqrt((*v)) + p.eps);
}