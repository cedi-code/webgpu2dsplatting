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

struct AtomicGrad {  
    pos: array<atomic<i32>, 2>,
    scale: array<atomic<i32>, 2>,
    rot: atomic<i32>,
    color: array<atomic<i32>, 3>,
    alpha: atomic<i32>,
};

struct GradGauss {
    pos : vec2f,
    scale : vec2f,
    rot : f32
};

struct GaussParams {
    pos : vec2f,
    scale : vec2f,
    rot : f32,
};

struct AdamParams {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
};

struct Uniform {
    adamP: AdamParams,
    alphaCutOff : f32,
    posGradThresh : f32,
    scaleGradThresh : f32,
    eigenVAspectThresh : f32,
    eigenVAspectThreshMax : f32,
    lossWeight : f32, // 0 means L2 only, 1 means L1 only
};