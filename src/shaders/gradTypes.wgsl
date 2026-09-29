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
    pos: atomic<vec2f>,
    scale : atomic<vec2f>,
    rot : atomic<f32>,
    color : atomic<vec3f>,
    alpha: atomic<f32>,
};

struct GradGauss {
    pos : vec2f,
    scale : vec2f,
    rot : f32
}

struct GaussParams {
    pos : vec2f,
    scale : vec2f,
    rot : f32,
};