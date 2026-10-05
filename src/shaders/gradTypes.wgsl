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



