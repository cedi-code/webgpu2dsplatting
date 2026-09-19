struct Params {
    pos : vec2f,
    scale : vec2f,
    rot : f32,
    color : vec3f,
    alpha: f32,
};

struct SimpleVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) grid: vec2f,
    @location(1) color: vec4f,
};

const tile = array<vec2f, 6>(
    // triangle 1
    vec2f( -1.0,  1.0),  // left top
    vec2f( -1.0,  -1.0), // left bottom
    vec2f( 1.0,  -1.0),  // right bottom

    // triangle 2
    vec2f( -1.0,  1.0), // left top
    vec2f( 1.0,  -1.0), // right bottom
    vec2f( 1.0,  1.0),  // right top
);

@group(0) @binding(0) var<storage, read> output: array<Params>;

@vertex fn vs(
    @builtin(vertex_index) i : u32,
    @builtin(instance_index) j : u32,
) -> SimpleVertexShaderOutput {

    let splat = output[j];
    let R : mat2x2f = rotMat(splat.rot);
    // scale is inverse!
    let s : vec2f = 4.0*exp(-splat.scale);
    // convert [0,1] -> [-1, 1] for x
    let t : vec2f = (splat.pos * vec2f(2.0, -2.0)) - vec2f(1.0, -1.0); 
    
    let posGauss : vec2f = transpose(R) * (s * tile[i]) + t;

    let col : vec3f = vecSigmoid(splat.color);
    let alpha : f32 = sigmoid(splat.alpha, 4.0);

    let gaussZ : f32 = 1.0 - f32(j+1) / 1000.0;

    return SimpleVertexShaderOutput(
        vec4f(posGauss, gaussZ, 1.0),
        vec2f(tile[i]),
        vec4f(col, alpha),
    );
}