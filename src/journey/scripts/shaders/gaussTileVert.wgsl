struct Splat {
  @location(0) position: vec2f,
  @location(1) scale: vec2f,
  @location(2) color: vec3f,
  @location(3) rotation: f32, 
};

struct SimpleVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) grid: vec2f,
    @location(1) color: vec3f,
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

fn rotMat(r: f32) -> mat2x2f {
    return mat2x2f(
        cos(r), sin(r), // column 0
        -sin(r), cos(r) // column 1
    ); 
}

@vertex fn vs(
    @builtin(vertex_index) i : u32,
    @builtin(instance_index) j : u32,
    splat : Splat,
) -> SimpleVertexShaderOutput {

    let R : mat2x2f = rotMat(splat.rotation);
    let s : vec2f = splat.scale;
    let t : vec2f = splat.position;
    
    let posGauss : vec2f = R * (s * tile[i]) + t;

    return SimpleVertexShaderOutput(
        vec4f(posGauss, 1.0 - f32(j+1) / 100.0, 1.0),
        vec2f(tile[i]),
        splat.color,
    );
}