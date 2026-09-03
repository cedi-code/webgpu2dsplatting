
struct MyUniforms {
    scale : vec2f,
    rotation: f32,
    mean : vec2f,
}

struct SimpleVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) grid: vec2f,
};

@group(0) @binding(0) var<uniform> uniforms : MyUniforms;

@vertex fn vs(
@builtin(vertex_index) vi : u32
    ) -> SimpleVertexShaderOutput {
    let pos = array(
        // triangle 1
        vec2f( -1.0,  1.0),  // left top
        vec2f( -1.0,  -1.0), // left bottom
        vec2f( 1.0,  -1.0),  // right bottom

        // triangle 2
        vec2f( -1.0,  1.0), // left top
        vec2f( 1.0,  -1.0), // right bottom
        vec2f( 1.0,  1.0),  // right top

    );
    let r = uniforms.rotation;
    let rotMat : mat2x2f = mat2x2f(cos(r), sin(r), -sin(r), cos(r));
    let posGauss : vec2f = rotMat * (uniforms.scale * pos[vi]) + uniforms.mean;

    return SimpleVertexShaderOutput(
        vec4f(posGauss, 0.0, 1.0),
        vec2f(pos[vi])
    );
}