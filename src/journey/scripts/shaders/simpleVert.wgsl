struct SimpleVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

struct Vertex {
    @location(0) position: vec2f,
    @location(1) color : vec3f
}
@vertex fn vs(
    vert: Vertex,
) -> SimpleVertexShaderOutput {
    return SimpleVertexShaderOutput(
        vec4f(vert.position, 0.0, 1.0),
        vec4f(vert.color, 1.0)
    );
}