struct SimpleVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@fragment fn fs(fsIn : SimpleVertexShaderOutput) -> @location(0) vec4f {
    return fsIn.color;
}