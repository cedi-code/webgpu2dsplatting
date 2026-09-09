const tile = array(
    // triangle 1
    vec2f( -1.0,  1.0),  // left top
    vec2f( -1.0,  -1.0), // left bottom
    vec2f( 1.0,  -1.0),  // right bottom

    // triangle 2
    vec2f( -1.0,  1.0), // left top
    vec2f( 1.0,  -1.0), // right bottom
    vec2f( 1.0,  1.0),  // right top

);

struct vsOut {
    @builtin(position) p: vec4f,
    @location(0) texCoord: vec2f,
};

@vertex fn vs(
    @builtin(vertex_index) i : u32
) -> vsOut {

    return vsOut(
        vec4(tile[i], 0.0, 1.0),
        vec2(
            (tile[i] + vec2(1.0,-1.0)) * vec2(0.5, -0.5) // transform to uv space [-1,1] -> [0,1]
            ));
}