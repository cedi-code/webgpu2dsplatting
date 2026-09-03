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

@vertex fn vs(
    @builtin(vertex_index) i : u32
) -> @builtin(position) vec4f {

    return vec4f(tile[i], 0.0, 1.0);
}