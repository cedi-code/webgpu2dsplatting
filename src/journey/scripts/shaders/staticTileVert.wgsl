@vertex fn vs(
@builtin(vertex_index) vi : u32
    ) -> @builtin(position) vec4f {
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
    return vec4f(pos[vi], 0.0, 1.0);
}