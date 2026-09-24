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


fn rotMat(r: f32) -> mat2x2f {
    return mat2x2f(
        cos(r), sin(r), // column 0
        -sin(r), cos(r) // column 1
    ); 
}

fn dSigmoid(x : f32, off: f32) -> f32 {
    return exp(-x + off) / ((1.0 + exp(-x + off))*(1.0 + exp(-x + off)));
}

fn sigmoid(x : f32, off : f32) -> f32 {
    return 1.0 / (1.0 + exp(-x + off));
}

fn vecSigmoid(x : vec3f) -> vec3f {
    return vec3f(
        sigmoid(x.r, 4.0),
        sigmoid(x.g, 4.0),
        sigmoid(x.b, 4.0),
    );
}

fn g(p : GaussParams, x : vec2f) -> f32 {

    let d = x - p.pos;
    let a = exp(p.scale) * (transpose(rotMat(p.rot)) * d);
    let D2 = dot(a, a);

    return exp(-0.5 * D2);
}


fn EvalGradGauss(p : GaussParams, x: vec2f) -> GradGauss {

    let gauss = g(p,x);
    let dist = (x - p.pos);

    //  v = R^T*(x-q)
    let R = rotMat(p.rot);
    let v = transpose(R)*dist;
    let sDiag = mat2x2f(exp(2.0 * p.scale).x, 0.0, 0.0, exp(2.0 * p.scale).y);

    // \Sigma = R * S^2 * R^T

    // -0.5 g * \Sigma  * -2 * (x-q)
    let gradQ = gauss * (R * sDiag * v);

    // -0.5 g * 2 * v^t * s * v 
    let gradS = -1.0 * gauss * v * v * exp(2.0 * p.scale);
    
    //d/d\theta (x-u)^T * RSS^TR^T * (x-u) <=>  v^T * (L*S2 - S2*L) * v => dR/d\theta = L*R  => 
    //  -0.5 g * 2*v_x*v_y*(s_x^2 - s_y^2)
    let gradR = -1.0 * gauss * v.x * v.y * (exp(2.0 * p.scale.x) - exp(2.0 * p.scale.y));

    return GradGauss(gradQ, gradS, gradR);
}