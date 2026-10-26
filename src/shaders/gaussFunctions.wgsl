
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

fn sigmoidInv(y : f32, off : f32) -> f32 {
    return off - log(max(1.0e-5,1.0 - y)) + log(max(1.0e-5,y));
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

fn eigenValues(varMat : mat2x2f) -> vec2f {
    // varMat is symetric
    // first eigen value is det(A - \lambda * I) = 0
    // A = ( a, c // c, b )
    // det(A - lambda * I) = (a - \lambda) * (b - lambda) - c^2 = 0
    // (ab - (a + b)\lambda - lambda^2 - c^2 = 0
    // lambda^2 - (a + b)\lambda + (ab - c^2) = 0
    // mitternachts formel:
    // lambda_1,2 = 0.5 * ((a+b) +- sqrt((a+b)^2 - 4*(ab - c^2)))
    // we want the bigger eigen value and sqrt > 0 so we take +?
    // lambda_1 = 0.5 * (a+b + sqrt((a-b)^2 + 4c^2))
    
    let a = varMat[0].x;
    let b = varMat[1].y;
    let c = varMat[0].y;

    let D = max(0.0, (a-b)*(a-b) + 4.0 * c * c);
    let Dsqrt = sqrt(D);
    let lambda1 = 0.5 * ((a + b) + Dsqrt);
    let lambda2 = 0.5 * ((a + b) - Dsqrt);

    return vec2f(lambda1, lambda2);
}

fn firstEigenVec(varMat : mat2x2f, lambda1 : f32, lambda2 : f32) -> vec2f {

    // v1 = [ (\lambda_1 - b), c ]

    let a = varMat[0].x;
    let b = varMat[1].y;
    let c = varMat[0].y;

    let v1 = vec2f(lambda1 - b , c);
    let v2 = vec2f(c , lambda1 - a);

    let v1Len2 = dot(v1, v1);

    let v3 = select(v1, v2, v1Len2 < 1.0e-10);
    
    let v3Len2 = dot(v3, v3);

    let scale = sqrt(max(0.0, lambda1));

    return normalize(select(v3, vec2f(1,0), v3Len2 < 1e-10)) * scale;
}
