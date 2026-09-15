struct PosGradSample {
    sample : vec2f
};

struct GradResult {
    expected : vec2f,
    result: vec2f,
    truncationError : vec2f,

};

struct Uniform {
    numSamples: f32,
    stepH : f32,
    epsilon: f32,
};

@group(0) @binding(0) var<storage, read> inputGauss: array<GaussParams>;
@group(0) @binding(1) var<storage, read> inputSamples: array<PosGradSample>;
@group(0) @binding(2) var<uniform> u : Uniform;
@group(0) @binding(3) var<storage, read_write> output: array<GradResult>;


fn numericalDiffGauss(gauss : GaussParams, x : vec2f) -> vec2f {

    var p = GaussParams(gauss.pos, gauss.scale, gauss.rot);

    let g0 = g(p,x);

    p.pos = p.pos + vec2f(u.stepH, 0.0);
    let g1x = g(p,x);

    p.pos += vec2f(-u.stepH, u.stepH);
    let g1y = g(p,x);


    return vec2f(
        (g1x - g0) / (u.stepH),
        (g1y - g0) / (u.stepH)
    );
}

fn upperBoundM1(p : GaussParams, x : vec2f) -> vec2f {

    // this is a rough approx
    const c = 10.0;
    let s4 = exp(4.0 * p.scale);
    let dist2 = (x - p.pos) * (x - p.pos);

    // sloppy approx, could be better
    return abs(c * s4 * g(p,x) * dist2);
}

fn upperBoundM2(gauss : GaussParams, x : vec2f) -> vec2f {

    var p = GaussParams(gauss.pos, gauss.scale, gauss.rot);

    // could also be false, but not by much
    var gMax = g(p,x);
    p.pos += vec2f(u.stepH, 0.0);
    gMax = max(g(p,x), gMax);

    p.pos += vec2f(-u.stepH, u.stepH);
    return vec2f(max(g(p,x), gMax));
}

fn calcTruncationError(p : GaussParams, x : vec2f) -> vec2f {

    let m1 = upperBoundM1(p, x);
    let m2 = upperBoundM2(p, x);

    // h / 2 M1 + 2eps / h * M2
    return ((u.stepH / 2.0) * m1) + ((2.0 * u.epsilon / u.stepH) * m2);
}

@compute @workgroup_size(1) fn computeGD() {

    for (var y = 0u; y < u32(u.numSamples); y++) {

        let gauss = inputGauss[0];
        let x = inputSamples[y].sample;
        output[y].expected = numericalDiffGauss(gauss, x);        
        output[y].result = EvalGradGauss(gauss, x).pos;
        output[y].truncationError = calcTruncationError(gauss, x);        

    }

}