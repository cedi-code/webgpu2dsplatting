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
    numProperties : f32,
    stepH : f32,
    epsilon: f32,
};

@group(0) @binding(0) var<storage, read> inputGauss: array<GaussParams>;
@group(0) @binding(1) var<storage, read> inputSamples: array<PosGradSample>;
@group(0) @binding(2) var<uniform> u : Uniform;
@group(0) @binding(3) var<storage, read_write> output: array<GradResult>;


fn numericalDiff(f1 : vec2f, f0 : f32) -> vec2f {
    return (f1 - vec2f(f0)) / u.stepH;
}

fn calcTruncationError(m1 : vec2f, m2 : vec2f) -> vec2f {

    // h / 2 M1 + 2eps / h * M2
    return ((u.stepH / 2.0) * m1) + ((2.0 * u.epsilon / u.stepH) * m2);
}


fn calcScaleM1(p : GaussParams, x : vec2f) -> vec2f {

    // this is a rough approx
    const c = 10.0;
    let s2 = exp(4.0 * p.scale);
    let dist = (x - p.pos);
    let dist4 = dist * dist * dist * dist;

    // sloppy approx, could be better
    return abs(c * s2 * g(p,x) * dist4);
}


fn calcPosM1(p : GaussParams, x : vec2f) -> vec2f {

    // this is a rough approx
    const c = 10.0;
    let s4 = exp(4.0 * p.scale);
    let dist2 = (x - p.pos) * (x - p.pos);

    // sloppy approx, could be better
    return abs(c * s4 * g(p,x) * dist2);
}

fn calcRotM1(p : GaussParams, x : vec2f) -> f32 {

    // this is a pure vibes approx, not good
    const c = 10.0;
    let dist = (x - p.pos);
    let distSomething = dist.x * dist.x * dist.y * dist.y;

    let scaleDiff = (exp(2.0 * p.scale.x) - exp(2.0 * p.scale.y)) * (exp(2.0 * p.scale.x) - exp(2.0 * p.scale.y)) ;

    return abs(c * g(p,x) * distSomething * scaleDiff);
}

fn calcM2(g0 : f32, g1x : f32, g1y : f32) -> vec2f {
    // could also be false, but not by much
    var gMax = max(g1x, g1y);
    return vec2f(max(g0, gMax));
}


@compute @workgroup_size(1) fn computeGD() {

    for (var y = 0u; y < u32(u.numSamples * u.numProperties); y+=3u) {

        let p = inputGauss[0];
        let x = inputSamples[y % u32(u.numProperties)].sample;

        let g0 = g(p,x);
        let hx = vec2f(u.stepH, 0.0);
        let hy = vec2f(0.0, u.stepH);

        let g1xPos      = g(GaussParams(p.pos + hx, p.scale, p.rot), x);
        let g1yPos      = g(GaussParams(p.pos + hy, p.scale, p.rot), x);
        let g1Pos = vec2f(g1xPos, g1yPos);

        let g1xScale    = g(GaussParams(p.pos, p.scale + hx, p.rot), x);
        let g1yScale    = g(GaussParams(p.pos, p.scale + hy, p.rot), x);
        let g1Scale = vec2f(g1xScale, g1yScale);

        let g1xRot       = g(GaussParams(p.pos, p.scale, p.rot + u.stepH), x);
        let g1Rot = vec2f(g1xRot, 0.0);

        output[y].expected = numericalDiff(g1Pos, g0);        
        output[y].result = EvalGradGauss(p, x).pos;
        output[y].truncationError = calcTruncationError(calcPosM1(p,x), calcM2(g0, g1xPos, g1yPos)); 

        output[y+1].expected = numericalDiff(g1Scale, g0);        
        output[y+1].result = EvalGradGauss(p, x).scale;
        output[y+1].truncationError = calcTruncationError(calcScaleM1(p,x), calcM2(g0, g1xScale, g1yScale));
        
        // hack, just ignore the y axis (waste data but i dont care)
        output[y+2].expected = numericalDiff(g1Rot, g0);        
        output[y+2].result = vec2f(EvalGradGauss(p, x).rot);
        output[y+2].truncationError = calcTruncationError(vec2f(calcRotM1(p,x),0.0), calcM2(g0, g1xRot, 0.0)); 
    }


}