struct Unfirom {
    adamP : AdamParams,
    activationFlag : i32,
    adamFlag : i32,
}

@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var goalTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output: GaussParams;
@group(0) @binding(3) var<storage, read_write> lossOutput : f32;
@group(0) @binding(4) var<uniform> uniforms : Unfirom;
@group(0) @binding(5) var<storage, read_write> adamMemory : MyAdamMemory;

// only for adam grad
const NUM_GAUSS = 1;
struct Grad {  
    pos: vec2f,
    scale : vec2f,
    rot : f32,
    color : vec3f,
    alpha: f32,
};
struct MyAdamMemory {
    t : u32,
    m : array<GradGauss, NUM_GAUSS>,
    v : array<GradGauss, NUM_GAUSS>,
}


const xRAY = false;

fn Loss(gColor: vec4f, imgC: vec4f) -> f32 {
    return (
        (gColor.r - imgC.r) * (gColor.r - imgC.r) + 
        (gColor.g - imgC.g) * (gColor.g - imgC.g) + 
        (gColor.b - imgC.b) * (gColor.b - imgC.b)
    ) / 3.0;;
}

fn Luminance(color : vec3f) -> f32 {
    return (color.r + color.g + color.b) / 3.0;
}

fn GradLoss(
    x: vec2f, 
    grad : ptr<function, GradGauss>,
    color: vec3f,
) {
    let gauss = g(output, x);
    let dist = (x - output.pos);

    var gradSample = EvalGradGauss(output, x);
    if (uniforms.activationFlag <= 0) {
        // technically wrong but proofs the point
        gradSample.scale = -1.0 * gauss * dist * dist * output.scale; 
    }


    let diff = 2.0 * Luminance((vec3f(gauss) - color));

    (*grad).pos += gradSample.pos * diff;
    (*grad).scale += gradSample.scale * diff;
    (*grad).rot += gradSample.rot * diff;
}
@compute @workgroup_size(1) fn computeGD() {

    let size = vec2u(128, 128); // static choosen size
    let n = f32(size.y * size.x);
    let sizeSample = vec2f(size);
    
    const nGauss = 1;

    var gradients : GradGauss = GradGauss();

    // loss + backwards pass
    var currLoss = 0.0;
    var gradsLoss = vec3f(0.0);
    for (var y = 0u; y < size.y; y++) {
        for (var x = 0u; x < size.x; x++) {
            let uv = vec2f(vec2u(x, y)) / sizeSample;
            let color = textureSampleLevel(goalTexture, ourSampler, uv, 0.0);
            
            let gColor = vec4f(vec3f(g(output, uv)), 1.0);
            currLoss += Loss(gColor, color);

            // params
            GradLoss(uv, &gradients, color.rgb);                     
        }
    }
    currLoss /= n;
    lossOutput = currLoss;

    gradients.pos    /= n;
    gradients.scale  /= n;
    gradients.rot    /= n;
    let initalParams = output;

    var newQ = vec2f(0.0);
    var newS = vec2f(0.0);
    var newR = 0.0;

    if(uniforms.adamFlag > 0) {
        // adam step
        adamMemory.t += 1u;
        var moment = adamMemory.m;
        var varian = adamMemory.v;
        var gradientAdam = array<GradGauss, NUM_GAUSS>(gradients);
        // ugly hardcoded adam memory holder        
        adamStepGradGauss(uniforms.adamP, f32(adamMemory.t), &moment, &varian, &gradientAdam);
        adamMemory.m = moment;
        adamMemory.v = varian;

        newQ = initalParams.pos - gradientAdam[0].pos;
        newS = initalParams.scale - gradientAdam[0].scale;
        newR = initalParams.rot; // - gradientAdam[0].rot;

    }
    else {
        newQ = initalParams.pos - 0.01 * gradients.pos;
        newS = initalParams.scale - 0.5 * gradients.scale;
        newR = initalParams.rot - 0.01 * gradients.rot;
    }

    output = GaussParams(newQ, newS, newR);
}