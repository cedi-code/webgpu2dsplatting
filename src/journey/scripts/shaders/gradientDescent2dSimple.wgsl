@group(0) @binding(0) var ourSampler: sampler;
@group(0) @binding(1) var goalTexture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> output: GaussParams;
@group(0) @binding(3) var<storage, read_write> lossOutput : f32;


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

    let gradSample = EvalGradGauss(output, x);

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

    let newQ = initalParams.pos - 0.01 * gradients.pos;
    let newS = initalParams.scale - 0.5 * gradients.scale;
    let newR = initalParams.rot - 0.01 * gradients.rot;

    output = GaussParams(newQ, newS, newR);
}