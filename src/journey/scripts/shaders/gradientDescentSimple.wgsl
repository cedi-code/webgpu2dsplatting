@group(0) @binding(0) var<storage, read_write> dataOutput: array<f32>;
@group(0) @binding(1) var<storage, read> dataY: array<f32>;


fn f(x: f32, q : f32) -> f32 {
    return exp(-0.25*(x-q)*(x-q));
}

const N = 100;

fn Loss(q: f32) -> f32 {

    var sum = 0.0;

    for(var i = 0u; i < N; i++) {
        let x = (20.0*f32(i))/f32(N) - 10.0; 
        let y = dataY[i];

        let diff = (f(x,q) - y);
        sum += diff * diff;
    }
    sum /= f32(N);

    return sum;
}

fn gradL(q : f32) -> f32 {

    var sum = 0.0;

    for(var i = 0u; i < N; i++) {
        let x = (10.0*f32(i))/f32(N) - 5.0; 
        let y = dataY[i];

        let diff = (f(x,q) - y);
        sum += diff * f(x,q) * (x - q);
    }
    sum *= 4.0 / f32(N);

    return sum;
}

@compute @workgroup_size(1) fn computeGD() {
    var q = dataOutput[0]; // inital guess "hack"

    let maxSteps = u32(arrayLength(&dataOutput));
    for(var i = 0u; i < maxSteps; i++) {
        dataOutput[i] = Loss(q);

        let step = gradL(q);
        q -= 0.3 * step;

        if(step == 0.0) {
            break;
        }
    }
    dataOutput[0] = q;

}