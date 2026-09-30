// https://toji.dev/webgpu-best-practices/compute-vertex-data#synchronizing-data-access-with-atomics
const QUANTIZE_FACTOR: f32 = 32768.0;
const DEQUANTIZE_FACTOR: f32 = 1.0 / 32768.0;


struct AtomicGrad {  
    pos: array<atomic<i32>, 2>,
    scale: array<atomic<i32>, 2>,
    rot: atomic<i32>,
    color: array<atomic<i32>, 3>,
    alpha: atomic<i32>,
};

fn atomicAddVec2f(ptr_val: ptr<storage, array<atomic<i32>, 2>, read_write>, value: vec2f) {
    let q = vec2i(value * QUANTIZE_FACTOR);
    atomicAdd(&((*ptr_val)[0]), q.x);
    atomicAdd(&((*ptr_val)[1]), q.y);
}

fn atomicAddVec3f(ptr_val: ptr<storage, array<atomic<i32>, 3>, read_write>, value: vec3f) {
    let q = vec3i(value * QUANTIZE_FACTOR);
    atomicAdd(&((*ptr_val)[0]), q.x);
    atomicAdd(&((*ptr_val)[1]), q.y);
    atomicAdd(&((*ptr_val)[2]), q.z);
}

fn atomicAddf32(ptr_val: ptr<storage, atomic<i32>, read_write>, value: f32) {
    let q = i32(value * QUANTIZE_FACTOR);
    atomicAdd(ptr_val, q);
}

fn loadf32(p: ptr<storage, atomic<i32>, read_write>) -> f32 {
    return f32(atomicLoad(p)) * DEQUANTIZE_FACTOR;
}

fn loadVec2f(p: ptr<storage, array<atomic<i32>, 2>, read_write>) -> vec2f {
    let x = f32(atomicLoad(&((*p)[0])));
    let y = f32(atomicLoad(&((*p)[1])));
    return vec2f(x, y) * DEQUANTIZE_FACTOR;
}

fn loadVec3f(p: ptr<storage, array<atomic<i32>, 3>, read_write>) -> vec3f {
    let x = f32(atomicLoad(&((*p)[0])));
    let y = f32(atomicLoad(&((*p)[1])));
    let z = f32(atomicLoad(&((*p)[2])));
    return vec3f(x, y, z) * DEQUANTIZE_FACTOR;
}