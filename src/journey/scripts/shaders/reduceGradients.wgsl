struct Uniforms {
  stride : u32,
};

@group(0) @binding(3) var<storage, read_write> lossOutput : array<array<f32, nSteps>>;
@group(0) @binding(5) var<storage, read_write> gradientChunks : array<array<Grad, nGauss>>; 
@group(0) @binding(6) var<storage, read_write> adamMemory : AdamMemory;

@group(1) @binding(0) var<uniform> uni : Uniforms;


@compute @workgroup_size(nGauss, 1, 1)
fn cs(
  @builtin(local_invocation_id) local_invocation_id: vec3u,
  @builtin(workgroup_id) workgroup_id : vec3u
  ) {
  let j = local_invocation_id.x;
  let i = workgroup_id.x;

  let chunk0 = i * uni.stride * 2;
  let chunk1 = chunk0 + uni.stride;

  gradientChunks[chunk0][j].pos   += gradientChunks[chunk1][j].pos;
  gradientChunks[chunk0][j].scale += gradientChunks[chunk1][j].scale;
  gradientChunks[chunk0][j].rot   += gradientChunks[chunk1][j].rot;
  gradientChunks[chunk0][j].color += gradientChunks[chunk1][j].color;
  gradientChunks[chunk0][j].alpha += gradientChunks[chunk1][j].alpha; 

  if(j > 0) {
    return;
  }

  let lossI = adamMemory.t % nSteps;
  let numLossChu = arrayLength(&lossOutput);
  lossOutput[chunk0][lossI] += lossOutput[chunk1][lossI];
}