struct RadixMemory {
    splatOffsets : array<u32, NUM_DISPATCHES>,
    offsets : array<u32, numChunksW * numChunksH>,
    data2 : array<u32, MAX_SLOTS>,
    data : array<u32, MAX_SLOTS>,
};

struct Uniforms {
  stride : u32,
};

@group(0) @binding(3) var<storage, read_write> lossOutput : array<array<f32, nSteps>>;
@group(0) @binding(5) var<storage, read_write> gradientChunks : array<array<Grad, MAX_SPLATS_CHUNK>>; 
@group(0) @binding(6) var<storage, read_write> adamMemory : AdamMemory;
@group(0) @binding(8) var<storage, read_write> radixMem : RadixMemory;

@group(1) @binding(0) var<uniform> uni : Uniforms;

var<workgroup> splats0: array<u32, MAX_SPLATS_CHUNK>;
var<workgroup> splats1: array<u32, MAX_SPLATS_CHUNK>;

@compute @workgroup_size(MAX_SPLATS_CHUNK, 1, 1)
fn cs(
  @builtin(local_invocation_id) local_invocation_id: vec3u,
  @builtin(workgroup_id) workgroup_id : vec3u
  ) {
  let j = local_invocation_id.x;
  let i = workgroup_id.x;

  let chunk0 = i * uni.stride * 2;
  let chunk1 = chunk0 + uni.stride;


  let offData0 = radixMem.offsets[chunk0];
  let prevOff0 = select(0u, radixMem.offsets[chunk0-1u], chunk0 > 0u);
  let numSplats0 = min((offData0 - prevOff0), MAX_SPLATS_CHUNK);


  let offData1 = radixMem.offsets[chunk1];
  let prevOff1 = select(0u, radixMem.offsets[chunk1-1u], chunk1 > 0u);
  let numSplats1 = min((offData1 - prevOff1), MAX_SPLATS_CHUNK);

  if(j < numSplats0) {
    let currI0 = offData0 + j; 
    splats0[j] = radixMem.data2[2u * currI0 + 1u];
  }
  // no race condition possible since we are reading from a diffrent tile / chunk
  if(j < numSplats1) {
    let currI1 = offData1 + j; 
    splats1[j] = radixMem.data2[2u * currI1 + 1u];
  }
  workgroupBarrier();

  // splats0 and splats1 are already sorted!

  // wait but i need to sum up all the possible splats!
  // so gradientChunks needs to be much much bigger / contain every splat!
  // so its not enough to just have MAX_SPLATS_CHUNK as a work size! oh oh!
  
  // do another radix sort to get the old order back! haha! lmao :cry:

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