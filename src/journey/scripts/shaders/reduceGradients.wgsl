
@group(0) @binding(5) var<storage, read_write> gradientChunks : array<array<Grad, nGauss>>; 

@compute @workgroup_size(nGauss, 1, 1)
fn cs(@builtin(local_invocation_id) local_invocation_id: vec3u) {
  let j = local_invocation_id.x;
  var sum = Grad(vec2f(0), vec2f(0), 0, vec3f(0), 0);
  let numChunks = arrayLength(&gradientChunks);
  for (var i = 0u; i < numChunks; i++) {
    sum.pos    += gradientChunks[i][j].pos;
    sum.scale  += gradientChunks[i][j].scale;
    sum.rot    += gradientChunks[i][j].rot;
    sum.color  += gradientChunks[i][j].color;
    sum.alpha  += gradientChunks[i][j].alpha;
  }
  gradientChunks[0][j] = sum;
}