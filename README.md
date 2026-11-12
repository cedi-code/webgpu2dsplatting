<div align="center">
  <img src="assets/titleTextCropped.gif" alt="Image to Gaussian Splat" />
</div>

# WebGPU Image to Gaussian Splat

From scratch interactive Image to Gaussian Splat for WebGPU, following the same techniques of the
original [3D Gaussian Splatting paper](https://arxiv.org/pdf/2308.04079)


## How was this made?

The following articles document the step-by-step journey of making img2splat in WebGPU.

### Steps making img2splat possible in webgpu

 - [ ] [Render a 2d gaussian](https://cedi-code.github.io/webgpu-img2splat/journey-primitive.html) — *chapter 4*

 - [ ] [Optimization Step](https://cedi-code.github.io/webgpu-img2splat/journey-optimization.html) — *chapter 5.1*
 - [ ] [Parallelize the Optimization](https://cedi-code.github.io/webgpu-img2splat/journey-parallel.html) - *chapter 6*
 - [ ] [The densification and refinment step](https://cedi-code.github.io/webgpu-img2splat/journey-densification.html) — *chapter 5.2*

**poc done**
- [ ] Experiment with depth
- [ ] Performance Profiling (finding more bottelnecks)
**finish up**

## Technical stuff

- **WebGPU optimization:** Render and optimize 2D Gaussian splats with WGSL
  render and compute shaders, including parallel forward, backward, and
  gradient-reduction passes.
- **Parallel radix sort:** Organize splat/chunk pairs on the GPU with prefix
  sums, parallel radix sorting, and chunk offsets so each image tile processes
  only the splats that overlap it.
- **Adam updates:** Apply Adam-based parameter updates to splat position,
  color, scale, rotation, and opacity during image optimization.
- **Splat densification:** Refine scenes by duplicating under-reconstructed
  splats, splitting over-sized splats, and pruning splats that are no longer
  useful.
- **Depth and checks:** Experiment with depth-aware rendering and validate
  shader gradients against numerical gradient checks.

> SSIM is discussed as a possible future loss in the journey, but it is not
> implemented in the current renderer.

## Folder structure

```text
.
├── assets/                  # blog and example images
├── src/
│   ├── journey/             # Technical implementation and articles
│   │   ├── markdowns/       # blog explanations
│   │   └── scripts/         # blog+main scripts
│   │       └── shaders/     # blog+main WGSL shaders
│   ├── config/              # Renderer configuration
│   ├── myutils/             # Shared utility helpers
│   └── tests/               # Shader and gradient tests
├── index.html
├── package.json
└── vite.config.ts
```

## How to run

### Requirements

- Node.js and npm
- WebGPU browser

```bash
npm install
npm run dev
```

```bash
npm run build
npm run preview
```
