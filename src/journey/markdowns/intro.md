# How was this made?
in the following articles I will document and explain my journey step by step for how I made the img2splat in webgpu.
The art of sideprojects in the current AI-landscape is a bit overrated imo, so I will explicitly not use any AI-tools (except for some frontend styling and large recfactors).

The goal here is not to get a perfect splatter, but for me to more deeply understanding on how WebGPU and Gaussian Splatting works.

I just followed the description of the original [3DGS paper](https://arxiv.org/pdf/2308.04079) without looking at how they implemented it.

### Steps making img2splat possible in webgpu

 - [ ] [Render a 2d gaussian](journey-primitive.html) — *chapter 4*

 - [ ] [Optimization Step](journey-optimization.html) — *chapter 5.1*
 - [ ] [Parallelize the Optimization](journey-parallel.html) - *chapter 6*
 - [ ] [The densification and refinment step](journey-densification) — *chapter 5.2*

**poc done**
- [ ] Experiment with depth
- [ ] Performance Profiling (finding more bottelnecks)
**finish up**

###### *markdown rendering is powerd by [marked](https://www.npmjs.com/package/marked), [shiki](https://www.npmjs.com/package/shiki) and [Katex](https://www.npmjs.com/package/katex)*