

# How was this made?
in the following articles I will document and explain my journey step by step for how I made a gaussian splatting pipeline from scratch in webgpu.

The goal here is not to get a perfect splatter, but for me (and you) to more deeply understanding on how WebGPU and Gaussian Splatting works.

For this blog we just follow the description of the original [3DGS paper](https://arxiv.org/pdf/2308.04079) without looking at how they implemented it. Some basic knowledge of shaders, typescript and calculus 2 is recommended but not really needed. 

### Live Demo
live demo can be found [here](index.html) (WebGPU must be enabled in your browser).

### Steps making img2splat possible in webgpu

 - [ ] [Render a 2d gaussian](journey-primitive.html) — *chapter 4*

 - [ ] [Optimization Step](journey-optimization.html) — *chapter 5.1*
 - [ ] [Parallelize the Optimization](journey-parallel.html) - *chapter 6*
 - [ ] [The densification and refinement step](journey-densification) — *chapter 5.2*

**poc done**
- [ ] Experiment with depth
- [ ] Performance Profiling (finding more bottlenecks)
**finish up**

### AI Usage
The art of sideprojects and making things from scratch in the current AI-landscape is a bit overrated imo.
Around 80-90% of the code, all of the math derivations, and all of the text in the blog are written by me (cedi).

The idea here was not to create a product ASAP, but to really get familiar with a topic and get lost, stuck and get my hands dirty :)

I used AI tools for :
- Frontend
- refactors aswell as templating for atomic operations
- discussion/advice: I often used it as an advisor when in doubt of my approaches, code is still "hand written". 

Copilots code is marked in the git commit history.


###### *markdown rendering is powerd by [marked](https://www.npmjs.com/package/marked), [shiki](https://www.npmjs.com/package/shiki) and [Katex](https://www.npmjs.com/package/katex)*