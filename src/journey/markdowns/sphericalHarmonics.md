## Spherical Harmonics?
Idea of  Spherical Harmonics (SH) is to define a function $f(x)$ which input lies on a unit sphere using a certain set of basis functions which are of polynomial form.

SH function are written as follows:

$$
Y^{m}_{l}(x) \quad \quad l \text{: degree} \quad m \text{: order}
$$

where $m \in [-l,l]$ and $l$ describes the degree of the polynomial.

Since a function basis set is infinite, what makes SH so special is that:
1. they are polynomial functions (so easy to compute)
2. great for approximations since the bigger $l$ the more higher freq. we capture

$$
f(x) \approx \sum_{l=0}^{N}\sum_{m=-l}^{l}{a_{l,m}Y^{m}_{l}}(x)
$$

so to approximate $f(x)$ we just need to find coefficents $a_{l,m}$, which is done by doing a function dot product between the function $f$ and a sh function $Y^{m}_{l}$:

$$
\langle f, Y^{m}_{l} \rangle = \int_{\Omega} f(x)Y^{m}_{l}(x) \,dx = a_{l,m}
$$

for 3dgs getting the color of a splat is view angle dependent, so it makes sense to have a function that takes in view angle $f(\theta, \phi) \rightarrow \R³$ and outputs a color.

$$
 \text{3dgs, the color of the splat is view dependant } (\theta, \phi)
$$

### 2dgs case
so how can we apply this for 2dgs? there is a problem, and a easy solution:

$$
 \text{ we dont have a viewing angle } \implies \text{color function is constant}
$$

so sadly we are not involving ourselfs with SH, but if you are intressted, i highly recommend the blog by gpfault: [Introduction to Spherical Harmonics for Graphics Programmers](https://gpfault.net/posts/sph.html) 


### Alpha
there is one parameter that is still missing in our splat struct, which is the $\alpha$ value of the splat.
```wgsl
    struct Splat { 
        @location(0) position: vec2f, 
        @location(1) scale: vec2f, 
        @location(2) color: vec3f, 
        @location(3) rotation: f32, 
        @location(4) alpha: f32, // [!code ++] 
    };
```
why not just make `color : vec4` instead of adding a new param?
doing this would not fix the alignment + in the optimization step we optimize for $\alpha$ seperatly, but its just cosmetic.

update the buffer aswell as passing it as a interstage variable:
```typescript
vBuilder.add(0, "position", "float32x2")  
        .add(1, "scale", "float32x2") 
        .add(2, "color", "float32x3") 
        .add(3, "rotation", "float32") 
        .add(4, "alpha", "float32");   // [!code ++] 
```
and `gaussTileVert.wgsl`:
```wgsl
    return SimpleVertexShaderOutput(
        vec4f(posGauss, 1.0 - f32(j+1) / n, 1.0),
        vec2f(tile[i]),
        vec4f(splat.color, splat.alpha),  // [!code ++] 
    );
```
and multiply it in the `gaussFrag.wgsl`:
```wgsl
@fragment fn fs(
    fsIn : SimpleVertexShaderOutput
) -> @location(0) vec4f {

    let gauss = exp(-6.0 * dot(fsIn.grid , fsIn.grid));
    let c = fsIn.color; // [!code ++] 
    let alpha : f32 = gauss * c.a; // [!code ++] 

    return vec4f(vec3f(1.0)*G, G); // [!code --] 
    return vec4f(vec3f(c.rgb) * alpha, gauss * alpha); // [!code ++]
}
```
and that it, we managed to create a splat primitive in webgpu :))

* [X] Render a 2d gaussian — *chapter 4*
    * [x] ~~WebGPU setup (rendering simple trinalge)~~
    * [x] ~~Drawing a Gaussian~~
    * [X] ~~Transforming Gaussians~~
    * [X] ~~spherical harmonics??~~

next we look into the optimization step, how to make our splat adapt to our input image.