## Drawing a Gaussian
### Definition
Our primitive in 2dgs is a elliptical Gaussian. It has two properties
* $\mu$ the mean, which represents the position / center
* $\Sigma$ the variance, which represents the shape

So at a position $x$ the value of the Gaussian can be evaluated as follows:
$$G(x) = \frac{1}{2\pi|\Sigma|^{\frac{1}{2}}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$$

### Drawing plan
Rendering this on the screen works quite diffrently than using our familiar approach with triangles since a $G(x)$ is sampled during the fragmentation step:
![classic](../../../assets/graphics3d_pipe.webp)
We have to sample $G(x)$ for the whole screen and therefore we do the following "trick":

* **1. Vertex Shader:**
  * `DRAW:` 2 triangles forming a tile that covers the whole screen
```text
[-1, 1] --------- [1, 1]
   |  \             |
   |    \     T2    |
   |      \         |
   |  T1    \       |
   |          \     |
[-1,-1] -------- [1,-1]
```
* **2. Rasterizer:**
  * `IN:`   Primitive boundaries
  * `WORK:` Interpolate position $x$
  * `OUT:`  Fragments covering the whole screen

* **3. Fragment Shader:**
  * `IN:`   $x$ + properties ($ \mu_i , \Sigma_i$)
  * `EVAL:` $G(x)$
  * `OUT:`  Image of the 2d Gaussian

###### *sidenote: since we are working with 2d gaussians we dont have todo any projection step*

### Shaders
Ok, lets implement! Our vertex shader is now very simple and just renders a single static tile (`staticTileVert.wgsl`):
```wgsl
const tile = array(
    // triangle 1
    vec2f( -1.0,  1.0),  // left top
    vec2f( -1.0,  -1.0), // left bottom
    vec2f( 1.0,  -1.0),  // right bottom

    // triangle 2
    vec2f( -1.0,  1.0), // left top
    vec2f( 1.0,  -1.0), // right bottom
    vec2f( 1.0,  1.0),  // right top

);

@vertex fn vs(
    @builtin(vertex_index) i : u32
) -> @builtin(position) vec4f {

    return vec4f(tile[i], 0.0, 1.0);
}
```

Now for sampeling $G(x)$ we can remove normalization term which made the integral of Gaussian sum up to one.
removing it we get the following:
$$G'(x) = e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$$

$G(x)$ gives us a color in `frag` where we can expect the following:

white when $x$ is near $\mu$
$$|x-\mu| \approx 0 \implies G'(x) \approx 1$$
black pixels when $x$ is far from $\mu$

$$|x-\mu| >> 0 \implies G'(x) \approx 0$$

how to we get **$x$**?
given that our `vs` drew a tile that covers the whole screen, `@builtin(position)` gives us the position of the current pixel of our canvas element, which we can use as our sampling variable **$x$**

So here is a implementation of $G'(x)$ in `simpleGaussFrag.wgsl`:
```wgsl
@fragment fn fs(
    @builtin(position) p : vec4f
    ) -> @location(0) vec4f {
        
    const mean : vec2f = canvas.xy * 0.5; // center of screen
    const sigmaInv : mat2x2f = [20.0, 0.0, 0.0, 20.0]; // diagonal matrix

    let d = p.xy - mean;    
    let D2 = dot(d * sigmaInv, d); 
    let gauss = exp(-0.5 * D2);

    return vec4f(gauss, gauss, gauss, 1.0);
}
```


For the script bellow I also added a uniform buffer to make it interactable, but now we can render a GS primitive! :)

* [x] ~~WebGPU setup (rendering simple trinalge)~~
* [x] ~~Drawing a Gaussian~~