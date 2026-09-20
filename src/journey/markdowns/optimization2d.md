## Gradient 2d (+ Texture loading)
Now that we can optimize $f(x)$ for $\mu$ in a 1D gaussian, lets do it in 2d!
Lets extend our definiton of $f(x)$ to $g(x): \R² \rightarrow \R $:

$$
g(x) = e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$

$\mu$ is now a 2d vector, and I added the the matrix $\Sigma$ which I will is parametized with $s$, keeping it simple for now:

$$
\mu = \begin{bmatrix}
\mu_x \\
\mu_y  
\end{bmatrix} 
\quad
s = \begin{bmatrix}
s_x \\
s_y  
\end{bmatrix} 
\quad
\Sigma^{-1}(s) =
\begin{bmatrix}
s_x^2 & 0 \\
0 & s_y^2 
\end{bmatrix} 
$$

Since will now optimize over $\mu$ and $s$ which are both 2d vectors, to make our code a bit more readable, I will create a struct that stores our parameters:

```wgsl
struct GaussParams {
    pos : vec2f,
    scale : vec2f,
};
``` 

And $g(x;\mu, s)$ as a function that takes in our struct:

```wgsl 
fn g(p : GaussParams, x : vec2f) -> f32 {

    let d = x - p.pos;
    let a = p.scale * d;
    let D2 = dot(a, a);

    return exp(-0.5 * D2);
}
``` 
Initalizing them $\mu_0$, $s_0$ randomly, we can us the update rule to approximate goal image $y^*$

$$
    \mu_{t+1} = \mu_t - \eta * \nabla_{\mu} L(\mu_{t}, s_t) 
    \\ \quad \\
    s_{t+1} = s_t - \eta * \nabla_s L(\mu_{t}, s_t)

$$

Lets change our gradient descent algorithm to now optimize over these two 2d vectors. I will store the inital $\mu_0$ and $s_0$ in a storage buffer `io` of type `GaussParams`, where we will also put in the resulting $\mu_N$, $s_N$ 

```wgsl
@group(0) @binding(0) var<storage, read_write> io: GaussParams;

@compute @workgroup_size(1) fn gradientDescent() {
    var q = io.pos; 
    var s = io.scale;
    loop {
        let step : GaussParams = gradL(GaussParams(q,s));
        q -= n * step.pos;
        s -= n * step.scale;
        break if (step.norm() >= 0.01);
    }

    io = GaussParams(q, s);
}
```

Our L2 Loss in 2D looks as follows  (assuming $y^*$ is square x-dim = y-dim):

$$
L(\mu, s) = \frac{1}{(N+1)^2}\sum_{i=0}^{N}\sum_{j=0}^{N}(g(x_{ij}; \mu, s) - y_{ij})^2
$$
Now for the implementation of the gradient, we have to loop over x-dim but also the y-dim:

```wgsl
fn gradL(param : GaussParam) -> GaussParams {

    var grad = GaussParams(); // init to zero
    
    for(var i = 0u; i < N; i++) {
        for(var j = 0u; j < N; j++) {
            let x = vec2f(j,i); 
            let y = dataY[i][j];

            let diff = (g(x,p) - y);
            p = gGrad(x,p);
        }
    }
    grad.pos *= 4.0 / f32(N*N);
    grad.scale *= 4.0 / f32(N*N);

    return param;
}
```




### Using Texture as Y
this texture for our example:

<img src="assets/testImage.jpg" width="200" height="200">


The following steps are mostly copied from [Webgpu fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-textures.html), it goes a bit more into the details of textures.

First loading the image in typescript is straight forward
```typescript
async function loadImageBitmap(url : string) : : Promise<ImageBitmap>  {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: 'none'});
}
``` 
now that we have the image data, we need to create storage space on the gpu and copy that data over, so we can read it later in our shaders.

```typescript
const url = 'assets/testImage.jpg';
const source = await loadImageBitmap(url);

// create storage space
const texture = ctx.device.createTexture({
    label: url,
    format: 'rgba8unorm',
    size: [source.width, source.height],
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});

// copying data
ctx.device.queue.copyExternalImageToTexture(
    { source, flipY: true },
    { texture },
    { width: source.width, height: source.height }
);
```
<!-- these is this wierd `flipY: true` flag which is there because Texture coordinates start from the top right corner and go from [0,1]. flipping the img, makes reading it more intuitive, y axis grows => image up direction. -->

Now that we have our texture stored in the buffer, we just need to bind it.

```typescript
const bindGroupTexture = ctx.device.createBindGroup({
    label: 'bindGroup texture',
    layout: pipeLineForward.getBindGroupLayout(0),
    entries: [
        { binding: 1, resource: texture },
    ]
});
```
<!-- 
no display, we dont care (for now, its more important to use compute shader and refrence that sampler makes it much faster, layouts can be said later on (maybe just a link)) -->

## Remove this part
Now to display the texture, we will make a vs `tileVert.wgsl`, that renders a tile covering the whole screen and passes the uv coordiantes to the fragment shader (by first converting NDC-space [-1,1] -> uv-space [0,1]):

```wgsl
const tile = array(...);

struct vsOut {
    @builtin(position) p: vec4f,
    @location(0) texCoord: vec2f,
};

@vertex fn vs(
    @builtin(vertex_index) i : u32
) -> vsOut {
    // transform to uv space [-1,1] -> [0,1]
    let uvCoord = (tile[i] + vec2(1.0)) * 0.5;
    return vsOut(vec4(tile[i], 0.0, 1.0), uvCoord);
}
```

and return out the texture colors in the fragment shader `textureFrag.wgsl`:
```wgsl
...
@group(0) @binding(1) var ourTexture: texture_2d<f32>;

@fragment fn fs(
    in : vsOut
    ) -> @location(0) vec4f {
    
    let pUV = in.p.xy * vec2f(1.0/uniforms.canvasDim);

    let size = textureDimensions(ourTexture, 0);
    let pos = vec2u(pUV * size);
    
    var resultColor = textureLoad(ourTexture, position, 0);

    return resultColor;;
}
```

and this would draw us our `testImage.jpg` on our screen. 