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

Our L2 Loss in 2D looks as follows  (assuming $M = (N+1)^2$):

$$
\begin{align*}
L(\mu, s) &= \frac{1}{M}||g(\mu, s)- y^*||_F^2
\\
&= \frac{1}{(N+1)^2}\sum_{i=0}^{N}\sum_{j=0}^{N}(g(x_{ij}; \mu, s) - y_{ij})^2

\end{align*}

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
            grad += diff * gGrad(x,p);
        }
    }
    grad.pos *= 4.0 / f32(N*N);
    grad.scale *= 4.0 / f32(N*N);

    return grad;
}
```
For the gradients of $\nabla_{\mu} g$ and $\nabla_{s} g$ the derrivations can be seen behind the spoiler tag aswell as the implementation of `gGrad(x,p)`.

Cool. This Code could optimize our parameters...but we need something to optimize for, our $y^*$. The part I left out, what is even `dataY[i][j]`?

### Using Texture as Y
for this example use this texture for our $y^* = I^*$:

<img src="assets/testImage.jpg" width="200" height="200">

This means we want our gaussian paramers $g(x;\mu, s)$ to approximate the shape in the image above. For that we need to be able to read its info and loss function changes to:

$$
L(\mu, s) = \frac{1}{M}||g(\mu, s)- I^*||_F^2
$$

We dont need to change a whole lot in our code since images are basically just 2d arrays that contain color information. So the only change needed is loading that image to read it in our shader.

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
this wierd `flipY: true` flag which is there because Texture coordinates start from the top right corner and go from [0,1]. flipping the img, makes reading it more intuitive, y axis grows => image up direction.

Now that we have our texture stored in the buffer, we just need to bind it.

```typescript
const bindGroupWorker = ctx.device.createBindGroup({
    label: 'bindGroup gd',
    layout: pipeLineForward.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: paramBuffer },
        { binding: 1, resource: texture },  // [!code ++]
    ]
});
```
<!-- 
no display, we dont care (for now, its more important to use compute shader and refrence that sampler makes it much faster, layouts can be said later on (maybe just a link)) -->


Now to read the texture, in the compute shader we just have to change two lines in `gradLoss`:


```wgsl
@group(0) @binding(1) var ourTexture: texture_2d<f32>;  // [!code ++]
...
    let N = textureDimensions(ourTexture, 0); // [!code ++]
    for(var i = 0u; i < N.y; i++) {
        for(var j = 0u; j < N.x; j++) {
            let x = vec2f(j,i); 
            let y = dataY[i][j];  // [!code --]
            let y = textureLoad(ourTexture, vec2u(j,i), 0); // [!code ++]

            let diff = (g(x,p) - y);
            grad += gGrad(x,p);
        }
    }
...
```

### Performance Problems
running this now it should all work out, $g$ approximating our image like we expect, but depending on your hardware, you might experience two things:

- the gd runs very slow
- the compute shader does not run at all

The reason this might happen is that we allocate too much memory in a single GPU thread, webgpu might not even load our compute shader, so before we make our gd multithreaded, we can already reduce the load by doing gd loop in typescript, essentially removing the loop condition:

```wgsl
...
@compute @workgroup_size(1) fn gradientDescent() {
    var q = io.pos; 
    var s = io.scale;
    loop {  // [!code --]
    let step : GaussParams = gradL(GaussParams(q,s));
    q -= n * step.pos;
    s -= n * step.scale;
        break if (step.norm() >= 0.01);  // [!code --]
    }  // [!code --]

    io = GaussParams(q, s);
}
``` 
only doing a single step per execution. and calling the worker from cpu a couple times:
```typescript
const steps = 10; // [!code ++]
for(let i = 0; i < steps; i++) { // [!code ++]
    ...
    pass.dispatchWorkgroups(1); 
    ...
} // [!code ++]
``` 


another thing which takes a huge hit on performance is actually our texture read `textureLoad`, which loads our texture for each sample $x_{ij}$!


If you already familiar with textures, then you was wondering why we didnt use a **sampler**! lets set that up, first create a sampler in cpu side and bind it

```typescript
    const sampler = ctx.device.createSampler(); // [!code ++]

    const bindGroupWorker = ctx.device.createBindGroup({
        label: 'bindGroup workbuffer',
        layout: pipelineCompute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
            { binding: 1, resource: texture },
            { binding: 2, resource: sampler }, // [!code ++]
    });
```

Then updating our `gradLoss` to use the sampler instead, and we can even define our own size $N$ since the sampler will interpolate the color of $I$ at the position x.

```wgsl
...
@group(0) @binding(1) var ourTexture: texture_2d<f32>;
@group(0) @binding(2) var ourSampler: sampler; // [!code ++]
...
    let N = textureDimensions(ourTexture, 0); // [!code --]
    let N = vec2f(128, 128); // [!code ++]
    
    for(var i = 0u; i < N.y; i++) {
        for(var j = 0u; j < N.x; j++) {
            let x = vec2f(j,i); 
            let y = textureLoad(ourTexture, vec2u(j,i), 0); // [!code --]
            let y = textureSampleLevel(goalTexture, ourSampler, vec2u(j,i), 0); // [!code ++]

            ...
        }
    }
...
``` 
And thats it, this should already run ok for a single gaussian :)
But there is one more thing...

### Activation Functions

todo: have our toy example show both losses, one with activation function and one without.
