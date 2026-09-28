## SSIM-loss?

The paper uses a specific loss:

$$
\mathcal{L} = (1-\lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}
$$

where for $\lambda$ is just a weight bias which the paper sets to $\lambda = 0.2$. The $\mathcal{L}_1$ is called the L1-Loss. which is quite simple:

$$
L_1 = \frac{1}{N}\sum^N_{i=1}|f(x_i) - y_i|
$$

The D-SSIM is a variation of SSIM, which just 

$$
\mathcal{L}_{\text{DSSIM}} = \frac{1- \mathcal{L}_{\text{SSIM}}}{2}
$$

wich maps the range of SSIM from [-1, 1] to D-SSIM [0,1], what the output means exactly is discussed bellow.

### Structural Similarity Index Measure (SSIM)
If one wants to know more about SSIM, one can read this [hitchhikers Guide](https://arxiv.org/pdf/2101.06354). But I will shortly explain it here. 

SSIM takes two inputs and outputs a scalar value. $I_1, I_2 \in \R^{3 \times N \times N}$ are two square rgb images we are comparing

- $\text{SSIM}(I_1, I_2) \approx 1 \implies$ two images are very similar


- $\text{SSIM}(I_1, I_2) \approx -1 \implies$ two images are the complete oposite ($I_1 \approx -I_2$)

- $\text{SSIM}(I_1, I_2) \approx 0 \implies$ two images have nothing in common.

so how is SSIM defined? it sums up a function $Q$ over every pixel (in our case our images are square):

$$
\text{SSIM}(I_1, I_2) = \frac{1}{N^2}\sum^{N}_{i=1}\sum^{N}_{j=1} Q(i,j)
$$

$$
Q(i,j) = l(i,j) * c(i,j) * s(i,j)
$$

where the terms mean similarty of:

- $l$ luminance
- $c$ contrast
- $s$ structure

note that there is no color term. before defining these terms we define mean $\mu_I(i,j)$  and variance $\sigma_I(i,j)$ of a point on a image:


$$
\mu_I(i, j) = \sum_{u=-W}^{W} \sum_{v=-W}^{W} w(u, v) \, I(i + u, \, j + v)
$$

where $W$ is a window size and $w$ is a window function which could be a gaussian. variance

$$
\sigma_I^2(i, j) = \sum_{u=-W}^{W} \sum_{v=-W}^{W} w(u, v) \Big[ I(i + u, \, j + v) - \mu_I(i, j) \Big]^2
$$

ok now we can define the terms from before:

$$
l = \frac{2\mu_1\mu_2 + C_1}{\mu_1^2 + \mu_2^2 + C_1}
$$

$$
c = \frac{2\sigma_1\sigma_2 + C_2}{\sigma_1^2 + \sigma_2^2 + C_2}
$$

$$
s = \frac{\sigma_{12} + C_3}{\sigma_1\sigma_2 + C_3}
$$

where $C_1, C_2, C_3$ are constants one can choose.

Aight thats all we need to know. we could now compute SSIM.

### SSIM problems

one issue with our current implementation is with the $\mu_I(i,j)$ and $\sigma_I(i,j)$ need information about the pixels around it, which in the current implementation is not possible:
-  we are computing the final image color while calculating the gradients at the same time 

but
- computing the gradient of SSIM requires information of the final image pixels around it.

Basically we need a forward pass (that computes the final image $I$). We could do this inside the compute shader but I decided we can reuse the instancing logic from [before](https://cedi-code.github.io/webgpu2dsplatting/journey-primitive.html).

### Forward pass
Todo this we need the following:

1. execute the forward step pipeline
2. storing the result in a texture
3. reading the texture in backwards pass

So first "execute the forward step pipeline". 
we will not use the instance buffer, since this we can read the param buffer directly.
changes in `gaussTileVert.wgsl`
```wgsl
...

@group(0) @binding(0) var<storage, read> output: array<Params>;  // [!code ++]

@vertex fn vs(
    @builtin(vertex_index) i : u32,
    @builtin(instance_index) j : u32,
    splat : Splat, // [!code --]
) -> vsOutput {

    let splat = output[j];  // [!code ++]
...
}
```

next we create render pipeline and binding the Param buffer to it and do a pass over the pipeline.

```typescript
...
    const pipeLineForward = ctx.device.createRenderPipeline({ 
        label: 'forward pipeline',
        layout: 'auto',
        ... /*(with alpha blending) */ 
    });
    ...

    const bindGroupForward = ctx.device.createBindGroup({
        label: 'bindGroup forward',
        layout: pipeLineForward.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
        ]
    });
    ...
    render(ctx, pipeLineForward, bindGroupForward, undefined, 6, numSplats);
...

```
trying to call render will not work, the problem is because in our `pipeLineForward` we have `layout: 'auto'`. when having multipe pipelines we have to specify a layout, how todo that is explained in the webgpu fundamentals: [WebGPU Bind Group Layouts](https://webgpufundamentals.org/webgpu/lessons/webgpu-bind-group-layouts.html)

Second point: "storing the result in a texture". We create the texture (of type `GPUTextureUsage.RENDER_ATTACHMENT`), but also need to specify to render into that texture. 
```typescript

    // create texture:
    const textureForward = ctx.device.createTexture({
        label: 'forwardpass texture',
        format: 'rgba8unorm',
        size: [256, 256],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // add render pass descriptor
    const textureForwardView = textureForward.createView();
    textureForwardView.label = 'forward texture view';

    const renderPassDescriptorTexture : GPURenderPassDescriptor= {
        label: 'forwardpass texture',
        colorAttachments: [
            {
                clearValue: [0.0, 0.0, 0.0, 1.0],
                loadOp: 'clear',
                storeOp: 'store',
                view: textureForwardView,
            },
        ],
    ...
    // render in texture (forward)
    ctx.renderPassDescriptor = renderPassDescriptorTexture;
    render(...);

    // compute shader (backwards pass)
    ...
``` 

and as last "reading the texture in backwards pass", we add the binding our new forward texture 

```typescript
    const bindGroup = ctx.device.createBindGroup({
        entries: [
            ...
            { binding: 6, resource: textureForward }, // [!code ++]
        ]
    });
```

and read it in our compute shader:

```wgsl
@group(0) @binding(6) var forwardTexture : texture_2d<f32>;  // [!code ++]

...
@compute @workgroup_size(1) fn computeGD() {
    ...
    for (var y = 0u; y < size.y; y++) {
        for (var x = 0u; x < size.x; x++) {
            let gColorRaw = textureSampleLevel(forwardTexture, ourSampler, uv, 0.0);  // [!code ++]

            ...

        }
    }
``` 

