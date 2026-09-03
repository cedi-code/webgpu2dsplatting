## Transforming Gaussians
### Problem
while thinking of how to continue with rendering multiple Gaussians, there was something not right...

From our current standpoint we would need to:
- *loop throu every gaussian* **per fragment**
```glsl
@fragment fn fs(
    @builtin(position) p : vec4f
    ) -> @location(0) vec4f {
        
   for(var gauss in gaussianBuffer) { // [!code ++]  
        ...  // sampling of G(x)
   } // [!code ++]  

    return vec4f(...);
}
```
each pixel would need todo this itteration loop, reading from memory, which would get very expensive the more splats we have. 

### Move transform to vertex shader
**idea:** if our position $x$ (in the fragment shader) would have the following form:

$$
x' = Ax + \mu
$$

assuming $AA^T = \Sigma$ then pluggin $x'$ into $G(x)$ simplifies to this:

$$
\begin{align*}
G(x') &= exp(-\frac{1}{2}(x'-\mu)^T\Sigma^{-1}(x'-\mu)) \\
      &= exp(-\frac{1}{2}((Ax)^T\Sigma^{-1}(Ax)) \\
      &= exp(-\frac{1}{2}((Ax)^T(AA^T)^{-1}(Ax)) \\
      &= e^{-\frac{1}{2}x^Tx}
\end{align*}
$$

**not needing to read $\mu$ and $\Sigma$ from memory in the fragment shader!**

the vertex shader would need to pass a $x'$ to the fragment shader. this changes the draw plan from:
- ~~- *loop throu every splat* **per fragment**~~
- *loop throu every splat* **per instance**


our fragment shader would only need to evaluate a scalar value `gaussFrag.wgsl`:
```wgsl
@fragment fn fs(
    in : vsOutput
    ) -> @location(0) vec4f {
    
    let G = exp(-0.5 * dot(in.x, in.x));

    return vec4f(vec3f(G), 1.0); 
}
```
and our vertex shader can be called per instance/splat `gaussTileVert.wgsl`:
```wgsl
...
@vertex fn vs(
@builtin(vertex_index) i : u32
    ) -> vsOutput { 
    
    let x = A * tile[i] + q;

    return vsOutput(...);
}
```
but how do we get $A$ such that $AA^T = \Sigma$?

From the paper we define a gaussian to have a elipsoid shape:

$$
\Sigma = RSS^TR^T
$$

where $R$ for us is a 2d rotation matrix and $S$ a diagonal scalar matrix. This gives us what we were looking for:

$$
   A = RS \implies x' = (RS)x + \mu
$$

our fragment shader still needs to sample 

```wgsl
    let x = R * S * tile[i] + q; // [!code ++]  
```
### Instance Buffer
now instead of having a loop in the vs, we store each gaussian (splat) as a instance and pass a instance buffer to it. This can be done as follows:
1. Define a struct how instance data is stored (in `gaussTileVert.wgsl`):
    ```wgsl
    struct Splat { // [!code ++]  
        @location(0) position: vec2f, // [!code ++]  
        @location(1) scale: vec2f, // [!code ++] 
        @location(2) color: vec3f, // [!code ++] 
        @location(3) rotation: f32,  // [!code ++] 
    };
    ```
2. create a buffer desciptor of that struct (exactly like the vertex buffer)
    ```typescript
        const vBuilder = new VertexBufferDescriptorBuilder( // [!code ++] 
            "Splat Instances", n, "instance" // [!code ++] 
        ); // [!code ++] 
        vBuilder.add(0, "position", "float32x2") // [!code ++] 
                .add(1, "scale", "float32x2") // [!code ++] 
                .add(2, "color", "float32x3") // [!code ++] 
                .add(3, "rotation", "float32"); // [!code ++] 
        const pipeLine = {
            vertex: { ...
                buffers: [
                    vBuilder.buildLayout(), // [!code ++] 
                ],
            }, ...
    ```
3. fill that buffer with splat data:
    ```typescript
    const vDesc = vBuilder.build(); // [!code ++] 
    const buff = bufferManager.createBuffer(vDesc); // [!code ++] 
    const val = new Float32Array(vDesc.unitSize * n); // [!code ++] 
    for(let i = 0; i < n; i++) { // [!code ++] 
        val.set(...); // [!code ++] 
        ... // [!code ++] 
    } // [!code ++] 
    ctx.device.queue.writeBuffer(buff, 0, val); // [!code ++] 
    ```
4. and reading buffer data in the vs:
    ```wgsl
    @vertex fn vs(
        @builtin(vertex_index) i : u32,
        splat : Splat, // [!code ++]
    ) -> vsOutput { 
        
        let R : mat2x2f = rotMat(splat.rotation); // [!code ++]  
        let s : vec2f = splat.scale; // [!code ++]  
        let t : vec2f = splat.position; // [!code ++]  

        let x = R * S * tile[i] + q;
    ```
5. and also telling the render pass to draw instances:
    ```typescript
        pass.setVertexBuffer(0, buff); 
        ...
        pass.draw(6, 1); // [!code --] 
        pass.draw(6, n); // [!code ++] 
    ```

now one can draw `n` gaussians/splats on the canvas :)

### Blending
webGPU does not do alpha blending by default and it needs to be specified via the pipeline description:
```typescript
const pipeLineDesc :  GPURenderPipelineDescriptor = {
...
    fragment: {
        ...
        targets: [{ 
            blend: { // [!code ++] 
                color: { // [!code ++] 
                    operation: 'add', // [!code ++] 
                    srcFactor: 'one', // [!code ++] 
                    dstFactor: 'one-minus-src-alpha', // [!code ++] 
                },
                alpha: { // [!code ++] 
                    operation: 'add', // [!code ++] 
                    srcFactor: 'one', // [!code ++] 
                    dstFactor: 'one-minus-src-alpha', // [!code ++] 
                }, // [!code ++] 
            } // [!code ++] 
        }],
        ...
```
and since alpha values work premultiplied, we have to adapt our fs aswell 
`gaussFrag.wgsl`:

```wgsl
@fragment fn fs(
    in : vsOutput
    ) -> @location(0) vec4f {
    
    let G = exp(-0.5 * dot(in.x, in.x));

    return vec4f(vec3f(G), 1.0);  // [!code --] 
    return vec4f(vec3f(1.0)*G, G); // [!code ++] 
}
```

I left out the actual output of the vertex shader, for the alpha blend to work, the render order has to hold.
This can be done by changing the `z` value of our tile position, using the `instance_index` as "depth" indicator for now (will probably have to change this later) in `gaussTileVert.wgsl`. We also define `vsOutput` to have
- `@builtin(position) position` : position of our tile (splat) in world space
- `@location(0) x` : position in local space, where we can sample $ x' \sim \mathcal{N}(0, I)$ from.


```wgsl
...
struct vsOutput { // [!code ++]  
    @builtin(position) position: vec4f, // [!code ++]  
    @location(0) x: vec2f, // [!code ++]  
}; // [!code ++]  

@vertex fn vs(
    @builtin(vertex_index) i : u32,
    @builtin(instance_index) j : u32, // [!code ++]  
    splat : Splat,
) -> vsOutput {

    let R : mat2x2f = rotMat(splat.rotation);
    let s : vec2f = splat.scale;
    let t : vec2f = splat.position;
    let x = R * (s * tile[i]) + t;

    return x; // [!code --]
    return output( // [!code ++]  
        vec4f(x, 1.0 - f32(j+1) / n, 1.0), // [!code ++]  
        vec2f(tile[i]), // [!code ++]  
    ); // [!code ++]  
}
```
ah and for the fs, since the tiles only go from [-1, 1] the gaussian bleeds out on the edge, so we can just put a bigger constant in the exponent to make it more steep:
`gaussFrag.wgsl`:
```wgsl
    let G = exp(-0.5 * dot(in.x, in.x)); // [!code --]   
    let G = exp(-6.0 * dot(in.x, in.x)); // [!code ++] 
```
and here you have it, when enabling "opaque" in the demo you can see that our splats are just tiles with gaussians drawn on top of them:

* [x] ~~WebGPU setup (rendering simple trinalge)~~
* [x] ~~Drawing a Gaussian~~
* [X] ~~Transforming Gaussians~~