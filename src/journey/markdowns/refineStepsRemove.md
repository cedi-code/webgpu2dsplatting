## Refine

What you may have noticed in the previous example is that a lot of splats just move out of the screen and "dissapear".

![insert gif here]()

Yet, we still loop over them for our sorting. 
So what we could do is remove these splats that leave our boundry after a couple of gradient descent steps.

```
for(let i = 0; i < nSteps; i++)
    forwardPass()
    setupBoxArray() 
    radixSort() 
    backwardsPass()
    reducePass()
    adamPass()

    if i % refinment       // [!code ++]
        refineSplatsPass() // [!code ++]
```

### Remove

we can do a simple check if our mean is over a certain threshold of screen, we mark that splat as "remove".

so we dispatch for each splat the following: `splatRefine.wgsl` 
```wgsl
@group(0) @binding(0) var<storage, read_write> splats: array<Params, NUM_SPLATS>;

@compute @workgroup_size(GROUPSIZE, 1, 1)
fn cs(
    @builtin(global_invocation_id) global_invocation_id : vec3u
    ) {

    let i = splats[global_invocation_id.x];

    let offScreen =
        any(splats[i].pos < vec2f(screenMin)) ||
        any(splats[i].pos > vec2f(screenMax));

    // offScreen is true => remove splat

    }
```

to remove a splat and not loop over the the splats off screen, we can sadly not just create a linked list with gpu buffers and pop that splat of the list. Therefore we create 2 new global variables:
| variable   | desciption   |
|---|---|
| `const NUM_SPLATS`  | maximum number of splats we allow in our buffer   |
|  `let activeSplats : u32` | current number of splats we have in our scene  |

and we do the same trick as in radixSetup, where we re-calculate the offset each splat has in memory.

```wgsl
@group(0) @binding(0) var<storage, read_write> splats: array<Params, NUM_SPLATS>;

var<workgroup> sums: array<u32, NUM_SPLATS>; // [!code ++]

@compute @workgroup_size(NUM_SPLATS, 1, 1)
fn cs(
    @builtin(global_invocation_id) global_invocation_id : vec3u
    ) {

    let i = global_invocation_id.x;

    let offScreen =
        any(splats[i].pos < vec2f(screenMin)) ||
        any(splats[i].pos > vec2f(screenMax));

    sums[i] = select(1u, 0u, offScreen); // [!code ++]

    workgroupBarrier(); // [!code ++]
    prefixSum();        // [!code ++]

    // sums has the new offset for splat i !

    }
``` 

now knowing the offset we just need to write the non-deleted splats into there new positions in the buffer!


```wgsl
@group(0) @binding(0) var<storage, read_write> splats: array<Params, NUM_SPLATS>;
    ...
    let splatCopy = splats[i];
    ...
    workgroupBarrier(); 
    ...
    let newI = sums[i] - 1u;

    if(!offScreen) {
        splats[newI] = splatCopy;
    }
``` 

and we also need to update our current splatCount `activeSplats`! So we let the last thread assign the total sum of all the threads.

```wgsl
...
workgroupBarrier(); 
if(i == NUM_SPLATS-1) {
    activeSplats = sums[i]; // sum[i] here contains the sum of all threads.
}
``` 
easy! now to actually utilize this, we need to tell our other passes to only use up to `activeSplats`-calls! and not loop over the whole `NUM_SPLATS`. So basically replacing `NUM_SPLATS` with `activeSplats` that is not memory related! :)

Actually its not that easy! when we look at our forward pass, we call it instaced based from the CPU! 

```typescript
// forward pass
pass.setPipeline(pipeLineForward);
pass.setBindGroup(0, bindGroupForward);
pass.draw(6, numSplats); // draw(#verticies, #instances)
...
// reduce pass
pass.dispatchWorkgroups(numSplats); 
``` 
but our `activeSplats` lies in GPU memory, so reading it back after every refine step would be expensive!

### Indirect Draw

there is a blog already explaining this better then me by Toji: [WebGPU Indirect Draw best practices](https://toji.dev/webgpu-best-practices/indirect-draws.html).

But in short, instead of passing the argument from the CPU, we can directly read the num of draw calls from a GPU buffer!

in that way, we can calculate on the GPU side how many draw calls we want to execute by storing them in a specific buffer!

```typescript
const drawBuffer = ctx.device.createBuffer({ // [!code ++]
    size: drawArgsDesc.sizeBytes, // [!code ++]
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,  // [!code ++]
}); // [!code ++]
...

// forward pass
pass.setPipeline(pipeLineForward);
pass.setBindGroup(0, bindGroupForward);
pass.draw(6, numSplats); // draw(#verticies, #instances) // [!code --]
pass.drawIndirect(drawBuffer, 0); // [!code ++]

...
// reduce pass
pass.dispatchWorkgroups(numSplats);  // [!code --]
pass.dispatchWorkgroupsIndirect(drawBuffer, 0);  // [!code ++]
```

We now can dynamically remove splats (and also technically add) from our render!

Small note, we also mark splats to be removed with low alpha values, which we can easly do by just adding another flag:

```wgsl
    ...
    let alphaLow = splats[i].alpha < uniforms.alphaThresh;
    let removeFlag = offScreen || alphaLow;
    ...
```

### Densification

the quality of the image highly depens on how many splats we had in the beginning! for example lets consider this image:

<img src="assets/fourBlobs.jpg" width="200">

if we only initalize with 2 splats, we will probably sturggle to capture the higher details of this image.

