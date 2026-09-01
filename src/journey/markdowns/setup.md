## Render a 2d gaussian
Lets start at the first step, just rendering a elipsoid

II will break down this step in the following substeps:

### Render a 2d gaussian
* [ ] WebGPU setup (rendering simple trinalge)
* [ ] Drawing a Gaussian
* [ ] Transforming Gaussians
* [ ] spherical harmonics??

Cool. Lets start coding.

## WebGPU setup 
first lets try to render a simple Triangle using WebGPU.

For any WebGPU information, I will use the awesome website [webgpu-fundamentals (wgpu-f)](https://webgpufundamentals.org/). 

### Utils
Instead of starting from scratch, I will use some boilerplatte of a older project I made. This will ease the managament of buffers and setup for now. but might need some changes later on in the project.

`ContextHelper.ts` contains 2 functions for setup:
```typescript
function getWebGPUctx(...) {
    // sets up the webGPU context and populates the ctx struct
    // ctx has device, context and canvas (and more)
}
function render(...) {
    // makes a render pass using the parameters
}
```


`BufferHelper.ts` provides a Builder pattern for ease of buffer creation:
```typescript
interface BufferManager {
    // takes a descriptor and creates a buffer in memory
    function createBuffer(descriptor : BufferDescriptor): GPUBuffer;
}

class VertexBufferDescriptorBuilder {
    // adds attribute of shader struct
    function add(); 

    // creates a descriptor
    function build(): VertexBufferDescriptor;

    // creates a layout for pipeline
    function buildLayout(): GPUVertexBufferLayout;
}

interface VertexBufferDescriptor  {
    unitSize;
    size;
    stepMode;
    attributes: number[];
}
```
 
#### Buffer Builder Example
lets say our struct in our shader looks as follows:
```wgsl
struct Vertex {
  @location(0) position: vec3f,
  @location(1) uv: vec2f,
  @location(2) color: vec4f,
}
```
using our buffer helper, we can create the buffer and populate it with data as follows:
```typescript
import { bufferManager, VertexBufferDescriptorBuilder } from 'BufferHelper';

// 1. ctx setup
bufferManager.init();

// 2. creating the builder mirroring our struct
const myBuild = new VertexBufferDescriptorBuilder(n);
myBuild.add(0, "position", "float32x3")
       .add(1, "uv", "float32x2")
       .add(2, "color", "float32x4");

const myDesc = myBuild.build();

// 3. creating buffer in memory
const myBuffer = bufferManager.createBuffer(myDesc);

// 4. populating data on CPU
 const data = new Float32Array(myDesc.size);

// offsets
const att = myDesc.attributes;
const posOff = att[0].offset;
const uvOff = att[1].offset;
const colorOff = att[2].offset;

// populate data
for(let i = 0; i < myDesc.size; i+=myDesc.unitSize) {
    data[i + posOff] = vec3(...);
    data[i + uvOff] = vec2(...);
    data[i + posOff] = vec4(...);
}

// 5. write data in buffer
device.queue.writeBuffer(myBuff, 0, data);

```

### Shaders
to draw a triangle we can keep it very simple but still already have it read from a buffer.

vertex shader reads from its buffer and passes it to FS (`simpleVert.wgsl`)
```wgsl
struct SimpleVertexShaderOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

struct Vertex {
    @location(0) position: vec2f,
    @location(1) color : vec3f
}
@vertex fn vs(
    vert: Vertex,
) -> SimpleVertexShaderOutput {
    return SimpleVertexShaderOutput(
        vec4f(vert.position, 0.0, 1.0),
        vec4f(vert.color, 1.0)
    );
}
```
and even simpler for the `simpleFrag.wgsl`:
```wgsl
@fragment fn fs(fsIn : SimpleVertexShaderOutput) -> @location(0) vec4f {
    return fsIn.color;
}
```

### Triangle
now everything can come together by:
1. creating ctx and shaders
    ```typescript
    const ctx = await getWebGPUctx();
    const vsModule = ctx.device.createShaderModule({
        code: // simpleVert.wgsl
    });
    const fsModule = ctx.device.createShaderModule({
        code: // simpleFrag.wgsl
    });
    ```
2. creating triangle buffer builder
    ```typescript
    const bufferBuilder = new VertexBufferDescriptorBuilder(3)
        .add(0, "position", "float32x2")
        .add(1, "color", "float32x3");
    ```
3. creating pipeline + renderPassDescriptor
    ```typescript
    const pipeline = ctx.device.createRenderPipeline({
        ...
        vertex: { ..., buffers: [bufferBuilder.buildLayout() ]},
        ...
    })
    const renderPassDescriptor : GPURenderPassDescriptor= { ... }
    ```
4. creating vertex buffer
    ```typescript
    bufferManager.init(ctx);
    const vBDesc = bufferBuilder.build();
    const vertexBuffer = bufferManager.createBuffer(vBDesc);
    ```
5. populating + writing in the vertex buffer
    ```typescript
    const data = new Float32Array(vBDesc.size);

    const unit = vBDesc.unitSize;
    const attrib = vBDesc.attributes;
    const pOff = attrib[0].offset; // position offset
    const colorOffset = attrib[1].offset; // color offset

    // vertex 0
    let i = 0;
    data.set([ 0.0, 0.5], i + pOff); // top center
    data.set([1,0,0], i + cOff); // red

    // vertex 1
    i += unit;
    data.set([-0.5, -0.5], i + pOff); // left bottom
    data.set([0,1,0], i + colorOffset); // green
    
    // vertex 2
    i += unit;
    data.set([0.5, -0.5], i + pOff); // right bottom
    data.set([0,0,1], i + colorOffset); // blue

    // write data
    ctx.device.queue.writeBuffer(vertexBuffer, 0, data);
    ```
6. rendering it out
    ```typescript
    render(ctx, pipeline, vertexBuffer, 3);
    ```
et voila, with about ~100 LOC (and the help of the utils functions) we have a triangle:
* [x] ~~WebGPU setup (rendering simple trinalge)~~