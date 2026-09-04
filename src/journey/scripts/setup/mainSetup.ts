import { bufferManager, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import shaderCodeVert from '../shaders/simpleVert.wgsl?raw'
import shaderCodeFrag from '../shaders/simpleFrag.wgsl?raw'


async function main() {

    const ctx = await getWebGPUctx({ canvasId: "setup"});
    if(!ctx) {
        return;
    }
    
    const vsModule = ctx.device.createShaderModule({
        label: ' simple vertex shader',
        code:  shaderCodeVert,
    });

    const fsModule = ctx.device.createShaderModule({
        label: 'simple triangle fragment shader',
        code: shaderCodeFrag,
    });

    const numVerticies : number = 3;
    const triangleBufferBuilder = new VertexBufferDescriptorBuilder("Triangle Vertex Buffer", numVerticies, "vertex")
        .add(0, "position", "float32x2")
        .add(1, "color", "float32x3");

    const pipeline = ctx.device.createRenderPipeline({
        label: 'hardcoded checkerboard triangle',
        layout: 'auto',
        vertex: {
            entryPoint: 'vs',
            module: vsModule,
            buffers: [
                triangleBufferBuilder.buildLayout(),
            ],
        },
        fragment: {
            entryPoint: 'fs',
            module: fsModule,
            targets: [{ format: ctx.presentationFormat }],
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        }
    });

    const renderPassDescriptor : GPURenderPassDescriptor= {
        label: 'basic renderpass',
        colorAttachments: [
            {
                clearValue: [0.3, 0.3, 0.3, 1.0],
                loadOp: 'clear',
                storeOp: 'store',
                view: ctx.context.getCurrentTexture().createView(),
            },
        ],
        depthStencilAttachment: {
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
            view: ctx.context.getCurrentTexture().createView(),
        }
    };
    ctx.renderPassDescriptor = renderPassDescriptor;

    bufferManager.init(ctx.device);

    const triangleVBDesc = triangleBufferBuilder.build();
    const vertexBuffer = bufferManager.createBuffer(triangleVBDesc);

    // == setting up Vertex data ==
    const vertexData = new Float32Array(numVerticies * triangleVBDesc.unitSize);

    const attrib = triangleVBDesc.attributes;
    const posOffset = attrib[0].offset;
    const colorOffset = attrib[1].offset;

    

    let currIndex = 0;
    vertexData.set([0.0, 0.5], currIndex + posOffset); // left bottom
    vertexData.set([1.0,0.0,0.0], currIndex + colorOffset); // red

    currIndex += triangleVBDesc.unitSize;
    vertexData.set([-0.5, -0.5], currIndex + posOffset); // right bottom
    vertexData.set([0,1,0], currIndex + colorOffset); // green
    
    currIndex += triangleVBDesc.unitSize;
    vertexData.set([0.5, -0.5], currIndex + posOffset); // top center
    vertexData.set([0,0,1], currIndex + colorOffset); // blue
    
    // == seting up vertex data DONE ==
    
    ctx.device.queue.writeBuffer(vertexBuffer, 0, vertexData);

    render(ctx, pipeline, undefined, vertexBuffer, numVerticies);
}

main();