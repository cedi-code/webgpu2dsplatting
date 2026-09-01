import { bufferManager, UniformBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

async function main() {

    const ctx = await getWebGPUctx({ canvasId: "draw-elipse"});
    if(!ctx) {
        return;
    }
    
    const responseVert = await fetch('src/journey/scripts/shaders/staticTileVert.wgsl');
    const shaderCodeVert = await responseVert.text();
    const responseFrag = await fetch('src/journey/scripts/shaders/simpleGaussFrag.wgsl');
    const shaderCodeFrag = await responseFrag.text();

    const vsModule = ctx.device.createShaderModule({
        label: ' simple vertex shader',
        code:  shaderCodeVert,
    });

    const fsModule = ctx.device.createShaderModule({
        label: 'simple triangle fragment shader',
        code: shaderCodeFrag,
    });

    const numVerticies : number = 6; // hardcoded in the shader, ugly but short
    const pipeline = ctx.device.createRenderPipeline({
        label: 'hardcoded checkerboard triangle',
        layout: 'auto',
        vertex: {
            entryPoint: 'vs',
            module: vsModule,
            buffers: [],
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

    // == uniform buffer setup ==
    bufferManager.init(ctx.device);

    const uBuilder = new UniformBufferDescriptorBuilder("My Uniform Buffer", "uniform");
    uBuilder.add("canvasDim", "vec2f");
    const uDesc = uBuilder.build();

    const uBuffer = bufferManager.createBuffer(uDesc);

    const uValues = new Float32Array(uDesc.size);
    uValues.set([1.0/ctx.canvas.width, 1.0/ctx.canvas.height], uDesc.attributes[0].offset);

    ctx.device.queue.writeBuffer(uBuffer, 0, uValues);

    const bindGroup = ctx.device.createBindGroup({
        label: 'my bind group',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uBuffer }},
        ]
    });


    render(ctx, pipeline, bindGroup, undefined, numVerticies);
}

main();