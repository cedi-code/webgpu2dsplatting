import { Pane } from 'tweakpane';
import { bufferManager, UniformBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

// shaders
import shaderCodeVert from '../shaders/staticTileVert.wgsl?raw'
import shaderCodeFrag from '../shaders/simpleGaussFrag.wgsl?raw'


async function main() {

    const ctx = await getWebGPUctx({ canvasId: "draw-elipse"});
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
    uBuilder.add("canvasDim", "vec2f")
            .add("diagMat", "mat2x2f")
            .add("mean", "vec2f");
    const uDesc = uBuilder.build();

    const uBuffer = bufferManager.createBuffer(uDesc);

    const uValues = new Float32Array(uDesc.size);
    const att = uDesc.attributes;
    const canv = ctx.canvas;

    uValues.set(
        [1.0/canv.width, 1.0/canv.height]
    , att[0].offset); // canvas Dim

    uValues.set(
        [20.0, 0.0, 
         0.0, 20.0]
    , att[1].offset); // variance Mat

    uValues.set(
        [0.5,0.5]
    , att[2].offset); // mean vec

    ctx.device.queue.writeBuffer(uBuffer, 0, uValues);

    const bindGroup = ctx.device.createBindGroup({
        label: 'my bind group',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uBuffer }},
        ]
    });

    // == uniform buffer done ==

    render(ctx, pipeline, bindGroup, undefined, numVerticies);

    // == inputs == 
    let updateUniform = (index : number, value : ArrayLike<number>) => {
        uValues.set(value, uDesc.attributes[index].offset);
        ctx.device.queue.writeBuffer(uBuffer, 0, uValues);
    }

    const PARAMS = {
        diagonal: { x: 0.05, y: 0.05 },
        mean: { x: 0.5, y : 0.5}
    };

    const pane = new Pane({
        container: document.getElementById("variance-slider") as HTMLElement,
    });

    pane.addBinding(PARAMS, 'diagonal', {
        picker: 'inline',
        expanded: true,
        
        x: { min: 0, max: 0.1, step: 0.001 },
        y: { min: 0, max: 0.1, step: 0.001, inverted: true },
    })
    .on('change', (ev) => {
        updateUniform(1, [1.0/ev.value.x, 0,0, 1.0/ev.value.y]);
        render(ctx, pipeline, bindGroup, undefined, numVerticies);
    });

    pane.addBinding(PARAMS, 'mean', {
        picker: 'inline',
        expanded: true,
        
        x: { min: 0, max: 1.0, step: 0.01 },
        y: { min: 0, max: 1.0, step: 0.01, inverted: false },
    })
    .on('change', (ev) => {
        updateUniform(2, [ev.value.x, ev.value.y]);
        render(ctx, pipeline, bindGroup, undefined, numVerticies);
    });

}

main();
