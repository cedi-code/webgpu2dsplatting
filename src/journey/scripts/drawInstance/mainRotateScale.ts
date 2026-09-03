import { Pane } from 'tweakpane';
import { bufferManager, UniformBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

async function main() {

    const ctx = await getWebGPUctx({ canvasId: "rotate-scale-elipse"});
    if(!ctx) {
        return;
    }
    
    const responseVert = await fetch('src/journey/scripts/shaders/gaussTileVert.wgsl');
    const shaderCodeVert = await responseVert.text();
    const responseFrag = await fetch('src/journey/scripts/shaders/gaussFrag.wgsl');
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
            targets: [{ 
                format: ctx.presentationFormat ,
                blend: {
                    color: {
                        dstFactor: 'one-minus-src-alpha',
                    },
                    alpha: {
                        dstFactor: 'one-minus-src-alpha',
                    },
                }
            }],

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
    uBuilder.add("scale", "vec2f")
            .add("rotation", "f32")
            .add("mean", "vec2f");
    const uDesc = uBuilder.build();

    const uBuffer = bufferManager.createBuffer(uDesc);

    const uValues = new Float32Array(uDesc.size);
    const att = uDesc.attributes;

    uValues.set(
        [2.0, 2.0]
    , att[0].offset); // scale Mat

    uValues.set([0]
    , att[1].offset); // rotation

    uValues.set(
        [0,0]
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
        scale: { x: 0.5, y: 0.5 },
        rotation: 0.0,
        position: { x: 0, y : 0}
    };

    const pane = new Pane({
        container: document.getElementById("rotation-sliders") as HTMLElement,
    });

    pane.addBinding(PARAMS, 'scale', {
        picker: 'inline',
        expanded: true,
        
        x: { min: 0, max: 2.0, step: 0.01 },
        y: { min: 0, max: 2.0, step: 0.01, inverted: true },
    })
    .on('change', (ev) => {
        updateUniform(0, [1.0/ev.value.x, 1.0/ev.value.y]);
        render(ctx, pipeline, bindGroup, undefined, numVerticies);
    });

    pane.addBinding(PARAMS, 'rotation', {
        min: 0,
        max: 2* Math.PI,
    })
    .on('change', (ev) => {
        updateUniform(1, [ev.value]);
        render(ctx, pipeline, bindGroup, undefined, numVerticies);
    });

    pane.addBinding(PARAMS, 'position', {
        picker: 'inline',
        expanded: true,
        
        x: { min: -1.0, max: 1.0, step: 0.01 },
        y: { min: -1.0, max: 1.0, step: 0.01, inverted: true },
    })
    .on('change', (ev) => {
        updateUniform(2, [ev.value.x, ev.value.y]);
        render(ctx, pipeline, bindGroup, undefined, numVerticies);
    });

}

main();
