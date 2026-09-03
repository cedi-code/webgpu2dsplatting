import { Pane } from 'tweakpane';
import { bufferManager, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

const rnd = (min : number, max : number) : number => {
        return Math.random() * (max - min) + min;
}

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
    const numInstances = 4;    

    bufferManager.init(ctx.device);

    const vBuilder = new VertexBufferDescriptorBuilder("Splat Instances", numInstances, "instance");

    vBuilder.add(0, "position", "float32x2")
            .add(1, "scale", "float32x2")
            .add(2, "color", "float32x3")
            .add(3, "rotation", "float32");
    
    const pipeLineDesc :  GPURenderPipelineDescriptor = {
        label: 'hardcoded checkerboard triangle',
        layout: 'auto',
        vertex: {
            entryPoint: 'vs',
            module: vsModule,
            buffers: [
                vBuilder.buildLayout(),
            ],
        },
        fragment: {
            entryPoint: 'fs',
            module: fsModule,
            targets: [{ 
                format: ctx.presentationFormat ,
                blend: {
                    color: {
                        operation: 'add',
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                    },
                    alpha: {
                        operation: 'add',
                        srcFactor: 'one',
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
    };
    const pipelineA = ctx.device.createRenderPipeline(pipeLineDesc);

    pipeLineDesc.fragment!.targets.at(0)!.blend!.color.dstFactor = 'zero';
    pipeLineDesc.fragment!.targets.at(0)!.blend!.alpha.dstFactor = 'zero';

    const pipelineB = ctx.device.createRenderPipeline(pipeLineDesc);

    let pipeline = pipelineA;

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

    // == buffer data setup ==
    const vDesc = vBuilder.build();

    const instanceBuff = bufferManager.createBuffer(vDesc);
    const instanceValues = new Float32Array(vDesc.unitSize * numInstances);
    /*
    struct Splat {
        @location(0) position: vec2f,
        @location(1) scale: vec2f,
        @location(2) color: vec3f,
        @location(3) rotation: f32, 
    };
    */
    for(let i = 0; i < numInstances; i++) {
        const att = vDesc.attributes;
        const off = vDesc.unitSize * i;

        instanceValues.set([rnd(-1,1), rnd(-1,1)],      off + att[0].offset);
        instanceValues.set([rnd(0,1),  rnd(0,1) ],      off + att[1].offset);
        instanceValues.set(
            [Math.round(rnd(0,1)), Math.round(rnd(0,1)), Math.round(rnd(0,1))],
                                                        off + att[2].offset);
        instanceValues.set([rnd(0,2*Math.PI)],         off + att[3].offset);
    }

    ctx.device.queue.writeBuffer(instanceBuff, 0, instanceValues);

    // == buffer done ==

    render(ctx, pipeline, undefined, instanceBuff, numVerticies, numInstances);

    // == inputs == 
    let updateInstance = (index : number, value : ArrayLike<number>) => {
        instanceValues.set(value, (numInstances-1)*vDesc.unitSize + vDesc.attributes[index].offset);
        ctx.device.queue.writeBuffer(instanceBuff, 0, instanceValues);
    }

    const PARAMS = {
        scale: { x: 0.5, y: 0.5 },
        rotation: 0.0,
        position: { x: 0, y : 0},
        opaque: false,
    };

    const pane = new Pane({
        container: document.getElementById("rotation-sliders") as HTMLElement,
    });

    pane.addBinding(PARAMS, 'scale', {
        picker: 'inline',
        expanded: true,
        
        x: { min: 0, max: 1.0, step: 0.01 },
        y: { min: 0, max: 1.0, step: 0.01, inverted: true },
    })
    .on('change', (ev) => {
        updateInstance(1, [ev.value.x, ev.value.y]);
        render(ctx, pipeline, undefined, instanceBuff, numVerticies, numInstances);
    });

    pane.addBinding(PARAMS, 'rotation', {
        min: 0,
        max: 2* Math.PI,
    })
    .on('change', (ev) => {
        updateInstance(3, [ev.value]);
        render(ctx, pipeline, undefined, instanceBuff, numVerticies, numInstances);
    });

    pane.addBinding(PARAMS, 'position', {
        picker: 'inline',
        expanded: true,
        
        x: { min: -1.0, max: 2.0, step: 0.01 },
        y: { min: -1.0, max: 2.0, step: 0.01, inverted: true },
    })
    .on('change', (ev) => {
        updateInstance(0, [ev.value.x, ev.value.y]);
        render(ctx, pipeline, undefined, instanceBuff, numVerticies, numInstances);
    });

    pane.addBinding(PARAMS, 'opaque')
    .on('change', (ev) => {
        pipeline = ev.value ? pipelineB : pipelineA;
        render(ctx, pipeline, undefined, instanceBuff, numVerticies, numInstances);        
    })
}

main();
