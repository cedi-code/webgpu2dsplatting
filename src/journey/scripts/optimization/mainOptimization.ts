import { Pane } from 'tweakpane';

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import shaderCodeCompute from '../shaders/gradientDescent2d.wgsl?raw';
import simpleTileVert from '../shaders/staticTileVert.wgsl?raw';
import simpleTextureFrag from '../shaders/simpleTextureFrag.wgsl?raw';
import type { UniformBufferDescriptor } from '../../../mytypes';


async function loadImageBitmap(url : string) {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: 'none'});
}

type FlatParams = {
    posX: number;
    posY: number;
    scaleX: number;
    scaleY: number;
    rot: number;
    r: number;
    g: number;
    b: number;
    alpha: number;
};

function parseParams(data : Float32Array, desc : UniformBufferDescriptor, round? : number) {
    const posOff    = desc.attributes[0].offset;
    const scaleOff  = desc.attributes[1].offset;
    const rotOff    = desc.attributes[2].offset;
    const colorOff  = desc.attributes[3].offset;
    const alphaOff  = desc.attributes[4].offset;    
    
    // the + makes it go back to nuber
    const r = (n: number) : number => +n.toFixed(round ?? 3);
    return {
            posX:   r(data[posOff]),
            posY:   r(data[posOff + 1]),
            scaleX: r(data[scaleOff]),
            scaleY: r(data[scaleOff + 1]),
            rot:    r(data[rotOff]),
            r:      r(data[colorOff]),
            g:      r(data[colorOff + 1]),
            b:      r(data[colorOff + 2]),
            alpha:  r(data[alphaOff]),
        };
}

async function main() {

    const ctx = await getWebGPUctx({ canvasId: "main"});
    if(!ctx) {
        return;
    }

    console.log(ctx.canvas.width );
    console.log(ctx.canvas.height);

    const csModule = ctx.device.createShaderModule({
        label: '1d gs module',
        code: shaderCodeCompute 
    });

    
    const vsModule = ctx.device.createShaderModule({
        label: '2d static tile',
        code: simpleTileVert 
    });


    const fsModule = ctx.device.createShaderModule({
        label: 'simple texture frag impl',
        code: simpleTextureFrag 
    });
    
    // number of splats
    const numParams = 2;

    const paramBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', numParams);
    paramBuilder.add('pos', "vec2f")
                .add('scale', "vec2f")
                .add('rot', "f32")
                .add('color', "vec3f")
                .add('alpha', "f32");
    const paramDesc = paramBuilder.build();

    console.log(paramDesc);
    
    const uniBuild = new UniformBufferDescriptorBuilder('gd uniform', "uniform");
    uniBuild.add('stepSize', 'f32');    
    const uniDesc = uniBuild.build();
        
    // == defining the binding layouts
    const bindGroupLayoutDescriptorsFragment = ctx.device.createBindGroupLayout(
        {
            entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {
                type: "filtering", // type for 'rgba8unorm'
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {
                    sampleType: "float", // type for 'rgba8unorm'
                    viewDimension: "2d",
                    multisampled: false,
                },
            },
            { // uniforms
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: {
                    type: 'storage',
                    minBindingSize: paramDesc.sizeBytes,
                },
                },
            ],
        },
    );

    const bindGroupLayoutCompute = ctx.device.createBindGroupLayout({
        entries: [
            { // dataOutput
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: paramDesc.sizeBytes,
            },
            },
            {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            sampler: {
                    type: "filtering", // type for 'rgba8unorm'
                },
            },
            { // goal texture
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            texture: {
                    sampleType: "float", // type for 'rgba8unorm'
                    viewDimension: "2d",
                    multisampled: false,
            },
            },
            { // uniforms
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'uniform',
                minBindingSize: uniDesc.sizeBytes,
            },
            },
        ],
    });

    
    const pipelineLayoutDraw = ctx.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayoutDescriptorsFragment ],
    });


    const pipelineLayoutCompute = ctx.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayoutCompute ],
    });

    // == creating the pipelines
    const pipeLineDraw = ctx.device.createRenderPipeline({
        label: 'texture result pipeline',
        layout: pipelineLayoutDraw,
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

    const pipelineCompute = ctx.device.createComputePipeline({
        label: 'gd compute pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: csModule,
        },
        
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

    // creating buffer
    bufferManager.init(ctx.device);

    const paramBuffer = bufferManager.createBuffer(paramDesc);

    const posOff    = paramDesc.attributes[0].offset;
    const scaleOff  = paramDesc.attributes[1].offset;
    const rotOff    = paramDesc.attributes[2].offset;
    const colorOff  = paramDesc.attributes[3].offset;
    const alphaOff  = paramDesc.attributes[4].offset;

    const input = new Float32Array(paramDesc.size);
    const unitS = paramDesc.unitSize!;
    let nextUnit = unitS;

    // gauss 1
    input.set([0.9, 0.3], posOff);
    input.set([1.5, 1.5], scaleOff);
    input.set([0.0], rotOff);
    input.set([8.0, 1.0, 1.0], colorOff);
    input.set([1.0], alphaOff);

    // gauss 2
    input.set([0.3, 0.6], nextUnit + posOff);
    input.set([0.7, 0.7], nextUnit + scaleOff);
    input.set([0.0], nextUnit + rotOff);
    input.set([1.0, 1.0, 8.0], nextUnit + colorOff);
    input.set([1.0], nextUnit + alphaOff);

    const prtyPrint = (data : Float32Array) => {
        const result : FlatParams[] = []
        for(let i = 0; i < numParams; i++) {
            const splatData = data.subarray(unitS*i, unitS*(i+1));
            result.push(parseParams(splatData, paramDesc));
        }
        console.table(result);
    }
    prtyPrint(input);

    ctx.device.queue.writeBuffer(paramBuffer, 0, input);

    const resultBuffer = ctx.device.createBuffer({
        label: 'result buffer',
        size: paramDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });


    // == uniform stuff for interaction ==

    const uniBuff = bufferManager.createBuffer(uniDesc);
    const uniVal = new Float32Array([0.1]); // stepsize
    
    ctx.device.queue.writeBuffer(uniBuff, 0, uniVal , 0);

    // == texture stuff

    const testImageUrl = 'assets/testImage2splats.jpg';
    const source = await loadImageBitmap(testImageUrl);
    const texture = ctx.device.createTexture({
        label: testImageUrl,
        format: 'rgba8unorm',
        size: [source.width, source.height],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    ctx.device.queue.copyExternalImageToTexture(
        { source, flipY: true },
        { texture },
        { width: source.width, height: source.height }
    );

    const sampler = ctx.device.createSampler();

    // ==

    const bindGroup = ctx.device.createBindGroup({
        label: 'bindGroup workbuffer',
        layout: pipelineCompute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
            { binding: 1, resource: sampler },
            { binding: 2, resource: texture },
            { binding: 3, resource: uniBuff },
        ]
    });

    const bindGroupDraw = ctx.device.createBindGroup({
        label: 'bindGroup draw',
        layout: pipeLineDraw.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: sampler },
            { binding: 1, resource: texture },
            { binding: 2, resource: paramBuffer },
        ]
    });

    type Output = {
        initalQ : number,
        finalQ: number,
        lossLog: number,
    }

    const PARAMS_OUT : Output = {
        initalQ : 0.5,
        finalQ: 0.0,
        lossLog: 0.0,
    }

    let updateResults = async (params_out : Output) : Promise<Float32Array<ArrayBuffer>> => {
        // read results
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(resultBuffer.getMappedRange());
        
        params_out.finalQ = result[1];        
        input.set(result, 0);

        // unmap getMapped range is only valid buffer until we call unmap, the length will be set to 0
        resultBuffer.unmap();        

        return result;
    }

    render(ctx, pipeLineDraw, bindGroupDraw, undefined, 6, 1);

    let runGD = async () => {

        const steps = 10;
        for(let i = 0; i < steps; i++) {

            ctx.device.queue.writeBuffer(paramBuffer, 0, input);

            const encoder = ctx.device.createCommandEncoder({
                label: 'gd encoder',
            });
            const pass = encoder.beginComputePass({
                label: 'dumb gradient descent compute pass',
            });
            pass.setPipeline(pipelineCompute);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(1);
            pass.end();

            // mapping result to my buffer
            encoder.copyBufferToBuffer(paramBuffer, 0, resultBuffer, 0, resultBuffer.size);

            // run the work lmao
            const commandBuffer = encoder.finish();
            ctx.device.queue.submit([commandBuffer]);

            // this updates the input values, not clean
            await updateResults(PARAMS_OUT);

            ctx.device.queue.writeBuffer(paramBuffer, 0, input);

            render(ctx, pipeLineDraw, bindGroupDraw, undefined, 6, 1);
        }
        prtyPrint(input);

        
    }
    // == interactive suff, not really needed
    {
        // const PARAMS = {
        //     stepSize: 0.3,
        // };
        
        const pane = new Pane({
            container: document.getElementById("gd-sliders") as HTMLElement,
        });

        // pane.addBinding(PARAMS, 'stepSize', {
        //     min: 0.001,
        //     max: 1.0,
        // }).on('change', (ev) => {
        //     let s = ev.value;
        //     uniVal[0] = s;
        //     ctx.device.queue.writeBuffer(uniBuff, 0, uniVal, 0);
        //     runGD();
        // });

        pane.addBinding(PARAMS_OUT, 'initalQ', {
            readonly: true,
        });

        pane.addBinding(PARAMS_OUT, 'finalQ', {
            readonly: true,
        });

        pane.addButton({
            title: 'gd step',
            label: 'step'
        }).on('click', async () => {
            await runGD();
        });
    }

    ctx.device.lost.then((info) => {
        console.error("Device lost lmao, reason:", info.reason, info.message);
    });
    
}

main();