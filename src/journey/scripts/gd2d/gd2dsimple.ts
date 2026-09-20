import { Pane } from 'tweakpane';

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { type UniformBufferDescriptor } from '../../../mytypes/BufferDescriptors';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import { parseParams, type FlatParams, createLossPlot } from '../../../myutils/LogHelpers';


import shaderCodeCompute from '../shaders/gradientDescent2dSimple.wgsl?raw';
import shaderGaussFunctions from '../../../shaders/gaussFunctions.wgsl?raw';
import staticTileVert from '../shaders/staticTileVert.wgsl?raw';
import textureFrag from '../shaders/simpleTextureFrag.wgsl?raw';

async function loadImageBitmap(url : string) : Promise<ImageBitmap> {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: 'none'});
}


let lossData : number[]  = []
let lossSteps: number[] = []
let plotHTMLElem = document.getElementById('loss-plot-2d') ?? document.body;
let lossPlot = createLossPlot(plotHTMLElem);

async function main() {

    const ctx = await getWebGPUctx({ canvasId: "gd2dsimple"});
    if(!ctx) {
        return;
    }

    console.log(ctx.canvas.width );
    console.log(ctx.canvas.height);
    

    const csModule = ctx.device.createShaderModule({
        label: 'simple 2d gs module',
        code: (shaderGaussFunctions + shaderCodeCompute) 
    });


    const vsModule = ctx.device.createShaderModule({
        label: '2d static tile',
        code: staticTileVert,
    });

    const fsModule = ctx.device.createShaderModule({
        label: 'simple texture frag impl',
        code: (shaderGaussFunctions + textureFrag) 
    });
    
    // number of splats
    const numSplats = 1;

    const paramBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', numSplats);
    paramBuilder.add('pos', "vec2f")
                .add('scale', "vec2f")
                .add('rot', "f32")
    const paramDesc = paramBuilder.build();

    const lossBuilder = new UniformBufferDescriptorBuilder('loss storage buffer', 'storage', 'copy_src_dst');
    lossBuilder.add('loss', "f32");
    const lossBuffDesc = lossBuilder.build();
        
    // == defining the binding layouts
    const bindGroupLayoutDescriptorsForward = ctx.device.createBindGroupLayout(
        {
            entries: [
                { 
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {
                        type: "filtering", // type for 'rgba8unorm'
                    },
                },
                { // goal texture
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {
                        sampleType: "float", // type for 'rgba8unorm'
                        viewDimension: "2d",
                        multisampled: false,
                },
                },
                {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: {
                    type: 'read-only-storage',
                    minBindingSize: paramDesc.sizeBytes,

                },
                },
            ],
        },
    );
    const bindGroupLayoutCompute = ctx.device.createBindGroupLayout({
        entries: [
            {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            sampler: {
                    type: "filtering", // type for 'rgba8unorm'
                },
            },
            { // goal texture
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            texture: {
                    sampleType: "float", // type for 'rgba8unorm'
                    viewDimension: "2d",
                    multisampled: false,
            },
            },
            { // params
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: paramDesc.sizeBytes,
            },
            },
            { // lossOutput
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: lossBuffDesc.sizeBytes,
            },
        },
        ],
    });

    
    const pipelineLayoutForward = ctx.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayoutDescriptorsForward ],
    });


    const pipelineLayoutCompute = ctx.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayoutCompute ],
    });

    // == creating the pipelines
    const pipeLineForward = ctx.device.createRenderPipeline({
        label: 'texture result pipeline',
        layout: pipelineLayoutForward,
        vertex: {
            entryPoint: 'vs',
            module: vsModule,
            buffers: [],
        },
        fragment: {
            entryPoint: 'fs',
            module: fsModule,
            targets: [{ 
                format: ctx.presentationFormat,
            }],
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

    const renderPassDescriptorScreen : GPURenderPassDescriptor= {
        label: 'basic renderpass',
        colorAttachments: [
            {
                clearValue: [0.0, 0.0, 0.0, 1.0],
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

    ctx.renderPassDescriptor = renderPassDescriptorScreen;

    // creating buffer
    bufferManager.init(ctx.device);

    const paramBuffer = bufferManager.createBuffer(paramDesc);

    const posOff    = paramDesc.attributes[0].offset;
    const scaleOff  = paramDesc.attributes[1].offset;
    const rotOff    = paramDesc.attributes[2].offset;

    const input = new Float32Array(paramDesc.size);
    const unitS = paramDesc.unitSize!;

    // gauss 1
    input.set([0.5, 0.5], posOff);
    input.set([1.8, 1.8], scaleOff);
    input.set([0.0], rotOff);

    ctx.device.queue.writeBuffer(paramBuffer, 0, input);

    // loss result buffer
    const lossBuffer = bufferManager.createBuffer(lossBuffDesc);
    const lossV = new Float32Array(lossBuffDesc.size);
    ctx.device.queue.writeBuffer(lossBuffer, 0, lossV);


    const resultBuffer = ctx.device.createBuffer({
        label: 'result buffer',
        size: paramDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const lossResultBuffer = ctx.device.createBuffer({
        label: 'loss result buffer',
        size: lossBuffDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });


    // == texture stuff

    const testImageUrl = 'assets/testImage.jpg';
    const source = await loadImageBitmap(testImageUrl);
    console.log(source);
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

    const bindGroupWorker = ctx.device.createBindGroup({
        label: 'bindGroup workbuffer',
        layout: pipelineCompute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: sampler },
            { binding: 1, resource: texture },
            { binding: 2, resource: paramBuffer },
            { binding: 3, resource: lossBuffer },
                ]
    });

    const bindGroupTexture = ctx.device.createBindGroup({
        label: 'bindGroup texture',
        layout: pipeLineForward.getBindGroupLayout(0),
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

    let updateLossResults = async () => {

        // read loss output
        await lossResultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(lossResultBuffer.getMappedRange());
        lossData.push(result[0].valueOf());
        const lastStep = lossSteps.at(-1) ?? 0;
        lossSteps.push(lastStep + 1);
        

        lossPlot.setData([lossSteps, lossData]);

        // unmap getMapped range is only valid buffer until we call unmap, the length will be set to 0
        lossResultBuffer.unmap();  
    }

    // renders on screen
    ctx.renderPassDescriptor = renderPassDescriptorScreen;
    render(ctx, pipeLineForward, bindGroupTexture, undefined, 6, numSplats);

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
            pass.setBindGroup(0, bindGroupWorker);
            pass.dispatchWorkgroups(1);
            pass.end();

            // mapping result to my buffer
            encoder.copyBufferToBuffer(paramBuffer, 0, resultBuffer, 0, resultBuffer.size);
            encoder.copyBufferToBuffer(lossBuffer, 0, lossResultBuffer, 0, lossResultBuffer.size);


            // run the work lmao
            const commandBuffer = encoder.finish();
            ctx.device.queue.submit([commandBuffer]);

            // this updates the input values, not clean
            await updateResults(PARAMS_OUT);

            // updates loss plot
            await updateLossResults();

            ctx.device.queue.writeBuffer(paramBuffer, 0, input);


            // display on canvas
            ctx.renderPassDescriptor = renderPassDescriptorScreen;
            render(ctx, pipeLineForward, bindGroupTexture, undefined, 6, numSplats);
        }        
    }
    // == interactive suff, not really needed
    {
        // const PARAMS = {
        //     stepSize: 0.3,
        // };
        
        const pane = new Pane({
            container: document.getElementById("gd-sliders-2d") as HTMLElement,
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