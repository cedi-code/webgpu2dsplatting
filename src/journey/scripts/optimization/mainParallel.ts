import { Pane } from 'tweakpane';

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { type UniformBufferDescriptor } from '../../../mytypes/BufferDescriptors';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import { parseParams, type FlatParams, createLossPlot } from '../../../myutils/LogHelpers';


import shaderCodeCompute from '../shaders/gradientBackwardsCompute.wgsl?raw';
import shaderGaussFunctions from '../../../shaders/gaussFunctions.wgsl?raw';
import gaussTileVertStorage from '../shaders/gaussTileVertStorage.wgsl?raw';
import gaussFrag from '../shaders/gaussFrag.wgsl?raw';
import adamShad from '../../../shaders/adam.wgsl?raw';
import adamCompute from '../shaders/adamOnGradients.wgsl?raw';
import shaderGradTypes from '../../../shaders/gradTypes.wgsl?raw';
import atomicTypes from '../../../shaders/atomicTypes.wgsl?raw';

async function loadImageBitmap(url : string) : Promise<ImageBitmap> {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: 'none'});
}


let lossData : number[]  = []
let lossSteps: number[] = []
let plotHTMLElem = document.getElementById('loss-plot') ?? document.body;
let lossPlot = createLossPlot(plotHTMLElem);

const MACHINE_EPSILON = 1.19e-07;


async function main() {

    const ctx = await getWebGPUctx({ canvasId: "main"});
    if(!ctx) {
        return;
    }

    console.log(ctx.canvas.width );
    console.log(ctx.canvas.height);

    // constants
    const CONSTANTS = {
        NUM_GAUSS : 2,
        xRAY : false,
        nGauss : 2,
        sampleDim : 128,
        EPSILON : MACHINE_EPSILON,
    };

    const sharedConstants = Object.entries(CONSTANTS)
    .map(([k, v]) => `const ${k} = ${v};`)
    .join('\n');

    const csModule = ctx.device.createShaderModule({
        label: 'backwards pass',
        code: (
            sharedConstants + 
            shaderGradTypes + 
            atomicTypes + 
            shaderGaussFunctions + 
            shaderCodeCompute
        ) 
    });

    const adamModule = ctx.device.createShaderModule({
        label: 'adam pass',
        code: (
            sharedConstants + 
            shaderGradTypes + 
            atomicTypes + 
            adamShad  + 
            adamCompute
        ) 
    });
    
    const vsModule = ctx.device.createShaderModule({
        label: '2d static tile',
        code: (
            shaderGradTypes +
            shaderGaussFunctions + 
            gaussTileVertStorage
        ),
    });


    const fsModule = ctx.device.createShaderModule({
        label: 'simple texture frag impl',
        code: (gaussFrag) 
    });
    
    // number of splats
    const numSplats = 2;

    const paramBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', numSplats);
    paramBuilder.add('pos', "vec2f")
                .add('scale', "vec2f")
                .add('rot', "f32")
                .add('color', "vec3f")
                .add('alpha', "f32");
    const paramDesc = paramBuilder.build();

    const lossBuilder = new UniformBufferDescriptorBuilder('loss storage buffer', 'storage', 'copy_src_dst');
    lossBuilder.add('loss', "f32");
    const lossBuffDesc = lossBuilder.build();

    const adamMemoryBuilder = new UniformBufferDescriptorBuilder('adam memory', 'storage', 'copy_dst');

    // hacky way, please remove after paralleization
    const hackySizeAdam = 2 * paramDesc.size + 4.0; // + 4.0 because of the t variable
    const adamMemoryDesc : UniformBufferDescriptor = {
        attributes : [],
        label: 'hacky adam memory',
        size: hackySizeAdam,
        sizeBytes: hackySizeAdam * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    };
    
    const uniBuild = new UniformBufferDescriptorBuilder('gd uniform', "uniform");
    uniBuild.add('lr', "f32")
            .add('b1', "f32")
            .add('b2', "f32")
            .add('eps', "f32");    
    const uniDesc = uniBuild.build();
        
    // == defining the binding layouts
    const bindGroupLayoutDescriptorsForward = ctx.device.createBindGroupLayout(
        {
            entries: [
            { // uniforms
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {
                    type: 'read-only-storage',
                    minBindingSize: paramDesc.sizeBytes,

                },
                },
            ],
        },
    );

    const bindGroupLayoutAdamStepCompute = ctx.device.createBindGroupLayout({
        entries: [
            { // dataOutput
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: paramDesc.sizeBytes,
            },
            },
            { // gradients
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: paramDesc.sizeBytes, // should be same size as params!
            },
            },
            { // adam memory
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: adamMemoryDesc.sizeBytes,
            },
            },
            { // uniforms
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'uniform',
                minBindingSize: uniDesc.sizeBytes,
            }
            },
            ],
    });

    const bindGroupLayoutBackwardsCompute = ctx.device.createBindGroupLayout({
        entries: [
            { // dataOutput
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: paramDesc.sizeBytes,
            },
            },
            { // sampler
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
            { // lossOutput
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: lossBuffDesc.sizeBytes,
            },
            },
            {
            binding: 4, // forward texture
            visibility: GPUShaderStage.COMPUTE,
            texture: {
                    sampleType: "float", // type for 'rgba8unorm'
                    viewDimension: "2d",
                    multisampled: false,
            },
            },
            { // gradients
            binding: 5,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: paramDesc.sizeBytes, // should be same size as params!
            },
            },
        ],
    });

    
    const pipelineLayoutForward = ctx.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayoutDescriptorsForward ],
    });


    const pipelineLayoutBackwardsCompute = ctx.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayoutBackwardsCompute ],
    });

    const pipelineLayoutAdamCompute = ctx.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayoutAdamStepCompute ],
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
    });

    const pipelineBackwardsCompute = ctx.device.createComputePipeline({
        label: 'gd compute pipeline',
        layout: pipelineLayoutBackwardsCompute,
        compute: {
            module: csModule,
        },
    });


    const pipelineAdamCompute = ctx.device.createComputePipeline({
        label: 'gd compute pipeline',
        layout: pipelineLayoutAdamCompute,
        compute: {
            module: adamModule,
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

    const textureForward = ctx.device.createTexture({
        label: 'forwardpass texture',
        format: 'rgba8unorm',
        size: [256, 256],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const textureForwardView = textureForward.createView();
    textureForwardView.label = 'forward texture view';

    const renderPassDescriptorTexture : GPURenderPassDescriptor= {
        label: 'texture renderpass',
        colorAttachments: [
            {
                clearValue: [0.0, 0.0, 0.0, 1.0],
                loadOp: 'clear',
                storeOp: 'store',
                view: textureForwardView,
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
    input.set([4.0], alphaOff);

    // gauss 2
    input.set([0.3, 0.6], nextUnit + posOff);
    input.set([0.7, 0.7], nextUnit + scaleOff);
    input.set([0.0], nextUnit + rotOff);
    input.set([1.0, 1.0, 8.0], nextUnit + colorOff);
    input.set([4.0], nextUnit + alphaOff);

    const prtyPrint = (data : Float32Array) => {
        const result : FlatParams[] = []
        for(let i = 0; i < numSplats; i++) {
            const splatData = data.subarray(unitS*i, unitS*(i+1));
            result.push(parseParams(splatData, paramDesc));
        }
        console.table(result);
    }
    prtyPrint(input);

    ctx.device.queue.writeBuffer(paramBuffer, 0, input);

    // gradient buffer
    let gradDesc = paramDesc;
    gradDesc.label = 'gradient buffer';
    const gradientBuffer = bufferManager.createBuffer(paramDesc);

    ctx.device.queue.writeBuffer(gradientBuffer, 0, new Int32Array(gradDesc.size));

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

    const adamMemBuffer = bufferManager.createBuffer(adamMemoryDesc);

    // not really nessesary? no, remove adam anyway soon
    const adamMemV = new Float32Array(adamMemoryDesc.size);
    ctx.device.queue.writeBuffer(adamMemBuffer, 0, adamMemV, 0);

    // == uniform stuff for interaction ==

    const uniBuff = bufferManager.createBuffer(uniDesc);
    const uniVal = new Float32Array([
        0.05,
        0.9,
        0.999,
        MACHINE_EPSILON
    ]); // adam params
    
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

    const bindGroupBackwards = ctx.device.createBindGroup({
        label: 'bindGroup backwards pass',
        layout: pipelineBackwardsCompute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
            { binding: 1, resource: sampler },
            { binding: 2, resource: texture },
            { binding: 3, resource: lossBuffer },
            { binding: 4, resource: textureForward },
            { binding: 5, resource: gradientBuffer },
        ]
    });

    const bindGroupAdam = ctx.device.createBindGroup({
        label: 'bindGroup adam pass',
        layout: pipelineAdamCompute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
            { binding: 1, resource: gradientBuffer },
            { binding: 2, resource: adamMemBuffer},
            { binding: 3, resource: uniBuff },
        ]
    });


    const bindGroupForward = ctx.device.createBindGroup({
        label: 'bindGroup forward',
        layout: pipeLineForward.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
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


    // render in texture
    ctx.renderPassDescriptor = renderPassDescriptorTexture;
    render(ctx, pipeLineForward, bindGroupForward, undefined, 6, numSplats, true);

    // renders on screen
    ctx.renderPassDescriptor = renderPassDescriptorScreen;
    render(ctx, pipeLineForward, bindGroupForward, undefined, 6, numSplats);

    let runGD = async () => {
        
        const steps = 100;
        const start = performance.now();
        console.log("start");

        const encoder = ctx.device.createCommandEncoder({
            label: 'gd encoder',
        });

        for(let i = 0; i < steps; i++) {

            // reset the gradients, maybe do this in the shader and not here?
            encoder.clearBuffer(gradientBuffer, 0, gradDesc.sizeBytes);
            encoder.clearBuffer(lossBuffer, 0, lossBuffDesc.sizeBytes);

            const pass = encoder.beginComputePass({
                label: 'simple backwards compute pass',
            });

            // backwards
            pass.setPipeline(pipelineBackwardsCompute);
            pass.setBindGroup(0, bindGroupBackwards);
            pass.dispatchWorkgroups(CONSTANTS.sampleDim, CONSTANTS.sampleDim);

            // adam
            pass.setPipeline(pipelineAdamCompute);
            pass.setBindGroup(0, bindGroupAdam);
            pass.dispatchWorkgroups(1);

            pass.end();
            
            // forward pass
            const passRender = encoder.beginRenderPass(renderPassDescriptorTexture);
            passRender.setPipeline(pipeLineForward);
            passRender.setBindGroup(0, bindGroupForward);
            passRender.draw(6, numSplats);

            passRender.end();
        }

        // copy buffer to cpu (expensive! 0.1s)
        encoder.copyBufferToBuffer(lossBuffer, 0, lossResultBuffer, 0, lossResultBuffer.size);
        // run the work lmao
        const commandBuffer = encoder.finish();
        ctx.device.queue.submit([commandBuffer]);
        
        // get loss
        await updateLossResults();

        // display on canvas
        ctx.renderPassDescriptor = renderPassDescriptorScreen;
        render(ctx, pipeLineForward, bindGroupForward, undefined, 6, numSplats);

        await ctx.device.queue.onSubmittedWorkDone();

        let elapsed = performance.now() - start;
        console.log("elapsed time:", elapsed / 1000.0);
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