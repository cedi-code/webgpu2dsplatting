import { Pane } from 'tweakpane';

import { vec2, vec3 } from "wgpu-matrix";

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { type UniformBufferDescriptor } from '../../../mytypes/BufferDescriptors';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import { parseParams, type FlatParams, createLossPlot } from '../../../myutils/LogHelpers';

import { defaultSplatConfig, createShaderModuleDescriptors } from '../../../config';
import { DEFAULT_GUI_VISIBILITY, PARALLEL_MODULES } from '../../../config/defaults';
import type { GuiVisibility, SplattingContext } from '../../../config/types';
import type { ctxWebGPU } from '../../../mytypes';
import type uPlot from 'uplot';
import { exampleSplatsRedBlueInit, imageGridSplats, loadImageBitmap, randomSplatInit } from '../../../myutils/MyUtils';


export type mainCtx = {
    webgpu : ctxWebGPU;
    config : SplattingContext;
    init : "random" | "grid" | "example";
    gui : Partial<GuiVisibility>;
    image : ImageBitmap;
    pane : Pane;
    plot? : uPlot;
} 


export async function main(settings : mainCtx) {

    const ctx = settings.webgpu;
    const splatConfig = settings.config;
    const guiVisibility = settings.gui;
    const source = settings.image; // image
    const pane = settings.pane;
    const lossPlot = settings.plot;
    const initMethod = settings.init;

    const moduleDescriptors = createShaderModuleDescriptors(PARALLEL_MODULES, splatConfig);

    if(!ctx) {
        return;
    }
    const FPS = 30;
    let lossData : number[]  = []
    let lossSteps: number[] = []
    // prints the current constants
    console.table(splatConfig.setup);
    console.table(splatConfig.dynamic);
    console.table(splatConfig.refine);


    const paramBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', splatConfig.setup.numSplats);
    paramBuilder.add('pos', "vec2f")
                .add('scale', "vec2f")
                .add('rot', "f32")
                .add('color', "vec3f")
                .add('alpha', "f32");
    const paramDesc = paramBuilder.build();

    const accumulatedGradStorageBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', splatConfig.setup.numSplats);
    accumulatedGradStorageBuilder.add('pos', "vec2f")
                .add('scale', "vec2f")
                .add('rot', "f32")
                .add('color', "vec3f")
                .add('alpha', "f32");
    const accumulatedGradStorageDesc = accumulatedGradStorageBuilder.build();
    const momentumPingPongDesc: UniformBufferDescriptor = {
        attributes: [],
        label: 'refine momentum ping-pong buffer',
        size: accumulatedGradStorageDesc.size * 2,
        sizeBytes: accumulatedGradStorageDesc.sizeBytes * 2,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    };

    // i still cant do arrays of structs...
    const gradientStorageBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', splatConfig.maxSlots );
    gradientStorageBuilder.add('pos', "vec2f")
                        .add('scale', "vec2f")
                        .add('rot', "f32")
                        .add('color', "vec3f")
                        .add('alpha', "f32");
    const gradStorageDesc = gradientStorageBuilder.build();

    const lossBuilder = new UniformBufferDescriptorBuilder('loss storage buffer', 'storage', 'copy_src_dst', splatConfig.loss.maxSteps * splatConfig.numChunks);
    lossBuilder.add('loss', "f32");
    const lossBuffDesc = lossBuilder.build();
    console.log("loss array size:", splatConfig.loss.maxSteps * splatConfig.numChunks);

    const adamMemoryBuilder = new UniformBufferDescriptorBuilder('adam memory', 'storage', 'copy_dst');

    // hacky way, please remove after paralleization
    const hackySizeAdam = 2 * paramDesc.size + 4.0;
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
            .add('eps', "f32")
            .add('alphaCutOff', "f32")
            .add('gradientPosVari', "f32")
            .add('gradinetScalVari', "f32")
            .add('opacityReset', "f32")
            .add('scaleDivideConst', "f32")
            .add('weightedLoss', "f32");

    const uniDesc = uniBuild.build();

    const drawArgsBuilder = new UniformBufferDescriptorBuilder('active splats', "storage", "copy_src_dst");
    drawArgsBuilder.add('vertexCount', "u32")
            .add('instanceCount', "u32")
            .add('vertexOff', "u32")
            .add('instanceOff', "u32")
            .add('padding', "u32")
            .add('workgroupX', "u32")
            .add('workgroupY', "u32")
            .add('workgroupZ', "u32");

    const drawArgsDesc = drawArgsBuilder.build();

    const radixBuild = new UniformBufferDescriptorBuilder('radix memory', "storage", "copy_src_dst");
    radixBuild.add("splatOffsets", { type : "u32", size : splatConfig.radixSetupDispatches })
              .add("offsets", { type : "u32", size: splatConfig.numChunks })
              .add("data", { type : "u32", size : splatConfig.maxSlots })
              .add("data2", { type : "u32", size : splatConfig.maxSlots })
              .add("splatOffsetsTotal", { type : "u32", size : splatConfig.setup.numSplats });

    const radixDesc = radixBuild.build();

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

    const bindGroupLayoutCompute = ctx.device.createBindGroupLayout({
        label: 'bingroup layout compute',
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
                minBindingSize: gradStorageDesc.sizeBytes, 
            },
            },
            { // adam memory
            binding: 6,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: adamMemoryDesc.sizeBytes,
            },
            },
            { // uniforms
            binding: 7,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'uniform',
                minBindingSize: uniDesc.sizeBytes,
            }
            },
            { // radix memory
            binding: 8,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: radixDesc.sizeBytes,
            }
            },
            { // active splats
            binding: 9,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: drawArgsDesc.sizeBytes,
            }
            },
        ],
    });


    const bindGroupLayoutComputeReduce = ctx.device.createBindGroupLayout({
        label: 'bindlayout reduce',
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
                minBindingSize: gradStorageDesc.sizeBytes, 
            },
            },
            { // adam memory
            binding: 6,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: adamMemoryDesc.sizeBytes,
            },
            },
            { // uniforms
            binding: 7,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'uniform',
                minBindingSize: uniDesc.sizeBytes,
            }
            },
            { // radix memory
            binding: 8,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: radixDesc.sizeBytes,
            }
            },
            { // active splats
            binding: 9,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'read-only-storage',
                minBindingSize: drawArgsDesc.sizeBytes,
            }
            },
        ],
    });

    const bindGroupLayoutSplatCheck = ctx.device.createBindGroupLayout({
        entries: [
            {
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
                buffer: {
                    type: 'storage',
                    minBindingSize: momentumPingPongDesc.sizeBytes,
                },
            },
        ],
    });    
    const pipelineLayoutForward = ctx.device.createPipelineLayout({
        label: 'layout forward',
        bindGroupLayouts: [ bindGroupLayoutDescriptorsForward ],
    });

    const pipelineLayoutCompute = ctx.device.createPipelineLayout({
        label: 'layout compute',
        bindGroupLayouts: [ bindGroupLayoutCompute ],
    });

    const pipelineLayoutComputeReduceLayout = ctx.device.createPipelineLayout({
        label: 'special reduce layout',
        bindGroupLayouts: [ bindGroupLayoutComputeReduce ],
    });

    const pipelineLayoutSplatCheck = ctx.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayoutCompute, bindGroupLayoutSplatCheck ],
    });


    // == creating the pipelines
    const moduleForwardVertex = ctx.device.createShaderModule(
        moduleDescriptors.modules['2d static tile']
    );
    const moduleForwardFragment = ctx.device.createShaderModule(
        moduleDescriptors.modules['simple texture frag impl']
    );
    const pipelineForwardDescriptor : GPURenderPipelineDescriptor = {
        label: 'texture result pipeline',
        layout: pipelineLayoutForward,
        vertex: {
            entryPoint: 'vs',
            module: moduleForwardVertex,
            buffers: [],
        },
        fragment: {
            entryPoint: 'fs',
            module: moduleForwardFragment,
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
            constants: {
                withFade: 1,
            }
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        }
    };
    const pipeLineForward = ctx.device.createRenderPipeline(pipelineForwardDescriptor);

    pipelineForwardDescriptor.fragment!.constants = { withFade: 0 };

    const pipeLineForwardBlob = ctx.device.createRenderPipeline(pipelineForwardDescriptor);  

    const moduleRadixSetup = ctx.device.createShaderModule(
        moduleDescriptors.modules['radix setup pass']
    );
    const pipelineRadixSetup1 = ctx.device.createComputePipeline({
        label: 'radix setup pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: moduleRadixSetup,
            entryPoint: 'registerSplats',
            constants: {
                first_pass: 1,
            }
        },
    });

    const pipelineRadixSetup2 = ctx.device.createComputePipeline({
        label: 'radix setup pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: moduleRadixSetup,
            entryPoint: 'registerSplats',
            constants: {
                first_pass: 0,
            }
        },
    });

    const moduleSplatCheck = ctx.device.createShaderModule(
        moduleDescriptors.modules['splat refine pass']
    );
    const pipelineSplatCheck1 = ctx.device.createComputePipeline({
        label: 'splat check pipelie',
        layout: pipelineLayoutSplatCheck,
        compute: {
            module: moduleSplatCheck,
            entryPoint: 'cs',
            constants: {
                first_pass: 1,
            }
        },
    });

    const pipelineSplatCheck2 = ctx.device.createComputePipeline({
        label: 'splat refine pipelie',
        layout: pipelineLayoutSplatCheck,
        compute: {
            module: moduleSplatCheck,
            entryPoint: 'cs',
            constants: {
                first_pass: 0,
            }
        },
    });

    const moduleRadix = ctx.device.createShaderModule(
        moduleDescriptors.modules['radix pass']
    );
    const pipelineRadix : GPUComputePipeline[] = []
    for(let i : number = 0; i < splatConfig.numRadixSorts; i++) {
        pipelineRadix.push(
            ctx.device.createComputePipeline({
                label: 'radix compute pipeline',
                layout: pipelineLayoutCompute,
                compute: {
                    module: moduleRadix,
                    entryPoint: 'radixSort',
                    constants: {
                        pass_nr : i
                    }
                },
            })
        );
    }


    const moduleRadixOffset = ctx.device.createShaderModule(
        moduleDescriptors.modules['radix offset pass']
    );
    const pipelineRadixOffset = ctx.device.createComputePipeline({
        label: 'radix offset compute pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: moduleRadixOffset,
        },
    });

    const moduleBackwardsCompute = ctx.device.createShaderModule(
        moduleDescriptors.modules['backwards pass']
    );
    const pipelineBackwardsCompute = ctx.device.createComputePipeline({
        label: 'gd compute pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: moduleBackwardsCompute,
        },
    });

    const moduleReduceCompute = ctx.device.createShaderModule(
        moduleDescriptors.modules['reduce pass']
    );
    const pipelineReduceCompute  = ctx.device.createComputePipeline({
        label: 'reduce gradients compute pipeline',
        layout: pipelineLayoutComputeReduceLayout,
        compute: {
            module: moduleReduceCompute,
        },
    }); 

    const moduleReduceLossCompute = ctx.device.createShaderModule(
        moduleDescriptors.modules['loss reduce']
    );
    const pipelineReduceLossCompute = ctx.device.createComputePipeline({
        label: 'reduce loss pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: moduleReduceLossCompute,
        }
    });

    const moduleAdamCompute = ctx.device.createShaderModule(
        moduleDescriptors.modules['adam pass']
    );
    const pipelineAdamCompute = ctx.device.createComputePipeline({
        label: 'adam compute pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: moduleAdamCompute,
        },
    });

    const moduleOpacityReset = ctx.device.createShaderModule(
        moduleDescriptors.modules['opacity reset']
    );
    const pipelineOpacityReset = ctx.device.createComputePipeline({
        label: 'opacity fix pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: moduleOpacityReset,
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
        size: [ctx.canvas.width, ctx.canvas.height],
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

    const paramBuffer = bufferManager.createBuffer(paramDesc, ctx.device);
    const splatCheckBuffer = bufferManager.createBuffer({
        ...paramDesc,
        label: 'splat check ping-pong buffer',
    }, ctx.device);
    const momentumPingPongBuffer = bufferManager.createBuffer(momentumPingPongDesc, ctx.device);


    const texture = ctx.device.createTexture({
        label: 'loaded image',
        format: 'rgba8unorm',
        size: [source.width, source.height],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    ctx.device.queue.copyExternalImageToTexture(
        { source, flipY: false },
        { texture },
        { width: source.width, height: source.height }
    );

    const input = new Float32Array(paramDesc.size);

    switch(initMethod) {
        case 'example':
            exampleSplatsRedBlueInit(paramDesc, input);
            break;
        case 'random':
            randomSplatInit(splatConfig.dynamic.activeSplats, paramDesc, input);
            break;
        case 'grid':
            imageGridSplats(
                splatConfig.dynamic.activeSplats, 
                source.width, 
                source.height, 
                paramDesc, 
                input);

    }

    ctx.device.queue.writeBuffer(paramBuffer, 0, input);

    // gradient buffer
    let gradDesc = gradStorageDesc;
    gradDesc.label = 'gradient buffer';
    const gradientBuffer = bufferManager.createBuffer(gradDesc, ctx.device);

    ctx.device.queue.writeBuffer(gradientBuffer, 0, new Float32Array(gradDesc.size));

    // loss result buffer (atomic uint array)
    const lossBuffer = bufferManager.createBuffer(lossBuffDesc, ctx.device);
    const lossV = new Float32Array(lossBuffDesc.size);
    ctx.device.queue.writeBuffer(lossBuffer, 0, lossV);

    const lossResultBuffer = ctx.device.createBuffer({
        label: 'loss result buffer',
        size: lossBuffDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const adamMemBuffer = bufferManager.createBuffer(adamMemoryDesc, ctx.device);

    // adam memory
    const adamMemV = new Float32Array(adamMemoryDesc.size);
    ctx.device.queue.writeBuffer(adamMemBuffer, 0, adamMemV, 0);

    // == uniform stuff for interaction ==

    const uniBuff = bufferManager.createBuffer(uniDesc, ctx.device);
    const uniV = new Float32Array(uniDesc.size);
    console.log(uniDesc.sizeBytes);
    const MACHINE_EPSILON = 1.19e-07; // this should be somewhere else
    uniV.set([
        0.05,
        0.9,
        0.999,
        MACHINE_EPSILON,
        splatConfig.refine.alphaCutoff, // alpha cutoff
        splatConfig.refine.gradientPosThresh, // positional gradient cutoff
        splatConfig.refine.gradientScaleThresh, // scale gradient cutoff
        splatConfig.refine.opacityResetV, // how diagnoal my splat can be
        splatConfig.refine.scaleDivideConst,
        0.0, // loss wieght L2 <-> L1
    ], 0);
    
    ctx.device.queue.writeBuffer(uniBuff, 0, uniV, 0);

    const updateUniform = (uniformName: string, value: number) => {
        const uniform = uniDesc.attributes.find(({ name }) => name === uniformName);
        if (!uniform) {
            console.error("could not find the uniform with name:", uniformName);
            return;
        }

        uniV[uniform.offset] = value;
        ctx.device.queue.writeBuffer(
            uniBuff,
            uniform.offsetBytes,
            new Float32Array([value]),
        );
    };

    // draw buffer for indirect draw
    const drawBuffer = ctx.device.createBuffer({
        size: drawArgsDesc.sizeBytes,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 
    });
    const drawArgs = new Uint32Array(drawArgsDesc.size);
    drawArgs[0] = 6; // vertex count
    drawArgs[1] = splatConfig.dynamic.activeSplats; // instance count
    drawArgs[2] = 0; // first vertex (offset)
    drawArgs[3] = 0; // first instance (offset)
    drawArgs[4] = splatConfig.dynamic.activeSplats; // workgroupX
    drawArgs[5] = 1; // workgroupY
    drawArgs[6] = 1; // workgroupZ
    drawArgs[7] = 0; // hacky padding
    

    ctx.device.queue.writeBuffer(drawBuffer, 0, drawArgs, 0);

        // for testing purposes
    const activeSplatsBuffRead = ctx.device.createBuffer({
        label: 'active splat read buffer',
        size: drawArgsDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // radix buffer setup

    const radixBuff = bufferManager.createBuffer(radixDesc, ctx.device);
    const radixCounters = new Uint32Array(radixDesc.attributes[1].offset);
    ctx.device.queue.writeBuffer(radixBuff, 0, radixCounters, 0);

    // for testing purposes
    const radixBuffRead = ctx.device.createBuffer({
        label: 'loss result buffer',
        size: radixDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    // == texture stuff



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
            { binding: 6, resource: adamMemBuffer},
            { binding: 7, resource: uniBuff },
            { binding: 8, resource: radixBuff },
            { binding: 9, resource: drawBuffer },
        ]
    });

    const bindGroupReduce = ctx.device.createBindGroup({
        label: 'bindGroup backwards pass',
        layout: pipelineReduceCompute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
            { binding: 1, resource: sampler },
            { binding: 2, resource: texture },
            { binding: 3, resource: lossBuffer },
            { binding: 4, resource: textureForward },
            { binding: 5, resource: gradientBuffer },
            { binding: 6, resource: adamMemBuffer},
            { binding: 7, resource: uniBuff },
            { binding: 8, resource: radixBuff },
            { binding: 9, resource: drawBuffer },
        ]
    });

    const bindGroupSplatCheck = ctx.device.createBindGroup({
        label: 'bindGroup splat check ping-pong',
        layout: pipelineSplatCheck1.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: splatCheckBuffer },
            { binding: 1, resource: momentumPingPongBuffer },
        ],
    });

    const bindGroupForward = ctx.device.createBindGroup({
        label: 'bindGroup forward',
        layout: pipeLineForward.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
        ]
    });
    

    const PARAMS_GUI = {
        blobs : false,
        lossWeight : 0.0,
        nSteps : splatConfig.loss.maxSteps,
        speed : 0, // copilot added
        refineSteps : splatConfig.dynamic.refineAfter,
        doRefine : splatConfig.dynamic.refineActive,
        opacitySteps: splatConfig.dynamic.opacityResetAfter,
        doOpacitiy : splatConfig.dynamic.opacityResetActive,
    }



    let readAndPrintBuff = async (
        buff: GPUBuffer, 
        TypedArrayConstructor: new (buffer: ArrayBuffer) => Float32Array | Uint32Array
    ) => {
        await buff.mapAsync(GPUMapMode.READ);
        const result = new TypedArrayConstructor(buff.getMappedRange());
        const resultData : number[] = []
        result.forEach((value : number) => {
            resultData.push(value);
        });
        console.log(resultData);
        buff.unmap();
    }

    let lossStepsReported = 0;     // copilot added

    let updateLossResults = async () => {
        console.log(lossPlot);
        if(!lossPlot)
            return;
        // read loss output
        await lossResultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(lossResultBuffer.getMappedRange());
        const newLosses = result.subarray(lossStepsReported, completedSteps);
        lossData.push(...newLosses);
        const lastStep = lossSteps.at(-1) ?? 0;
        lossSteps.push(...[...Array(newLosses.length).keys()].map((v) => v + lastStep));
        lossStepsReported = completedSteps;
        
        lossPlot.setData([lossSteps, lossData]);

        // unmap getMapped range is only valid buffer until we call unmap, the length will be set to 0
        lossResultBuffer.unmap();  
    }


    // render in texture
    ctx.renderPassDescriptor = renderPassDescriptorTexture;
    render(ctx, pipeLineForward, bindGroupForward, undefined, 6, splatConfig.dynamic.activeSplats, true, drawBuffer);

    // renders on screen
    ctx.renderPassDescriptor = renderPassDescriptorScreen;
    render(ctx, pipeLineForward, bindGroupForward, undefined, 6, splatConfig.dynamic.activeSplats, false, drawBuffer);

    let refineStep = 0;
    let opacityStep = 0;
    let completedSteps = 0;         // copilot added
    let runInProgress = false;      // copilot added
    let cancelGD = () => {};        // copilot added
    let updateGDButton = (running: boolean) => {};   // copilot added
    let runGD = async () => {
        if (runInProgress) {
            return;
        }

        if (completedSteps >= PARAMS_GUI.nSteps) {
            completedSteps = 0;
            lossStepsReported = 0;
            refineStep = 0;
            opacityStep = 0;
        }

        runInProgress = true;
        updateGDButton(true);
        const start = performance.now();
        console.log("start");

        const encodeStep = (encoder: GPUCommandEncoder) => {
            // reset the gradients, maybe do this in the shader and not here?
            encoder.clearBuffer(gradientBuffer, 0, gradDesc.sizeBytes);

            const pass = encoder.beginComputePass({
                label: 'simple backwards compute pass',
            });

            // radix setup
            pass.setPipeline(pipelineRadixSetup1);
            pass.setBindGroup(0, bindGroupBackwards);
            pass.dispatchWorkgroups(splatConfig.radixSetupDispatches);
            // again the same thing
            pass.setPipeline(pipelineRadixSetup2);
            pass.dispatchWorkgroups(splatConfig.radixSetupDispatches);

            // radix sort
            pipelineRadix.forEach((pipeline : GPUComputePipeline) => {
                pass.setPipeline(pipeline);
                pass.dispatchWorkgroups(1); // only one instance, this could be improved? with using storage buffers more?
            });

            // calculate chunk offsets
            pass.setPipeline(pipelineRadixOffset);
            pass.dispatchWorkgroups(splatConfig.groupChunkRatio);

            // backwards
            pass.setPipeline(pipelineBackwardsCompute);
            pass.dispatchWorkgroups(splatConfig.chunksX, splatConfig.chunksY);

            // // reduce
            pass.setPipeline(pipelineReduceCompute);
            pass.setBindGroup(0, bindGroupReduce);
            pass.dispatchWorkgroupsIndirect(drawBuffer, 16); // hacky but start at the workgroup offset
            // pass.dispatchWorkgroups(activeSplats);

            // reduce loss
            pass.setPipeline(pipelineReduceLossCompute);
            pass.setBindGroup(0, bindGroupBackwards);
            pass.dispatchWorkgroups(1);
            
            // adam
            pass.setPipeline(pipelineAdamCompute);
            pass.dispatchWorkgroups(splatConfig.radixSetupDispatches);
            pass.end();

            const passRender = encoder.beginRenderPass(renderPassDescriptorTexture);
            passRender.setPipeline(pipeLineForward);
            passRender.setBindGroup(0, bindGroupForward);
            passRender.drawIndirect(drawBuffer, 0);
            passRender.end();

            // ctx.renderPassDescriptor = renderPassDescriptorScreen;
            // render(ctx, pipeLineForward, bindGroupForward, undefined, 6, activeSplats, false, drawBuffer);
            opacityStep++;
            if(opacityStep >= PARAMS_GUI.opacitySteps && PARAMS_GUI.doOpacitiy) {
                console.log("opacity Reset");
                const passOpacity = encoder.beginComputePass({
                    label: 'refine step compute pass',
                });
                passOpacity.setPipeline(pipelineOpacityReset);
                passOpacity.setBindGroup(0, bindGroupBackwards);
                passOpacity.dispatchWorkgroups(splatConfig.radixSetupDispatches);
                passOpacity.end();
                opacityStep = 0;
            }

            refineStep++;
            if(refineStep >= PARAMS_GUI.refineSteps && PARAMS_GUI.doRefine) {
                console.log("refine")
                // pass that marks all splats outside of viewport as dead + marks all splats with high positional variance (should still be in the gradientBuffer)
                const passRefine = encoder.beginComputePass({
                    label: 'refine step compute pass',
                });
                passRefine.setPipeline(pipelineSplatCheck1);
                passRefine.setBindGroup(0, bindGroupBackwards);
                passRefine.setBindGroup(1, bindGroupSplatCheck);
                passRefine.dispatchWorkgroups(splatConfig.radixSetupDispatches);
                // again the same thing
                passRefine.setPipeline(pipelineSplatCheck2);
                passRefine.dispatchWorkgroups(splatConfig.radixSetupDispatches);
                passRefine.end();
                refineStep = 0;
            }
        };

        let lossBufferCleared = completedSteps > 0;
        // ===
        // copilot added "submitSteps"
        // ===
        const submitSteps = (stepCount: number) => {
            const encoder = ctx.device.createCommandEncoder({
                label: 'gd encoder',
            });
            if (!lossBufferCleared) {
                encoder.clearBuffer(lossBuffer, 0, lossBuffDesc.sizeBytes);
                lossBufferCleared = true;
            }
            for(let i = 0; i < stepCount; i++) {
                encodeStep(encoder);
            }
            ctx.device.queue.submit([encoder.finish()]);
        };

        let readbackInProgress = false;
        // ===
        // copilot added "finishRun"
        // ===
        const finishRun = async (isFinalRun: boolean) => {
            if (readbackInProgress) {
                return;
            }
            readbackInProgress = true;
            const readbackEncoder = ctx.device.createCommandEncoder({
                label: 'gd readback encoder',
            });
            readbackEncoder.copyBufferToBuffer(lossBuffer, 0, lossResultBuffer, 0, lossResultBuffer.size);
            readbackEncoder.copyBufferToBuffer(radixBuff, 0, radixBuffRead, 0, radixBuffRead.size);
            readbackEncoder.copyBufferToBuffer(drawBuffer, 0, activeSplatsBuffRead, 0, activeSplatsBuffRead.size);
            ctx.device.queue.submit([readbackEncoder.finish()]);
            await ctx.device.queue.onSubmittedWorkDone();
            await updateLossResults();
            await readAndPrintBuff(activeSplatsBuffRead, Uint32Array);
            console.log("elapsed time:", (performance.now() - start) / 1000.0);
            readbackInProgress = false;
            if (isFinalRun) {
                runInProgress = false;
                updateGDButton(false);
            }
        };

        if (PARAMS_GUI.speed === 0) {
            submitSteps(PARAMS_GUI.nSteps - completedSteps);
            completedSteps = PARAMS_GUI.nSteps;
            await finishRun(true);
            ctx.renderPassDescriptor = renderPassDescriptorScreen;
            render(ctx, PARAMS_GUI.blobs ? pipeLineForwardBlob : pipeLineForward, bindGroupForward, undefined, 6, splatConfig.dynamic.activeSplats, false, drawBuffer);
            return;
        }

        let lastFrameTime = performance.now();
        let lastFinishTime = lastFrameTime;
        let stepAccumulator = 0;
        let animationFrameId: number | undefined;
        // ===
        // copilot added "scheduleFrame"
        // ===
        const scheduleFrame = (frameTime: number) => {
            if (completedSteps >= PARAMS_GUI.nSteps) {
                runInProgress = false;
                updateGDButton(false);
                return;
            }

            const elapsedSeconds = Math.max(0, frameTime - lastFrameTime) / 1000;
            lastFrameTime = frameTime;
            const stepsThisFrame = PARAMS_GUI.speed === 1
                ? 1
                : (() => {
                    stepAccumulator += elapsedSeconds * Math.pow(FPS, 2 - PARAMS_GUI.speed);
                    const steps = Math.floor(stepAccumulator);
                    stepAccumulator -= steps;
                    return steps;
                })();

            const remainingSteps = PARAMS_GUI.nSteps - completedSteps;
            const stepCount = Math.min(remainingSteps, stepsThisFrame);
            if (stepCount > 0) {
                submitSteps(stepCount);
                completedSteps += stepCount;
                ctx.renderPassDescriptor = renderPassDescriptorScreen;
                render(ctx, PARAMS_GUI.blobs ? pipeLineForwardBlob : pipeLineForward, bindGroupForward, undefined, 6, splatConfig.dynamic.activeSplats, false, drawBuffer);
                if (completedSteps < PARAMS_GUI.nSteps && frameTime - lastFinishTime >= 1000) {
                    lastFinishTime = frameTime;
                    void finishRun(false);
                }
            }
            animationFrameId = requestAnimationFrame(scheduleFrame);
        };

        cancelGD = () => {
            if (animationFrameId !== undefined) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = undefined;
            }
            runInProgress = false;
            updateGDButton(false);
        };
        animationFrameId = requestAnimationFrame(scheduleFrame);
    }
    // == interactive suff, not really needed
    {
        
        const blobsBinding = pane.addBinding(PARAMS_GUI, 'blobs', {
            label: 'show blobs',
        });
        blobsBinding.hidden = !guiVisibility.blobs;
        blobsBinding.on('change', (ev) => {
            ctx.renderPassDescriptor = renderPassDescriptorScreen;
            render(ctx, PARAMS_GUI.blobs ? pipeLineForwardBlob : pipeLineForward, bindGroupForward, undefined, 6, splatConfig.dynamic.activeSplats, false, drawBuffer);
        });

        const lossWeightBinding = pane.addBinding(PARAMS_GUI, 'lossWeight', {
            label: 'L2 <-> L1',
            min: 0.0,
            max: 1.0,
            step: 0.1,
        });
        lossWeightBinding.hidden = !guiVisibility.lossWeight;

        const nStepsBinding = pane.addBinding(PARAMS_GUI, 'nSteps', {
            label: 'gd steps',
            min: 1,
            max: splatConfig.loss.maxSteps,
            step: 10,
        });
        nStepsBinding.hidden = !guiVisibility.nSteps;

        const speedBinding = pane.addBinding(PARAMS_GUI, 'speed', {
            label: 'gd speed',
            min: 0,
            max: 2,
            step: 0.1,
        });
        speedBinding.hidden = !guiVisibility.speed;

        const refinePane = pane.addFolder({
            title: 'refinement'
        });
        refinePane.hidden = !guiVisibility.refinement;

        const refineSlider = refinePane.addBinding(PARAMS_GUI, 'refineSteps', {
            label: 'refine after:',
            min: 1,
            max: 1000,
            step: 10,
        });

        refinePane.addBinding(PARAMS_GUI, "doRefine", {
            label: 'with refine'
        }).on('change', (ev) => {
            refineSlider.disabled = !ev.value;
        });

        const opacitiySlider = refinePane.addBinding(PARAMS_GUI, 'opacitySteps', {
            label: 'opacity after:',
            min: 1,
            max: 100,
        });

        refinePane.addBinding(PARAMS_GUI, "doOpacitiy", {
            label: 'with opacity refine'
        }).on('change', (ev) => {
            opacitiySlider.disabled = !ev.value;
        });

        const refineParams = refinePane.addFolder({
            title: 'refinment params',
            expanded: false,
        });
        refineParams.hidden = !guiVisibility.refinementParams;

        const refineUniformParams = {
            alphaCutOff: splatConfig.refine.alphaCutoff,
            posGradThresh: splatConfig.refine.gradientPosThresh,
            scaleGradThresh: splatConfig.refine.gradientScaleThresh,
            opacityReset: splatConfig.refine.opacityResetV,
            scaleDivideConst: splatConfig.refine.scaleDivideConst,
        };

        refineParams.addBinding(refineUniformParams, 'alphaCutOff', {
            step: splatConfig.refine.alphaCutoff / 10,
        }).on('change', (ev) => {
            updateUniform('alphaCutOff', ev.value);
        });

        refineParams.addBinding(refineUniformParams, 'posGradThresh', {
            step: splatConfig.refine.gradientPosThresh / 10,
        }).on('change', (ev) => {
            updateUniform('gradientPosVari', ev.value);
        });

        refineParams.addBinding(refineUniformParams, 'scaleGradThresh', {
            step: splatConfig.refine.gradientScaleThresh / 10,
        }).on('change', (ev) => {
            updateUniform('gradinetScalVari', ev.value);
        });

        refineParams.addBinding(refineUniformParams, 'opacityReset', {
            step: splatConfig.refine.opacityResetV / 10,
        }).on('change', (ev) => {
            updateUniform('opacityReset', ev.value);
        });

        refineParams.addBinding(refineUniformParams, 'scaleDivideConst', {
            step: splatConfig.refine.scaleDivideConst / 10,
        }).on('change', (ev) => {
            updateUniform('scaleDivideConst', ev.value);
        });

        const gdButton = pane.addButton({
            title: 'gradient descent',
            label: 'step'
        });
        gdButton.hidden = !guiVisibility.gradientDescent;
        updateGDButton = (running: boolean) => {
            gdButton.title = running ? 'stop gradient descent' : 'gradient descent';
            gdButton.label = running ? 'stop' : 'step';
        };
        gdButton.on('click', async () => {
            if (runInProgress) {
                cancelGD();
            } else {
                await runGD();
            }
        });
    }

    ctx.device.lost.then((info) => {
        console.error("Device lost lmao, reason:", info.reason, info.message);
    });
    
}


