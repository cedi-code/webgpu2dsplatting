import { Pane } from 'tweakpane';

import { vec2, vec3, type Vec2 } from "wgpu-matrix";

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { type ArrayType, type UniformBufferDescriptor } from '../../../mytypes/BufferDescriptors';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import { parseParams, type FlatParams, createLossPlot } from '../../../myutils/LogHelpers';

import { generateAtomicOperations } from '../../../myutils/ShaderTemplating';

import shaderCodeCompute from '../shaders/gradientBackwardsCompute.wgsl?raw';
import shaderGaussFunctions from '../../../shaders/gaussFunctions.wgsl?raw';
import gaussTileVertStorage from '../shaders/gaussTileVertStorage.wgsl?raw';
import gaussFrag from '../shaders/gaussFragBlob.wgsl?raw';
import adamShad from '../../../shaders/adam.wgsl?raw';
import adamCompute from '../shaders/adamOnGradients.wgsl?raw';
import shaderGradTypes from '../../../shaders/gradTypes.wgsl?raw';
import radixSetup from '../shaders/radixSetup.wgsl?raw';
import radixCompute from '../shaders/radixSort.wgsl?raw';
import radixTypes from '../../../shaders/radixTypes.wgsl?raw'
import reduceRadixCompute from '../shaders/reduceGradientsRadix.wgsl?raw';
import reduceLossCompute from '../shaders/reduceLoss.wgsl?raw';
import radixOffsets from '../shaders/radixChunkOffsets.wgsl?raw';

import splatRefine from '../shaders/splatRefine.wgsl?raw';


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

const canvasId = "main"
const canvasElem = (document.getElementById('main') as HTMLCanvasElement);
const canvasDim = {
    width : canvasElem.width, 
    height : canvasElem.height,
};

async function main() {

    const ctx = await getWebGPUctx({ canvasId: canvasId});
    if(!ctx) {
        return;
    }

    // number of splats
    const numSplats = 4000; // hard limit is 1 splat per pixel?
    const activeSplats = 1000;
    const sampleDim = 512;
    const chunkWidth = 16;
    const chunkHeight = 16;
    const chunksX = Math.floor(((sampleDim + chunkWidth - 1)  / chunkWidth));
    const chunksY = Math.floor(((sampleDim + chunkHeight - 1)  / chunkHeight));
    const numChunks = chunksX * chunksY;

    const maxSteps = 100;
    const refineAfter = 200;

    const gradientPosThresh = 0.0002;
    const gradientScaleThresh = 0.0002;
    const alphaCutoff = 0.2;

    const radixElements = 3; // splat id, chunk id, permutation id
    const groupSizeRadix = 256;
    const barriersPrefixSum = 8; // log2(groupSizeRadix)
    const radixSetupDispatches = Math.floor((numSplats + groupSizeRadix - 1) / groupSizeRadix);
    const maxSplatsPerChunk = Math.min(numSplats, 256);

    // logb(a) = logc(a) / locc(b)
    const numRadixSorts = 1 + Math.floor(Math.log(numChunks) / Math.log(groupSizeRadix));
    const groupChunkRatio = 1 + Math.floor(numChunks / groupSizeRadix)

    // constants
    const CONSTANTS = {
        NUM_SPLATS : numSplats,
        xRAY : false,
        sampleDim : sampleDim,
        EPSILON : MACHINE_EPSILON,
        chunkWidth: chunkWidth,
        chunkHeight: chunkHeight,
        numChunksW : chunksX,
        numChunksH : chunksY,
        nSteps : maxSteps,
        TILE_SCALE: 4,
        MAX_SPLATS_CHUNK : maxSplatsPerChunk,
        // 2 because it says which splat id first and then the "chunk/box id", depth is the splat id
        MAX_SLOTS: numChunks * radixElements * maxSplatsPerChunk,
        RADIX_ELEMS: radixElements, 
        R_CHUNK_INDEX: 0,
        R_SPLAT_INDEX: 1,
        R_TRANSPOSE_INDEX: 2,
        GROUPSIZE: groupSizeRadix,
        BARRIERS: barriersPrefixSum,
        NUM_DISPATCHES : radixSetupDispatches,
        NUM_SORTS : numRadixSorts,
        OFF_CANVAS_TRHESHOLD : 1.001, // distance of canvas border 1.0 means canvas border
        SCALE_DIVIDE_CONST : 1.6, // this is from the paper?
        OPACITY_PERCENTAGE : 0.6,
    };

    console.table(CONSTANTS);

    const sharedConstants = Object.entries(CONSTANTS)
    .map(([k, v]) => `const ${k} = ${v};`)
    .join('\n');

    const csBackwardsModule = ctx.device.createShaderModule({
        label: 'backwards pass',
        code: (
            sharedConstants +
            shaderGradTypes +
            radixTypes +
            generateAtomicOperations() + 
            shaderGaussFunctions + 
            adamShad +
            shaderCodeCompute
        )
    });

    const adamModule = ctx.device.createShaderModule({
        label: 'adam pass',
        code: (
            sharedConstants + 
            shaderGradTypes + 
            radixTypes +
            generateAtomicOperations() + 
            adamShad  + 
            adamCompute
        ) 
    });

    const radixSetupModule = ctx.device.createShaderModule({
        label: 'radix setup pass',
        code: (
            sharedConstants + 
            shaderGradTypes + 
            radixTypes +
            shaderGaussFunctions +
            radixSetup
        ) 
    });

    const splatCheckModule = ctx.device.createShaderModule({
        label: 'radix setup pass',
        code: (
            sharedConstants +
            shaderGradTypes +
            radixTypes +
            shaderGaussFunctions +
            adamShad + 
            splatRefine
        )
    });

    const radixModule = ctx.device.createShaderModule({
        label: 'radix pass',
        code: (
            radixTypes +
            sharedConstants +
            radixCompute
        )
    });

    const radixOffsetModule = ctx.device.createShaderModule({
        label: 'radix pass',
        code: (
            radixTypes +
            sharedConstants +
            radixOffsets
        )
    });

    const reduceRadixModule = ctx.device.createShaderModule({
        label: 'reduce pass',
        code: (
            sharedConstants + 
            shaderGradTypes + 
            radixTypes +
            adamShad +
            reduceRadixCompute
        ) 
    });

    const reduceLossModule = ctx.device.createShaderModule({
        label: 'loss reduce',
        code: (
            sharedConstants +
            shaderGradTypes + 
            adamShad +
            reduceLossCompute 
        )
    })
    
    const vsModule = ctx.device.createShaderModule({
        label: '2d static tile',
        code: (
            sharedConstants +
            shaderGradTypes +
            shaderGaussFunctions + 
            gaussTileVertStorage
        ),
    });




    const fsModule = ctx.device.createShaderModule({
        label: 'simple texture frag impl',
        code: (sharedConstants + gaussFrag) 
    });


    const paramBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', numSplats);
    paramBuilder.add('pos', "vec2f")
                .add('scale', "vec2f")
                .add('rot', "f32")
                .add('color', "vec3f")
                .add('alpha', "f32");
    const paramDesc = paramBuilder.build();

    const accumulatedGradStorageBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', numSplats);
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
    const gradientStorageBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', CONSTANTS.MAX_SLOTS );
    gradientStorageBuilder.add('pos', "vec2f")
                        .add('scale', "vec2f")
                        .add('rot', "f32")
                        .add('color', "vec3f")
                        .add('alpha', "f32");
    const gradStorageDesc = gradientStorageBuilder.build();

    // reduce uniform buffer
    const reduceBuilder = new UniformBufferDescriptorBuilder('reduce uniform', "uniform");
    const reduceUniDesc = reduceBuilder.add('stride', "u32").build();

    const lossBuilder = new UniformBufferDescriptorBuilder('loss storage buffer', 'storage', 'copy_src_dst', CONSTANTS.nSteps * numChunks);
    lossBuilder.add('loss', "f32");
    const lossBuffDesc = lossBuilder.build();
    console.log("loss array size:", CONSTANTS.nSteps * numChunks);

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
    radixBuild.add("splatOffsets", { type : "u32", size : CONSTANTS.NUM_DISPATCHES })
              .add("offsets", { type : "u32", size: numChunks })
              .add("data", { type : "u32", size: CONSTANTS.MAX_SLOTS })
              .add("data2", { type : "u32", size: CONSTANTS.MAX_SLOTS })
              .add("splatOffsetsTotal", { type : "u32", size : CONSTANTS.NUM_SPLATS });

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
        bindGroupLayouts: [ bindGroupLayoutDescriptorsForward ],
    });

    const pipelineLayoutCompute = ctx.device.createPipelineLayout({
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
    const pipelineForwardDescriptor : GPURenderPipelineDescriptor = {
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

    const pipelineRadixSetup1 = ctx.device.createComputePipeline({
        label: 'radix setup pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: radixSetupModule,
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
            module: radixSetupModule,
            entryPoint: 'registerSplats',
            constants: {
                first_pass: 0,
            }
        },
    });

    const pipelineSplatCheck1 = ctx.device.createComputePipeline({
        label: 'splat check pipelie',
        layout: pipelineLayoutSplatCheck,
        compute: {
            module: splatCheckModule,
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
            module: splatCheckModule,
            entryPoint: 'cs',
            constants: {
                first_pass: 0,
            }
        },
    });

    const pipelineRadix : GPUComputePipeline[] = []
    for(let i : number = 0; i < CONSTANTS.NUM_SORTS; i++) {
        pipelineRadix.push(
            ctx.device.createComputePipeline({
                label: 'radix compute pipeline',
                layout: pipelineLayoutCompute,
                compute: {
                    module: radixModule,
                    entryPoint: 'radixSort',
                    constants: {
                        pass_nr : i
                    }
                },
            })
        );
    }


    const pipelineRadixOffset = ctx.device.createComputePipeline({
        label: 'radix offset compute pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: radixOffsetModule,
        },
    });

    const pipelineBackwardsCompute = ctx.device.createComputePipeline({
        label: 'gd compute pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: csBackwardsModule,
        },
    });

    const pipelineReduceCompute  = ctx.device.createComputePipeline({
        label: 'reduce gradients compute pipeline',
        layout: pipelineLayoutComputeReduceLayout,
        compute: {
            module: reduceRadixModule,
        },
    }); 

    const pipelineReduceLossCompute = ctx.device.createComputePipeline({
        label: 'reduce loss pipeline',
        layout: pipelineLayoutCompute,
        compute: {
            module: reduceLossModule,
        }
    });

    const pipelineAdamCompute = ctx.device.createComputePipeline({
        label: 'adam compute pipeline',
        layout: pipelineLayoutCompute,
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
        size: [canvasDim.width, canvasDim.height],
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
    const splatCheckBuffer = bufferManager.createBuffer({
        ...paramDesc,
        label: 'splat check ping-pong buffer',
    });

    const momentumPingPongBuffer = bufferManager.createBuffer(momentumPingPongDesc);

    const testImageUrl = 'assets/jeff.jpg';
    const source = await loadImageBitmap(testImageUrl);
    const texture = ctx.device.createTexture({
        label: testImageUrl,
        format: 'rgba8unorm',
        size: [source.width, source.height],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    ctx.device.queue.copyExternalImageToTexture(
        { source, flipY: false },
        { texture },
        { width: source.width, height: source.height }
    );

    const posOff    = paramDesc.attributes[0].offset;
    const scaleOff  = paramDesc.attributes[1].offset;
    const rotOff    = paramDesc.attributes[2].offset;
    const colorOff  = paramDesc.attributes[3].offset;
    const alphaOff  = paramDesc.attributes[4].offset;

    const input = new Float32Array(paramDesc.size);
    const unitS = paramDesc.unitSize!;
    let nextUnit = unitS;

    let minimumS = 2;
    let randomSplatInit = (n : number) => {
        for(let i = 0; i < n; i++) {
            let rndPos = vec2.clamp(vec2.create(Math.random(), Math.random()), 0.1, 0.9);
            input.set(rndPos, posOff + unitS * i);

            let rndScale = vec2.add(
                vec2.create(minimumS,minimumS), 
                vec2.scale(vec2.create(Math.random(), Math.random()), 2.0)
            );
            input.set(rndScale, scaleOff + unitS * i);

            let rndRot = Math.PI * Math.random();
            input.set([rndRot], rotOff + unitS * i);

            let rndColor = vec3.add(
                vec3.create(1),
                vec3.scale(vec3.create(Math.random(), Math.random(), Math.random()), 9.0)
            );
            input.set(rndColor, colorOff + unitS * i);

            let rndAlpha = 4.0 + Math.random() * 4.0;
            input.set([rndAlpha], alphaOff + unitS * i);
        }
    }

    function imageGridSplats(
        n: number,
        width : number,
        height : number,
    ) {

        const cols = Math.ceil(Math.sqrt(n * width / height));
        const rows = Math.ceil(n / cols);
        const scaleX = width / cols;
        const scaleY = height / rows;
        const logScaleX = Math.log(3000 / scaleX);
        const logScaleY = Math.log(3000 / scaleY);

        for (let i = 0; i < n; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);

            const x0 = Math.floor(col * width / cols);
            const x1 = Math.floor((col + 1) * width / cols);
            const y0 = Math.floor(row * height / rows);
            const y1 = Math.floor((row + 1) * height / rows);

            input.set([
                (col + 0.5) / cols,
                (row + 0.5) / rows,
            ], posOff + unitS * i);

            input.set(
                [logScaleX, logScaleY],
                scaleOff + unitS * i
            );

            input.set([0], rotOff + unitS * i);

            input.set([
                 8,
                 8,
                 8,
            ], colorOff + unitS * i);

            input.set([4], alphaOff + unitS * i);
        }
    }

    imageGridSplats(activeSplats, source.width, source.height);

    // randomSplatInit(activeSplats);

    // // gauss 1
    // input.set([0.9, 0.3],       posOff);
    // input.set([1.5, 1.5],       scaleOff);
    // input.set([0.0],            rotOff);
    // input.set([8.0, 1.0, 1.0],  colorOff);
    // input.set([4.0],            alphaOff);

    // // gauss 2
    // input.set([0.3, 0.6],       nextUnit + posOff);
    // input.set([0.7, 1.0],       nextUnit + scaleOff);
    // input.set([0.0],            nextUnit + rotOff);
    // input.set([1.0, 1.0, 8.0],  nextUnit + colorOff);
    // input.set([4.0],            nextUnit + alphaOff);

    const prtyPrint = (data : Float32Array) => {
        const result : FlatParams[] = []
        for(let i = 0; i < Math.min(activeSplats, 30); i++) {
            const splatData = data.subarray(unitS*i, unitS*(i+1));
            result.push(parseParams(splatData, paramDesc));
        }
        console.table(result);
    }
    prtyPrint(input);

    ctx.device.queue.writeBuffer(paramBuffer, 0, input);

    // gradient buffer
    let gradDesc = gradStorageDesc;
    gradDesc.label = 'gradient buffer';
    const gradientBuffer = bufferManager.createBuffer(gradDesc);

    ctx.device.queue.writeBuffer(gradientBuffer, 0, new Float32Array(gradDesc.size));

    // loss result buffer (atomic uint array)
    const lossBuffer = bufferManager.createBuffer(lossBuffDesc);
    const lossV = new Float32Array(lossBuffDesc.size);
    ctx.device.queue.writeBuffer(lossBuffer, 0, lossV);

    const lossResultBuffer = ctx.device.createBuffer({
        label: 'loss result buffer',
        size: lossBuffDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const adamMemBuffer = bufferManager.createBuffer(adamMemoryDesc);

    // adam memory
    const adamMemV = new Float32Array(adamMemoryDesc.size);
    ctx.device.queue.writeBuffer(adamMemBuffer, 0, adamMemV, 0);

    // == uniform stuff for interaction ==

    const uniBuff = bufferManager.createBuffer(uniDesc);
    const uniV = new Float32Array(uniDesc.size);
    console.log(uniDesc.sizeBytes);
    uniV.set([
        0.05,
        0.9,
        0.999,
        MACHINE_EPSILON,
        alphaCutoff, // alpha cutoff
        gradientPosThresh, // positional gradient cutoff 
        gradientScaleThresh, // scale gradient cutoff
        0.0
    ], 0);
    
    ctx.device.queue.writeBuffer(uniBuff, 0, uniV, 0);

    // draw buffer for indirect draw
    const drawBuffer = ctx.device.createBuffer({
        size: drawArgsDesc.sizeBytes,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 
    });
    const drawArgs = new Uint32Array(drawArgsDesc.size);
    drawArgs[0] = 6; // vertex count
    drawArgs[1] = activeSplats; // instance count
    drawArgs[2] = 0; // first vertex (offset)
    drawArgs[3] = 0; // first instance (offset)
    drawArgs[4] = activeSplats; // workgroupX
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

    const radixBuff = bufferManager.createBuffer(radixDesc);
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
        nSteps : maxSteps,
        refineSteps : refineAfter,
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

    let updateLossResults = async () => {

        // read loss output
        await lossResultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(lossResultBuffer.getMappedRange());
        lossData.push(...result.subarray(0, PARAMS_GUI.nSteps));
        const lastStep = lossSteps.at(-1) ?? 0;
        lossSteps.push(...[...Array(PARAMS_GUI.nSteps).keys()].map((v) => v + lastStep));
        
        lossPlot.setData([lossSteps, lossData]);

        // unmap getMapped range is only valid buffer until we call unmap, the length will be set to 0
        lossResultBuffer.unmap();  
    }


    // render in texture
    ctx.renderPassDescriptor = renderPassDescriptorTexture;
    render(ctx, pipeLineForward, bindGroupForward, undefined, 6, activeSplats, true, drawBuffer);

    // renders on screen
    ctx.renderPassDescriptor = renderPassDescriptorScreen;
    render(ctx, pipeLineForward, bindGroupForward, undefined, 6, activeSplats, false, drawBuffer);

    let refineStep = 0;
    let runGD = async () => {
        const start = performance.now();
        console.log("start");

        const encoder = ctx.device.createCommandEncoder({
            label: 'gd encoder',
        });
        encoder.clearBuffer(lossBuffer, 0, lossBuffDesc.sizeBytes);

        for(let i = 0; i < PARAMS_GUI.nSteps; i++) {

            // reset the gradients, maybe do this in the shader and not here?
            encoder.clearBuffer(gradientBuffer, 0, gradDesc.sizeBytes);

            const pass = encoder.beginComputePass({
                label: 'simple backwards compute pass',
            });

            // radix setup
            pass.setPipeline(pipelineRadixSetup1);
            pass.setBindGroup(0, bindGroupBackwards);
            pass.dispatchWorkgroups(CONSTANTS.NUM_DISPATCHES);
            // again the same thing
            pass.setPipeline(pipelineRadixSetup2);
            pass.dispatchWorkgroups(CONSTANTS.NUM_DISPATCHES);

            // radix sort
            pipelineRadix.forEach((pipeline : GPUComputePipeline) => {
                pass.setPipeline(pipeline);
                pass.dispatchWorkgroups(1); // only one instance, this could be improved? with using storage buffers more?
            });

            // calculate chunk offsets
            pass.setPipeline(pipelineRadixOffset);
            pass.dispatchWorkgroups(groupChunkRatio);

            // backwards
            pass.setPipeline(pipelineBackwardsCompute);
            pass.dispatchWorkgroups(chunksX, chunksY);

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
            pass.dispatchWorkgroups(CONSTANTS.NUM_DISPATCHES);

            pass.end();
            
            // forward pass
            const passRender = encoder.beginRenderPass(renderPassDescriptorTexture);
            passRender.setPipeline(pipeLineForward);
            passRender.setBindGroup(0, bindGroupForward);
            passRender.drawIndirect(drawBuffer, 0);

            passRender.end();

            ctx.renderPassDescriptor = renderPassDescriptorScreen;
            render(ctx, pipeLineForward, bindGroupForward, undefined, 6, activeSplats, false, drawBuffer);
            refineStep++;
            if(refineStep >= PARAMS_GUI.refineSteps) {
                console.log("refine")
                // pass that marks all splats outside of viewport as dead + marks all splats with high positional variance (should still be in the gradientBuffer)
                const passRefine = encoder.beginComputePass({
                    label: 'refine step compute pass',
                });
                passRefine.setPipeline(pipelineSplatCheck1);
                passRefine.setBindGroup(0, bindGroupBackwards);
                passRefine.setBindGroup(1, bindGroupSplatCheck);
                passRefine.dispatchWorkgroups(CONSTANTS.NUM_DISPATCHES);
                // again the same thing
                passRefine.setPipeline(pipelineSplatCheck2);
                passRefine.dispatchWorkgroups(CONSTANTS.NUM_DISPATCHES);

                passRefine.end();
                refineStep = 0;
            }
        }
        
        // copy buffer to cpu (expensive!)
        encoder.copyBufferToBuffer(lossBuffer, 0, lossResultBuffer, 0, lossResultBuffer.size);
        encoder.copyBufferToBuffer(radixBuff, 0, radixBuffRead, 0, radixBuffRead.size);
        encoder.copyBufferToBuffer(drawBuffer, 0, activeSplatsBuffRead, 0, activeSplatsBuffRead.size);
        // run the work lmao
        const commandBuffer = encoder.finish();
        ctx.device.queue.submit([commandBuffer]);
        
        // display on canvas
        ctx.renderPassDescriptor = renderPassDescriptorScreen;
        
        render(ctx, PARAMS_GUI.blobs ? pipeLineForwardBlob : pipeLineForward, bindGroupForward, undefined, 6, activeSplats, false, drawBuffer);

        await ctx.device.queue.onSubmittedWorkDone();

        let elapsed = performance.now() - start;
        console.log("elapsed time:", elapsed / 1000.0);

        // get loss
        await updateLossResults();   
        
        // debug
        readAndPrintBuff(activeSplatsBuffRead, Uint32Array);
        
    }
    // == interactive suff, not really needed
    {
        
        const pane = new Pane({
            container: document.getElementById("gd-sliders") as HTMLElement,
        });



        pane.addBinding(PARAMS_GUI, 'blobs', {
            label: 'show blobs',
        }).on('change', (ev) => {
            let showBlob = ev.value;
            ctx.renderPassDescriptor = renderPassDescriptorScreen;
            render(ctx, PARAMS_GUI.blobs ? pipeLineForwardBlob : pipeLineForward, bindGroupForward, undefined, 6, activeSplats, false, drawBuffer);

        });

        pane.addBinding(PARAMS_GUI, 'lossWeight', {
            label: 'L2 <-> L1',
            min: 0.0,
            max: 1.0,
            step: 0.1,
        });

        pane.addBinding(PARAMS_GUI, 'nSteps', {
            label: 'gd steps',
            min: 1,
            max: CONSTANTS.nSteps,
            step: 1,
        });

        pane.addBinding(PARAMS_GUI, 'refineSteps', {
            label: 'refine after:',
            min: 1,
            max: 1000,
        });



        pane.addButton({
            title: 'gradient descent',
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