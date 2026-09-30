import { Pane } from 'tweakpane';

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { type UniformBufferDescriptor } from '../../../mytypes/BufferDescriptors';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import { createLossPlot } from '../../../myutils/LogHelpers';


import shaderCodeCompute from '../shaders/gradientDescent2d.wgsl?raw';
import shaderGaussFunctions from '../../../shaders/gaussFunctions.wgsl?raw';
import gaussTileVertStorage from '../shaders/gaussTileVertStorage.wgsl?raw';
import gaussFrag from '../shaders/gaussFrag.wgsl?raw';
import adamShad from '../../../shaders/adam.wgsl?raw';
import shaderGradTypes from '../../../shaders/gradTypes.wgsl?raw';


async function loadImageBitmap(url : string) : Promise<ImageBitmap> {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: 'none'});
}


let lossData : number[]  = []
let lossSteps: number[] = []
let plotHTMLElem = document.getElementById('loss-plot-color') ?? document.body;
let lossPlot = createLossPlot(plotHTMLElem);

const MACHINE_EPSILON = 1.19e-07;


async function main() {

    const ctx = await getWebGPUctx({ canvasId: "color-alpha-2"});
    if(!ctx) {
        return;
    }

    const csModule = ctx.device.createShaderModule({
        label: '2d gs module',
        code: (shaderGradTypes + adamShad + shaderGaussFunctions + shaderCodeCompute) 
    });
    
    const vsModule = ctx.device.createShaderModule({
        label: '2d static tile',
        code: shaderGradTypes + shaderGaussFunctions + gaussTileVertStorage,
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
            { // lossOutput
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: lossBuffDesc.sizeBytes,
            },
            },
            {
            binding: 5,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'storage',
                minBindingSize: adamMemoryDesc.sizeBytes,
            },
            },
            {
            binding: 6, // forward texture
            visibility: GPUShaderStage.COMPUTE,
            texture: {
                    sampleType: "float", // type for 'rgba8unorm'
                    viewDimension: "2d",
                    multisampled: false,
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

    const PARAMS = {
        pos1: { x: 0.9, y: 0.3 },
        scale1: { x: 0.24, y: 0.24 },
        rot1: 0.0,
        col1: { r: 255, g: 30 , b: 30, a: 0.5 },

        pos2: { x: 0.3, y: 0.6 },
        scale2: { x: 0.35, y: 0.35 },
        rot2: 0.0,
        col2: { r: 30, g: 30 , b: 255, a: 0.5 },
    };

    let rgbToColorArr = (param : {r : number, g : number, b : number, a : number}) => {
        return [
            param.r * 8.0 / 255.0,
            param.g * 8.0 / 255.0,
            param.b * 8.0 / 255.0,
        ];
    };

    let rgbToAlphArr = (param : {r : number, g : number, b : number, a : number}) => {
        return [
            param.a * 8.0,
        ];
    };

    let scaleToArr = (param : { x: number, y: number}) => {
        return [-Math.log(param.x + MACHINE_EPSILON), -Math.log(param.y + MACHINE_EPSILON)];
    };
    

    let resetInput = () => {
        // gauss 1
        input.set([PARAMS.pos1.x, PARAMS.pos1.y], posOff);
        input.set(scaleToArr(PARAMS.scale1), scaleOff);
        input.set([PARAMS.rot1], rotOff);
        input.set(rgbToColorArr(PARAMS.col1), colorOff);
        input.set(rgbToAlphArr(PARAMS.col1), alphaOff);

        // gauss 2
        input.set([PARAMS.pos2.x, PARAMS.pos2.y], posOff + nextUnit);
        input.set(scaleToArr(PARAMS.scale2), scaleOff + nextUnit);
        input.set([PARAMS.rot2], rotOff + nextUnit);
        input.set(rgbToColorArr(PARAMS.col2), colorOff + nextUnit);
        input.set(rgbToAlphArr(PARAMS.col2), alphaOff + nextUnit);

        ctx.device.queue.writeBuffer(paramBuffer, 0, input);

        lossData = [];
        lossSteps = [];
    };

    // set default
    resetInput();



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

    const testImageUrl = 'assets/overlappImage.jpg';
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
            { binding: 4, resource: lossBuffer },
            { binding: 5, resource: adamMemBuffer},
            { binding: 6, resource: textureForward },
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

    let updateResults = async () : Promise<Float32Array<ArrayBuffer>> => {
        // read results
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(resultBuffer.getMappedRange());
        
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

    let renderScreen = () => {
        ctx.renderPassDescriptor = renderPassDescriptorScreen;
        render(ctx, pipeLineForward, bindGroupForward, undefined, 6, numSplats);
    }

    let renderTexture = () => {
        ctx.renderPassDescriptor = renderPassDescriptorTexture;
        render(ctx, pipeLineForward, bindGroupForward, undefined, 6, numSplats, true);
    }



    renderTexture();
    renderScreen();

    let runGD = async () => {

        const steps = 100;
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
            encoder.copyBufferToBuffer(lossBuffer, 0, lossResultBuffer, 0, lossResultBuffer.size);


            // run the work lmao
            const commandBuffer = encoder.finish();
            ctx.device.queue.submit([commandBuffer]);

            // this updates the input values, not clean
            await updateResults();

            // updates loss plot
            await updateLossResults();

            ctx.device.queue.writeBuffer(paramBuffer, 0, input);

            // forward pass
            renderTexture();

            // display on canvas
            renderScreen();
        }
        
    }
    // == interactive suff, not really needed
    {
        // const PARAMS = {
        //     stepSize: 0.3,
        // };
        
        const pane = new Pane({
            container: document.getElementById("gd-sliders-2gauss") as HTMLElement,
        });

        const paneSplats = pane.addFolder({
            title: 'inital values',
            expanded: true,
        });

        const paneG1 = paneSplats.addFolder({
           title: 'splat 1',
           expanded: true, 
        });

        
        const paneG2 = paneSplats.addFolder({
           title: 'splat 2',
           expanded: false, 
        });

        paneG1.addBinding(PARAMS, 'pos1', {
            label: 'pos',
            picker: 'inline',
            expanded: true,
            
            x: { min: 0.0, max: 1.0, step: 0.01 },
            y: { min: 0.0, max: 1.0, step: 0.01 },
        })
        .on('change', (ev) => {
            resetInput();
            renderScreen();
        });

        paneG1.addBinding(PARAMS, 'scale1', {
            label: 'scale',
            picker: 'inline',
            expanded: true,
            
            x: { min: 0.0, max: 0.5, step: 0.01 },
            y: { min: 0.0, max: 0.5, step: 0.01 },
        })
        .on('change', (ev) => {
            resetInput();
            renderScreen();
        });

        paneG1.addBinding(PARAMS, 'rot1', {
            label: 'rot',
            min: 0,
            max: 2* Math.PI,
        })
        .on('change', (ev) => {
            resetInput();
            renderScreen();
        });

        paneG1.addBinding(PARAMS, 'col1', {
            label: 'color'
        })
        .on('change', (ev) => {
            resetInput();
            renderScreen();
        });

        paneG2.addBinding(PARAMS, 'pos2', {
            label: 'pos',
            picker: 'inline',
            expanded: true,
            
            x: { min: 0.0, max: 1.0, step: 0.01 },
            y: { min: 0.0, max: 1.0, step: 0.01 },
        })
        .on('change', (ev) => {
            resetInput();
            renderScreen();
        });

        paneG2.addBinding(PARAMS, 'scale2', {
            label: 'scale',
            picker: 'inline',
            expanded: true,
            
            x: { min: 0.0, max: 0.5, step: 0.01 },
            y: { min: 0.0, max: 0.5, step: 0.01 },
        })
        .on('change', (ev) => {
            resetInput();
            renderScreen();
        });

        paneG2.addBinding(PARAMS, 'rot2', {
            label: 'rot',
            min: 0,
            max: 2* Math.PI,
        })
        .on('change', (ev) => {
            resetInput();
            renderScreen();
        });

        paneG2.addBinding(PARAMS, 'col2', {
            label: 'color'
        })
        .on('change', (ev) => {
            resetInput();
            renderScreen();
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