import { Pane } from 'tweakpane';

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import shaderCodeCompute from '../shaders/gradientDescent2d.wgsl?raw';
import simpleTileVert from '../shaders/staticTileVert.wgsl?raw';
import simpleTextureFrag from '../shaders/simpleTextureFrag.wgsl?raw';


async function loadImageBitmap(url : string) {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: 'none'});
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
    
    const paramBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_src_dst', 2);
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
    
    const uBuilder = new UniformBufferDescriptorBuilder("My Uniform Buffer", "uniform");
    uBuilder.add('pos', "vec2f")
            .add('scale', "vec2f")
            .add('rot', "f32")
            .add('color', "vec3f")
            .add('alpha', "f32");

    const uDesc = uBuilder.build();
    
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
                    type: 'uniform',
                    minBindingSize: uDesc.sizeBytes,
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

    input.set([0.5, 0.5], posOff);
    input.set([0.7, 0.7], scaleOff);
    input.set([0.0], rotOff);
    input.set([8.0, 1.0, 1.0], colorOff);
    input.set([1.0], alphaOff);

    ctx.device.queue.writeBuffer(paramBuffer, 0, input);

    const resultBuffer = ctx.device.createBuffer({
        label: 'result buffer',
        size: paramDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });


    // == uniform stuff for interaction ==

    const uBuffer = bufferManager.createBuffer(uDesc);

    const uValues = new Float32Array(uDesc.size);
    const att = uDesc.attributes;

    let setResultInUnfirom = (input : Float32Array<ArrayBuffer>, index : number) => {
        let startI = paramDesc.unitSize! * index;
        let endI = startI + uValues.length;
        uValues.set(input.subarray(startI, endI), 0);
        ctx.device.queue.writeBuffer(uBuffer, 0, uValues);
    }

    setResultInUnfirom(input,0);

    const uniBuff = bufferManager.createBuffer(uniDesc);
    const uniVal = new Float32Array([0.1]); // stepsize
    
    ctx.device.queue.writeBuffer(uniBuff, 0, uniVal , 0);

    // == texture stuff

    const testImageUrl = 'assets/testImage.jpg';
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
            { binding: 2, resource: uBuffer },
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
        
        console.log("result", result);
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

            setResultInUnfirom(input, 0);

            render(ctx, pipeLineDraw, bindGroupDraw, undefined, 6, 1);

            ctx.device.queue.writeBuffer(paramBuffer, 0, input);
            
        }
        
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