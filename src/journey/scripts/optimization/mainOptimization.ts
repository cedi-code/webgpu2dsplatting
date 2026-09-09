import { Pane } from 'tweakpane';

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import shaderCodeCompute from '../shaders/gradientDescentSimple.wgsl?raw';
import simpleTileVert from '../shaders/staticTileVert.wgsl?raw';
import simpleTextureFrag from '../shaders/simpleTextureFrag.wgsl?raw';

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
        label: 'naiv gauss impl',
        code: simpleTextureFrag 
    });

    const numSamples = 100; // HAS TO BE CONSISTENT WITH SHADER!

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
                minBindingSize: numSamples * Float32Array.BYTES_PER_ELEMENT,
            },
            },
            { // dataY
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: 'read-only-storage',
                minBindingSize: numSamples * Float32Array.BYTES_PER_ELEMENT,
            },
            },
            { // uniforms
            binding: 2,
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



    
    const ySamples = new Float32Array(numSamples);

    ySamples.forEach((_,i) => {
        let x = (10.0*i)/numSamples - 5.0; 
        ySamples[i] = Math.exp(-0.25*(x-2.0)*(x-2.0));
    });

    const maxSize : number = 100;
    const input = new Float32Array(maxSize);
    const initalGuess = Math.random()* 8.0 - 4.0;
    input[0] = initalGuess;
    // creating buffer
    const workBuffer = ctx.device.createBuffer({
        label: 'my loss output buffer',
        size: input.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const yBuffer = ctx.device.createBuffer({
        label: 'y samples',
        size: ySamples.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    ctx.device.queue.writeBuffer(workBuffer, 0, input);
    ctx.device.queue.writeBuffer(yBuffer, 0, ySamples);

    const resultBuffer = ctx.device.createBuffer({
        label: 'result buffer',
        size: input.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // == uniform stuff for interaction ==

    bufferManager.init(ctx.device);



    const uniBuff = bufferManager.createBuffer(uniDesc);
    const uniVal = new Float32Array([0.3]);
    
    ctx.device.queue.writeBuffer(uniBuff, 0, uniVal , 0);

    // == texture stuff

    const kTextureWidth = 5;
    const kTextureHeight = 7;
    const _ = [255,   0,   0, 255];  // red
    const y = [255, 255,   0, 255];  // yellow
    const b = [  0,   0, 255, 255];  // blue
    const textureData = new Uint8Array([
        b, _, _, _, _,
        _, y, y, y, _,
        _, y, _, _, _,
        _, y, y, _, _,
        _, y, _, _, _,
        _, y, _, _, _,
        _, _, _, _, _,
    ].flat());

    const texture = ctx.device.createTexture({
        size: [kTextureWidth, kTextureHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    ctx.device.queue.writeTexture(
        { texture },
        textureData,
        { bytesPerRow: kTextureWidth * 4 },
        { width: kTextureWidth, height: kTextureHeight },
    );

    const sampler = ctx.device.createSampler();

    // ==

    const bindGroup = ctx.device.createBindGroup({
        label: 'bindGroup workbuffer',
        layout: pipelineCompute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: workBuffer },
            { binding: 1, resource: yBuffer },
            { binding: 2, resource: uniBuff },
        ]
    });

    const bindGroupDraw = ctx.device.createBindGroup({
        label: 'bindGroup draw',
        layout: pipeLineDraw.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: sampler },
            { binding: 1, resource: texture },
        ]
    });

    type Output = {
        initalQ : number,
        finalQ: number,
        lossLog: number,
    }

    const PARAMS_OUT : Output = {
        initalQ : initalGuess,
        finalQ: 0.0,
        lossLog: 0.0,
    }

    let updateResults = async (params_out : Output) => {
        // read results
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(resultBuffer.getMappedRange());

        params_out.finalQ = result[0];
        params_out.initalQ = initalGuess;
        
        // unmap getMapped range is only valid buffer until we call unmap, the length will be set to 0
        resultBuffer.unmap();        
    }


    let runGD = async () => {

        input[0] = initalGuess;
        ctx.device.queue.writeBuffer(workBuffer, 0, input);

        const encoder = ctx.device.createCommandEncoder({
        label: 'doubling encoder',
        });
        const pass = encoder.beginComputePass({
            label: 'doubling compute pass',
        });
        pass.setPipeline(pipelineCompute);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();

        // mapping result to my buffer
        encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, resultBuffer.size);

        // run the work lmao
        const commandBuffer = encoder.finish();
        ctx.device.queue.submit([commandBuffer]);

        updateResults(PARAMS_OUT);
    }

    await runGD();

    render(ctx, pipeLineDraw, bindGroupDraw, undefined, 6, 1);


    console.log(ctx.canvas.width );
    console.log(ctx.canvas.height);

    // == interactive suff, not really needed
    {
        const PARAMS = {
            stepSize: 0.3,
        };
        
        const pane = new Pane({
            container: document.getElementById("gd-sliders") as HTMLElement,
        });

        pane.addBinding(PARAMS, 'stepSize', {
            min: 0.01,
            max: 1.0,
        }).on('change', (ev) => {
            let s = ev.value;
            uniVal[0] = s;
            ctx.device.queue.writeBuffer(uniBuff, 0, uniVal, 0);
            runGD();
        });

        pane.addBinding(PARAMS_OUT, 'initalQ', {
            readonly: true,
        });

        pane.addBinding(PARAMS_OUT, 'finalQ', {
            readonly: true,
        });
    }
    
}

main();