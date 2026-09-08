import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import shaderCodeCompute from '../shaders/gradientDescentSimple.wgsl?raw';

async function main() {

    const ctx = await getWebGPUctx({ canvasId: "gd1d"});
    if(!ctx) {
        return;
    }

    const csModule = ctx.device.createShaderModule({
        label: '1d gs module',
        code: shaderCodeCompute 
    });

    const pipeline = ctx.device.createComputePipeline({
        label: 'doubling compute pipeline',
        layout: 'auto',
        compute: {
            module: csModule,
        },
    });
    const numSamples = 100; // HAS TO BE CONSISTENT WITH SHADER!
    
    const ySamples = new Float32Array(numSamples);

    ySamples.forEach((_,i) => {
        let x = (10.0*i)/numSamples - 5.0; 
        ySamples[i] = Math.exp(-0.25*(x-2.0)*(x-2.0));
    });

    const maxSize : number = 100;
    const input = new Float32Array(maxSize);
    const initalGuess = Math.random()* 5.0 - 2.5;
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


    const bindGroup = ctx.device.createBindGroup({
        label: 'bindGroup workbuffer',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: workBuffer },
            { binding: 1, resource: yBuffer },
        ]
    });

    // commands
    const encoder = ctx.device.createCommandEncoder({
    label: 'doubling encoder',
    });
    const pass = encoder.beginComputePass({
        label: 'doubling compute pass',
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(input.length);
    pass.end();

    // mapping result to my buffer
    encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, resultBuffer.size);

    // run the work lmao
    const commandBuffer = encoder.finish();
    ctx.device.queue.submit([commandBuffer]);

    // read results
    await resultBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(resultBuffer.getMappedRange());

    console.log('in:', input);
    console.log('out:', result);

    // unmap getMapped range is only valid buffer until we call unmap, the length will be set to 0
    resultBuffer.unmap();
}

main();