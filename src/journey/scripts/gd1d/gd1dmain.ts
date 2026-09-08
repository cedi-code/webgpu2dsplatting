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

    const input = new Float32Array([1,3,5]);

    // creating buffer
    const workBuffer = ctx.device.createBuffer({
        label: 'my first work buffer',
        size: input.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    ctx.device.queue.writeBuffer(workBuffer, 0, input);

    const resultBuffer = ctx.device.createBuffer({
        label: 'result buffer',
        size: input.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });


    const bindGroup = ctx.device.createBindGroup({
        label: 'bindGroup workbuffer',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: workBuffer }
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