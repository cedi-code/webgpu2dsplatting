import { ButtonBladeController } from '@tweakpane/core';
import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../myutils/BufferHelper';
import { getWebGPUctx, render } from '../myutils/ContextHelpers';


import shaderGaussFunctions from '../shaders/gaussFunctions.wgsl?raw';
import shaderStructs from '../shaders/structs.wgsl?raw';

import shaderTestCompute from './testShader.wgsl?raw';


async function main() {

    const ctx = await getWebGPUctx({ canvasId: "main"});
    if(!ctx) {
        return;
    }

    const csModule = ctx.device.createShaderModule({
        label: 'compute shader test',
        code: (shaderStructs + shaderGaussFunctions + shaderTestCompute)  
    });
    
    // == test pipeline

    const pipelineCompute = ctx.device.createComputePipeline({
        label: 'gd compute pipeline',
        layout: 'auto',
        compute: {
            module: csModule,
        },
        
    });

    // == creating buffer ==
    bufferManager.init(ctx.device);

    const numSplats = 1;
    const numXSamples = 3;

    const paramBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_dst', numSplats);
    paramBuilder.add('pos', "vec2f")
                .add('scale', "vec2f")
                .add('rot', "f32")
                .add('color', "vec3f")
                .add('alpha', "f32");

    const paramDesc = paramBuilder.build();

    const sampleBuilder = new UniformBufferDescriptorBuilder('sample storage buffer', 'storage', 'copy_dst', numXSamples);
    sampleBuilder.add('sample', "vec2f");

    const sampleDesc = sampleBuilder.build();
    
    const numTestSamples = 1;
    const storageBufferBuilderGrad = new UniformBufferDescriptorBuilder("test result buffer", "storage", "copy_src", numTestSamples);
    const gradSamplesDesc = storageBufferBuilderGrad
                            .add('expected', "vec2f")
                            .add('result', "vec2f")
                            .build();

    const testResultBuffer = bufferManager.createBuffer(gradSamplesDesc);
    const paramBuffer = bufferManager.createBuffer(paramDesc);
    const sampleBuffer = bufferManager.createBuffer()

    const resultBuffer = ctx.device.createBuffer({
        label: 'result buffer',
        size: gradSamplesDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroup = ctx.device.createBindGroup({
        label: 'bindGroup workbuffer',
        layout: pipelineCompute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
            { binding: 1, resource: resultBuffer },
        ]
    });

    // == fill buffers ==

    const posOff    = paramDesc.attributes[0].offset;
    const scaleOff  = paramDesc.attributes[1].offset;
    const rotOff    = paramDesc.attributes[2].offset;
    const colorOff  = paramDesc.attributes[3].offset;
    const alphaOff  = paramDesc.attributes[4].offset;

    const params = new Float32Array(paramDesc.size);
    // gauss 1
    params.set([0.5, 0.5], posOff);
    params.set([1.5, 1.5], scaleOff);
    params.set([0.0], rotOff);
    params.set([8.0, 8.0, 8.0], colorOff);
    params.set([8.0], alphaOff);

    ctx.device.queue.writeBuffer(paramBuffer, 0, params);

    let runGD = async () => {

        const encoder = ctx.device.createCommandEncoder({
            label: 'gd encoder',
        });
        const pass = encoder.beginComputePass({
            label: 'numerical gradient compute pass',
        });
        pass.setPipeline(pipelineCompute);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();

        // mapping result to my buffer
        encoder.copyBufferToBuffer(testResultBuffer, 0, resultBuffer, 0, resultBuffer.size);

        // run the work 
        const commandBuffer = encoder.finish();
        ctx.device.queue.submit([commandBuffer]);

        // this updates the input values, not clean
        // read results
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(resultBuffer.getMappedRange());
        
        // unmap getMapped range is only valid buffer until we call unmap, the length will be set to 0
        resultBuffer.unmap();                
    }
}

main();