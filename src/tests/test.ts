import { vec2, type Vec2 } from 'wgpu-matrix';
import { Pane } from 'tweakpane';
import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../myutils/BufferHelper';
import { getWebGPUctx, render } from '../myutils/ContextHelpers';
import { randVec2 } from '../myutils/MathHelpers';

import shaderGaussFunctions from '../shaders/gaussFunctions.wgsl?raw';

import shaderTestCompute from './testShader.wgsl?raw';

type GradResult = {
    numericalGrad : Vec2,
    derrivedGrad: Vec2,
    truncationError: Vec2,
};

async function main() {

    const ctx = await getWebGPUctx({ canvasId: "test"});
    if(!ctx) {
        return;
    }

    const csModule = ctx.device.createShaderModule({
        label: 'compute shader test',
        code: (shaderGaussFunctions + shaderTestCompute)  
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
    const numXSamples = 10;

    const paramBuilder = new UniformBufferDescriptorBuilder('params storage buffer', 'storage', 'copy_dst', numSplats);
    paramBuilder.add('pos', "vec2f")
                .add('scale', "vec2f")
                .add('rot', "f32")
                .add('color', "vec3f")
                .add('alpha', "f32");

    const paramDesc = paramBuilder.build();

    const sampleBuilder = new UniformBufferDescriptorBuilder('sample storage buffer', 'storage', 'copy_dst', numXSamples);
    sampleBuilder.add('sample', "vec2f")
    
    const sampleDesc = sampleBuilder.build();
    
    const unifromBuilder = new UniformBufferDescriptorBuilder('test unfirom', 'uniform');
    unifromBuilder.add('numSamples', "f32")
                  .add('numProperties', "f32")
                    .add('stepH', "f32")
                    .add('Epsilon', "f32");
    
    const uniformDesc = unifromBuilder.build();
    
    const numPropertiesTested = 3.0;
    const numTestSamples = numSplats * numXSamples * numPropertiesTested;
    const storageBufferBuilderGrad = new UniformBufferDescriptorBuilder("test result buffer", "storage", "copy_src", numTestSamples);
    const gradResultDesc = storageBufferBuilderGrad
                            .add('expected', "vec2f")
                            .add('result', "vec2f")
                            .add('truncationError', "vec2f")
                            .build();

    const testResultBuffer = bufferManager.createBuffer(gradResultDesc);
    const paramBuffer = bufferManager.createBuffer(paramDesc);
    const sampleBuffer = bufferManager.createBuffer(sampleDesc);
    const uniformBuffer = bufferManager.createBuffer(uniformDesc);

    const resultBuffer = ctx.device.createBuffer({
        label: 'result buffer',
        size: gradResultDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const bindGroup = ctx.device.createBindGroup({
        label: 'bindGroup workbuffer',
        layout: pipelineCompute.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: paramBuffer },
            { binding: 1, resource: sampleBuffer },
            { binding: 2, resource: uniformBuffer },
            { binding: 3, resource: testResultBuffer },
        ]
    });

    // == fill buffers ==
    if(paramDesc.unitSize == undefined || sampleDesc.unitSize == undefined) {
        console.error("storage buffer descriptors wrong?");
        return;
    }

    const posOff    = paramDesc.attributes[0].offset;
    const scaleOff  = paramDesc.attributes[1].offset;
    const rotOff    = paramDesc.attributes[2].offset;
    const colorOff  = paramDesc.attributes[3].offset;
    const alphaOff  = paramDesc.attributes[4].offset;

    const params = new Float32Array(paramDesc.size);
    // gauss 
    for(let i = 0; i < numSplats; i++) {

        let splatOff = paramDesc.unitSize * i;
        params.set([0.5, 0.5],      splatOff + posOff);
        params.set([1.5, 1.0],      splatOff + scaleOff);
        params.set([0.0],           splatOff + rotOff);
        params.set([8.0, 8.0, 8.0], splatOff + colorOff);
        params.set([8.0],           splatOff + alphaOff);

    }
    ctx.device.queue.writeBuffer(paramBuffer, 0, params);

    // fill sample x positions
    
    const samples = new Float32Array(sampleDesc.size);

    for(let i = 0; i < numXSamples; i++) {
        samples.set(randVec2(0,1), sampleDesc.unitSize * i);
    }

    console.log("x samples", samples);

    ctx.device.queue.writeBuffer(sampleBuffer, 0, samples);
    // create unfirom for step size "h"

    const uniformsValues = new Float32Array(uniformDesc.size);
    const h = 1e-4;
    const MACHINE_EPSILON = 1.19e-07;

    console.log("EPSILON", MACHINE_EPSILON);

    uniformsValues.set([numXSamples],   uniformDesc.attributes[0].offset);
    uniformsValues.set([numPropertiesTested],   uniformDesc.attributes[1].offset);
    uniformsValues.set([h],             uniformDesc.attributes[2].offset);
    uniformsValues.set([MACHINE_EPSILON], uniformDesc.attributes[3].offset);

    console.log("uniform values:", uniformsValues)

    ctx.device.queue.writeBuffer(uniformBuffer, 0, uniformsValues);
    
    // figure out how to read back test result

    const PARAMS = {
        stepH : 4,
        errorDq : '..',
        TestResultGradQ : 'waiting...',
        TestResultGradS : 'waiting...',
        TestResultGradR : 'waiting...',
    };

    let evalResult = (result : GradResult[]) => {
        let testFail = [false, false, false];
        result.forEach( (res : GradResult, i : number) => {
            let errorVec2 = vec2.create();

            const propertyI = i % numPropertiesTested;


            
            const rotProperty = (i % numPropertiesTested) == 2;

            vec2.sub(res.numericalGrad, res.derrivedGrad, errorVec2);
            const errorGradX = Math.abs(res.numericalGrad[0] - res.derrivedGrad[0])  > res.truncationError[0];
            const errorGradY = Math.abs(res.numericalGrad[1] - res.derrivedGrad[1])  > res.truncationError[1];
            PARAMS.errorDq = (Math.abs(res.numericalGrad[0] - res.derrivedGrad[0])).toFixed(8);

            if(errorGradX || (errorGradY && !rotProperty)) {
                console.error("trunaction error", res.truncationError);
                console.error("numerical error", vec2.sub(res.numericalGrad, res.derrivedGrad, errorVec2));
                console.log("sampleX", samples[Math.floor(i / numPropertiesTested)]);
    
                console.error("property that failed", propertyI);
                console.error("dim that failed x,y", errorGradX, errorGradY);
                testFail[propertyI] = true;
            }
        })
        
        PARAMS.TestResultGradQ = testFail[0] ? 'failed' : 'pass';
        PARAMS.TestResultGradS = testFail[1] ? 'failed' : 'pass';
        PARAMS.TestResultGradR = testFail[2] ? 'failed' : 'pass';

    }

    let runTest = async () => {

        const encoder = ctx.device.createCommandEncoder({
            label: 'test encoder',
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

        // read results
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(resultBuffer.getMappedRange());
        let mappedResult : GradResult[] = [];
        for(let i = 0; i < gradResultDesc.size; i+=6) {
            mappedResult.push({
                numericalGrad : result.subarray(i, i+2),
                derrivedGrad : result.subarray(i+2, i+4),
                truncationError: result.subarray(i+4, i+6)
            });
        } 

        // unmap getMapped range is only valid buffer until we call unmap, the length will be set to 0
        evalResult(mappedResult);

        resultBuffer.unmap();                
        
    }



    const pane = new Pane({
        container: document.getElementById("output-window") as HTMLElement,
    });
    
    const paneParams = pane.addFolder({
        title: 'Test parameters'
    });

    const paneTests = pane.addFolder({
        title: 'Tests'
    });

    paneParams.addBinding(PARAMS, 'stepH', {
        label: 'step size 10^-x',
        min: 0,
        max: 8
    });

    paneTests.addButton({
        title: 'Run Tests',
        label: 'Run',
    }).on('click', () => {
        PARAMS.errorDq = '0.000000';
        runTest();
    });

    paneTests.addBinding(PARAMS, 'errorDq', {
        label: 'numerical vs analytical diff: ',
        readonly: true,
        bufferSize: numTestSamples
    });
    paneTests.addBinding(PARAMS, 'TestResultGradQ', {
        readonly: true,
        label: 'grad pos'
    });
    paneTests.addBinding(PARAMS, 'TestResultGradS', {
        readonly: true,
        label: 'grad scale'
    });
    paneTests.addBinding(PARAMS, 'TestResultGradR', {
        readonly: true,
        label: 'grad rot'
    });
}

main();