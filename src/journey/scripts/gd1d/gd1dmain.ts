import { Pane } from 'tweakpane';

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import shaderCodeCompute from '../shaders/gradientDescentSimple.wgsl?raw';

import { createLossPlot, creatGaussComparePlot } from '../../../myutils/LogHelpers';

async function main() {

    let lossData : number[]  = []
    let lossSteps: number[] = []

    let plotHTMLElem = document.getElementById('loss-plot-1d') ?? document.body;
    let lossPlot = createLossPlot(plotHTMLElem);
    
    let samplesPosX : number[] = [];
    let gaussCurveSolu : number[] = [];
    let gaussCurveCurr : number[] = [];



    let plotGaussHTMLElem = document.getElementById('plot-1d-gauss') ?? document.body;
    let plotGauss = creatGaussComparePlot(plotGaussHTMLElem);

    
    

    const PARAMS = {
        stepSize: 0.3,
        initalQ: -1.0,
        finalQ: 0.0,
    };

    const ctx = await getWebGPUctx({ canvasId: "compute"});
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
    const stepSize = 5;
    
    const ySamples = new Float32Array(numSamples);

    ySamples.forEach((_,i) => {
        let x = (10.0*i)/numSamples - 5.0; 
        samplesPosX.push(x);
        ySamples[i] = Math.exp(-0.25*(x-2.0)*(x-2.0));
        gaussCurveCurr.push(ySamples[i].valueOf());
        gaussCurveSolu.push(ySamples[i].valueOf());
    });

    plotGauss.setData([samplesPosX, gaussCurveCurr, gaussCurveSolu]);

    


    const maxSize : number = stepSize + 1.0;
    const input = new Float32Array(maxSize);
    input[0] = PARAMS.initalQ;
    PARAMS.finalQ = PARAMS.initalQ;
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
    const uniBuild = new UniformBufferDescriptorBuilder('gd uniform', "uniform");
    uniBuild.add('stepSize', 'f32');

    const uniDesc = uniBuild.build();

    const uniBuff = bufferManager.createBuffer(uniDesc);
    const uniVal = new Float32Array([0.3]);
    
    ctx.device.queue.writeBuffer(uniBuff, 0, uniVal , 0);

    // ==

    const bindGroup = ctx.device.createBindGroup({
        label: 'bindGroup workbuffer',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: workBuffer },
            { binding: 1, resource: yBuffer },
            { binding: 2, resource: uniBuff },
        ]
    });

    let updateGaussGraph = () => {
        // really dumb, but i dont care for now
        gaussCurveCurr = [];
        ySamples.forEach((_,i) => {
        let x = (10.0*i)/numSamples - 5.0; 
            samplesPosX.push(x);
            let q = PARAMS.finalQ;
            let g_x = Math.exp(-0.25*(x-q)*(x-q));
            gaussCurveCurr.push(g_x);
        });

        plotGauss.setData([samplesPosX, gaussCurveCurr,  gaussCurveSolu]);
    }

    let updateResults = async () => {
        // read results
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(resultBuffer.getMappedRange());

        PARAMS.finalQ = result[0];

        lossData.push(...result.subarray(1,stepSize).valueOf());
        const lastStep = lossSteps.at(-1) ?? 0;
        lossSteps.push(...Array.from(Array(stepSize).keys(),i => i + 1 + lastStep));
        
        lossPlot.setData([lossSteps, lossData]);

        updateGaussGraph();

        // unmap getMapped range is only valid buffer until we call unmap, the length will be set to 0
        resultBuffer.unmap();        
    }

    let runGD = async () => {

        input[0] = PARAMS.finalQ;
        ctx.device.queue.writeBuffer(workBuffer, 0, input);

        const encoder = ctx.device.createCommandEncoder({
        label: 'doubling encoder',
        });
        const pass = encoder.beginComputePass({
            label: 'doubling compute pass',
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();

        // mapping result to my buffer
        encoder.copyBufferToBuffer(workBuffer, 0, resultBuffer, 0, resultBuffer.size);

        // run the work lmao
        const commandBuffer = encoder.finish();
        ctx.device.queue.submit([commandBuffer]);

        updateResults();
    }

    await runGD();


    
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
    });

    pane.addBinding(PARAMS, 'initalQ', {
        readonly: false,
        min: -5.0,
        max: 5.0,
    }).on('change', (ev) => {
        let s = ev.value;
        input[0] = s;
        PARAMS.finalQ = s;
        ctx.device.queue.writeBuffer(workBuffer, 0, input);
        updateGaussGraph();
        lossData = [];
        lossSteps = [];
        lossPlot.setData([lossSteps, lossData]);
    });

    pane.addBinding(PARAMS, 'finalQ', {
        readonly: true,
    });

    pane.addButton({
        'title' : '5 steps',
        'label' : 'run gradient descent'
    }).on('click', () => {
        runGD();
    });


}

main();