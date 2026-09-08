import { Pane } from 'tweakpane';

import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../../../myutils/BufferHelper';

import { getWebGPUctx, render } from '../../../myutils/ContextHelpers';

import shaderCodeCompute from '../shaders/gradientDescentSimple.wgsl?raw';


function drawSillyLoss(loss : Float32Array<ArrayBuffer>, N: number, eps : number) {
  
  
  const canvas = document.getElementById('gd1d') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d');

  const height = N;

  const maxLoss = Math.max(...loss);
  const minLoss = 0.00005;

  const y_scaled = (l: number) => {
    return height * (Math.log(l) - Math.log(eps)) / (Math.log(maxLoss) - Math.log(minLoss));
  };
 
  if(ctx) {
    ctx.clearRect(0,0, N, height);
    ctx.fillStyle = '#fff';
    for (let x = 0; x < N; ++x) {
        let l = loss[x];
        
        if(l) {
            let lossLog = y_scaled(Math.max(l,eps));
            const v = lossLog;
            ctx.fillRect(x, height - v, 1, v);
        }
    }
  }
  else {
    console.log("plot failed");
  }
}

async function main() {

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
        result[0] = result[1]; // hacky + ugly
        
        drawSillyLoss(result, maxSize, 0.01);
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
        pass.setPipeline(pipeline);
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

main();