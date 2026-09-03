import {
    type ctx,
    type ctxFull,
    type ctxOptions,
} from "../mytypes/ContextTypes"

function render(
    ctx : ctx,
    pipeline : GPURenderPipeline, 
    bindGroup?: GPUBindGroup, 
    vertexBuffer?: GPUBuffer,
    numVert? : number,
    numInstances? : number,
    ) {

    // get current textrure from canvas
    const canvasTexture = ctx.context.getCurrentTexture();
    if(!ctx.renderPassDescriptor) return;
    for(let colorAttachment of ctx.renderPassDescriptor.colorAttachments) {
        if(!colorAttachment) continue; 
        colorAttachment.view = canvasTexture.createView();
    }
    if(ctx.renderPassDescriptor.depthStencilAttachment && ctx.depthTexture) {
        ctx.renderPassDescriptor.depthStencilAttachment.view = ctx.depthTexture.createView();
    }

    const encoder = ctx.device.createCommandEncoder({ label: 'my first encoder'});

    // make a render pass
    const pass = encoder.beginRenderPass(ctx.renderPassDescriptor);
    pass.setPipeline(pipeline);
    if(vertexBuffer) pass.setVertexBuffer(0, vertexBuffer); // we might have multiple vertex buffers!
    if(bindGroup) pass.setBindGroup(0, bindGroup); // we might have multiple bind groups!
    pass.draw(numVert || 0, numInstances || 1);
    pass.end();

    const commandBuffer = encoder.finish();
    ctx.device.queue.submit([commandBuffer]);
}

async function getWebGPUctx(options? : ctxOptions) : Promise< ctxFull| null> {
    const gpu = navigator.gpu;
    if(!gpu) {
        alert("browser needs WebGPU support");
        return null;
    }

    const adapter = await gpu.requestAdapter();
    if(!adapter) {
        alert("could not get gpu adapter");
        return null;
    }
    const device = await adapter?.requestDevice();
    if(!device) {
        alert("browser needs WebGPU support");
        return null;
    }

    // get webgpu context from canvas
    const canvas = options?.canvasId ? 
         document.getElementById(options.canvasId) as HTMLCanvasElement :
         document.querySelector('canvas');
    if(!canvas) {
        console.error("could not find canvas html element");
        if(options?.canvasId) {
            console.error("could not find canvas with name:", options.canvasId);
        }
        return null;
    }

    const context = canvas?.getContext('webgpu') as GPUCanvasContext;
    if(!context) {
        console.error("could not get webgpu context from canvas");
        return null;
    }
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: presentationFormat,
        alphaMode: "premultiplied",
    });

    const depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });

    const renderPassDescriptor = null;

    return { gpu, adapter, device, canvas, context, presentationFormat, depthTexture, renderPassDescriptor };

}

export { render, getWebGPUctx };