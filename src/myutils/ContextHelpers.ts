import {
    type ctxRender,
    type ctxWebGPU,
    type ctxOptions,
} from "../mytypes/ContextTypes"

function render(
    ctx : ctxRender,
    pipeline : GPURenderPipeline, 
    bindGroup?: GPUBindGroup, 
    vertexBuffer?: GPUBuffer,
    numVert? : number,
    numInstances? : number,
    ownAttachment? : boolean,
    drawBuffer? : GPUBuffer,
    ) {

    if(!ctx.renderPassDescriptor) return;
    // get current textrure from canvas
    if(!ownAttachment) {
        const canvasTexture = ctx.context.getCurrentTexture();
        for(let colorAttachment of ctx.renderPassDescriptor.colorAttachments) {
            if(!colorAttachment) continue; 
            colorAttachment.view = canvasTexture.createView();
        }
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
    if(drawBuffer) {
        pass.drawIndirect(drawBuffer, 0);
    }
    else {
        pass.draw(numVert || 0, numInstances || 1);
    }
    pass.end();

    const commandBuffer = encoder.finish();
    ctx.device.queue.submit([commandBuffer]);
}

function webgpuWarningSlow() {
    let modal = document.getElementById("webgpu-modal") as HTMLDialogElement | null;

    if (!modal) {
        modal = document.createElement("dialog");
        modal.id = "webgpu-modal";
        
        modal.innerHTML = `
            <h3 style="margin-top: 0; margin-bottom: 1rem;">WebGPU Supported but very slow!</h3>
            <p style="margin-bottom: 1rem;">Your browser supports WebGPU but its rendering on the CPU! </p>
            <p style="margin-bottom: 1.5rem;">The demos might be very slow!</p>

            <button id="webgpu-close-btn" >Close</button>
            `;
        
        document.body.appendChild(modal);
        const closeBtn = modal.querySelector("#webgpu-close-btn");
            closeBtn?.addEventListener("click", () => {
                modal?.close();
            });
    }

    modal.showModal();
}

function webgpuWarning() {
    let modal = document.getElementById("webgpu-modal") as HTMLDialogElement | null;

    if (!modal) {
        modal = document.createElement("dialog");
        modal.id = "webgpu-modal";
        
        modal.innerHTML = `
            <h3 style="margin-top: 0; margin-bottom: 1rem;">WebGPU Not Supported</h3>
            <p style="margin-bottom: 1rem;">Your browser needs WebGPU support to run this application.</p>
            <p style="margin-bottom: 1.5rem;">Please visit <a href="https://enablegpu.com" target="_blank" style="color: #0066cc; text-decoration: underline;">enablegpu.com</a> or check your browser flags.</p>
            <button id="webgpu-close-btn" >Close</button>
            `;
        
        document.body.appendChild(modal);
        const closeBtn = modal.querySelector("#webgpu-close-btn");
            closeBtn?.addEventListener("click", () => {
                modal?.close();
            });
    }

    modal.showModal();
}

async function getWebGPUctx(options? : ctxOptions) : Promise< ctxWebGPU| null> {
    const gpu = navigator.gpu;
    if(!gpu) {
        webgpuWarning();
        return null;
    }

    const adapter = await gpu.requestAdapter();
    if(!adapter) {
        webgpuWarning();
        return null;
    }
    const info = adapter.info;
    const text = `${info.vendor} ${info.architecture} ${info.device} ${info.description}`.toLowerCase();
    const isSoftware = /swiftshader|llvmpipe|lavapipe|softpipe|warp|software|cpu/.test(text);
    if (isSoftware) {
        console.warn('Running on a software renderer, performance will be poor.');
        webgpuWarningSlow();
    }
    const device = await adapter?.requestDevice();
    if(!device) {
        webgpuWarning();
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