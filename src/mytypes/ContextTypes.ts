export interface ctxRender {
    device : GPUDevice; 
    context : GPUCanvasContext;
    depthTexture: GPUTexture | null;
    renderPassDescriptor : GPURenderPassDescriptor | null; 
}
export interface ctxWebGPU extends ctxRender { 
    gpu : GPU;
    adapter: GPUAdapter;
    canvas: HTMLCanvasElement;
    presentationFormat : GPUTextureFormat;
} 

export interface ctxOptions {
    canvasId? : string;
    renderPassDescriptor? : GPURenderPassDescriptor;
    skipDepth? : boolean;
}