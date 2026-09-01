export interface ctx {
    device : GPUDevice; 
    context : GPUCanvasContext;
    depthTexture: GPUTexture | null;
    renderPassDescriptor : GPURenderPassDescriptor | null; 
}
export interface ctxFull extends ctx { 
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