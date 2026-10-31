export interface SplatSetupConfig {
    numSplats: number;
    sampleDim: number;
    chunkWidth: number;
    chunkHeight: number;
}

export interface SplatDynamicConfig {
    activeSplats: number;
    refineAfter: number;
    refineActive: boolean;
    opacityResetAfter: number;
    opacityResetActive : boolean;
}

export interface LossConfig {
    maxSteps: number;
}

export interface RefineDefaultsConfig {
    gradientPosThresh: number;
    gradientScaleThresh: number;
    opacityResetV: number;
    alphaCutoff: number;
    scaleDivideConst: number;
    offCanvasThreshold: number;
}

export interface RadixElement {
    name: string;
    index: number;
}

export interface RadixConfig {
    groupSizeRadix: number;
    elementNames: string[]; // the string order determends the index
}

export interface FragmentConstantsConfig {
    xRAY: boolean;
    opacityPercentage: number;
    tileScale: number;
}

export interface GuiVisibility {
    blobs: boolean;
    lossWeight: boolean;
    nSteps: boolean;
    speed: boolean;
    refinement: boolean;
    refinementParams: boolean;
    gradientDescent: boolean;
}

// Master Raw Input Config
export interface SplattingConfigInput {
    setup: SplatSetupConfig;
    dynamic: SplatDynamicConfig;
    loss: LossConfig;
    refine: RefineDefaultsConfig;
    radix: RadixConfig;
    fragment: FragmentConstantsConfig;
}


export type SplattingConfigOverrides = Partial<{
    [K in keyof SplattingConfigInput]: Partial<SplattingConfigInput[K]>;
}>;

// The Final Compiled Context (includes all computed/derived constants for shaders/buffers)
export interface SplattingContext extends SplattingConfigInput {
    chunksX: number;
    chunksY: number;
    numChunks: number;
    maxSplatsPerChunk: number;
    radixElementsCount: number;
    radixElements: RadixElement[];
    numBarriers : number;
    radixSetupDispatches: number;
    numRadixSorts: number;
    groupChunkRatio: number;
    maxSlots: number;
}

export type ShaderCodeSnippet =  string | ((ctx : SplattingContext) => string);

export interface ShaderModulesDesciptors {
    modules : Record<string, GPUShaderModuleDescriptor>;
}