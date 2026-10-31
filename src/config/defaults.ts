import type { ShaderModuleConfig } from "./shaders";
import { type GuiVisibility, type SplattingConfigInput } from "./types";

export const DEFAULT_CONFIG: SplattingConfigInput = {
    setup: {
        numSplats: 1000,
        sampleDim: 512,
        chunkWidth: 16,
        chunkHeight: 16,
    },
    dynamic: {
        activeSplats: 500,
        refineAfter: 300,
        refineActive : true,
        opacityResetAfter: 30,
        opacityResetActive : true,
    },
    loss: {
        maxSteps: 100,
    },
    refine: {
        gradientPosThresh: 0.002,
        gradientScaleThresh: 0.02,
        opacityResetV: 0.99,
        alphaCutoff: 0.1,
        scaleDivideConst: 1.6,
        offCanvasThreshold: 1.2,
    },
    radix: {
        groupSizeRadix: 256,
        elementNames: [
            "chunk",
            "splat",
            "transpose"
        ]
    },
    fragment: {
        xRAY: false,
        opacityPercentage: 0.6,
        tileScale: 4,
    },
};

export const PARALLEL_MODULES : ShaderModuleConfig[] = [
    {
        label: 'backwards pass',
        shaders: [
            "sharedConstants",
            "gradTypes",
            "radixTypes",
            "atomicOperations",
            "gaussFunctions",
            "adam",
            "gradientBackwardsCompute",
        ]
    },
    {
        label: 'adam pass',
        shaders: [
            "sharedConstants",
            "gradTypes",
            "radixTypes",
            "atomicOperations",
            "adam",
            "adamOnGradients",
        ]
    },
    {
        label: 'radix setup pass',
        shaders: [
            "sharedConstants",
            "gradTypes",
            "radixTypes",
            "gaussFunctions",
            "radixSetup",
        ]
    },
    {
        label: 'splat refine pass',
        shaders: [
            "sharedConstants",
            "gradTypes",
            "radixTypes",
            "gaussFunctions",
            "adam",
            "splatRefine",
        ]
    },
    {
        label: 'radix pass',
        shaders: [
            "radixTypes",
            "sharedConstants",
            "radixSort",
        ]
    },
    {
        label: 'radix offset pass',
        shaders: [
            "radixTypes",
            "sharedConstants",
            "radixChunkOffsets",
        ]
    },
    {
        label: 'reduce pass',
        shaders: [
            "sharedConstants",
            "gradTypes",
            "radixTypes",
            "adam",
            "reduceGradientsRadix",
        ]
    },
    {
        label: 'loss reduce',
        shaders: [
            "sharedConstants",
            "gradTypes",
            "adam",
            "reduceLoss",
        ]
    },
    {
        label: 'opacity reset',
        shaders: [
            "sharedConstants",
            "gradTypes",
            "opacityReset",
        ]
    },
    {
        label: '2d static tile',
        shaders: [
            "sharedConstants",
            "gradTypes",
            "gaussFunctions",
            "gaussTileVertStorage",
        ]
    },
    {
        label: 'simple texture frag impl',
        shaders: [
            "sharedConstants",
            "gaussFragBlob",
        ]
    }
];

export const DEFAULT_GUI_VISIBILITY: GuiVisibility = {
    blobs: true,
    lossWeight: true,
    nSteps: true,
    speed: true,
    refinement: true,
    refinementParams: true,
    gradientDescent: true,
};