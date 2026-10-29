import { type SplattingConfigInput } from "./types";

export const DEFAULT_CONFIG: SplattingConfigInput = {
    setup: {
        numSplats: 2000,
        sampleDim: 512,
        chunkWidth: 16,
        chunkHeight: 16,
    },
    dynamic: {
        activeSplats: 1000,
        refineAfter: 300,
        opacityResetAfter: 30,
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