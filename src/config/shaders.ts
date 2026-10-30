import type { ShaderCodeSnippet, SplattingContext } from "./types";
import { generateAtomicOperations } from "../myutils/ShaderTemplating";


const shadersJourney = import.meta.glob('../journey/scripts/shaders/*.wgsl', {
  query: '?raw',
  import: 'default',
  eager: true,
});

const shadersDefault = import.meta.glob('../shaders/*.wgsl', {
  query: '?raw',
  import: 'default',
  eager: true,
});


const rawShaders = { ...shadersJourney, ...shadersDefault };

// GEMINI: Transform the object keys from full paths to just filenames
const shaders = Object.fromEntries(
  Object.entries(rawShaders).map(([path, content]) => {
    const fileName = path.split('/').pop()?.replace('.wgsl', '') ?? path;
    return [fileName, content];
  })
);




const MACHINE_EPSILON = 1.19e-07;

export const CONSTANTS_F : ShaderCodeSnippet = (splatConfig : SplattingContext ) => {

    const radixIndex = (name: string): number => {
        const element = splatConfig.radixElements.find((radixElement) => radixElement.name === name);
        if (!element) {
            throw new Error(`Missing radix element "${name}" in splatConfig`);
        }
        return element.index;
    };

    // ADD NEW CONSTANTS HERE
    return Object.entries({
        NUM_SPLATS : splatConfig.setup.numSplats,
        xRAY : splatConfig.fragment.xRAY,
        sampleDim : splatConfig.setup.sampleDim,
        EPSILON : MACHINE_EPSILON,
        chunkWidth: splatConfig.setup.chunkWidth,
        chunkHeight: splatConfig.setup.chunkHeight,
        numChunksW : splatConfig.chunksX,
        numChunksH : splatConfig.chunksY,
        nSteps : splatConfig.loss.maxSteps,
        TILE_SCALE: splatConfig.fragment.tileScale,
        MAX_SPLATS_CHUNK : splatConfig.maxSplatsPerChunk,
        MAX_SLOTS: splatConfig.maxSlots,
        RADIX_ELEMS: splatConfig.radixElementsCount,
        R_CHUNK_INDEX: radixIndex('chunk'),
        R_SPLAT_INDEX: radixIndex('splat'),
        R_TRANSPOSE_INDEX: radixIndex('transpose'),
        GROUPSIZE: splatConfig.radix.groupSizeRadix,
        BARRIERS: splatConfig.numBarriers,
        NUM_DISPATCHES : splatConfig.radixSetupDispatches,
        NUM_SORTS : splatConfig.numRadixSorts,
        GROUP_CHUNK_RATIO: splatConfig.groupChunkRatio,
        OFF_CANVAS_TRHESHOLD : splatConfig.refine.offCanvasThreshold,
        OPACITY_PERCENTAGE : splatConfig.fragment.opacityPercentage,
        NUM_CHUNKS: splatConfig.numChunks
    }).map(([k, v]) => `const ${k} = ${v};`).join('\n');
};

// ADD NEW SHADER FILES HERE
export const registry = {
    sharedConstants: CONSTANTS_F,
    gradientBackwardsCompute : shaders['gradientBackwardsCompute'],
    gaussFunctions : shaders['gaussFunctions'],
    gaussTileVertStorage : shaders['gaussTileVertStorage'],
    gaussFragBlob : shaders['gaussFragBlob'],
    adam : shaders['adam'],
    adamOnGradients : shaders['adamOnGradients'],
    gradTypes : shaders['gradTypes'],
    radixSetup : shaders['radixSetup'],
    radixSort : shaders['radixSort'],
    radixTypes : shaders['radixTypes'],
    reduceGradientsRadix : shaders['reduceGradientsRadix'],
    reduceLoss : shaders['reduceLoss'],
    radixChunkOffsets : shaders['radixChunkOffsets'],
    opacityReset : shaders['opacityReset'],
    splatRefine : shaders['splatRefine'],
    atomicOperations : generateAtomicOperations(),
} satisfies Record<string, ShaderCodeSnippet>;

export type ShaderName = keyof typeof registry;

export interface ShaderModuleConfig {
    label: string;
    shaders: ShaderName[];
};