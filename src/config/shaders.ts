import type { ShaderCodeSnippet, SplattingContext } from "./types";
import { generateAtomicOperations } from "../myutils/ShaderTemplating";

import shaderCodeCompute from '../journey/scripts/shaders/gradientBackwardsCompute.wgsl?raw';
import gaussFrag from '../journey/scripts/shaders/gaussFragBlob.wgsl?raw';
import gaussTileVertStorage from '../journey/scripts/shaders/gaussTileVertStorage.wgsl?raw';
import adamCompute from '../journey/scripts/shaders/adamOnGradients.wgsl?raw';
import radixSetup from '../journey/scripts/shaders/radixSetup.wgsl?raw';
import radixCompute from '../journey/scripts/shaders/radixSort.wgsl?raw';
import reduceRadixCompute from '../journey/scripts/shaders/reduceGradientsRadix.wgsl?raw';
import reduceLossCompute from '../journey/scripts/shaders/reduceLoss.wgsl?raw';
import radixOffsets from '../journey/scripts/shaders/radixChunkOffsets.wgsl?raw';
import opacityReset from '../journey/scripts/shaders/opacityReset.wgsl?raw';

import shaderGaussFunctions from '../shaders/gaussFunctions.wgsl?raw';
import adamShad from '../shaders/adam.wgsl?raw';
import shaderGradTypes from '../shaders/gradTypes.wgsl?raw';
import radixTypes from '../shaders/radixTypes.wgsl?raw'

const MACHINE_EPSILON = 1.19e-07;

const CONSTANTS_F : ShaderCodeSnippet = (splatConfig : SplattingContext ) => {

    const radixIndex = (name: string): number => {
        const element = splatConfig.radixElements.find((radixElement) => radixElement.name === name);
        if (!element) {
            throw new Error(`Missing radix element "${name}" in splatConfig`);
        }
        return element.index;
    };

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
export const registry = {
    sharedConstants: CONSTANTS_F,
    gradientBackwardsCompute : shaderCodeCompute,
    gaussFunctions : shaderGaussFunctions,
    gaussTileVertStorage : gaussTileVertStorage,
    gaussFragBlob : gaussFrag,
    adam : adamShad,
    adamOnGradients : adamCompute,
    gradTypes : shaderGradTypes,
    radixSetup : radixSetup,
    radixSort : radixCompute,
    radixTypes : radixTypes,
    reduceGradientsRadix : reduceRadixCompute,
    reduceLoss : reduceLossCompute,
    radixChunkOffsets : radixOffsets,
    opacityReset : opacityReset,
    atomicOperations : generateAtomicOperations(),
} satisfies Record<string, ShaderCodeSnippet>;