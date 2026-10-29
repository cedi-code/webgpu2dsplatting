import { DEFAULT_CONFIG } from './defaults';
import { type SplattingConfigInput, type SplattingContext, type RadixElement } from './types';

export function createSplattingContext(
    userConfig?: Partial<{
        [K in keyof SplattingConfigInput]: Partial<SplattingConfigInput[K]>;
    }>
): SplattingContext {
    const setup = { ...DEFAULT_CONFIG.setup, ...userConfig?.setup };
    const dynamic = { ...DEFAULT_CONFIG.dynamic, ...userConfig?.dynamic };
    const loss = { ...DEFAULT_CONFIG.loss, ...userConfig?.loss };
    const refine = { ...DEFAULT_CONFIG.refine, ...userConfig?.refine };
    const radix = { ...DEFAULT_CONFIG.radix, ...userConfig?.radix };
    const fragment = { ...DEFAULT_CONFIG.fragment, ...userConfig?.fragment };

    const chunksX = Math.floor((setup.sampleDim + setup.chunkWidth - 1) / setup.chunkWidth);
    const chunksY = Math.floor((setup.sampleDim + setup.chunkHeight - 1) / setup.chunkHeight);
    const numChunks = chunksX * chunksY;

    const maxSplatsPerChunk = Math.min(setup.numSplats, 256);

    const numBarriers = Math.log2(radix.groupSizeRadix);

    // Map input objects to full RadixElement objects with auto-assigned indices
    const radixElements: RadixElement[] = radix.elementNames.map((el, index) => ({
        name: el,
        index, // based on position
    }));
    const radixElementsCount = radixElements.length;

    const radixSetupDispatches = Math.floor((setup.numSplats + radix.groupSizeRadix - 1) / radix.groupSizeRadix);
    const numRadixSorts = 1 + Math.floor(Math.log(numChunks) / Math.log(radix.groupSizeRadix));
    const groupChunkRatio = 1 + Math.floor(numChunks / radix.groupSizeRadix);
    
    // this might change!
    const maxSlots = numChunks * radixElementsCount * maxSplatsPerChunk;

    return {
        // config values
        setup,
        dynamic,
        loss,
        refine,
        radix,
        fragment,
        // Computed fields
        chunksX,
        chunksY,
        numChunks,
        maxSplatsPerChunk,
        radixElements,
        numBarriers,
        radixElementsCount,
        radixSetupDispatches,
        numRadixSorts,
        groupChunkRatio,
        maxSlots,
    };
}

export const splatConfig = createSplattingContext();