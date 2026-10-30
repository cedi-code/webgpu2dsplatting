import { DEFAULT_CONFIG, PARALLEL_MODULES } from './defaults';
import { registry, type ShaderModuleConfig } from './shaders';
import { type SplattingConfigOverrides,
    type SplattingContext, type RadixElement, type ShaderModulesDesciptors,
    type ShaderCodeSnippet} from './types';

export function createSplattingContext(
    userConfig?: SplattingConfigOverrides
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

export function createShaderModuleDescriptors(
    moduleConfig : ShaderModuleConfig[],
    splatCtx : SplattingContext
) : ShaderModulesDesciptors {

    // map operation, turns moduleConfig into ModuleDescriptor    
    return {
        modules: moduleConfig.reduce<Record<string, GPUShaderModuleDescriptor>>
        (
            (module, config) => {

                // goes throu each code snippet and figures out if it needs the shader context or not
                const codeArr = config.shaders.map((shaderName) : ShaderCodeSnippet =>
                {
                    return registry[shaderName];
                });

                const codeCombined = codeArr.reduce((strSum, currCode) => {
                    if(typeof currCode == 'function') {
                        return (strSum as string) + currCode(splatCtx);
                    }
                    else {
                        return (strSum as string) + currCode;
                    }
                }, "") as string;

                module[config.label] = {
                    label: config.label + "Module",
                    code: codeCombined,
                };

                return module;
            },
            {}
        )
    };
}

export const defaultSplatConfig = createSplattingContext(); 
export const defaultModuleDescriptorsParallel = createShaderModuleDescriptors(
    PARALLEL_MODULES,
    defaultSplatConfig
);