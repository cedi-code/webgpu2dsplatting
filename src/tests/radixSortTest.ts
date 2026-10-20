
import { bufferManager, UniformBufferDescriptorBuilder, VertexBufferDescriptorBuilder } from '../myutils/BufferHelper';
import { getWebGPUctx, render } from '../myutils/ContextHelpers';


import radixCompute from '../journey/scripts/shaders/radixSort.wgsl?raw';
import radixTypes from '../shaders/radixTypes.wgsl?raw';
import radixOffsets from '../journey/scripts/shaders/radixChunkOffsets.wgsl?raw';

const MACHINE_EPSILON = 1.19e-07;

async function main() {

    const ctx = await getWebGPUctx({ canvasId: "test"});
    if(!ctx) {
        return;
    }

    // number of splats
    const numSplats = 2; // hard limit is 1 splat per pixel?
    const sampleDim = 2;
    const chunkWidth = 1;
    const chunkHeight = 1;
    const chunksX = Math.floor(((sampleDim + chunkWidth - 1)  / chunkWidth));
    const chunksY = Math.floor(((sampleDim + chunkHeight - 1)  / chunkHeight));
    const numChunks = chunksX * chunksY;
    console.log(numChunks);

    const radixElements = 3; // splat id, chunk id, permutation id
    const groupSizeRadix = 256;
    const barriersPrefixSum = 8; // log2(groupSizeRadix)

    // constants
    const CONSTANTS = {
        NUM_SPLATS : numSplats,
        xRAY : false,
        sampleDim : sampleDim,
        EPSILON : MACHINE_EPSILON,
        chunkWidth: chunkWidth,
        chunkHeight: chunkHeight,
        numChunksW : chunksX,
        numChunksH : chunksY,
        TILE_SCALE: 4,
        MAX_SLOTS: numChunks * radixElements * 2,
        RADIX_ELEMS: radixElements, 
        R_CHUNK_INDEX: 0,
        R_SPLAT_INDEX: 1,
        R_TRANSPOSE_INDEX: 2,
        GROUPSIZE: groupSizeRadix,
        BARRIERS: barriersPrefixSum,
        NUM_DISPATCHES : 1,
        NUM_SORTS : 2,
    };

    console.table(CONSTANTS);

    const sharedConstants = Object.entries(CONSTANTS)
    .map(([k, v]) => `const ${k} = ${v};`)
    .join('\n');

    const radixModule = ctx.device.createShaderModule({
        label: 'radix pass',
        code: (
            radixTypes +
            sharedConstants +
            radixCompute
        )
    });

    const radixOffModule = ctx.device.createShaderModule({
        label: 'radix pass',
        code: (
            radixTypes +
            sharedConstants +
            radixOffsets
        )
    });

    const radixBuild = new UniformBufferDescriptorBuilder('radix memory', "storage", "copy_src_dst");

    radixBuild.add("splatOffsets", { type : "u32", size : CONSTANTS.NUM_DISPATCHES })
              .add("offsets", { type : "u32", size: numChunks })
              .add("data", { type : "u32", size: CONSTANTS.MAX_SLOTS })
              .add("data2", { type : "u32", size: CONSTANTS.MAX_SLOTS })
              .add("splatOffsetsTotal", { type : "u32", size : CONSTANTS.NUM_SPLATS });

    const radixDesc = radixBuild.build();

    const bindGroupLayoutDescriptorsRadix = ctx.device.createBindGroupLayout(
        {
            entries: [
            { // uniforms
                binding: 8,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'storage',
                    minBindingSize: radixDesc.sizeBytes,

                },
                },
            ],
        },
    );

    const pipelineLayoutRadix = ctx.device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayoutDescriptorsRadix ],
    });

        
    const pipelineRadix0 = ctx.device.createComputePipeline({
        label: 'radix compute pipeline pass 0',
        layout: pipelineLayoutRadix,
        compute: {
            module: radixModule,
            constants: {
                pass_nr: 0,
            }
        },
    });

    const pipelineRadix1 = ctx.device.createComputePipeline({
        label: 'radix compute pipeline pass 0',
        layout: pipelineLayoutRadix,
        compute: {
            module: radixModule,
            constants: {
                pass_nr: 1,
            }
        },
    });

    const pipelineOffset = ctx.device.createComputePipeline({
        label: 'offset calc',
        layout: pipelineLayoutRadix,
        compute: {
            module: radixOffModule,
        },
    });

    // creating buffer
    bufferManager.init(ctx.device);

    // radix buffer setup

    /*
    struct RadixMemory {
        splatOffsets : array<u32, NUM_DISPATCHES>,
        offsets : array<u32, numChunksW * numChunksH>,
        data2 : array<u32, MAX_SLOTS>,
        data : array<u32, MAX_SLOTS>,
        splatOffsetsTotal: array<u32, NUM_SPLATS>,
    };
    */

    const radixBuff = bufferManager.createBuffer(radixDesc);
    const radixV = new Uint32Array(radixDesc.size);
    const radixCounters = new Uint32Array(radixDesc.attributes[1].offset);

    // TODO: initalize radixV for testing
    const rSplatOff = radixDesc.attributes[0].offset;
    const rOffOff   = radixDesc.attributes[1].offset;
    const rData2Off = radixDesc.attributes[2].offset;
    const rDataOff  = radixDesc.attributes[3].offset;
    const rSplTot   = radixDesc.attributes[4].offset;

    const testSize = 6 * CONSTANTS.RADIX_ELEMS;
    radixV.set([testSize], rSplatOff);
    radixV.set([ // (splat id, chunk id, _)
        3, 98, 0,
        0, 98, 0,
        0, 98, 0,
        3, 99, 0,
        1, 99, 0,
        2, 99, 0,
    ], rDataOff);


    ctx.device.queue.writeBuffer(radixBuff, 0, radixV, 0);

    // for testing purposes
    const radixBuffRead = ctx.device.createBuffer({
        label: 'loss result buffer',
        size: radixDesc.sizeBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    // ==
    const bindGroupRadix = ctx.device.createBindGroup({
        label: 'bindGroup backwards pass',
        layout: pipelineRadix0.getBindGroupLayout(0),
        entries: [
            { binding: 8, resource: radixBuff },
        ]
    });

    let readAndPrintData = async (
        buff: GPUBuffer, 
        TypedArrayConstructor: new (buffer: ArrayBuffer) => Float32Array | Uint32Array
    ) => {
        await buff.mapAsync(GPUMapMode.READ);
        const result = new TypedArrayConstructor(buff.getMappedRange());
        const resultData : { 'chunkid' : number, 'splatid' : number, 'inv' : number }[] = []
        const data2Res = result.subarray(rDataOff, rSplTot); 
        for(let i = 0; i < CONSTANTS.MAX_SLOTS; i+=CONSTANTS.RADIX_ELEMS) {
            resultData.push(
                {
                    'chunkid' : data2Res[i + 0],
                    'splatid' : data2Res[i + 1],
                    'inv'     : data2Res[i + 2],
                }
            );
        }

        console.table(resultData);
        buff.unmap();
    }

    let readAndPrintOff = async (
        buff: GPUBuffer, 
        TypedArrayConstructor: new (buffer: ArrayBuffer) => Float32Array | Uint32Array
    ) => {
        await buff.mapAsync(GPUMapMode.READ);
        const result = new TypedArrayConstructor(buff.getMappedRange());
        const resultData : number[] = []
        const data2Res = result.subarray(rOffOff, rData2Off); 
        data2Res.forEach((elem : number) => {
            resultData.push(elem);
        })
        console.table(resultData);
        buff.unmap();
    }


    let runRadix = async () => {


        const encoder = ctx.device.createCommandEncoder({
            label: 'gd encoder',
        });

        const pass = encoder.beginComputePass({
            label: 'radix compute pass',
        });

        
        // radix sort
        pass.setPipeline(pipelineRadix0);
        pass.setBindGroup(0, bindGroupRadix);
        pass.dispatchWorkgroups(1); // only one instance, this could be improved! with using storage buffers more?

        pass.setPipeline(pipelineRadix1);
        pass.dispatchWorkgroups(1); // only one instance, this could be improved! with using storage buffers more?

        pass.setPipeline(pipelineOffset);
        const gr = Math.floor((numChunks + (CONSTANTS.GROUPSIZE -1)) / CONSTANTS.GROUPSIZE)
        pass.dispatchWorkgroups(gr);
        pass.end();


        // copy buffer to cpu (expensive!)
        encoder.copyBufferToBuffer(radixBuff, 0, radixBuffRead, 0, radixBuffRead.size);
        // run the work lmao
        const commandBuffer = encoder.finish();
        ctx.device.queue.submit([commandBuffer]);
        
        // print buffer  
        // readAndPrintData(radixBuffRead, Uint32Array);
        readAndPrintOff(radixBuffRead, Uint32Array);
    }

    runRadix();
    
}

main();