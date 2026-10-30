import { vec2, vec3 } from "wgpu-matrix";
import type { UniformBufferDescriptor } from "../mytypes";

async function loadImageBitmap(url : string) : Promise<ImageBitmap> {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: 'none'});
}

function  randomSplatInit(
    n : number,
    paramDesc : UniformBufferDescriptor,
    input : Float32Array) {

    if(paramDesc.attributes.length < 5 || !paramDesc.unitSize) {
        throw Error("cannot initalize Splats, wrong buffer descriptor");
    }
    const minimumS = 2;
    const posOff    = paramDesc.attributes[0].offset;
    const scaleOff  = paramDesc.attributes[1].offset;
    const rotOff    = paramDesc.attributes[2].offset;
    const colorOff  = paramDesc.attributes[3].offset;
    const alphaOff  = paramDesc.attributes[4].offset;
    const unitS = paramDesc.unitSize!;

    for(let i = 0; i < n; i++) {
        let rndPos = vec2.clamp(vec2.create(Math.random(), Math.random()), 0.1, 0.9);
        input.set(rndPos, posOff + unitS * i);

        let rndScale = vec2.add(
            vec2.create(minimumS,minimumS), 
            vec2.scale(vec2.create(Math.random(), Math.random()), 2.0)
        );
        input.set(rndScale, scaleOff + unitS * i);

        let rndRot = Math.PI * Math.random();
        input.set([rndRot], rotOff + unitS * i);

        let rndColor = vec3.add(
            vec3.create(1),
            vec3.scale(vec3.create(Math.random(), Math.random(), Math.random()), 9.0)
        );
        input.set(rndColor, colorOff + unitS * i);

        let rndAlpha = 4.0 + Math.random() * 4.0;
        input.set([rndAlpha], alphaOff + unitS * i);
    }
}

function imageGridSplats(
    n: number,
    width : number,
    height : number,
    paramDesc : UniformBufferDescriptor,
    input : Float32Array
) {
    if(paramDesc.attributes.length < 5 || !paramDesc.unitSize) {
        throw Error("cannot initalize Splats, wrong buffer descriptor");
    }

    const posOff    = paramDesc.attributes[0].offset;
    const scaleOff  = paramDesc.attributes[1].offset;
    const rotOff    = paramDesc.attributes[2].offset;
    const colorOff  = paramDesc.attributes[3].offset;
    const alphaOff  = paramDesc.attributes[4].offset;
    const unitS = paramDesc.unitSize;


    const cols = Math.ceil(Math.sqrt(n * width / height));
    const rows = Math.ceil(n / cols);
    const scaleX = width / cols;
    const scaleY = height / rows;
    const logScaleX = Math.log(3000 / scaleX);
    const logScaleY = Math.log(3000 / scaleY);

    for (let i = 0; i < n; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);

        const x0 = Math.floor(col * width / cols);
        const x1 = Math.floor((col + 1) * width / cols);
        const y0 = Math.floor(row * height / rows);
        const y1 = Math.floor((row + 1) * height / rows);

        input.set([
            (col + 0.5) / cols,
            (row + 0.5) / rows,
        ], posOff + unitS * i);

        input.set(
            [logScaleX, logScaleY],
            scaleOff + unitS * i
        );

        input.set([0], rotOff + unitS * i);

        input.set([
                8,
                8,
                8,
        ], colorOff + unitS * i);

        input.set([4], alphaOff + unitS * i);
    }
}

function exampleSplatsRedBlueInit(    
    paramDesc : UniformBufferDescriptor,
    input : Float32Array
    ) {

    if(paramDesc.attributes.length < 5 || !paramDesc.unitSize) {
        throw Error("cannot initalize Splats, wrong buffer descriptor");
    }

    const posOff    = paramDesc.attributes[0].offset;
    const scaleOff  = paramDesc.attributes[1].offset;
    const rotOff    = paramDesc.attributes[2].offset;
    const colorOff  = paramDesc.attributes[3].offset;
    const alphaOff  = paramDesc.attributes[4].offset;
    const unitS = paramDesc.unitSize;

    // gauss 1
    input.set([0.9, 0.3],       posOff);
    input.set([1.5, 1.5],       scaleOff);
    input.set([0.0],            rotOff);
    input.set([8.0, 1.0, 1.0],  colorOff);
    input.set([4.0],            alphaOff);

    // gauss 2
    input.set([0.3, 0.6],       unitS + posOff);
    input.set([0.7, 1.0],       unitS + scaleOff);
    input.set([0.0],            unitS + rotOff);
    input.set([1.0, 1.0, 8.0],  unitS + colorOff);
    input.set([4.0],            unitS + alphaOff);
}

export { loadImageBitmap, randomSplatInit, imageGridSplats, exampleSplatsRedBlueInit };
