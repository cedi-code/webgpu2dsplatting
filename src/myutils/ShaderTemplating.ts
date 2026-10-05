import { type UniformBaseType } from "../mytypes";
import { getByteBaseSize } from "./BufferHelper";

type Precision = 'i32' | 'u32';

const atomicVecType = (t: UniformBaseType, p: Precision): string => {
    const size = getByteBaseSize(t) / 4.0;
    return size > 1 ? `array<atomic<${p}>, ${size}>` : `atomic<${p}>`;
};

const atomicFnName = (t: UniformBaseType, p: Precision, space : string, operation : string) => {
    return `fn atomic${operation}${t}_${space}${p === 'u32' ? `_u` : ""}`
};

const generateAtomicOperations : (() => string) = () => {
    
    type Options = {
        spaces : string[],
        atomicTypes : UniformBaseType[],
        precision : Precision[],
    }
    const options : Options = {
        spaces : ['storage', 'workgroup'],
        atomicTypes : ['f32', 'vec2f', 'vec3f'],
        precision: [ 'i32', 'u32'],
    };


    // https://toji.dev/webgpu-best-practices/compute-vertex-data#synchronizing-data-access-with-atomics
    const QUANTIZE_FACTOR: number = 32768.0;
    const DEQUANTIZE_FACTOR: number = 1.0 / 32768.0;

    const QUANTIZE_FACTOR_UNSIGNED: number = QUANTIZE_FACTOR * 2.0;
    const DEQUANTIZE_FACTOR_UNSIGNED: number = DEQUANTIZE_FACTOR * 0.5;



    const generateAtomicAdd = (space : string, t : UniformBaseType, precision : 'u32' | 'i32') => {

        const size = getByteBaseSize(t) / 4.0;
        const isUnsigned = precision === 'u32';
        const factorName = isUnsigned ? 'QUANTIZE_FACTOR_UNSIGNED' : 'QUANTIZE_FACTOR';

        const accessMode = space === 'storage' ? ', read_write' : '';

        const qType = size === 1 ? precision : `vec${size}${precision[0]}`;

        const name = atomicFnName(t,precision,space,"Add");
        let functionHeader = `${name}
            (ptr_val: ptr<${space}, 
                    ${ atomicVecType(t, precision)}, 
                    ${accessMode}
                    >, 
            value: ${t}) 
        `;

        let functionQ = `let q = ${qType}(value * ${factorName});`;
        let functionAdds = ``
        if (size === 1) {
            functionAdds = 'atomicAdd(ptr_val, q);\n';
        } else {
            for (let i = 0; i < size; i++) {
                functionAdds += `atomicAdd(&((*ptr_val)[${i}]), q[${i}]);\n`;
            }
        }
        return functionHeader + `{\n` + functionQ + `\n` + functionAdds + `\n}`;
    };

    let result = ``;
    result += `const QUANTIZE_FACTOR: f32 = ${QUANTIZE_FACTOR.toFixed(1)};\n`;
    result += `const DEQUANTIZE_FACTOR: f32 = ${DEQUANTIZE_FACTOR};\n`;
    result += `const QUANTIZE_FACTOR_UNSIGNED: f32 = ${QUANTIZE_FACTOR_UNSIGNED.toFixed(1)};\n`;
    result += `const DEQUANTIZE_FACTOR_UNSIGNED: f32 = ${DEQUANTIZE_FACTOR_UNSIGNED};\n\n`;

    const generateAtomicLoad = (space: string, t: UniformBaseType, precision: 'u32' | 'i32') => {
            const size = getByteBaseSize(t) / 4.0;
            const isUnsigned = precision === 'u32';
            const dequantizeFactor = isUnsigned ? 'DEQUANTIZE_FACTOR_UNSIGNED' : 'DEQUANTIZE_FACTOR';
            const accessMode = space === 'storage' ? ', read_write' : '';

            const name = atomicFnName(t,precision,space,"Load");
            let functionHeader = `${name}
            (
                p: ptr<${space}, ${atomicVecType(t, precision)}${accessMode}>
            ) -> ${t}`;

            let body = ``;
            if (size === 1) {
                body = `return f32(atomicLoad(p)) * ${dequantizeFactor};\n`;
            } else {
                let args = ``;
                for (let i = 0; i < size; i++) {
                    args += (i > 0 ? `, ` : ``) + `f32(atomicLoad(&((*p)[${i}])))`;
                }
                body = `return ${t}(${args}) * ${dequantizeFactor};\n`;
            }
            return functionHeader + `{\n` + body + `\n}`;
    };

    for (let s = 0; s < options.spaces.length; s++) {
        const space = options.spaces[s];

        for (let t = 0; t < options.atomicTypes.length; t++) {
            const type : UniformBaseType = options.atomicTypes[t];

            for (let p = 0; p < options.precision.length; p++) {
                const prec = options.precision[p];

                result += generateAtomicAdd(space, type, prec) + `\n\n`;
                result += generateAtomicLoad(space, type, prec) + `\n\n`;
            }
        }
    }

    return result;
}

export { generateAtomicOperations }