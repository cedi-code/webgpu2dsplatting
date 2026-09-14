import type { UniformBufferDescriptor } from '../mytypes';
import { sigmoid } from './MathHelpers';

export type FlatParams = {
    posX: number;
    posY: number;
    scaleX: number;
    scaleY: number;
    rot: number;
    r: number;
    g: number;
    b: number;
    alpha: number;
};

export function parseParams(data : Float32Array, desc : UniformBufferDescriptor, round? : number) : FlatParams {
    const posOff    = desc.attributes[0].offset;
    const scaleOff  = desc.attributes[1].offset;
    const rotOff    = desc.attributes[2].offset;
    const colorOff  = desc.attributes[3].offset;
    const alphaOff  = desc.attributes[4].offset;    
    
    // the + makes it go back to nuber
    const r = (n: number) : number => +n.toFixed(round ?? 3);
    return {
            posX:   r(data[posOff]),
            posY:   r(data[posOff + 1]),
            scaleX: r(Math.exp(data[scaleOff])),
            scaleY: r(Math.exp(data[scaleOff + 1])),
            rot:    r(data[rotOff]),
            r:      r(sigmoid(data[colorOff])),
            g:      r(sigmoid(data[colorOff + 1])),
            b:      r(sigmoid(data[colorOff + 2])),
            alpha:  r(sigmoid(data[alphaOff])),
        };
}