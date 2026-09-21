import type { UniformBufferDescriptor } from '../mytypes';
import { sigmoid } from './MathHelpers';

import uPlot from 'uplot';


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
            r:      r(sigmoid(data[colorOff], 4.0)),
            g:      r(sigmoid(data[colorOff + 1], 4.0)),
            b:      r(sigmoid(data[colorOff + 2], 4.0)),
            alpha:  r(sigmoid(data[alphaOff], 4.0)),
        };
}

function maxLabelWidth(self : uPlot, axis : uPlot.Axis, values: string[]) {
    let ctx = self.ctx;
    let width = 0;

    if(!axis.font) {
        return width;
    }
    // Preserve the canvas state so the font cache stays valid.
    ctx.save();
    ctx.font = axis.font[0];

    for (let value of values ?? []) {
        if (value != null)
            width = Math.max(width, ctx.measureText(String(value)).width);
    }

    ctx.restore();

    return width / uPlot.pxRatio;
}



export function createLossPlot(plotHTMLBody : HTMLElement, opts? : uPlot.Options ) : uPlot {

    if(!opts) {
        opts = {
            title: "Loss graph",
            width: 300,
            height: 256,
            scales: {
                x: {
                    time: false,
                //	auto: false,
                //	range: [0, 6],
                },
            },
            series: [
                {
                    label: "step",
                },
                {
                    label: "loss",
                    stroke: "red",
                }
            ],
            axes: [
                {
                    label: "Steps",
                    // scale: '%',
                    values(self, splits) {
                        return splits.map(s => +s.toFixed(2));
                    }
                },
                {
                    label: "L2",
                    labelGap: 8,
                    // scale: '%',
                    stroke: "red",
                }
            ],
        };
    }
    let lossDataPlot : uPlot.AlignedData = [];

    return new uPlot(opts, lossDataPlot, plotHTMLBody);

    //     u.setData(getData(points, mult *= 10));    
}

export function creatGaussComparePlot(plotHTMLBody : HTMLElement) : uPlot {

    const opts : uPlot.Options = {
        title: "Curves",
        width: 512,
        height: 256,
        scales: {
            x: {
                time: false,
            },
        },
        series: [
            {
                label: "sample",
            },
            {
                label: "approx",
                stroke: "red",
            },
            {
                label: "solution",
                stroke: "blue",
            }
        ],
        axes: [
            {
                label: "x",
                // scale: '%',
                values(self, splits) {
                    return splits.map(s => +s.toFixed(2));
                }
            },
            {
                label: "f(x)",
                labelGap: 8,
                // scale: '%',
                stroke: "black",
            }
        ],
    };


    let lossDataPlot : uPlot.AlignedData = [];

    return new uPlot(opts, lossDataPlot, plotHTMLBody);

    //     u.setData(getData(points, mult *= 10));    
}
