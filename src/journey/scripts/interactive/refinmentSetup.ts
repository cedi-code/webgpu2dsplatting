import { Pane } from "tweakpane";
import { getWebGPUctx } from "../../../myutils/ContextHelpers";
import { createLossPlot } from "../../../myutils/LogHelpers";
import { main } from "../optimization/mainParallel";
import { createSplattingContext } from "../../../config";
import { loadImageBitmap } from "../../../myutils/MyUtils";
import type { GuiVisibility } from "../../../config/types";


const ctxWebgpuNoRefine = await getWebGPUctx({ canvasId: "no-refine"});

if(ctxWebgpuNoRefine) {
    const splatConfig = createSplattingContext({
        setup : { numSplats : 20, sampleDim : 64 },
        dynamic : { activeSplats : 2 },
        loss : { maxSteps : 100},
    });
   
    let plotHTMLElem = document.getElementById('loss-plot-no') ?? document.body
    let lossPlot = createLossPlot(plotHTMLElem);

    const testImageUrl = 'assets/fourBlobs.jpg';
    const source = await loadImageBitmap(testImageUrl);

    const pane = new Pane({
            container: document.getElementById("no-refine-slider") as HTMLElement,
    });



    const guiConfig : Partial<GuiVisibility> = { blobs: true, speed : true, gradientDescent : true, };
    console.log("running main");
    main({ 
        webgpu: ctxWebgpuNoRefine,
        config : splatConfig,
        init: 'example',
        gui : guiConfig,
        image : source,
        pane: pane,
        plot : lossPlot 
    });
}

const ctxWebgpuNoRefine2 = await getWebGPUctx({ canvasId: "with-refine"});

if(ctxWebgpuNoRefine2) {
     const splatConfig = createSplattingContext({
        setup : { numSplats : 200, sampleDim : 64 },
        dynamic : { 
            activeSplats : 2,
            refineAfter : 150, 
            opacityResetActive : false },
        loss : { maxSteps : 400},
    });

    let plotHTMLElem2 = document.getElementById('loss-plot-color') ?? document.body
    let lossPlot2 = createLossPlot(plotHTMLElem2);

    const testImageUrl = 'assets/fourBlobs.jpg';
    const source = await loadImageBitmap(testImageUrl);

    const pane = new Pane({
            container: document.getElementById("with-refine-slider") as HTMLElement,
    });



    const guiConfig : Partial<GuiVisibility> = {
        blobs: true, 
        speed : true,
        gradientDescent : true, 
        refinement : true,
        
    };
    console.log("running main");
    main({ 
        webgpu: ctxWebgpuNoRefine2,
        config : splatConfig,
        init: 'example',
        gui : guiConfig,
        image : source,
        pane: pane,
        plot : lossPlot2 
    });
}
