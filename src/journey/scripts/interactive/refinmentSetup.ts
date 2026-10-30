import { Pane } from "tweakpane";
import { getWebGPUctx } from "../../../myutils/ContextHelpers";
import { createLossPlot } from "../../../myutils/LogHelpers";
import { main } from "../optimization/mainParallel";
import { createSplattingContext } from "../../../config";
import { loadImageBitmap } from "../../../myutils/MyUtils";
import type { GuiVisibility } from "../../../config/types";

const ctxWebgpuNoRefine = await getWebGPUctx({ canvasId: "no-refine"});

if(ctxWebgpuNoRefine) {
    let plotHTMLElem = document.getElementById('loss-plot-no') ?? document.body
    let lossPlot = createLossPlot(plotHTMLElem);

    const testImageUrl = 'assets/fourBlobs.jpg';
    const source = await loadImageBitmap(testImageUrl);

    const pane = new Pane({
            container: document.getElementById("no-refine-slider") as HTMLElement,
    });

    const splatConfig = createSplattingContext({
        setup : { numSplats : 20, sampleDim : 128 },
        dynamic : { activeSplats : 2 },
        loss : { maxSteps : 50},
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
