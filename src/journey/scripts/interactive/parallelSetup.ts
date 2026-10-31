import { Pane } from "tweakpane";
import { getWebGPUctx } from "../../../myutils/ContextHelpers";
import { createLossPlot } from "../../../myutils/LogHelpers";
import { main } from "../optimization/mainParallel";
import { loadImageBitmap } from "../../../myutils/MyUtils";
import { createSplattingContext, defaultSplatConfig } from "../../../config";
import { DEFAULT_GUI_VISIBILITY } from "../../../config/defaults";
import type { GuiVisibility } from "../../../config/types";

const ctxWebgpu = await getWebGPUctx({ canvasId: "parallel-jeff"});

if(ctxWebgpu) {
    let plotHTMLElem = document.getElementById('loss-plot-jeff') ?? document.body
    let lossPlot = createLossPlot(plotHTMLElem);

    const splatConfig = createSplattingContext({
        setup : { numSplats : 1100, sampleDim : 256 },
        dynamic : { 
            activeSplats : 1000, 
            refineActive : false, 
            opacityResetActive : false
        },
        loss : { maxSteps : 100},
    });

    const guiConfig : Partial<GuiVisibility> = { blobs: true, speed : true, gradientDescent : true, };
    

    const testImageUrl = 'assets/jeff.jpg';
    const source = await loadImageBitmap(testImageUrl);

    const pane = new Pane({
            container: document.getElementById("slider-jeff") as HTMLElement,
    });

    main({ 
        webgpu: ctxWebgpu,
        config : splatConfig,
        init: "grid",
        gui : guiConfig,
        image : source,
        pane: pane,
        plot : lossPlot 
    });
}