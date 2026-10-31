import { Pane } from "tweakpane";
import { getWebGPUctx } from "../../../myutils/ContextHelpers";
import { createLossPlot } from "../../../myutils/LogHelpers";
import { main } from "../optimization/mainParallel";
import { loadImageBitmap } from "../../../myutils/MyUtils";
import { defaultSplatConfig } from "../../../config";
import { DEFAULT_GUI_VISIBILITY } from "../../../config/defaults";

const ctxWebgpu = await getWebGPUctx({ canvasId: "main"});

if(ctxWebgpu) {
    let plotHTMLElem = document.getElementById('loss-plot') ?? document.body
    let lossPlot = createLossPlot(plotHTMLElem);

    const testImageUrl = 'assets/cat.jpg';
    const source = await loadImageBitmap(testImageUrl);

    const pane = new Pane({
            container: document.getElementById("gd-sliders") as HTMLElement,
    });

    main({ 
        webgpu: ctxWebgpu,
        config : defaultSplatConfig,
        init: "grid",
        gui : DEFAULT_GUI_VISIBILITY,
        image : source,
        pane: pane,
        plot : lossPlot 
    });
}