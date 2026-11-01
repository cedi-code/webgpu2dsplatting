import { Pane } from "tweakpane";
import { getWebGPUctx } from "../../../myutils/ContextHelpers";
import { createLossPlot } from "../../../myutils/LogHelpers";
import { main, type MainController } from "../optimization/mainParallel";
import { loadImageBitmap } from "../../../myutils/MyUtils";
import { defaultSplatConfig } from "../../../config";
import { DEFAULT_GUI_VISIBILITY } from "../../../config/defaults";

export let splatController: MainController | undefined;

export async function handleMediaUpload(file: File): Promise<void> {
    if (!splatController) {
        throw new Error("The splat controller is not initialized.");
    }

    const source = await createImageBitmap(file);
    await splatController.setSource(source);
}

const ctxWebgpu = await getWebGPUctx({ canvasId: "main"});

if(ctxWebgpu) {
    let plotHTMLElem = document.getElementById('loss-plot') ?? document.body
    let lossPlot = createLossPlot(plotHTMLElem);

    const testImageUrl = 'assets/cat.jpg';
    const source = await loadImageBitmap(testImageUrl);

    const pane = new Pane({
            container: document.getElementById("gd-sliders") as HTMLElement,
    });

    splatController = await main({ 
        webgpu: ctxWebgpu,
        config : defaultSplatConfig,
        init: "grid",
        gui : DEFAULT_GUI_VISIBILITY,
        image : source,
        pane: pane,
        plot : lossPlot 
    });

    const mediaUpload = document.getElementById("main-media-upload") as HTMLInputElement;

    mediaUpload.addEventListener("change", () => {
        const [file] = mediaUpload.files ?? [];
        if (!file) {
            return;
        }

        void handleMediaUpload(file).catch((error: unknown) => {
            console.error("Could not upload source image.", error);
        });
    });
    

}