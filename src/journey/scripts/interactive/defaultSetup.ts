import { Pane } from "tweakpane";
import { getWebGPUctx } from "../../../myutils/ContextHelpers";
import { createLossPlot } from "../../../myutils/LogHelpers";
import { main, type MainController } from "../optimization/mainParallel";
import { loadImageBitmap } from "../../../myutils/MyUtils";
import { defaultSplatConfig } from "../../../config";
import { DEFAULT_GUI_VISIBILITY } from "../../../config/defaults";
import EmblaCarousel from "embla-carousel";
import Autoplay from "embla-carousel-autoplay";

export let splatController: MainController | undefined;

const sourceImages = new Map<string, ImageBitmap>();
let sourceChangeInProgress = false;

export async function handleMediaUpload(file: File): Promise<void> {
    if (!splatController) {
        throw new Error("The splat controller is not initialized.");
    }

    const source = await createImageBitmap(file);
    await splatController.setSource(source);
}

async function selectSource(sourceUrl: string, selectedSlide: HTMLElement, startOptimization: boolean): Promise<void> {
    if (!splatController) {
        throw new Error("The splat controller is not initialized.");
    }
    if (sourceChangeInProgress || splatController.getGDProgress().running) {
        return;
    }

    sourceChangeInProgress = true;

    try {
        let source = sourceImages.get(sourceUrl);
        if (!source) {
            source = await loadImageBitmap(sourceUrl);
            sourceImages.set(sourceUrl, source);
        }

        await splatController.setSource(source);
        document.querySelectorAll(".source-slide").forEach((slide) => {
            slide.classList.toggle("is-selected", slide === selectedSlide);
        });
        if (startOptimization) {
            await splatController.runGD();
        }
    } finally {
        sourceChangeInProgress = false;
    }
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
    const progressBar = document.getElementById("gd-progress") as HTMLElement;
    const progressBarFill = document.getElementById("gd-progress-bar") as HTMLElement;
    const progressStatus = document.getElementById("gd-progress-status") as HTMLElement;
    const sourceSlides = document.getElementById("source-slides");
    const carousel = sourceSlides?.closest(".embla");
    const sourceCarousel = carousel as HTMLElement | null;
    const carouselSwitchDelay = 10 * 1000;
    const carouselScrollDuration = 100;
    const embla = carousel ? EmblaCarousel(carousel.querySelector(".embla__viewport") as HTMLElement, {
        loop: false,
        duration: carouselScrollDuration,
    }, []) : undefined;

    const updateProgress = (): void => {
        const progress = splatController?.getGDProgress();
        if (!progress) {
            return;
        }

        const percentage = progress.totalSteps > 0
            ? Math.min(100, (progress.completedSteps / progress.totalSteps) * 100)
            : 0;
        const roundedPercentage = Math.round(percentage);
        progressBarFill.style.width = `${percentage}%`;
        progressStatus.textContent = progress.running
            ? `${roundedPercentage}%`
            : percentage >= 100 ? "Complete" : "Ready";
        progressBar.setAttribute("aria-valuenow", String(roundedPercentage));

        const isLocked = sourceChangeInProgress || progress.running;
        sourceCarousel?.classList.toggle("is-loading", isLocked);
        sourceSlides?.querySelectorAll<HTMLButtonElement>(".source-slide").forEach((slide) => {
            slide.disabled = isLocked;
        });
        mediaUpload.disabled = isLocked;

        if (progress.running) {
            requestAnimationFrame(updateProgress);
        }
    };

    const selectActiveSlide = (): void => {
        if (!embla) {
            return;
        }

        const slide = embla.slideNodes()[embla.selectedScrollSnap()];
        const sourceUrl = slide?.dataset.source;
        if (!slide || !sourceUrl) {
            return;
        }

        void selectSource(sourceUrl, slide, false).catch((error: unknown) => {
            console.error("Could not select autoplay source image.", error);
        });
    };

    embla?.on("select", selectActiveSlide);
    selectActiveSlide();

    sourceSlides?.addEventListener("click", (event) => {
        const slide = (event.target as HTMLElement).closest<HTMLElement>(".source-slide[data-source]");
        const sourceUrl = slide?.dataset.source;
        if (!slide || !sourceUrl) {
            return;
        }

        void selectSource(sourceUrl, slide, true).then(updateProgress).catch((error: unknown) => {
            console.error("Could not select source image.", error);
        });
    });

    mediaUpload.addEventListener("change", () => {
        const [file] = mediaUpload.files ?? [];
        if (!file) {
            return;
        }

        if (sourceChangeInProgress || splatController?.getGDProgress().running) {
            mediaUpload.value = "";
            return;
        }

        const objectUrl = URL.createObjectURL(file);
        const slide = document.createElement("button");
        slide.type = "button";
        slide.className = "embla__slide source-slide is-selected";
        slide.dataset.source = objectUrl;
        slide.innerHTML = `<img src="${objectUrl}" alt="Uploaded source image">`;
        sourceSlides?.insertBefore(slide, sourceSlides.lastElementChild ?? null);
        embla?.reInit();

        sourceChangeInProgress = true;
        void handleMediaUpload(file).then(async () => {
            document.querySelectorAll(".source-slide").forEach((item) => {
                item.classList.toggle("is-selected", item === slide);
            });
            await splatController?.runGD();
        }).then(updateProgress).finally(() => {
            sourceChangeInProgress = false;
            updateProgress();
        }).catch((error: unknown) => {
            URL.revokeObjectURL(objectUrl);
            slide.remove();
            embla?.reInit();
            console.error("Could not upload source image.", error);
        });
        mediaUpload.value = "";
    });

    const settingsToggle = document.querySelector<HTMLButtonElement>(".settings-toggle");
    const resetButton = document.getElementById("journey-reset") as HTMLButtonElement;
    const settingsPanel = document.getElementById("settings-panel");
    resetButton.addEventListener("click", () => {
        if (!splatController || sourceChangeInProgress) {
            return;
        }

        resetButton.disabled = true;
        void splatController.reset().then(updateProgress).catch((error: unknown) => {
            console.error("Could not reset the journey renderer.", error);
        }).finally(() => {
            resetButton.disabled = false;
        });
    });

    settingsToggle?.addEventListener("click", () => {
        if (!settingsPanel) {
            return;
        }
        const isExpanded = settingsToggle.getAttribute("aria-expanded") === "true";
        settingsToggle.setAttribute("aria-expanded", String(!isExpanded));
        settingsToggle.textContent = isExpanded ? "Show more settings" : "Hide settings";
        settingsPanel.hidden = isExpanded;
    });

}