window.PDFModal = (() => {
  let elements = {};
  let currentRenderTask = null;
  let isClosed = false;

  function init() {
    elements = {
      modal: document.getElementById("pdfModal"),
      viewer: document.getElementById("pdfViewer"),
      title: document.getElementById("pdfModalTitle"),
      subtitle: document.getElementById("pdfModalSubtitle"),
      close: document.getElementById("pdfModalClose"),
    };

    if (window.pdfjsLib) {
      pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
    }

    elements.close?.addEventListener("click", close);

    elements.modal?.addEventListener("click", (event) => {
      if (event.target === elements.modal) close();
    });

    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") close();
    });
  }

  function buildSearchText(text) {
    return (text || "")
      .replace(/[ \t]+/g, " ")
      .replace(/\n+/g, " ")
      .trim()
      .slice(0, 80);
  }

  function normalizeText(text) {
    return (text || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/\s+/g, " ")
      .toLowerCase()
      .trim();
  }

  async function renderPage(pdf, pageNumber, searchText) {
    if (isClosed) return;
    const page = await pdf.getPage(pageNumber);
    if (isClosed) return;
    const baseViewport = page.getViewport({ scale: 1 });

    const availableWidth = elements.viewer.clientWidth - 48;
    const scale = Math.max(
      0.6,
      Math.min(1.35, availableWidth / baseViewport.width),
    );
    const viewport = page.getViewport({ scale });

    const pageWrap = document.createElement("div");
    pageWrap.className = "pdf-page-wrap";

    const canvas = document.createElement("canvas");
    canvas.className = "pdf-canvas";

    const context = canvas.getContext("2d");
    canvas.width = viewport.width;
    canvas.height = viewport.height;

    const textLayer = document.createElement("div");
    textLayer.className = "pdf-text-layer";
    textLayer.style.width = `${viewport.width}px`;
    textLayer.style.height = `${viewport.height}px`;
    textLayer.style.setProperty("--scale-factor", scale);

    pageWrap.style.width = `${viewport.width}px`;
    pageWrap.style.height = `${viewport.height}px`;
    pageWrap.style.setProperty("--scale-factor", scale);

    pageWrap.appendChild(canvas);
    pageWrap.appendChild(textLayer);
    const existingWrap = elements.viewer.querySelector(
      `.pdf-page-wrap[data-page="${pageNumber}"]`,
    );

    if (existingWrap) {
      existingWrap.replaceWith(pageWrap);
    } else {
      elements.viewer.appendChild(pageWrap);
    }

    pageWrap.dataset.page = String(pageNumber);

    currentRenderTask = page.render({
      canvasContext: context,
      viewport,
    });

    await currentRenderTask.promise;
    if (isClosed) return;

    const textContent = await page.getTextContent();
    if (isClosed) return;

    await pdfjsLib.renderTextLayer({
      textContentSource: textContent,
      container: textLayer,
      viewport,
      textDivs: [],
    }).promise;
    if (isClosed) return;

    highlightSearch(textLayer, searchText);
  }

  function highlightSearch(textLayer, searchText) {
    if (!searchText) return;

    const normalizedSearch = normalizeText(searchText);
    if (!normalizedSearch) return;

    const spans = Array.from(textLayer.querySelectorAll("span"));

    for (const span of spans) {
      const original = span.textContent || "";
      const normalized = normalizeText(original);

      if (!normalized) continue;

      if (
        normalizedSearch.includes(normalized) ||
        normalized.includes(normalizedSearch)
      ) {
        span.classList.add("pdf-highlight");
      }
    }
  }

  async function createPagePlaceholders(pdf) {
    const targetPage = await pdf.getPage(1);
    const baseViewport = targetPage.getViewport({ scale: 1 });

    const availableWidth = elements.viewer.clientWidth - 48;
    const scale = Math.max(
      0.6,
      Math.min(1.35, availableWidth / baseViewport.width),
    );

    const viewport = targetPage.getViewport({ scale });

    for (let i = 1; i <= pdf.numPages; i++) {
      const placeholder = document.createElement("div");
      placeholder.className = "pdf-page-wrap pdf-page-placeholder";
      placeholder.dataset.page = String(i);
      placeholder.style.width = `${viewport.width}px`;
      placeholder.style.height = `${viewport.height}px`;

      elements.viewer.appendChild(placeholder);
    }
  }

  async function open({ pdfUrl, title = "PDF", text = "", page = 1, onError }) {
    const fail = (message) => {
      onError?.(message);
    };

    if (!pdfUrl) {
      fail("No hay PDF asociado.");
      return;
    }

    if (!window.pdfjsLib) {
      fail("pdf.js no está cargado.");
      return;
    }

    isClosed = false;

    const pageNumber = Number(page || 1);
    const searchText = buildSearchText(text);

    if (elements.title) elements.title.textContent = title;
    if (elements.subtitle)
      elements.subtitle.textContent = searchText
        ? `Buscando: "${searchText}"`
        : `Página ${pageNumber}`;

    if (elements.viewer) {
      elements.viewer.innerHTML = `<div class="pdf-loading">Cargando PDF...</div>`;
    }

    try {
      const pdf = await pdfjsLib.getDocument(pdfUrl).promise;

      if (isClosed) return;
      elements.modal?.classList.remove("hidden");

      if (elements.viewer) elements.viewer.innerHTML = "";

      await createPagePlaceholders(pdf);
      if (isClosed) return;

      await renderPage(pdf, pageNumber, searchText);
      if (isClosed) return;

      const targetPage = elements.viewer.querySelector(
        `.pdf-page-wrap[data-page="${pageNumber}"]`,
      );
      targetPage?.scrollIntoView({
        behavior: "auto",
        block: "start",
      });

      setTimeout(async () => {
        for (let i = 1; i <= pdf.numPages; i++) {
          if (isClosed) return;

          if (i === pageNumber) continue;

          await renderPage(pdf, i, searchText);
        }
      }, 0);
    } catch (error) {
      console.error(error);

      if (isClosed) return;
      const rawMessage = String(error?.message || error || "");

      let friendlyMessage = "No se pudo cargar el PDF.";

      if (rawMessage.includes("Missing PDF")) {
        friendlyMessage =
          "El archivo PDF asociado a este documento no fue encontrado.";
      } else if (rawMessage.includes("Failed to fetch")) {
        friendlyMessage =
          "No se pudo conectar con el servidor para descargar el PDF.";
      }

      close();
      fail(friendlyMessage);
    }
  }

  function close() {
    isClosed = true;
    elements.modal?.classList.add("hidden");

    if (currentRenderTask) {
      currentRenderTask.cancel?.();
      currentRenderTask = null;
    }

    if (elements.viewer) {
      elements.viewer.innerHTML = "";
    }
  }

  return {
    init,
    open,
    close,
  };
})();

PDFModal.init();

function getPdfUrlFromChunkId(chunkId) {
  const docId = chunkId.split("#")[0];
  return `${API_BASE}/pdfs/${docId}.pdf`;
}
