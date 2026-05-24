(() => {
  const TOPBAR_HTML_URL = "/topbar.html";
  const RELOAD_STATE_KEY = "csic_graphrag_reload_state_v2";
  const RELOAD_TOAST_DISMISSED_KEY = "csic_graphrag_reload_toast_dismissed_v1";
  const POLL_INTERVALS_MS = [10000, 20000, 50000];

  let toastHideTimer = null;
  let pollTimer = null;
  let pollAttempt = 0;
  let reloadJob = null;

  function getApiBase() {
    return (window.APP_CONFIG?.API_BASE || "").replace(/\/+$/, "");
  }

  function escapeHtml(value) {
    return (value ?? "")
      .toString()
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function getToast() {
    return document.getElementById("topbarToast");
  }

  function markReloadToastDismissed() {
    try {
      sessionStorage.setItem(RELOAD_TOAST_DISMISSED_KEY, "true");
    } catch {}
  }

  function clearReloadToastDismissed() {
    try {
      sessionStorage.removeItem(RELOAD_TOAST_DISMISSED_KEY);
    } catch {}
  }

  function isReloadToastDismissed() {
    try {
      return sessionStorage.getItem(RELOAD_TOAST_DISMISSED_KEY) === "true";
    } catch {
      return false;
    }
  }

  function showToast(message, type = "success", { sticky = false } = {}) {
    const toast = getToast();
    if (!toast) return;

    clearTimeout(toastHideTimer);
    toast.className = "topbar-toast";
    toast.classList.add(type);
    toast.classList.remove("hidden");

    toast.innerHTML = `
      <div class="topbar-toast-content">
        <div class="topbar-toast-message">${message}</div>
        <button
          type="button"
          class="topbar-toast-close"
          aria-label="Cerrar mensaje"
          title="Cerrar"
        >
          ✕
        </button>
      </div>
    `;

    const closeBtn = toast.querySelector(".topbar-toast-close");
    if (closeBtn) {
      closeBtn.addEventListener("click", () => {
        if (type === "loading") {
          markReloadToastDismissed();
        }
        hideToast();
      });
    }

    requestAnimationFrame(() => toast.classList.add("show"));

    if (!sticky) {
      toastHideTimer = setTimeout(hideToast, 3400);
    }
  }

  function countEntities(files) {
    const proyectos = new Set();
    const grupos = new Set();
    for (const { path } of files) {
      const parts = path.replace(/\\/g, "/").split("/");
      const root = parts[0] || "";
      const isProyecto = /^proyectos[_\s]?\d{4}/i.test(root);
      const isGrupo = /grupos/i.test(root) || /^gi[_\s]/i.test(root);
      for (let i = 1; i < parts.length - 1; i++) {
        if (/^\d+$/.test(parts[i])) {
          if (isProyecto) proyectos.add(`${root}/${parts[i]}`);
          else if (isGrupo) grupos.add(`${root}/${parts[i]}`);
          break;
        }
      }
    }
    return { proyectos: proyectos.size, grupos: grupos.size };
  }

  function formatEntityLabel({ proyectos, grupos }) {
    const parts = [];
    if (proyectos > 0)
      parts.push(`${proyectos} ${proyectos === 1 ? "proyecto" : "proyectos"}`);
    if (grupos > 0)
      parts.push(
        `${grupos} ${grupos === 1 ? "grupo de investigación" : "grupos de investigación"}`,
      );
    return parts.length > 0 ? parts.join(" y ") : null;
  }

  function showLoadingToast(current, total, uploadLabel = null) {
    const toast = getToast();
    if (!toast) return;
    const alreadyVisible = toast.classList.contains("show");

    clearTimeout(toastHideTimer);
    toast.className = "topbar-toast loading";
    if (alreadyVisible) toast.classList.add("show");
    toast.classList.remove("hidden");

    const subtitleUpload = uploadLabel
      ? `Subiendo ${uploadLabel}…`
      : "Subiendo archivos…";
    const subtitle =
      current === null || total === 0 || current === 0
        ? subtitleUpload
        : current >= total && total > 0
          ? `${current} de ${total} archivos — construyendo grafo…`
          : `${current} de ${total} archivos`;

    toast.innerHTML = `
      <div class="topbar-toast-content">
        <div class="topbar-toast-message">
          <div style="margin-bottom:6px;font-weight:700">Cargando archivos…</div>
          <div style="font-size:12px;opacity:0.85;margin-bottom:6px">${subtitle}</div>
          <div style="background:rgba(0,0,0,0.08);border-radius:999px;height:6px;overflow:hidden;position:relative">
            <div class="topbar-progress-indeterminate"></div>
          </div>
        </div>
        <button type="button" class="topbar-toast-close" aria-label="Cerrar mensaje" title="Cerrar">✕</button>
      </div>
    `;
    const closeBtn = toast.querySelector(".topbar-toast-close");
    if (closeBtn) {
      closeBtn.addEventListener("click", () => {
        markReloadToastDismissed();
        hideToast();
      });
    }
    if (!alreadyVisible)
      requestAnimationFrame(() => toast.classList.add("show"));
  }

  function maybeShowLoadingToast(current, total, uploadLabel = null) {
    if (isReloadToastDismissed()) return;
    showLoadingToast(current, total, uploadLabel);
  }

  // Hace el POST con FormData usando XMLHttpRequest para poder reportar progreso de upload.
  function uploadWithProgress(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable)
          onProgress({ loaded: e.loaded, total: e.total });
      });

      xhr.addEventListener("load", () => {
        const text = xhr.responseText || "";
        let data = {};
        try {
          data = text ? JSON.parse(text) : {};
        } catch {
          data = {};
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(data);
        } else {
          const detail =
            data.detail || data.message || text || `HTTP ${xhr.status}`;
          reject(new Error(`${url} -> ${xhr.status} ${detail}`.trim()));
        }
      });

      xhr.addEventListener("error", () =>
        reject(new Error("Error de red durante la carga.")),
      );
      xhr.addEventListener("abort", () =>
        reject(new Error("Carga cancelada.")),
      );

      xhr.send(formData);
    });
  }

  function hideToast() {
    const toast = getToast();
    if (!toast) return;

    clearTimeout(toastHideTimer);
    toast.classList.remove("show");
    setTimeout(() => toast.classList.add("hidden"), 180);
  }

  function saveReloadState(state) {
    try {
      sessionStorage.setItem(RELOAD_STATE_KEY, JSON.stringify(state));
    } catch {}
  }

  function readReloadState() {
    try {
      return JSON.parse(sessionStorage.getItem(RELOAD_STATE_KEY) || "null");
    } catch {
      return null;
    }
  }

  function clearReloadState() {
    try {
      sessionStorage.removeItem(RELOAD_STATE_KEY);
    } catch {}
  }

  function emitReloadEvent(type, detail = {}) {
    window.dispatchEvent(new CustomEvent(type, { detail }));
  }

  function getReloadButton() {
    return document.getElementById("reloadFilesBtn");
  }

  function updateReloadButtonUI() {
    const button = getReloadButton();
    if (!button) return;

    const isRunning = Boolean(reloadJob);
    button.disabled = isRunning;
    button.textContent = isRunning ? "Cargando…" : "Cargar archivos";
    button.dataset.loading = String(isRunning);
  }

  async function parseJsonResponse(response, endpoint) {
    const text = await response.text().catch(() => "");
    let data = {};

    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = {};
    }

    if (!response.ok) {
      const detail =
        data.detail || data.message || text || `HTTP ${response.status}`;
      throw new Error(`${endpoint} -> ${response.status} ${detail}`.trim());
    }

    return data;
  }

  function formatReloadSuccessMessage(payload) {
    const result = payload?.result ?? payload ?? {};
    const processedCount =
      result.processed_count ??
      result.processed?.length ??
      result.files_processed;
    const errorsCount =
      result.errors_count ??
      (Array.isArray(result.errors) ? result.errors.length : undefined);
    const skippedCount = Array.isArray(result.skipped)
      ? result.skipped.length
      : 0;

    const csvLoaded = result.csv_loaded === true;
    const parts = ["Carga finalizada."];

    if (typeof processedCount === "number" && processedCount > 0) {
      parts.push(`Procesados: <strong>${processedCount}</strong>.`);
    } else if (csvLoaded && processedCount === 0) {
      parts.push("CSV cargado.");
    }

    if (skippedCount > 0) {
      parts.push(`Salteados: <strong>${skippedCount}</strong>.`);
    }

    if (typeof errorsCount === "number" && errorsCount > 0) {
      parts.push(`Errores: <strong>${errorsCount}</strong>.`);
    }

    return parts.join(" ");
  }

  function _fileRefFromPath(path) {
    const parts = path.replace(/\\/g, "/").split("/");
    const filename = parts[parts.length - 1];
    const projectId = parts.slice(1).find((p) => /^\d+$/.test(p));
    return { filename, projectId };
  }

  function _buildResultsHTML(skipped, errors) {
    const sections = [];

    const missingCsv = errors.find((e) => e.type === "MissingProyectosCSV");
    if (missingCsv) {
      sections.push(`
        <div class="results-warning-banner">
          No hay un CSV de proyectos cargado en el sistema. Suba uno antes de cargar archivos.
        </div>`);
    }

    const fileErrors = errors.filter((e) => e.type === "ProcessFileError");
    if (fileErrors.length > 0) {
      const items = fileErrors
        .map((e) => {
          const { filename, projectId } = _fileRefFromPath(e.file || "");
          const ref = projectId
            ? ` <span class="results-dim">· proyecto ${escapeHtml(projectId)}</span>`
            : "";
          return `<li class="results-friendly-item">${escapeHtml(filename)}${ref}</li>`;
        })
        .join("");
      sections.push(`
        <div class="results-friendly-section">
          <p class="results-friendly-label results-label-error">
            Los siguientes archivos no pudieron procesarse:
          </p>
          <ul class="results-friendly-list">${items}</ul>
        </div>`);
    }

    if (skipped.length > 0) {
      const seen = new Set();
      const unique = skipped.filter((s) => {
        const key = `${s.id_formulario}|${s.year}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      const items = unique
        .map((s) => {
          const id = escapeHtml(s.id_formulario);
          const year = escapeHtml(s.year);
          return `<li class="results-friendly-item">Proyecto ${id} <span class="results-dim">· ${year}</span></li>`;
        })
        .join("");
      sections.push(`
        <div class="results-friendly-section">
          <p class="results-friendly-label results-label-warning">
            Los siguientes proyectos no están registrados en el CSV y sus archivos fueron ignorados:
          </p>
          <ul class="results-friendly-list">${items}</ul>
        </div>`);
    }

    return sections.join("");
  }

  function openResultsModal(skipped, errors) {
    const modal = document.getElementById("resultsModal");
    const backdrop = document.getElementById("resultsBackdrop");
    const body = document.getElementById("resultsModalBody");
    if (!modal || !backdrop || !body) return;
    if (!skipped.length && !errors.length) return;

    body.innerHTML = _buildResultsHTML(skipped, errors);
    backdrop.classList.remove("hidden");
    modal.classList.remove("hidden");
  }

  function closeResultsModal() {
    const modal = document.getElementById("resultsModal");
    const backdrop = document.getElementById("resultsBackdrop");
    if (!modal || !backdrop) return;
    backdrop.classList.add("hidden");
    modal.classList.add("hidden");
  }

  function stopPolling(resetAttempt = true) {
    clearTimeout(pollTimer);
    pollTimer = null;
    if (resetAttempt) pollAttempt = 0;
  }

  function nextPollInterval() {
    if (pollAttempt < 5) return POLL_INTERVALS_MS[0];
    if (pollAttempt < 15) return POLL_INTERVALS_MS[1];
    return POLL_INTERVALS_MS[2];
  }

  async function fetchReloadStatus(jobId) {
    const endpoint = `${getApiBase()}/ui/upload/status?job_id=${encodeURIComponent(jobId)}`;
    const response = await fetch(endpoint);
    return parseJsonResponse(response, endpoint);
  }

  async function syncReloadStateFromServer() {
    const base = getApiBase();
    if (!base) return;

    try {
      const endpoint = `${base}/ui/upload/status`;
      const response = await fetch(endpoint);
      const data = await parseJsonResponse(response, endpoint);

      if (
        data?.job_id &&
        (data.status === "queued" || data.status === "running")
      ) {
        reloadJob = { job_id: data.job_id, status: data.status };

        const prevState = readReloadState();
        const storedLabel = prevState?.uploadLabel ?? null;
        saveReloadState({
          job_id: data.job_id,
          status: data.status,
          startedAt: data?.started_at ?? null,
          createdAt: data?.created_at ?? null,
          message: data?.message ?? "Carga en curso...",
          uploadLabel: storedLabel,
        });

        updateReloadButtonUI();
        scheduleStatusPoll(data.job_id);
        const p = data?.progress ?? null;
        maybeShowLoadingToast(p ? p.current : 0, p ? p.total : 0, storedLabel);
        return;
      }

      reloadJob = null;
      clearReloadState();
      clearReloadToastDismissed();
      updateReloadButtonUI();
    } catch {
      reloadJob = null;
      updateReloadButtonUI();
    }
  }

  function handleReloadStatus(statusData) {
    const jobId = statusData?.job_id;
    const status = statusData?.status;

    if (!jobId) {
      reloadJob = null;
      clearReloadToastDismissed();
      updateReloadButtonUI();
      stopPolling();
      return;
    }

    if (status === "queued" || status === "running") {
      reloadJob = { job_id: jobId, status };

      const prevState = readReloadState();
      const storedLabel = prevState?.uploadLabel ?? null;
      saveReloadState({
        job_id: jobId,
        status,
        startedAt: statusData?.started_at ?? null,
        createdAt: statusData?.created_at ?? null,
        message: statusData?.message ?? "Carga en curso...",
        uploadLabel: storedLabel,
      });

      updateReloadButtonUI();
      const prog = statusData?.progress ?? null;
      maybeShowLoadingToast(
        prog ? prog.current : 0,
        prog ? prog.total : 0,
        storedLabel,
      );
      scheduleStatusPoll(jobId);
      return;
    }

    if (status === "success") {
      const state = {
        job_id: jobId,
        status: "success",
        startedAt: statusData?.started_at ?? null,
        finishedAt: statusData?.finished_at ?? null,
        result: statusData?.result ?? null,
        message: statusData?.message || formatReloadSuccessMessage(statusData),
      };

      saveReloadState(state);
      reloadJob = null;
      clearReloadToastDismissed();
      updateReloadButtonUI();
      stopPolling();
      showToast(formatReloadSuccessMessage(statusData), "success");

      const skipped = statusData?.result?.skipped ?? [];
      const allErrors = Array.isArray(statusData?.result?.errors)
        ? statusData.result.errors
        : [];
      const userErrors = allErrors.filter(
        (e) =>
          e.type === "ProcessFileError" || e.type === "MissingProyectosCSV",
      );
      if (skipped.length > 0 || userErrors.length > 0) {
        openResultsModal(skipped, userErrors);
      }

      emitReloadEvent("reload:finish", state);
      return;
    }

    if (status === "error") {
      const message = statusData?.message
        ? `Error al cargar archivos.<br><span style="font-weight:700;opacity:.9">${escapeHtml(statusData.message)}</span>`
        : "Error al cargar archivos.";

      const state = {
        job_id: jobId,
        status: "error",
        startedAt: statusData?.started_at ?? null,
        finishedAt: statusData?.finished_at ?? null,
        error: statusData?.error ?? "unknown_error",
        message,
      };

      saveReloadState(state);
      reloadJob = null;
      clearReloadToastDismissed();
      updateReloadButtonUI();
      stopPolling();
      showToast(message, "error", { sticky: true });
      emitReloadEvent("reload:error", state);
      return;
    }

    reloadJob = null;
    clearReloadToastDismissed();
    updateReloadButtonUI();
    stopPolling();
  }

  function scheduleStatusPoll(jobId) {
    stopPolling(false);
    pollAttempt += 1;

    pollTimer = setTimeout(async () => {
      try {
        handleReloadStatus(await fetchReloadStatus(jobId));
      } catch (error) {
        const errMessage = error?.message || String(error);

        if (
          errMessage.includes("404") &&
          errMessage.includes("No hay cargas registradas")
        ) {
          clearReloadState();
          clearReloadToastDismissed();
          reloadJob = null;
          updateReloadButtonUI();
          stopPolling();
          hideToast();
          return;
        }

        const message = `Error consultando estado de carga.<br><span style="font-weight:700;opacity:.9">${escapeHtml(errMessage)}</span>`;
        clearReloadToastDismissed();
        showToast(message, "error", { sticky: true });
        emitReloadEvent("reload:error", {
          job_id: jobId,
          error: errMessage,
          message,
        });
        reloadJob = null;
        updateReloadButtonUI();
        stopPolling();
      }
    }, nextPollInterval());
  }

  let collectedFolderFiles = []; // [{ path: string, file: File }]
  let collectedZipFiles = []; // [File] — ZIPs que el backend extrae

  async function openUploadModal() {
    const modal = document.getElementById("uploadModal");
    const backdrop = document.getElementById("uploadBackdrop");
    if (!modal || !backdrop) return;
    backdrop.classList.remove("hidden");
    modal.classList.remove("hidden");
    document.body.classList.add("upload-modal-open");

    const statusEl = document.getElementById("csvSystemStatus");
    if (statusEl) {
      statusEl.textContent = "";
      try {
        const data = await fetch(`${getApiBase()}/ui/csv/status`).then((r) =>
          r.json(),
        );
        statusEl.textContent = data.has_proyectos_csv
          ? "(ya hay una en el sistema)"
          : "(requerida, no hay ninguna en el sistema)";
        statusEl.className = data.has_proyectos_csv
          ? "upload-optional upload-csv-ok"
          : "upload-optional upload-csv-missing";
      } catch {
        statusEl.textContent = "";
      }
    }
  }

  function closeUploadModal() {
    const modal = document.getElementById("uploadModal");
    const backdrop = document.getElementById("uploadBackdrop");
    if (!modal || !backdrop) return;
    backdrop.classList.add("hidden");
    modal.classList.add("hidden");
    document.body.classList.remove("upload-modal-open");

    collectedFolderFiles = [];
    collectedZipFiles = [];
    const foldersInput = document.getElementById("foldersInput");
    const csvInput = document.getElementById("csvInput");
    const foldersSelected = document.getElementById("foldersSelected");
    const csvSelected = document.getElementById("csvSelected");
    if (foldersInput) foldersInput.value = "";
    if (csvInput) csvInput.value = "";
    if (foldersSelected) foldersSelected.innerHTML = "";
    if (csvSelected) csvSelected.textContent = "";
  }

  function updateFoldersDisplay() {
    const container = document.getElementById("foldersSelected");
    if (!container) return;

    if (collectedFolderFiles.length === 0 && collectedZipFiles.length === 0) {
      container.innerHTML = "";
      return;
    }

    const rootFolders = new Map();
    for (const f of collectedFolderFiles) {
      const root = f.path.split("/")[0];
      if (!rootFolders.has(root)) rootFolders.set(root, []);
      rootFolders.get(root).push(f);
    }

    const parts = [];
    for (const [folder, files] of rootFolders) {
      const { proyectos, grupos } = countEntities(files);
      const labelParts = [];
      if (proyectos > 0)
        labelParts.push(
          `${proyectos} ${proyectos === 1 ? "proyecto" : "proyectos"}`,
        );
      if (grupos > 0)
        labelParts.push(`${grupos} ${grupos === 1 ? "grupo" : "grupos"}`);
      const label =
        labelParts.length > 0
          ? labelParts.join(" y ")
          : `${files.length} archivo${files.length !== 1 ? "s" : ""}`;
      parts.push(
        `<span class="upload-folder-tag" data-folder="${escapeHtml(folder)}">` +
          `${escapeHtml(folder)} (${label})` +
          `<button type="button" class="upload-folder-tag-remove" aria-label="Quitar ${escapeHtml(folder)}" data-folder="${escapeHtml(folder)}">✕</button>` +
          `</span>`,
      );
    }
    for (const zip of collectedZipFiles) {
      parts.push(
        `<span class="upload-folder-tag" data-zip="${escapeHtml(zip.name)}">` +
          `${escapeHtml(zip.name)} (ZIP)` +
          `<button type="button" class="upload-folder-tag-remove" aria-label="Quitar ${escapeHtml(zip.name)}" data-zip="${escapeHtml(zip.name)}">✕</button>` +
          `</span>`,
      );
    }

    container.innerHTML = parts.join("");

    container.querySelectorAll(".upload-folder-tag-remove").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.zip) {
          collectedZipFiles = collectedZipFiles.filter(
            (f) => f.name !== btn.dataset.zip,
          );
        } else {
          const folder = btn.dataset.folder;
          collectedFolderFiles = collectedFolderFiles.filter(
            ({ path }) => path.split("/")[0] !== folder,
          );
        }
        updateFoldersDisplay();
      });
    });
  }

  function isAcceptedFile(name) {
    if (name.startsWith(".")) return false;
    const lower = name.toLowerCase();
    return lower.endsWith(".pdf") || lower.endsWith(".odt");
  }

  async function collectFilesFromHandle(dirHandle, pathPrefix) {
    const result = [];
    const prefix = pathPrefix + dirHandle.name + "/";
    for await (const [name, entry] of dirHandle) {
      if (name.startsWith(".") || name === "__MACOSX") continue;
      if (entry.kind === "file") {
        if (!isAcceptedFile(name)) continue;
        const file = await entry.getFile();
        result.push({ path: prefix + name, file });
      } else if (entry.kind === "directory") {
        result.push(...(await collectFilesFromHandle(entry, prefix)));
      }
    }
    return result;
  }

  async function collectFilesFromEntry(entry, pathPrefix) {
    if (entry.isFile) {
      if (!isAcceptedFile(entry.name)) return [];
      return new Promise((resolve, reject) => {
        entry.file(
          (file) => resolve([{ path: pathPrefix + file.name, file }]),
          reject,
        );
      });
    }

    if (entry.isDirectory) {
      const prefix = pathPrefix + entry.name + "/";
      const reader = entry.createReader();
      const allFiles = [];

      await new Promise((resolve, reject) => {
        function readNextBatch() {
          reader.readEntries(async (entries) => {
            if (!entries.length) {
              resolve();
              return;
            }
            for (const e of entries) {
              const files = await collectFilesFromEntry(e, prefix);
              allFiles.push(...files);
            }
            readNextBatch();
          }, reject);
        }
        readNextBatch();
      });

      return allFiles;
    }

    return [];
  }

  const CSV_REQUIRED_COLS = [
    "anio",
    "id_formulario",
    "titulo",
    "documento",
    "tipo_documento",
    "pais_documento",
    "area",
    "nombres",
    "apellidos",
    "sexo",
    "calidad",
    "descripcion",
    "palabras_claves",
    "palabras_claves2",
    "palabras_claves3",
  ];

  function readCsvHeader(file) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const firstLine = (e.target?.result ?? "").split(/\r?\n/)[0] ?? "";
        const cols = firstLine.split(",").map((c) =>
          c
            .trim()
            .replace(/^["']|["']$/g, "")
            .toLowerCase(),
        );
        resolve(cols);
      };
      reader.onerror = () => resolve([]);
      reader.readAsText(file.slice(0, 4096));
    });
  }

  async function submitUpload() {
    const csvInput = document.getElementById("csvInput");
    const csvFile = csvInput?.files?.[0] ?? null;

    if (collectedFolderFiles.length === 0 && !csvFile) {
      showToast("Arrastre al menos una carpeta o seleccione un CSV.", "error");
      return;
    }

    // Solo enviar archivos cuya carpeta raíz sea proyectos_YYYY o grupos/gi,
    // y que además tengan una subcarpeta numérica (id_formulario).
    const projectFiles = collectedFolderFiles.filter(({ path }) => {
      const parts = path.replace(/\\/g, "/").split("/");
      const root = parts[0] || "";
      const isProyecto = /^proyectos[_\s]?\d{4}/i.test(root);
      const isGrupo = /grupos/i.test(root) || /^gi[_\s]/i.test(root);
      if (!isProyecto && !isGrupo) return false;
      return parts.slice(1, -1).some((p) => /^\d+$/.test(p));
    });

    const validationErrors = [];

    if (
      projectFiles.length === 0 &&
      collectedZipFiles.length === 0 &&
      !csvFile
    ) {
      if (collectedFolderFiles.length > 0) {
        const invalidRoots = [
          ...new Set(
            collectedFolderFiles.map(
              (f) => f.path.replace(/\\/g, "/").split("/")[0],
            ),
          ),
        ];
        const names = invalidRoots.map((r) => `"${r}"`).join(", ");
        validationErrors.push(
          `La carpeta ${names} no cumple el formato esperado. Revise la estructura en "Formato esperado".`,
        );
      } else {
        showToast(
          "Arrastre al menos una carpeta o seleccione un CSV.",
          "error",
        );
        return;
      }
    } else if (
      projectFiles.length === 0 &&
      collectedZipFiles.length === 0 &&
      collectedFolderFiles.length > 0
    ) {
      const invalidRoots = [
        ...new Set(
          collectedFolderFiles.map(
            (f) => f.path.replace(/\\/g, "/").split("/")[0],
          ),
        ),
      ];
      const names = invalidRoots.map((r) => `"${r}"`).join(", ");
      validationErrors.push(
        `La carpeta ${names} no cumple el formato esperado. Revise la estructura en "Formato esperado".`,
      );
    }

    if (csvFile) {
      const cols = await readCsvHeader(csvFile);
      const missing = CSV_REQUIRED_COLS.filter((c) => !cols.includes(c));
      if (missing.length > 0) {
        validationErrors.push(
          `Faltan las columnas ${missing.map((c) => `<code>${escapeHtml(c)}</code>`).join(", ")} en el CSV "${escapeHtml(csvFile.name)}".`,
        );
      }
    }

    if (validationErrors.length > 0) {
      showToast(validationErrors.join("<br>"), "error", { sticky: true });
      return;
    }

    // Construir el FormData ANTES de cerrar el modal, porque closeUploadModal
    // resetea collectedFolderFiles a [] y limpia los inputs.
    const formData = new FormData();
    for (const { path, file } of projectFiles) {
      formData.append("files", file);
      formData.append("file_paths", path);
    }
    for (const zip of collectedZipFiles) {
      formData.append("files", zip);
      formData.append("file_paths", zip.name);
    }
    if (csvFile) {
      formData.append("csv_file", csvFile);
    }

    const uploadLabel = formatEntityLabel(countEntities(projectFiles));
    closeUploadModal();
    await startReloadInBackground(formData, uploadLabel);
  }

  async function startReloadInBackground(formData, uploadLabel = null) {
    if (reloadJob) return reloadJob;

    const base = getApiBase();
    if (!base) {
      showToast("No se encontró la configuración de la API.", "error");
      return null;
    }

    const endpoint = `${base}/ui/upload`;
    reloadJob = { status: "starting" };
    pollAttempt = 0;
    clearReloadToastDismissed();
    updateReloadButtonUI();
    showLoadingToast(null, 0, uploadLabel);

    try {
      const data = await uploadWithProgress(endpoint, formData, () => {});
      const jobId = data?.job_id;

      if (!jobId) throw new Error("La API no devolvió job_id.");

      reloadJob = { job_id: jobId, status: data?.status ?? "queued" };

      saveReloadState({
        job_id: jobId,
        status: data?.status ?? "queued",
        startedAt: null,
        createdAt: new Date().toISOString(),
        message: data?.message ?? "Carga iniciada en segundo plano.",
        uploadLabel: uploadLabel ?? null,
      });

      updateReloadButtonUI();
      const prog = data?.progress ?? null;
      maybeShowLoadingToast(
        prog ? prog.current : 0,
        prog ? prog.total : 0,
        uploadLabel,
      );
      emitReloadEvent("reload:start", {
        job_id: jobId,
        status: data?.status ?? "queued",
        message: data?.message,
      });
      scheduleStatusPoll(jobId);
      return data;
    } catch (error) {
      reloadJob = null;
      clearReloadToastDismissed();
      updateReloadButtonUI();
      stopPolling();

      const message = `Error al iniciar la carga.<br><span style="font-weight:700;opacity:.9">${escapeHtml(error?.message || String(error))}</span>`;
      clearReloadState();
      showToast(message, "error", { sticky: true });
      emitReloadEvent("reload:error", {
        error: error?.message || String(error),
        message,
      });
      throw error;
    }
  }

  function restoreReloadToastFromState() {
    const state = readReloadState();
    if (!state) return;

    if (
      ["queued", "running", "starting"].includes(state.status) &&
      state.job_id
    ) {
      reloadJob = { job_id: state.job_id, status: state.status };
      pollAttempt = 0;
      updateReloadButtonUI();

      fetchReloadStatus(state.job_id)
        .then((statusData) => {
          handleReloadStatus(statusData);
        })
        .catch(() => {
          clearReloadState();
          clearReloadToastDismissed();
          reloadJob = null;
          updateReloadButtonUI();
          stopPolling();
          hideToast();
        });

      return;
    }

    if (state.status === "success") {
      clearReloadToastDismissed();
      showToast(state.message || "Carga finalizada.", "success");
      const skipped = state?.result?.skipped ?? [];
      if (skipped.length > 0) {
        openResultsModal(skipped);
      }
      clearReloadState();
      return;
    }

    if (state.status === "error") {
      clearReloadToastDismissed();
      showToast(state.message || "Error al cargar archivos.", "error");
      clearReloadState();
      return;
    }

    clearReloadState();
    clearReloadToastDismissed();
  }

  function attachReloadButtonHandler() {
    const button = getReloadButton();
    if (!button || button.dataset.bound === "true") {
      updateReloadButtonUI();
      return;
    }

    button.dataset.bound = "true";
    button.addEventListener("click", openUploadModal);
    updateReloadButtonUI();

    document
      .getElementById("uploadModalClose")
      ?.addEventListener("click", closeUploadModal);
    document
      .getElementById("uploadBackdrop")
      ?.addEventListener("click", closeUploadModal);
    document
      .getElementById("uploadCancelBtn")
      ?.addEventListener("click", closeUploadModal);
    document
      .getElementById("uploadSubmitBtn")
      ?.addEventListener("click", submitUpload);

    document
      .getElementById("resultsModalClose")
      ?.addEventListener("click", closeResultsModal);
    document
      .getElementById("resultsBackdrop")
      ?.addEventListener("click", closeResultsModal);
    document
      .getElementById("resultsCloseBtn")
      ?.addEventListener("click", closeResultsModal);

    const formatToggle = document.getElementById("uploadFormatToggle");
    const formatHelp = document.getElementById("uploadFormatHelp");
    formatToggle?.addEventListener("click", () => {
      const isHidden = formatHelp?.classList.contains("hidden");
      formatHelp?.classList.toggle("hidden", !isHidden);
      formatToggle.setAttribute("aria-expanded", String(isHidden));
    });

    const dropzone = document.getElementById("foldersDropzone");
    if (dropzone) {
      dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
      dropzone.addEventListener("dragleave", (e) => {
        if (!dropzone.contains(e.relatedTarget)) {
          dropzone.classList.remove("dragover");
        }
      });
      dropzone.addEventListener("drop", async (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        const items = e.dataTransfer?.items;
        if (!items) return;
        const newFiles = [];
        for (const item of items) {
          if (item.kind !== "file") continue;
          const entry = item.webkitGetAsEntry?.();
          if (!entry) continue;
          if (entry.isFile && entry.name.toLowerCase().endsWith(".zip")) {
            await new Promise((resolve) =>
              entry.file((f) => {
                collectedZipFiles.push(f);
                resolve();
              }, resolve),
            );
            continue;
          }
          try {
            const files = await collectFilesFromEntry(entry, "");
            newFiles.push(...files);
          } catch (err) {
            console.warn("Error leyendo entrada:", err);
          }
        }
        collectedFolderFiles.push(...newFiles);
        updateFoldersDisplay();
      });

      dropzone.addEventListener("click", async (e) => {
        if (
          e.target !== dropzone &&
          !dropzone
            .querySelector(".upload-dropzone-content")
            ?.contains(e.target)
        )
          return;
        if (window.showDirectoryPicker) {
          try {
            const dirHandle = await window.showDirectoryPicker({
              mode: "read",
            });
            const files = await collectFilesFromHandle(dirHandle, "");
            collectedFolderFiles.push(...files);
            updateFoldersDisplay();
          } catch (err) {
            if (err.name !== "AbortError")
              console.warn("Error seleccionando carpeta:", err);
          }
        } else {
          document.getElementById("foldersInput")?.click();
        }
      });
    }

    // Fallback input webkitdirectory (cuando showDirectoryPicker no está disponible)
    const foldersInput = document.getElementById("foldersInput");
    foldersInput?.addEventListener("change", () => {
      for (const file of Array.from(foldersInput.files ?? [])) {
        if (!isAcceptedFile(file.name)) continue;
        collectedFolderFiles.push({
          path: file.webkitRelativePath || file.name,
          file,
        });
      }
      foldersInput.value = "";
      updateFoldersDisplay();
    });

    const csvInput = document.getElementById("csvInput");
    csvInput?.addEventListener("change", () => {
      const sel = document.getElementById("csvSelected");
      const name = csvInput.files?.[0]?.name ?? "";
      if (sel) {
        if (name) {
          sel.innerHTML =
            `<span class="upload-folder-tag">` +
            `${escapeHtml(name)}` +
            `<button type="button" class="upload-folder-tag-remove" aria-label="Quitar CSV">✕</button>` +
            `</span>`;
          sel
            .querySelector(".upload-folder-tag-remove")
            ?.addEventListener("click", () => {
              csvInput.value = "";
              sel.innerHTML = "";
            });
        } else {
          sel.innerHTML = "";
        }
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      const uploadModal = document.getElementById("uploadModal");
      if (uploadModal && !uploadModal.classList.contains("hidden")) {
        closeUploadModal();
        return;
      }
      const resultsModal = document.getElementById("resultsModal");
      if (resultsModal && !resultsModal.classList.contains("hidden")) {
        closeResultsModal();
      }
    });
  }

  async function loadTopbar() {
    const response = await fetch(TOPBAR_HTML_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`GET ${TOPBAR_HTML_URL} -> ${response.status}`);
    }

    const mountPoint = document.getElementById("topbar");
    if (mountPoint) {
      mountPoint.innerHTML = await response.text();
    }

    const page = document.body.dataset.page;
    document.querySelector(`[data-page="${page}"]`)?.classList.add("active");

    window.SettingsUI?.mount();
    attachReloadButtonHandler();
    await syncReloadStateFromServer();
    restoreReloadToastFromState();
  }

  loadTopbar().catch((error) => {
    console.error("Error cargando topbar:", error);
  });

  window.TopbarUI = {
    startReloadInBackground,
    openUploadModal,
    closeUploadModal,
    showToast,
    hideToast,
    getReloadState: readReloadState,
  };
})();
