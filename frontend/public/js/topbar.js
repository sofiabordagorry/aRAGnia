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

  function maybeShowReloadToast(message) {
    if (isReloadToastDismissed()) return;
    showToast(message, "loading", { sticky: true });
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
    button.textContent = isRunning ? "Recargando…" : "Recargar archivos";
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

    const parts = ["Recarga finalizada."];

    if (typeof processedCount === "number") {
      parts.push(`Procesados: <strong>${processedCount}</strong>.`);
    }

    if (typeof errorsCount === "number" && errorsCount > 0) {
      parts.push(`Errores: <strong>${errorsCount}</strong>.`);
    }

    return parts.join(" ");
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

        saveReloadState({
          job_id: data.job_id,
          status: data.status,
          startedAt: data?.started_at ?? null,
          createdAt: data?.created_at ?? null,
          message: data?.message ?? "Recarga en curso...",
        });

        updateReloadButtonUI();
        scheduleStatusPoll(data.job_id);
        maybeShowReloadToast(
          "Hay una recarga en curso… Podés seguir usando el chat mientras termina.",
        );
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

      saveReloadState({
        job_id: jobId,
        status,
        startedAt: statusData?.started_at ?? null,
        createdAt: statusData?.created_at ?? null,
        message: statusData?.message ?? "Recarga en curso...",
      });

      updateReloadButtonUI();
      maybeShowReloadToast(
        "Recargando archivos en segundo plano… Podés seguir usando el chat mientras termina.",
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
      emitReloadEvent("reload:finish", state);
      return;
    }

    if (status === "error") {
      const message = statusData?.message
        ? `Error al recargar archivos.<br><span style="font-weight:700;opacity:.9">${escapeHtml(statusData.message)}</span>`
        : "Error al recargar archivos.";

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
          errMessage.includes("No hay recargas registradas")
        ) {
          clearReloadState();
          clearReloadToastDismissed();
          reloadJob = null;
          updateReloadButtonUI();
          stopPolling();
          hideToast();
          return;
        }

        const message = `Error consultando estado de recarga.<br><span style="font-weight:700;opacity:.9">${escapeHtml(errMessage)}</span>`;
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

  async function startReloadInBackground() {
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
    maybeShowReloadToast("Iniciando recarga en segundo plano…");

    try {
      const data = await parseJsonResponse(
        await fetch(endpoint, { method: "POST" }),
        endpoint,
      );
      const jobId = data?.job_id;

      if (!jobId) throw new Error("La API no devolvió job_id.");

      reloadJob = { job_id: jobId, status: data?.status ?? "queued" };

      saveReloadState({
        job_id: jobId,
        status: data?.status ?? "queued",
        startedAt: null,
        createdAt: new Date().toISOString(),
        message: data?.message ?? "Recarga iniciada en segundo plano.",
      });

      updateReloadButtonUI();
      maybeShowReloadToast(
        "Recargando archivos en segundo plano… Podés seguir usando el chat mientras termina.",
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

      const message = `Error al iniciar la recarga.<br><span style="font-weight:700;opacity:.9">${escapeHtml(error?.message || String(error))}</span>`;
      saveReloadState({
        status: "error",
        error: error?.message || String(error),
        message,
      });
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
      showToast(state.message || "Recarga finalizada.", "success");
      clearReloadState();
      return;
    }

    if (state.status === "error") {
      clearReloadToastDismissed();
      showToast(state.message || "Error al recargar archivos.", "error", {
        sticky: true,
      });
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
    button.addEventListener("click", startReloadInBackground);
    updateReloadButtonUI();
  }

  async function loadTopbar() {
    const response = await fetch(TOPBAR_HTML_URL);
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
    showToast,
    hideToast,
    getReloadState: readReloadState,
  };
})();
