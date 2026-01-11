const STORAGE_KEY = "rag_graphrag_ui_v1";
const API_BASE = "http://localhost:8000";

const DEFAULTS = {
  graphrag_enabled: true,
  theme: "light", // "light" | "dark"
  enter_to_send: true,
  compact_mode: false,
};

const els = {
  queryInput: document.getElementById("queryInput"),
  askBtn: document.getElementById("askBtn"),
  chat: document.getElementById("chat"),
  status: document.getElementById("status"),
  modeBadge: document.getElementById("modeBadge"),

  chunksPanel: document.getElementById("chunksPanel"),
  chunksList: document.getElementById("chunksList"),

  settingsBtn: document.getElementById("settingsBtn"),
  modal: document.getElementById("settingsModal"),
  backdrop: document.getElementById("modalBackdrop"),
  closeModalBtn: document.getElementById("closeModalBtn"),
  saveBtn: document.getElementById("saveBtn"),
  resetBtn: document.getElementById("resetBtn"),

  graphToggle: document.getElementById("graphToggle"),
  themeLightBtn: document.getElementById("themeLightBtn"),
  themeDarkBtn: document.getElementById("themeDarkBtn"),

  enterToSend: document.getElementById("enterToSend"),
  compactMode: document.getElementById("compactMode"),
};

/* ========= Settings (localStorage) ========= */

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw);
    return { ...DEFAULTS, ...parsed };
  } catch {
    return { ...DEFAULTS };
  }
}

function saveSettings(s) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}

let settings = loadSettings();
let draft = { ...settings };

/* ========= UI helpers ========= */

function setStatus(text) {
  els.status.textContent = text;
}

function applyTheme(theme) {
  document.body.classList.toggle("dark", theme === "dark");
  els.themeLightBtn.classList.toggle("active", theme === "light");
  els.themeDarkBtn.classList.toggle("active", theme === "dark");
}

function applyCompact(on) {
  document.body.classList.toggle("compact", !!on);
}

function modeLabel(s) {
  return s.graphrag_enabled ? "Modo: GraphRAG" : "Modo: RAG";
}

function escapeHtml(str) {
  return (str ?? "")
    .toString()
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/* ========= Chat UI ========= */

function clearConversationUI() {
  els.chat.innerHTML = "";
  renderChunks([]);
}

function addMessage(type, textOrNode) {
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  if (typeof textOrNode === "string") div.textContent = textOrNode;
  else div.appendChild(textOrNode);

  els.chat.appendChild(div);
  return div;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "typing";
  wrap.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
  return addMessage("assistant", wrap);
}

function renderChunks(chunks) {
  if (!Array.isArray(chunks) || chunks.length === 0) {
    els.chunksList.innerHTML = "";
    els.chunksPanel.classList.add("hidden");
    return;
  }

  els.chunksPanel.classList.remove("hidden");
  els.chunksList.innerHTML = "";

  chunks.forEach((c, idx) => {
    const title = c.id ?? `Chunk ${idx + 1}`;
    const score = typeof c.score === "number" ? c.score.toFixed(3) : "";
    const text = c.text ?? c.content ?? "";

    const card = document.createElement("div");
    card.className = "chunk-card";
    card.innerHTML = `
      <div class="chunk-head">
        <div class="chunk-title" title="${escapeHtml(title)}">${escapeHtml(
          title
        )}</div>
        <div class="chunk-score">${score ? `score ${score}` : ""}</div>
      </div>
      <div class="chunk-text">${escapeHtml(text)}</div>
    `;
    els.chunksList.appendChild(card);
  });
}

/* ========= Modal (draft) ========= */

function syncDraftFromSettings() {
  draft = { ...settings };
}

function renderDraftToModal() {
  els.graphToggle.checked = !!draft.graphrag_enabled;

  applyTheme(draft.theme);

  els.enterToSend.checked = !!draft.enter_to_send;
  els.compactMode.checked = !!draft.compact_mode;

  applyCompact(draft.compact_mode);
}

function openModal() {
  syncDraftFromSettings();
  renderDraftToModal();
  els.backdrop.classList.remove("hidden");
  els.modal.classList.remove("hidden");
  document.body.classList.add("modal-open");
}

function closeModal() {
  els.backdrop.classList.add("hidden");
  els.modal.classList.add("hidden");
  document.body.classList.remove("modal-open");
}

function closeModalWithoutSaving() {
  applyTheme(settings.theme);
  applyCompact(settings.compact_mode);
  closeModal();
}

function commitDraft() {
  settings = { ...draft };
  saveSettings(settings);

  els.modeBadge.textContent = modeLabel(settings);
  applyTheme(settings.theme);
  applyCompact(settings.compact_mode);

  setStatus("Settings guardados.");
}

function resetAll() {
  draft = { ...DEFAULTS };
  renderDraftToModal();
  setStatus("Draft reseteado (guardá para aplicar).");
}

/* ========= Settings listeners ========= */

els.graphToggle.addEventListener("change", () => {
  draft.graphrag_enabled = els.graphToggle.checked;
});

els.themeLightBtn.addEventListener("click", () => {
  draft.theme = "light";
  renderDraftToModal();
});

els.themeDarkBtn.addEventListener("click", () => {
  draft.theme = "dark";
  renderDraftToModal();
});

els.enterToSend.addEventListener("change", () => {
  draft.enter_to_send = els.enterToSend.checked;
});

els.compactMode.addEventListener("change", () => {
  draft.compact_mode = els.compactMode.checked;
  applyCompact(draft.compact_mode);
});

els.saveBtn.addEventListener("click", () => {
  commitDraft();
  closeModal();
});

els.resetBtn.addEventListener("click", resetAll);

els.settingsBtn.addEventListener("click", openModal);
els.backdrop.addEventListener("click", closeModalWithoutSaving);
els.closeModalBtn.addEventListener("click", closeModalWithoutSaving);

/* ========= Query behavior ========= */

function currentEndpoint() {
  const path = settings.graphrag_enabled ? "/graphrag/query" : "/rag/query";
  return `${API_BASE}${path}`;
}

async function ask() {
  const q = els.queryInput.value.trim();
  if (!q) {
    setStatus("Escribí una query.");
    els.queryInput.focus();
    return;
  }

  // requisito: al hacer nueva pregunta, borrar la anterior
  clearConversationUI();

  addMessage("user", q);
  els.queryInput.value = "";

  els.askBtn.disabled = true;
  setStatus("Consultando...");

  const typingMsg = addTyping();

  const url = currentEndpoint();
  const payload = { query: q };

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const t = await res.text().catch(() => "");
      throw new Error(
        `HTTP ${res.status} ${res.statusText}${t ? ` - ${t}` : ""}`
      );
    }

    const data = await res.json();

    typingMsg.remove();
    addMessage("assistant", data.answer ?? "(sin answer)");

    renderChunks(data.chunks ?? data.context ?? []);
    setStatus("Listo.");
  } catch (err) {
    typingMsg.remove();
    addMessage(
      "assistant",
      "Error llamando al backend.\n\n" + String(err?.message || err)
    );
    setStatus("Error.");
  } finally {
    els.askBtn.disabled = false;
    els.queryInput.focus();
  }
}

els.askBtn.addEventListener("click", ask);

els.queryInput.addEventListener("keydown", (e) => {
  if (!settings.enter_to_send) return;
  if (e.key === "Enter") {
    e.preventDefault();
    ask();
  }
});

/* ========= Init ========= */

function getUrlParam(name) {
  try {
    const u = new URL(window.location.href);
    return u.searchParams.get(name);
  } catch {
    return null;
  }
}

function init() {
  els.modeBadge.textContent = modeLabel(settings);
  applyTheme(settings.theme);
  applyCompact(settings.compact_mode);
  renderChunks([]);

  // soporte para venir desde history.html sin sessionStorage
  const prefill = getUrlParam("q");
  const autoAsk = getUrlParam("ask");

  if (prefill) {
    els.queryInput.value = prefill;
    if (autoAsk === "1") setTimeout(() => ask(), 0);
  }
}

init();
