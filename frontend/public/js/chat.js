const API_BASE = window.APP_CONFIG?.API_BASE || "http://localhost:8000";
const DEFAULTS = {
  graphrag_enabled: true,
  enter_to_send: true,
};

const els = {
  queryInput: document.getElementById("queryInput"),
  askBtn: document.getElementById("askBtn"),
  chat: document.getElementById("chat"),
  modeBadge: document.getElementById("modeBadge"),
};

let settings = getSettings();

function getSettings() {
  return { ...DEFAULTS, ...(window.SettingsUI?.getSettings?.() || {}) };
}

function modeLabel(currentSettings) {
  return currentSettings.graphrag_enabled ? "Modo: GraphRAG" : "Modo: RAG";
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

function scrollChatToBottom() {
  if (els.chat) {
    els.chat.scrollTop = els.chat.scrollHeight;
  }
}

function clearConversationUI() {
  if (els.chat) els.chat.innerHTML = "";
}

function addMessage(type, content) {
  const element = document.createElement("div");
  element.className = `msg ${type}`;

  if (typeof content === "string") {
    if (type === "assistant" && typeof marked !== "undefined") {
      element.innerHTML = marked.parse(content);
    } else {
      element.textContent = content;
    }
  } else {
    element.appendChild(content);
  }

  els.chat?.appendChild(element);
  scrollChatToBottom();
  return element;
}

function addTyping() {
  const indicator = document.createElement("div");
  indicator.className = "typing-indicator";
  indicator.innerHTML = `
    <span class="dot"></span>
    <span class="dot"></span>
    <span class="dot"></span>
  `;

  const message = addMessage("assistant", indicator);
  message.classList.add("typing");
  return message;
}

function getEntitySnippet(text, entities) {
  const before = 80;
  const after = 160;
  const terms = [];

  for (const [id, label] of entities) {
    if (label !== "Investigador") continue;

    terms.push(id);
    const parts = id.trim().split(/\s+/);
    if (parts.length > 1) terms.push(parts.at(-1));
  }

  const lowerText = text.toLowerCase();

  for (const term of terms) {
    if (!term) continue;

    const index = lowerText.indexOf(term.toLowerCase());
    if (index === -1) continue;

    const start = Math.max(0, index - before);
    const end = Math.min(text.length, index + term.length + after);

    return `${start > 0 ? "…" : ""}${text.slice(start, end)}${end < text.length ? "…" : ""}`;
  }

  return `${text.slice(0, 300)}${text.length > 300 ? "…" : ""}`;
}

function renderChunks(chunks, chunkToEntities, messageElement) {
  if (!messageElement || !Array.isArray(chunks) || chunks.length === 0) return;

  const wrapper = document.createElement("div");
  wrapper.className = "msg-sources";

  const toggle = document.createElement("button");
  toggle.className = "sources-toggle";
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");
  toggle.innerHTML = `Fuentes <span class="sources-count">${chunks.length}</span><span class="sources-icon"></span>`;
  toggle.addEventListener("click", () => {
    const isOpen = wrapper.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(isOpen));
  });

  const list = document.createElement("div");
  list.className = "sources-list";

  chunks.forEach((chunk, index) => {
    const title = chunk.id ?? `Chunk ${index + 1}`;
    const text = (chunk.text ?? chunk.content ?? "")
      .replace(/[ \t]+/g, " ")
      .replace(/\n[ \t]+/g, "\n")
      .trim();

    const entities = chunkToEntities[chunk.id] || [];
    const tags =
      entities.length > 0 && settings.graphrag_enabled
        ? `<div class="source-entities">${entities
            .map(
              ([id, label]) =>
                `<span class="entity-tag" title="${escapeHtml(label)}">${escapeHtml(id)}</span>`,
            )
            .join(" ")}</div>`
        : "";

    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <div class="source-num">${index + 1}</div>
      <div class="source-body">
        <div class="source-title" title="${escapeHtml(title)}">${escapeHtml(title)}</div>
        <div class="source-preview">${escapeHtml(getEntitySnippet(text, entities))}</div>
        ${tags}
      </div>
    `;

    list.appendChild(item);
  });

  wrapper.append(toggle, list);
  messageElement.appendChild(wrapper);
  scrollChatToBottom();
}

function currentEndpoint() {
  return `${API_BASE}${settings.graphrag_enabled ? "/graphrag/query" : "/rag/query"}`;
}

async function ask() {
  const query = els.queryInput?.value.trim();
  if (!query) {
    els.queryInput?.focus();
    return;
  }

  clearConversationUI();
  addMessage("user", query);

  if (els.queryInput) els.queryInput.value = "";
  if (els.askBtn) els.askBtn.disabled = true;

  const typingMessage = addTyping();

  try {
    const response = await fetch(currentEndpoint(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(
        `HTTP ${response.status} ${response.statusText}${text ? ` - ${text}` : ""}`,
      );
    }

    const data = await response.json();
    typingMessage.remove();

    const assistantMessage = addMessage(
      "assistant",
      data.answer ?? "(sin answer)",
    );
    renderChunks(
      data.chunks ?? data.context ?? [],
      data.chunk_to_entities || {},
      assistantMessage,
    );
  } catch (error) {
    typingMessage.remove();
    addMessage(
      "assistant",
      `Error llamando al backend.\n\n${error?.message || error}`,
    );
  } finally {
    if (els.askBtn) els.askBtn.disabled = false;
    els.queryInput?.focus();
    scrollChatToBottom();
  }
}

function getUrlParam(name) {
  try {
    return new URL(window.location.href).searchParams.get(name);
  } catch {
    return null;
  }
}

function init() {
  settings = getSettings();
  if (els.modeBadge) els.modeBadge.textContent = modeLabel(settings);

  const prefill = getUrlParam("q");
  const autoAsk = getUrlParam("ask");

  if (prefill && els.queryInput) {
    els.queryInput.value = prefill;
    if (autoAsk === "1") setTimeout(ask, 0);
  }

  els.askBtn?.addEventListener("click", ask);
  els.queryInput?.addEventListener("keydown", (event) => {
    settings = getSettings();
    if (settings.enter_to_send && event.key === "Enter") {
      event.preventDefault();
      ask();
    }
  });

  window.addEventListener("settings:changed", (event) => {
    settings = event.detail || getSettings();
    if (els.modeBadge) els.modeBadge.textContent = modeLabel(settings);
  });
}

init();
