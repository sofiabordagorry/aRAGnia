const API_BASE = window.APP_CONFIG?.API_BASE;

const els = {
  historyList: document.getElementById("historyList"),
  historyEmpty: document.getElementById("historyEmpty"),
  clearHistoryBtn: document.getElementById("clearHistoryBtn"),
  searchInput: document.getElementById("searchInput"),
  // filterAll: document.getElementById("filterAll"),
  // filterRag: document.getElementById("filterRag"),
  // filterGraph: document.getElementById("filterGraph"),
  drawerRoot: document.getElementById("historyDrawer"),
  drawerBackdrop: document.getElementById("historyDrawerBackdrop"),
  drawerCloseBtn: document.getElementById("drawerCloseBtn"),
  // drawerMode: document.getElementById("drawerMode"),
  drawerDate: document.getElementById("drawerDate"),
  drawerQuestion: document.getElementById("drawerQuestion"),
  drawerCypherSection: document.getElementById("drawerCypherSection"),
  drawerCypher: document.getElementById("drawerCypher"),
  drawerAnswer: document.getElementById("drawerAnswer"),
  drawerChunks: document.getElementById("drawerChunks"),
};

let filterMode = "all";
let searchText = "";
let historyCache = [];
let cacheLoaded = false;

if (!API_BASE) {
  console.error("API_BASE no está definido.");
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

function normMode(value) {
  return (value ?? "").toString().trim().toLowerCase();
}

function toMillis(value) {
  if (!value) return null;
  if (typeof value === "number") return value;

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.getTime();
}

function formatDate(value) {
  const millis = toMillis(value);
  return millis ? new Date(millis).toLocaleString() : "";
}

function normalizeEntity(raw) {
  return {
    entity_id: raw?.entity_id ?? raw?.id ?? "",
    entity_label: raw?.entity_label ?? raw?.label ?? "",
  };
}

function normalizeChunk(raw) {
  return {
    id: raw?.id ?? "",
    chunk_id: raw?.chunk_id ?? raw?.chunkId ?? "",
    chunk_text: raw?.chunk_text ?? raw?.chunk ?? raw?.text ?? "",
    score: raw?.score ?? null,
    entities: Array.isArray(raw?.entities)
      ? raw.entities.map(normalizeEntity)
      : [],
  };
}

function normalizeHistoryItem(raw) {
  return {
    id: raw?.id,
    mode: normMode(raw?.type ?? raw?.mode),
    q: raw?.query_text ?? raw?.q ?? raw?.query ?? "",
    cypherQuery: raw?.cypher_query ?? raw?.cypherQuery ?? "",
    answer: raw?.response ?? raw?.answer ?? raw?.answer_text ?? "",
    ts: raw?.created_at ?? raw?.ts ?? raw?.timestamp ?? null,
    chunks: Array.isArray(raw?.chunks) ? raw.chunks.map(normalizeChunk) : [],
  };
}

async function apiGetHistory() {
  const response = await fetch(`${API_BASE}/ui/history`);
  if (!response.ok) {
    throw new Error(
      `GET /ui/history -> ${response.status} - ${await response.text()}`,
    );
  }
  return response.json();
}

async function apiDeleteAll() {
  const response = await fetch(`${API_BASE}/ui/history`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(
      `DELETE /ui/history -> ${response.status} - ${await response.text()}`,
    );
  }
}

async function apiDeleteItem(id) {
  const response = await fetch(
    `${API_BASE}/ui/history/item?id=${encodeURIComponent(id)}`,
    {
      method: "DELETE",
    },
  );

  if (!response.ok) {
    throw new Error(
      `DELETE /ui/history/item -> ${response.status} - ${await response.text()}`,
    );
  }
}

function setActiveFilter() {
  els.filterAll?.classList.toggle("active", filterMode === "all");
  els.filterRag?.classList.toggle("active", filterMode === "rag");
  els.filterGraph?.classList.toggle("active", filterMode === "graphrag");
}

function matches(item) {
  const term = searchText.trim().toLowerCase();
  if (filterMode !== "all" && normMode(item.mode) !== filterMode) return false;
  if (!term) return true;

  return [item.q, item.answer, item.cypherQuery].some((value) =>
    (value ?? "").toLowerCase().includes(term),
  );
}

function drawerOpen() {
  if (!els.drawerRoot) return;
  els.drawerRoot.classList.add("open");
  els.drawerRoot.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function drawerClose() {
  if (!els.drawerRoot) return;
  els.drawerRoot.classList.remove("open");
  els.drawerRoot.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function renderChunks(chunks, mode) {
  if (!els.drawerChunks) return;
  els.drawerChunks.innerHTML = "";

  if (!Array.isArray(chunks) || chunks.length === 0) {
    els.drawerChunks.innerHTML = '<div class="drawer-box">Sin chunks.</div>';
    return;
  }

  chunks.forEach((chunk) => {
    const card = document.createElement("div");
    card.className = "chunk-card";

    // const modeHtml =
    //   mode === "graphrag"
    //     ? '<span class="chunk-meta-pill chunk-meta-graph">Graph chunk</span>'
    //     : '<span class="chunk-meta-pill chunk-meta-rag">RAG chunk</span>';

    const modeHtml = '<span class="chunk-meta-pill chunk-meta-graph">Graph chunk</span>';
    
    const scoreHtml =
      chunk.score !== null && chunk.score !== undefined && chunk.score !== ""
        ? `<span class="chunk-meta-pill"><strong>score:</strong> ${escapeHtml(chunk.score)}</span>`
        : "";

    const entitiesHtml =
      Array.isArray(chunk.entities) && chunk.entities.length
        ? `
        <div class="chunk-entities">
          <div class="chunk-entities-title">Entidades</div>
          <div class="chunk-entities-list">
            ${chunk.entities
              .map(
                (entity) => `
                  <span class="entity-pill">
                    <span class="entity-label">${escapeHtml(entity.entity_label || "Entidad")}</span>
                    <span class="entity-id">${escapeHtml(entity.entity_id || "")}</span>
                  </span>`,
              )
              .join("")}
          </div>
        </div>`
        : "";

    card.innerHTML = `
      <div class="chunk-top">
        <div class="chunk-top-left">
          ${modeHtml}
          ${scoreHtml}
        </div>
        <span class="chunk-id">${escapeHtml(chunk.chunk_id || chunk.id || "")}</span>
      </div>
      <div class="chunk-text">${escapeHtml(chunk.chunk_text ?? "")}</div>
      ${entitiesHtml}
    `;

    els.drawerChunks.appendChild(card);
  });
}

function renderDrawer(item) {
  if (!els.drawerRoot) return;

  // els.drawerMode.textContent =
  //   item.mode === "graphrag"
  //     ? "GraphRAG"
  //     : item.mode === "rag"
  //       ? "RAG"
  //       : "Desconocido";
  els.drawerDate.textContent = formatDate(item.ts);
  els.drawerQuestion.textContent = item.q ?? "";
  els.drawerAnswer.textContent = item.answer ?? "";

  const hasCypher = Boolean(item.cypherQuery?.trim());
  if (els.drawerCypherSection) els.drawerCypherSection.hidden = !hasCypher;
  if (els.drawerCypher)
    els.drawerCypher.textContent = hasCypher ? item.cypherQuery : "";

  renderChunks(item.chunks, item.mode);
}

function openHistoryDrawerById(id) {
  const item = historyCache.find((entry) => String(entry.id) === String(id));
  if (!item) return;
  renderDrawer(item);
  drawerOpen();
}

function renderFromCache() {
  if (!els.historyList || !els.historyEmpty) return;

  const items = historyCache.filter(matches);
  els.historyList.innerHTML = "";
  els.historyEmpty.style.display = items.length ? "none" : "block";

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "history-row";
    row.title = "Click para ver detalle";

    // const modeLabel =
    //   item.mode === "graphrag" ? "GraphRAG" : item.mode === "rag" ? "RAG" : "?";
    const extraMeta =
      item.mode === "graphrag" && item.cypherQuery
        ? '<span class="history-extra">Cypher</span>'
        : "";

    row.innerHTML = `
      <div class="history-main">
        <div class="history-q">${escapeHtml(item.q)}</div>
        <div class="history-meta">
          ${extraMeta}
          <span class="history-date">${formatDate(item.ts)}</span>
        </div>
      </div>
      <div class="history-ops">
        <button class="danger-btn" data-id="${escapeHtml(item.id)}" type="button">✕</button>
      </div>
    `;

    row.addEventListener("click", (event) => {
      if (event.target.closest("button")) return;
      openHistoryDrawerById(item.id);
    });

    row.querySelector("button")?.addEventListener("click", async () => {
      try {
        await apiDeleteItem(item.id);
        historyCache = historyCache.filter((entry) => entry.id !== item.id);
        renderFromCache();
        drawerClose();
      } catch (error) {
        console.error(error);
      }
    });

    els.historyList.appendChild(row);
  });
}

async function loadHistoryOnce() {
  if (cacheLoaded) return;

  try {
    const raw = await apiGetHistory();
    historyCache = Array.isArray(raw) ? raw.map(normalizeHistoryItem) : [];
    historyCache.sort((a, b) => (toMillis(b.ts) ?? 0) - (toMillis(a.ts) ?? 0));
    cacheLoaded = true;
    renderFromCache();
  } catch (error) {
    console.error(error);
    if (els.historyEmpty) els.historyEmpty.style.display = "block";
    if (els.historyList) els.historyList.innerHTML = "";
  }
}

els.clearHistoryBtn?.addEventListener("click", async () => {
  try {
    await apiDeleteAll();
    historyCache = [];
    renderFromCache();
    drawerClose();
  } catch (error) {
    console.error(error);
  }
});

els.searchInput?.addEventListener("input", (event) => {
  searchText = event.target.value || "";
  renderFromCache();
});

// els.filterAll?.addEventListener("click", () => {
//   filterMode = "all";
//   setActiveFilter();
//   renderFromCache();
// });

// els.filterRag?.addEventListener("click", () => {
//   filterMode = "rag";
//   setActiveFilter();
//   renderFromCache();
// });

// els.filterGraph?.addEventListener("click", () => {
//   filterMode = "graphrag";
//   setActiveFilter();
//   renderFromCache();
// });

els.drawerBackdrop?.addEventListener("click", drawerClose);
els.drawerCloseBtn?.addEventListener("click", drawerClose);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && els.drawerRoot?.classList.contains("open")) {
    drawerClose();
  }
});

// setActiveFilter();
loadHistoryOnce();
