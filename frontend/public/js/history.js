const API_BASE = "http://localhost:8000";
const SETTINGS_KEY = "rag_graphrag_ui_v1";

const els = {
  historyList: document.getElementById("historyList"),
  historyEmpty: document.getElementById("historyEmpty"),
  clearHistoryBtn: document.getElementById("clearHistoryBtn"),
  searchInput: document.getElementById("searchInput"),
  filterAll: document.getElementById("filterAll"),
  filterRag: document.getElementById("filterRag"),
  filterGraph: document.getElementById("filterGraph"),
};

let filterMode = "all"; // all | rag | graphrag
let searchText = "";

/* ================= THEME ================= */

function loadThemeFromSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    const settings = JSON.parse(raw);
    if (settings?.theme === "dark") {
      document.body.classList.add("dark");
    } else {
      document.body.classList.remove("dark");
    }
  } catch {}
}

/* ================= API ================= */

async function apiGetHistory() {
  const res = await fetch(`${API_BASE}/ui/history`);
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`GET /ui/history -> ${res.status} - ${t}`);
  }
  return res.json();
}

async function apiDeleteAll() {
  await fetch(`${API_BASE}/ui/history`, { method: "DELETE" });
}

async function apiDeleteItem(id) {
  await fetch(`${API_BASE}/ui/history/item?id=${id}`, { method: "DELETE" });
}

/* ================= RENDER ================= */

function fmtDate(ts) {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return "";
  }
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

function setActiveFilter() {
  els.filterAll.classList.toggle("active", filterMode === "all");
  els.filterRag.classList.toggle("active", filterMode === "rag");
  els.filterGraph.classList.toggle("active", filterMode === "graphrag");
}

function matches(item) {
  const q = (item?.q ?? "").toLowerCase();
  const mode = (item?.mode ?? "").toLowerCase();

  if (filterMode !== "all" && mode !== filterMode) return false;
  if (searchText.trim() && !q.includes(searchText.toLowerCase())) return false;

  return true;
}

async function render() {
  const all = await apiGetHistory();
  const items = all.filter(matches);

  els.historyList.innerHTML = "";
  els.historyEmpty.style.display = items.length ? "none" : "block";

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "history-row";

    row.innerHTML = `
      <div class="history-main">
        <div class="history-q">${escapeHtml(item.q)}</div>
        <div class="history-meta">
          <span class="history-pill">${
            item.mode === "graphrag" ? "GraphRAG" : "RAG"
          }</span>
          <span class="history-date">${fmtDate(item.ts)}</span>
        </div>
      </div>

      <div class="history-ops">
        <button class="danger-btn" data-id="${item.id}">✕</button>
      </div>
    `;

    row.querySelector("button").addEventListener("click", async () => {
      await apiDeleteItem(item.id);
      render();
    });

    els.historyList.appendChild(row);
  });
}

/* ================= EVENTS ================= */

els.clearHistoryBtn.addEventListener("click", async () => {
  await apiDeleteAll();
  render();
});

els.searchInput.addEventListener("input", (e) => {
  searchText = e.target.value || "";
  render();
});

els.filterAll.addEventListener("click", () => {
  filterMode = "all";
  setActiveFilter();
  render();
});
els.filterRag.addEventListener("click", () => {
  filterMode = "rag";
  setActiveFilter();
  render();
});
els.filterGraph.addEventListener("click", () => {
  filterMode = "graphrag";
  setActiveFilter();
  render();
});

/* ================= INIT ================= */

loadThemeFromSettings();
setActiveFilter();
render();
