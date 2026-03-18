const GRAPH_API_BASE = window.APP_CONFIG?.API_BASE || "http://localhost:8000";

(function initGraphPanel() {
  const els = {
    svg: document.getElementById("graphSvg"),
    empty: document.getElementById("graphEmptyState"),
    status: document.getElementById("graphStatus"),
    refreshBtn: document.getElementById("refreshGraphBtn"),
    nodeCount: document.getElementById("graphNodeCount"),
    edgeCount: document.getElementById("graphEdgeCount"),
    aliasCount: document.getElementById("graphAliasCount"),
    legend: document.getElementById("graphLegend"),
    selection: document.getElementById("graphSelection"),
    aliasStatus: document.getElementById("aliasStatus"),
    aliasEntityCount: document.getElementById("aliasEntityCount"),
    aliasList: document.getElementById("aliasList"),
    aliasSearchInput: document.getElementById("aliasSearchInput"),
    aliasSection: document.getElementById("aliasSection"),
    entityStatus: document.getElementById("entityStatus"),
    entityResultCount: document.getElementById("entityResultCount"),
    entitySearchInput: document.getElementById("entitySearchInput"),
    entityTypeSelect: document.getElementById("entityTypeSelect"),
    entityList: document.getElementById("entityList"),
    entityPanel: document.getElementById("entityPanel"),
    tabEntitiesBtn: document.getElementById("tabEntitiesBtn"),
    tabAliasesBtn: document.getElementById("tabAliasesBtn"),
  };

  if (!els.svg) return;

  const LABEL_COLORS = {
    Investigador: "#0f766e",
    Proyecto: "#1d4ed8",
    Topico: "#c2410c",
    Documento: "#6d28d9",
    Anio: "#334155",
    Entidad: "#0f172a",
  };

  const state = {
    snapshot: {
      nodes: [],
      edges: [],
      summary: { node_count: 0, edge_count: 0, alias_edge_count: 0 },
    },
    aliasData: {
      pairs: [],
      entities: [],
      summary: { pair_count: 0, entity_count: 0 },
    },
    entityCatalog: [],
    aliasOnly: false,
    selectedNodeId: null,
    selectedEntityId: null,
    loadingGraph: false,
    loadingAliases: false,
    loadingEntities: false,
    aliasLoadedOnce: false,
    activeSidePanel: "entities",
    entitySearch: "",
    entityType: "Investigador",
    aliasSearch: "",
    zoomScale: 1,
    panX: 0,
    panY: 0,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    panStartX: 0,
    panStartY: 0,
  };

  const ZOOM_STEP = 0.2;
  const ZOOM_MIN = 0.3;
  const ZOOM_MAX = 4.0;

  function escapeHtml(value) {
    return (value ?? "")
      .toString()
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function debounce(fn, waitMs) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), waitMs);
    };
  }

  function setGraphStatus(message, isError = false) {
    if (!els.status) return;
    els.status.textContent = message;
    els.status.classList.toggle("error", isError);
  }

  function setEntityStatus(message, isError = false) {
    if (!els.entityStatus) return;
    els.entityStatus.textContent = message;
    els.entityStatus.classList.toggle("error", isError);
  }

  function setAliasStatus(message, isError = false) {
    if (!els.aliasStatus) return;
    els.aliasStatus.textContent = message;
    els.aliasStatus.classList.toggle("error", isError);
  }

  function updateMetrics(summary) {
    const safeSummary = summary || {
      node_count: 0,
      edge_count: 0,
      alias_edge_count: 0,
    };
    if (els.nodeCount)
      els.nodeCount.textContent = `${safeSummary.node_count} nodos`;
    if (els.edgeCount)
      els.edgeCount.textContent = `${safeSummary.edge_count} relaciones`;
    if (els.aliasCount)
      els.aliasCount.textContent = `${safeSummary.alias_edge_count} aliases`;
  }

  function setActiveSidePanel(panelName) {
    state.activeSidePanel = panelName === "aliases" ? "aliases" : "entities";
    const entityActive = state.activeSidePanel === "entities";
    const nextAliasOnly = !entityActive;
    const changedAliasMode = state.aliasOnly !== nextAliasOnly;

    state.aliasOnly = nextAliasOnly;

    els.entityPanel?.classList.toggle("active", entityActive);
    els.aliasSection?.classList.toggle("active", !entityActive);

    els.tabEntitiesBtn?.classList.toggle("active", entityActive);
    els.tabAliasesBtn?.classList.toggle("active", !entityActive);

    if (els.tabEntitiesBtn) {
      els.tabEntitiesBtn.setAttribute(
        "aria-selected",
        entityActive ? "true" : "false",
      );
    }
    if (els.tabAliasesBtn) {
      els.tabAliasesBtn.setAttribute(
        "aria-selected",
        !entityActive ? "true" : "false",
      );
    }

    if (!entityActive && !state.aliasLoadedOnce) {
      loadAliases();
    }

    if (changedAliasMode && state.selectedEntityId) {
      loadNeighborhood(state.selectedEntityId);
    }

    if (!state.selectedEntityId) {
      setGraphStatus(
        state.aliasOnly
          ? "Vista de alias activa. Selecciona una entidad para ver solo relaciones POSIBLE_ALIAS."
          : "Vista de entidades activa. Selecciona una entidad para ver su vecindad completa.",
      );
      renderGraph();
    }
  }

  function toggleEmptyState(show, message) {
    if (!els.empty) return;
    if (message) els.empty.textContent = message;
    els.empty.classList.toggle("hidden", !show);
  }

  function apiUrl(path, params = {}) {
    const url = new URL(`${GRAPH_API_BASE}${path}`);
    Object.entries(params).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      const stringValue = String(value);
      if (stringValue === "") return;
      url.searchParams.set(key, stringValue);
    });
    return url.toString();
  }

  async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(
        `HTTP ${response.status} ${response.statusText}${text ? ` - ${text}` : ""}`,
      );
    }
    return response.json();
  }

  function getLabelColor(label) {
    return LABEL_COLORS[label] || LABEL_COLORS.Entidad;
  }

  function truncateLabel(value, maxLength = 18) {
    if (!value || value.length <= maxLength) return value;
    return `${value.slice(0, maxLength - 1)}...`;
  }

  function buildLegend(nodes) {
    if (!els.legend) return;
    const counts = new Map();
    nodes.forEach((node) =>
      counts.set(node.label, (counts.get(node.label) || 0) + 1),
    );
    els.legend.innerHTML = Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(
        ([label, count]) => `
          <span class="legend-pill">
            <span class="legend-swatch" style="background:${getLabelColor(label)}"></span>
            ${escapeHtml(label)} · ${count}
          </span>
        `,
      )
      .join("");
  }

  function buildLayout(nodes, width, height) {
    const positions = new Map();
    const grouped = new Map();
    nodes.forEach((node) => {
      if (!grouped.has(node.label)) grouped.set(node.label, []);
      grouped.get(node.label).push(node);
    });

    const labels = Array.from(grouped.keys()).sort();
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.max(80, Math.min(width, height) / 2 - 56);

    if (nodes.length === 1) {
      positions.set(nodes[0].id, { x: centerX, y: centerY });
      return positions;
    }

    labels.forEach((label, groupIndex) => {
      const groupNodes = grouped.get(label) || [];
      const radius =
        labels.length === 1
          ? maxRadius * 0.65
          : 54 +
            (groupIndex * (maxRadius - 54)) / Math.max(labels.length - 1, 1);
      groupNodes.forEach((node, index) => {
        const angleOffset = groupIndex * 0.48;
        const angle =
          (2 * Math.PI * index) / Math.max(groupNodes.length, 1) + angleOffset;
        const jitter =
          groupNodes.length <= 2 ? 0 : (index % 2 === 0 ? 1 : -1) * 12;
        positions.set(node.id, {
          x: centerX + Math.cos(angle) * (radius + jitter),
          y: centerY + Math.sin(angle) * (radius + jitter),
        });
      });
    });

    return positions;
  }

  function applyTransform() {
    const viewport = els.svg.querySelector("#graphViewport");
    if (!viewport) return;
    viewport.setAttribute(
      "transform",
      `translate(${state.panX.toFixed(3)} ${state.panY.toFixed(3)}) scale(${state.zoomScale.toFixed(3)})`,
    );
  }

  // alias kept for callers that used the old name
  const applyZoom = applyTransform;

  function resetTransform() {
    const viewBox = els.svg.viewBox.baseVal;
    state.zoomScale = 1;
    state.panX = viewBox.width
      ? (viewBox.width / 2) * (1 - state.zoomScale)
      : 0;
    state.panY = viewBox.height
      ? (viewBox.height / 2) * (1 - state.zoomScale)
      : 0;
    applyTransform();
  }

  function zoomAtPoint(delta, svgX, svgY) {
    const prevScale = state.zoomScale;
    const nextScale = Math.min(
      ZOOM_MAX,
      Math.max(ZOOM_MIN, prevScale * (delta > 0 ? 1 / 1.12 : 1.12)),
    );
    if (nextScale === prevScale) return;
    // keep the point under cursor fixed: pan += origin * (prevScale - nextScale)
    state.panX = svgX - (svgX - state.panX) * (nextScale / prevScale);
    state.panY = svgY - (svgY - state.panY) * (nextScale / prevScale);
    state.zoomScale = nextScale;
    applyTransform();
  }

  function zoomGraph(direction) {
    const viewBox = els.svg.viewBox.baseVal;
    const cx = viewBox.width / 2;
    const cy = viewBox.height / 2;
    zoomAtPoint(direction === "in" ? -1 : 1, cx, cy);
  }

  function bindPanAndZoom() {
    const svg = els.svg;
    if (!svg) return;

    // --- Mouse drag ---
    svg.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      // don't start pan when clicking a node
      if (e.target.closest("[data-node-id]")) return;
      state.dragging = true;
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      state.panStartX = state.panX;
      state.panStartY = state.panY;
      svg.style.cursor = "grabbing";
      e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
      if (!state.dragging) return;
      state.panX = state.panStartX + (e.clientX - state.dragStartX);
      state.panY = state.panStartY + (e.clientY - state.dragStartY);
      applyTransform();
    });

    window.addEventListener("mouseup", () => {
      if (!state.dragging) return;
      state.dragging = false;
      svg.style.cursor = "grab";
    });

    // --- Touch drag ---
    svg.addEventListener(
      "touchstart",
      (e) => {
        if (e.touches.length !== 1) return;
        if (e.target.closest("[data-node-id]")) return;
        state.dragging = true;
        state.dragStartX = e.touches[0].clientX;
        state.dragStartY = e.touches[0].clientY;
        state.panStartX = state.panX;
        state.panStartY = state.panY;
      },
      { passive: true },
    );

    window.addEventListener(
      "touchmove",
      (e) => {
        if (!state.dragging || e.touches.length !== 1) return;
        state.panX =
          state.panStartX + (e.touches[0].clientX - state.dragStartX);
        state.panY =
          state.panStartY + (e.touches[0].clientY - state.dragStartY);
        applyTransform();
      },
      { passive: true },
    );

    window.addEventListener("touchend", () => {
      state.dragging = false;
    });

    // --- Wheel zoom (zoom toward cursor) ---
    svg.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const rect = svg.getBoundingClientRect();
        const viewBox = svg.viewBox.baseVal;
        // convert screen coords -> SVG viewBox coords
        const svgX = ((e.clientX - rect.left) / rect.width) * viewBox.width;
        const svgY = ((e.clientY - rect.top) / rect.height) * viewBox.height;
        zoomAtPoint(e.deltaY, svgX, svgY);
      },
      { passive: false },
    );
  }

  function renderSelection(snapshot) {
    if (!els.selection) return;
    if (!state.selectedNodeId) {
      els.selection.textContent = state.aliasOnly
        ? "Filtro POSIBLE_ALIAS activo. Selecciona una entidad para cargar solo sus relaciones de alias."
        : "Selecciona una entidad desde la card de exploración para visualizar su vecindad.";
      return;
    }

    const node = snapshot.nodes.find(
      (item) => item.id === state.selectedNodeId,
    );
    if (!node) {
      state.selectedNodeId = null;
      renderSelection(snapshot);
      return;
    }

    const related = snapshot.edges
      .filter((edge) => edge.source === node.id || edge.target === node.id)
      .map((edge) => {
        const peerId = edge.source === node.id ? edge.target : edge.source;
        const peer = snapshot.nodes.find((item) => item.id === peerId);
        return { display: peer?.display || peerId, label: edge.type };
      });

    const relationMarkup = related.length
      ? `<div class="selection-relations">${related
          .slice(0, 12)
          .map(
            (item) =>
              `<span class="selection-pill"><span class="selection-pill-label">${escapeHtml(item.label)}</span><span class="selection-pill-target">${escapeHtml(item.display)}</span></span>`,
          )
          .join("")}</div>`
      : "";

    els.selection.innerHTML = `
      <strong>${escapeHtml(node.display)}</strong> · ${escapeHtml(node.label)} · ${node.degree} conexiones
      ${relationMarkup}
    `;
  }

  function renderGraph() {
    const snapshot = state.snapshot;
    if (
      !snapshot ||
      !Array.isArray(snapshot.nodes) ||
      snapshot.nodes.length === 0
    ) {
      els.svg.innerHTML = "";
      applyZoom();
      buildLegend([]);
      renderSelection({ nodes: [], edges: [] });
      toggleEmptyState(
        true,
        state.selectedEntityId
          ? "No hay relaciones para la entidad seleccionada con el filtro actual."
          : "Grafo vacío: busca y selecciona una entidad para empezar.",
      );
      return;
    }

    toggleEmptyState(false);
    buildLegend(snapshot.nodes);
    const width = Math.max(720, Math.round(els.svg.clientWidth || 720));
    const height = Math.max(520, Math.round(els.svg.clientHeight || 520));
    els.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

    const positions = buildLayout(snapshot.nodes, width, height);
    const connectedNodeIds = new Set();
    if (state.selectedNodeId) {
      connectedNodeIds.add(state.selectedNodeId);
      snapshot.edges.forEach((edge) => {
        if (edge.source === state.selectedNodeId)
          connectedNodeIds.add(edge.target);
        if (edge.target === state.selectedNodeId)
          connectedNodeIds.add(edge.source);
      });
    }

    const edgeMarkup = snapshot.edges
      .map((edge) => {
        const source = positions.get(edge.source);
        const target = positions.get(edge.target);
        if (!source || !target) return "";
        const highlighted =
          state.selectedNodeId &&
          (edge.source === state.selectedNodeId ||
            edge.target === state.selectedNodeId);
        const dimmed = state.selectedNodeId && !highlighted;
        const edgeClasses = ["graph-edge"];
        if (edge.is_alias) edgeClasses.push("alias");
        if (highlighted) edgeClasses.push("highlighted");
        if (dimmed) edgeClasses.push("dimmed");
        return `
          <line class="${edgeClasses.join(" ")}" x1="${source.x.toFixed(2)}" y1="${source.y.toFixed(2)}" x2="${target.x.toFixed(2)}" y2="${target.y.toFixed(2)}">
            <title>${escapeHtml(edge.type)}: ${escapeHtml(edge.source)} -> ${escapeHtml(edge.target)}</title>
          </line>
        `;
      })
      .join("");

    const nodeMarkup = snapshot.nodes
      .map((node) => {
        const position = positions.get(node.id);
        if (!position) return "";
        const radius = Math.max(
          8,
          Math.min(18, 8 + Math.round(node.degree / 2)),
        );
        const nodeClasses = ["graph-node"];
        if (node.is_alias_candidate) nodeClasses.push("alias-candidate");
        if (state.selectedNodeId === node.id) nodeClasses.push("selected");
        if (state.selectedNodeId && !connectedNodeIds.has(node.id))
          nodeClasses.push("dimmed");
        return `
          <g class="${nodeClasses.join(" ")}" data-node-id="${escapeHtml(node.id)}" transform="translate(${position.x.toFixed(2)} ${position.y.toFixed(2)})">
            <circle class="graph-node-circle" r="${radius}" fill="${getLabelColor(node.label)}"></circle>
            <text class="graph-node-label" y="${radius + 18}">${escapeHtml(truncateLabel(node.display))}</text>
            <title>${escapeHtml(node.display)} (${escapeHtml(node.label)})</title>
          </g>
        `;
      })
      .join("");

    els.svg.innerHTML = `<g id="graphViewport">${edgeMarkup}${nodeMarkup}</g>`;
    applyZoom();
    els.svg.querySelectorAll("[data-node-id]").forEach((nodeElement) => {
      nodeElement.addEventListener("click", () => {
        const nodeId = nodeElement.getAttribute("data-node-id");
        state.selectedNodeId = state.selectedNodeId === nodeId ? null : nodeId;
        renderGraph();
      });
    });

    renderSelection(snapshot);
  }

  function renderEntityList() {
    if (!els.entityList) return;
    const entities = state.entityCatalog || [];
    if (!entities.length) {
      els.entityList.innerHTML =
        '<div class="alias-placeholder">No hay entidades para ese filtro.</div>';
      return;
    }

    els.entityList.innerHTML = entities
      .map(
        (entity) => `
          <button class="entity-item ${state.selectedEntityId === entity.id ? "active" : ""}" type="button" data-entity-id="${escapeHtml(entity.id)}">
            <span class="entity-item-main">
              <span class="entity-item-title">${escapeHtml(entity.display)}</span>
              <span class="entity-item-id">${escapeHtml(entity.id)}</span>
            </span>
            <span class="entity-pill">${escapeHtml(entity.label)}</span>
          </button>
        `,
      )
      .join("");

    els.entityList.querySelectorAll("[data-entity-id]").forEach((button) => {
      button.addEventListener("click", () => {
        const entityId = button.getAttribute("data-entity-id") || "";
        selectEntity(entityId);
      });
    });
  }

  async function loadEntityCatalog() {
    if (state.loadingEntities) return;
    state.loadingEntities = true;
    setEntityStatus("Buscando entidades...");

    try {
      const data = await fetchJson(
        apiUrl("/ui/graph/entities", {
          search: state.entitySearch,
          entity_label: state.entityType,
          limit: 40,
        }),
      );
      state.entityCatalog = data.entities || [];
      if (els.entityResultCount) {
        els.entityResultCount.textContent = `${data.summary?.result_count || 0} resultados`;
      }
      setEntityStatus("Selecciona una entidad para visualizarla en el grafo.");
      renderEntityList();
    } catch (error) {
      state.entityCatalog = [];
      if (els.entityResultCount)
        els.entityResultCount.textContent = "0 resultados";
      setEntityStatus(
        `No se pudieron cargar entidades: ${error?.message || error}`,
        true,
      );
      renderEntityList();
    } finally {
      state.loadingEntities = false;
    }
  }

  function groupAliasPairs(aliasData) {
    const groups = new Map();
    (aliasData?.pairs || []).forEach((pair) => {
      if (!groups.has(pair.source_id)) {
        groups.set(pair.source_id, {
          id: pair.source_id,
          name: pair.source_name,
          pairs: [],
        });
      }
      groups.get(pair.source_id).pairs.push(pair);
    });
    return Array.from(groups.values()).sort(
      (a, b) => b.pairs.length - a.pairs.length || a.name.localeCompare(b.name),
    );
  }

  function renderAliasList() {
    if (!els.aliasList) return;
    const aliasData = state.aliasData;
    if (
      !aliasData ||
      !Array.isArray(aliasData.pairs) ||
      aliasData.pairs.length === 0
    ) {
      els.aliasList.innerHTML =
        '<div class="alias-placeholder">No hay posibles alias para ese filtro.</div>';
      if (els.aliasEntityCount)
        els.aliasEntityCount.textContent = "0 entidades";
      return;
    }

    const groups = groupAliasPairs(aliasData);
    if (els.aliasEntityCount) {
      els.aliasEntityCount.textContent = `${aliasData.summary?.entity_count || 0} entidades`;
    }

    els.aliasList.innerHTML = groups
      .map(
        (group) => `
          <article class="alias-item">
            <div class="alias-item-head">
              <div>
                <div class="alias-item-title">${escapeHtml(group.name)}</div>
                <div class="alias-item-subtitle">${escapeHtml(group.id)} · ${group.pairs.length} conexiones posibles</div>
              </div>
              <span class="metric-pill">${group.pairs.length}</span>
            </div>
            <div class="alias-pair-list">
              ${group.pairs
                .map(
                  (pair) =>
                    `<button class="alias-pair-pill" type="button" data-node-id="${escapeHtml(pair.target_id)}">${escapeHtml(pair.target_name)}</button>`,
                )
                .join("")}
            </div>
          </article>
        `,
      )
      .join("");

    els.aliasList.querySelectorAll("[data-node-id]").forEach((button) => {
      button.addEventListener("click", () => {
        const entityId = button.getAttribute("data-node-id") || "";
        selectEntity(entityId);
      });
    });
  }

  async function loadAliases() {
    if (state.loadingAliases) return;
    state.loadingAliases = true;
    setAliasStatus("Buscando posibles alias...");

    try {
      state.aliasData = await fetchJson(
        apiUrl("/ui/graph/aliases", { search: state.aliasSearch }),
      );
      state.aliasLoadedOnce = true;
      setAliasStatus(
        `${state.aliasData.summary?.pair_count || 0} relaciones POSIBLE_ALIAS encontradas.`,
      );
      renderAliasList();
    } catch (error) {
      state.aliasData = {
        pairs: [],
        entities: [],
        summary: { pair_count: 0, entity_count: 0 },
      };
      setAliasStatus(
        `No se pudieron cargar los posibles alias: ${error?.message || error}`,
        true,
      );
      renderAliasList();
    } finally {
      state.loadingAliases = false;
    }
  }

  async function loadNeighborhood(entityId) {
    if (!entityId || state.loadingGraph) return;
    state.loadingGraph = true;
    if (els.refreshBtn) els.refreshBtn.disabled = true;
    toggleEmptyState(true, "Cargando vecindad de la entidad...");

    try {
      const data = await fetchJson(
        apiUrl("/ui/graph/neighborhood", {
          entity_id: entityId,
          alias_only: state.aliasOnly,
          relationship_limit: 420,
        }),
      );
      state.snapshot = data;
      state.selectedNodeId = entityId;
      // reset pan/zoom to center for new entity
      state.zoomScale = 1;
      state.panX = 0;
      state.panY = 0;
      updateMetrics(data.summary);
      setGraphStatus(
        state.aliasOnly
          ? "Filtro activo: solo se muestran relaciones POSIBLE_ALIAS de la entidad seleccionada."
          : "Vecindad de entidad cargada correctamente.",
      );
      renderEntityList();
      renderGraph();
    } catch (error) {
      state.snapshot = {
        nodes: [],
        edges: [],
        summary: { node_count: 0, edge_count: 0, alias_edge_count: 0 },
      };
      updateMetrics(state.snapshot.summary);
      setGraphStatus(
        `No se pudo cargar la vecindad: ${error?.message || error}`,
        true,
      );
      renderGraph();
    } finally {
      state.loadingGraph = false;
      if (els.refreshBtn) els.refreshBtn.disabled = false;
    }
  }

  async function selectEntity(entityId) {
    state.selectedEntityId = entityId;
    await loadNeighborhood(entityId);
  }

  function bindEvents() {
    const debouncedEntitySearch = debounce(() => {
      state.entitySearch = (els.entitySearchInput?.value || "").trim();
      loadEntityCatalog();
    }, 280);

    const debouncedAliasSearch = debounce(() => {
      state.aliasSearch = (els.aliasSearchInput?.value || "").trim();
      if (state.activeSidePanel !== "aliases") {
        setActiveSidePanel("aliases");
      }
      loadAliases();
    }, 280);

    els.entitySearchInput?.addEventListener("input", debouncedEntitySearch);
    els.entityTypeSelect?.addEventListener("change", () => {
      state.entityType = els.entityTypeSelect?.value || "";
      loadEntityCatalog();
    });

    els.aliasSearchInput?.addEventListener("input", debouncedAliasSearch);
    els.tabEntitiesBtn?.addEventListener("click", () =>
      setActiveSidePanel("entities"),
    );
    els.tabAliasesBtn?.addEventListener("click", () =>
      setActiveSidePanel("aliases"),
    );

    els.refreshBtn?.addEventListener("click", async () => {
      const refreshTasks = [loadEntityCatalog()];
      if (state.aliasLoadedOnce || state.activeSidePanel === "aliases") {
        refreshTasks.push(loadAliases());
      }
      await Promise.all(refreshTasks);
      if (state.selectedEntityId) {
        await loadNeighborhood(state.selectedEntityId);
      }
    });

    bindPanAndZoom();

    window.addEventListener("resize", () => {
      if (state.snapshot?.nodes?.length) renderGraph();
    });
  }

  bindEvents();
  setActiveSidePanel("entities");
  renderGraph();
  loadEntityCatalog();
})();
