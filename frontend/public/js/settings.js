window.APP_CONFIG = {
  API_BASE: "http://localhost:8000",
  SETTINGS_KEY: "rag_graphrag_ui_v1",
};

(() => {
  const STORAGE_KEY = window.APP_CONFIG.SETTINGS_KEY;

  const DEFAULTS = {
    graphrag_enabled: true,
    theme: "light",
    enter_to_send: true,
  };

  const els = {
    settingsBtn: null,
    modal: null,
    backdrop: null,
    closeModalBtn: null,
    saveBtn: null,
    resetBtn: null,
    themeLightBtn: null,
    themeDarkBtn: null,
    enterToSend: null,
  };

  let settings = loadSettings();
  let draft = { ...settings };
  let isBound = false;

  function loadSettings() {
    try {
      return {
        ...DEFAULTS,
        ...JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"),
      };
    } catch {
      return { ...DEFAULTS };
    }
  }

  function saveSettings(nextSettings) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(nextSettings));
  }

  function applyTheme(theme) {
    document.body.classList.toggle("dark", theme === "dark");
  }

  function emitChanged(nextSettings) {
    window.dispatchEvent(
      new CustomEvent("settings:changed", { detail: nextSettings }),
    );
  }

  function refreshEls() {
    els.settingsBtn = document.getElementById("settingsBtn");
    els.modal = document.getElementById("settingsModal");
    els.backdrop = document.getElementById("modalBackdrop");
    els.closeModalBtn = document.getElementById("closeModalBtn");
    els.saveBtn = document.getElementById("saveBtn");
    els.resetBtn = document.getElementById("resetBtn");
    els.themeLightBtn = document.getElementById("themeLightBtn");
    els.themeDarkBtn = document.getElementById("themeDarkBtn");
    els.enterToSend = document.getElementById("enterToSend");
  }

  function setThemeButtonsActive() {
    els.themeLightBtn?.classList.toggle("active", draft.theme === "light");
    els.themeDarkBtn?.classList.toggle("active", draft.theme === "dark");
  }

  function renderDraftToModal() {
    if (els.enterToSend) els.enterToSend.checked = draft.enter_to_send;

    applyTheme(draft.theme);
    setThemeButtonsActive();
  }

  function forceCloseModal() {
    document.body.classList.remove("modal-open");
    els.backdrop?.classList.add("hidden");
    els.modal?.classList.add("hidden");
  }

  function openModal() {
    if (!els.modal || !els.backdrop) return;

    draft = { ...settings };
    renderDraftToModal();
    els.backdrop.classList.remove("hidden");
    els.modal.classList.remove("hidden");
    document.body.classList.add("modal-open");
  }

  function closeModal({ revert = false } = {}) {
    if (!els.modal || !els.backdrop) return;

    if (revert) {
      applyTheme(settings.theme);
    }

    els.backdrop.classList.add("hidden");
    els.modal.classList.add("hidden");
    document.body.classList.remove("modal-open");
  }

  function commitDraft() {
    settings = { ...draft };
    saveSettings(settings);
    applyTheme(settings.theme);
    emitChanged(settings);
    closeModal();
  }

  function resetToDefaults() {
    draft = { ...DEFAULTS };
    renderDraftToModal();
  }

  function bindEvents() {
    els.settingsBtn?.addEventListener("click", openModal);
    els.closeModalBtn?.addEventListener("click", () =>
      closeModal({ revert: true }),
    );
    els.backdrop?.addEventListener("click", () => closeModal({ revert: true }));

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !els.modal?.classList.contains("hidden")) {
        closeModal({ revert: true });
      }
    });
    
    els.enterToSend?.addEventListener("change", (event) => {
      draft.enter_to_send = event.target.checked;
    });

    els.themeLightBtn?.addEventListener("click", () => {
      draft.theme = "light";
      applyTheme(draft.theme);
      setThemeButtonsActive();
    });

    els.themeDarkBtn?.addEventListener("click", () => {
      draft.theme = "dark";
      applyTheme(draft.theme);
      setThemeButtonsActive();
    });

    els.saveBtn?.addEventListener("click", commitDraft);
    els.resetBtn?.addEventListener("click", resetToDefaults);
  }

  function mount() {
    refreshEls();
    forceCloseModal();
    applyTheme(settings.theme);

    if (isBound || !els.settingsBtn || !els.modal || !els.backdrop) return;

    bindEvents();
    isBound = true;
  }

  window.SettingsUI = {
    mount,
    getSettings: () => ({ ...settings }),
    getStorageKey: () => STORAGE_KEY,
  };

  applyTheme(settings.theme);
})();
