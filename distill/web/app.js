/* distill web ui */

const $ = (s) => document.querySelector(s);
const form = $("#form");
const logEl = $("#log");
const progressEl = $("#progress");
const coursesEl = $("#courses");
const goBtn = $("#go");

let genStart = null;
let flowNodes = {};   // module_index → DOM node
let flowEdges = {};   // "fromIdx-toIdx" → SVG path element
let flowEdgeMeta = []; // [{pathEl, fromIdx, toIdx}] — null fromIdx = root dot
let flowActive = false;
let generating = false;

/* ── GitHub star count ───────────────────────────── */
fetch("https://api.github.com/repos/TAOGenna/Distill")
  .then(function (r) { return r.json(); })
  .then(function (data) {
    var el = $("#star-count");
    if (el && typeof data.stargazers_count === "number") {
      el.textContent = data.stargazers_count;
    }
  })
  .catch(function () {});
let totalModules = 0;
let completedModuleSet = new Set();
let activeModuleSet = new Set();
let elapsedTimer = null;
let liveCostUsd = 0;
let liveTokensIn = 0;
let liveTokensOut = 0;

/* ── Theme toggle ────────────────────────────────── */

function getEffectiveTheme() {
  var stored = localStorage.getItem("distill_theme");
  if (stored === "dark" || stored === "light") return stored;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  ISO_PALETTES = theme === "dark" ? ISO_PALETTES_DARK : ISO_PALETTES_LIGHT;
}

// Apply saved theme immediately
(function () {
  var stored = localStorage.getItem("distill_theme");
  if (stored) document.documentElement.setAttribute("data-theme", stored);
})();

if ($("#theme-toggle")) {
  $("#theme-toggle").addEventListener("click", function () {
    var current = getEffectiveTheme();
    var next = current === "dark" ? "light" : "dark";
    localStorage.setItem("distill_theme", next);
    applyTheme(next);
  });
}

/* ── Isometric icon builder ───────────────────────── */

var ISO_PALETTES_LIGHT = {
  generated: {
    bottom: { left: "#2e2e2e", right: "#444444", top: "#5a5a5a" },
    top:    { left: "#a5a5a5", right: "#666666", top: "#888888" },
  },
  pending: {
    bottom: { left: "#bebebe", right: "#bebebe", top: "#f0f0f0" },
    top:    { left: "#bebebe", right: "#bebebe", top: "#f0f0f0" },
  },
};

var ISO_PALETTES_DARK = {
  generated: {
    bottom: { left: "#888888", right: "#aaaaaa", top: "#cccccc" },
    top:    { left: "#cccccc", right: "#999999", top: "#e0e0e0" },
  },
  pending: {
    bottom: { left: "#444444", right: "#444444", top: "#555555" },
    top:    { left: "#444444", right: "#444444", top: "#555555" },
  },
};

function isDarkMode() {
  return getEffectiveTheme() === "dark";
}

var ISO_PALETTES = isDarkMode() ? ISO_PALETTES_DARK : ISO_PALETTES_LIGHT;

// Update palette when OS theme changes (only if no manual override)
if (window.matchMedia) {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    if (!localStorage.getItem("distill_theme")) {
      ISO_PALETTES = getEffectiveTheme() === "dark" ? ISO_PALETTES_DARK : ISO_PALETTES_LIGHT;
    }
  });
}

function isoIcon(size, palette) {
  var S = 13 * (size / 60);
  var cx = size / 2;
  var cy = size * 0.55;

  function pr(x, y, z) {
    return {
      x: cx + (x - y) * S * 0.866,
      y: cy + (x + y) * S * 0.5 - z * S,
    };
  }

  function poly(a, b, c, d, fill) {
    var p = [a, b, c, d]
      .map(function (v) { return v.x.toFixed(1) + "," + v.y.toFixed(1); })
      .join(" ");
    return '<polygon points="' + p + '" fill="' + fill + '"/>';
  }

  function blk(w, d, h, z0, col) {
    var svg = "";
    svg += poly(pr(-w, d, z0 + h), pr(w, d, z0 + h), pr(w, d, z0), pr(-w, d, z0), col.left);
    svg += poly(pr(w, -d, z0 + h), pr(w, -d, z0), pr(w, d, z0), pr(w, d, z0 + h), col.right);
    svg += poly(pr(-w, -d, z0 + h), pr(w, -d, z0 + h), pr(w, d, z0 + h), pr(-w, d, z0 + h), col.top);
    return svg;
  }

  var svg =
    '<svg viewBox="0 0 ' + size + " " + size +
    '" width="' + size + '" height="' + size +
    '" xmlns="http://www.w3.org/2000/svg">';
  svg += blk(1.1, 1.1, 0.35, 0, palette.bottom);
  svg += blk(0.75, 0.75, 0.35, 0.41, palette.top);
  svg += "</svg>";
  return svg;
}

/* ── Config + Setup ──────────────────────────────── */

let providerDefaults = {};
let apiKeySet = false;

function populateModelDropdowns(provider) {
  var defaults = providerDefaults[provider] || { design: "", generate: "" };
  var designSelect = $("#design-model");
  var generateSelect = $("#generate-model");

  designSelect.innerHTML = '<option value="' + esc(defaults.design) + '">' + esc(defaults.design) + '</option>';
  generateSelect.innerHTML = '<option value="' + esc(defaults.generate) + '">' + esc(defaults.generate) + '</option>';

  if (defaults.design !== defaults.generate) {
    designSelect.innerHTML += '<option value="' + esc(defaults.generate) + '">' + esc(defaults.generate) + '</option>';
    generateSelect.innerHTML += '<option value="' + esc(defaults.design) + '">' + esc(defaults.design) + '</option>';
  }
}

function updateSetupKeyVisibility(provider) {
  var row = $("#setup-key-row");
  if (row) row.style.display = (provider === "ollama" || provider === "claude_code" || provider === "mock") ? "none" : "";
  var effortRow = $("#effort-row");
  if (effortRow) effortRow.style.display = provider === "claude_code" ? "" : "none";
  var diagramsRow = $("#diagrams-row");
  if (diagramsRow) diagramsRow.style.display = provider === "claude_code" ? "" : "none";
}

function setDiagramsStatus(text, cls) {
  var el = $("#diagrams-status");
  if (!el) return;
  el.textContent = text;
  el.className = "diagrams-status" + (cls ? " " + cls : "");
}

function updateFormState() {
  if (generating) {
    goBtn.disabled = true;
    return;
  }
  var provider = $("#setup-provider") ? $("#setup-provider").value : "anthropic";
  var hasAuth = apiKeySet || provider === "ollama" || provider === "mock" || provider === "claude_code";
  var hasUrl = $("#url-input").value.trim().length > 0;
  var hasLevel = $("#level-input").value.trim().length > 0;

  goBtn.disabled = !(hasAuth && hasUrl && hasLevel);
}

function markConfigured() {
  apiKeySet = true;
  // Auto-collapse settings when configured
  var panel = $("#setup-panel");
  if (panel) panel.classList.add("collapsed");
  updateFormState();
}

/* ── Settings toggle ────────────────────────────── */

if ($("#setup-toggle")) {
  $("#setup-toggle").addEventListener("click", function () {
    var panel = $("#setup-panel");
    panel.classList.toggle("collapsed");
  });
}

/* ── Log toggle ─────────────────────────────────── */

if ($("#log-toggle")) {
  $("#log-toggle").addEventListener("click", function () {
    var section = $("#log-section");
    section.classList.toggle("collapsed");
    var log = $("#log");
    if (section.classList.contains("collapsed")) {
      log.classList.add("log-collapsed");
    } else {
      log.classList.remove("log-collapsed");
      log.scrollTop = log.scrollHeight;
    }
  });
}

// Load config
fetch("/api/config")
  .then((r) => r.json())
  .then((cfg) => {
    apiKeySet = cfg.api_key_set;

    if (cfg.output_dir) $("#cfg-output").value = cfg.output_dir;
    if (cfg.provider) {
      if ($("#setup-provider")) $("#setup-provider").value = cfg.provider;
      updateSetupKeyVisibility(cfg.provider);
    }
    if (cfg.api_key_masked && $("#setup-key")) {
      $("#setup-key").placeholder = cfg.api_key_masked;
    }
    if (cfg.effort && $("#cfg-effort")) {
      $("#cfg-effort").value = cfg.effort;
    }
    if (cfg.max_revision_cycles !== undefined) {
      $("#cfg-revision-cycles").value = cfg.max_revision_cycles;
    }
    if (cfg.provider_defaults) {
      providerDefaults = cfg.provider_defaults;
      populateModelDropdowns(cfg.provider || "anthropic");
    }
    if (cfg.design_model) {
      var ds = $("#design-model");
      if (!ds.querySelector('option[value="' + cfg.design_model + '"]')) {
        ds.innerHTML += '<option value="' + esc(cfg.design_model) + '">' + esc(cfg.design_model) + '</option>';
      }
      ds.value = cfg.design_model;
    }
    if (cfg.generate_model) {
      var gs = $("#generate-model");
      if (!gs.querySelector('option[value="' + cfg.generate_model + '"]')) {
        gs.innerHTML += '<option value="' + esc(cfg.generate_model) + '">' + esc(cfg.generate_model) + '</option>';
      }
      gs.value = cfg.generate_model;
    }

    // Diagrams mode
    var diagramsSel = $("#cfg-diagrams");
    if (diagramsSel) {
      var mode = cfg.diagram_mode || "ascii";
      if (mode === "excalidraw" && cfg.diagrams_available) {
        diagramsSel.value = "excalidraw";
        setDiagramsStatus("ready", "ok");
      } else {
        diagramsSel.value = "ascii";
        if (mode === "excalidraw" && !cfg.diagrams_available) {
          setDiagramsStatus("excalidraw not built", "error");
        }
      }
    }

    // If key is set, hide setup panel and enable form
    if (apiKeySet) {
      markConfigured();
    } else {
      updateFormState();
    }
  })
  .catch(() => { updateFormState(); });

// Setup panel save — shared logic for button click and blur
async function saveApiKey() {
  var provider = $("#setup-provider").value;
  var key = $("#setup-key") ? $("#setup-key").value.trim() : "";
  if (!key && provider !== "ollama") return;

  if (key) {
    await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key, provider: provider }),
    });
    $("#setup-key").value = "";
    $("#setup-key").placeholder = "****" + key.slice(-4);
  }
  markConfigured();
}

if ($("#setup-save")) {
  $("#setup-save").addEventListener("click", saveApiKey);
}

// Auto-save API key when user clicks away from the field
if ($("#setup-key")) {
  $("#setup-key").addEventListener("blur", saveApiKey);
}

if ($("#setup-provider")) {
  $("#setup-provider").addEventListener("change", async () => {
    var provider = $("#setup-provider").value;
    updateSetupKeyVisibility(provider);
    populateModelDropdowns(provider);
    // Persist provider to server immediately
    await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: provider }),
    });
    // Re-check if this provider has a key saved
    var cfg = await fetch("/api/config").then((r) => r.json());
    apiKeySet = cfg.api_key_set;
    if (cfg.api_key_masked && $("#setup-key")) {
      $("#setup-key").placeholder = cfg.api_key_masked;
    } else if ($("#setup-key")) {
      $("#setup-key").value = "";
      $("#setup-key").placeholder = "sk-...";
    }
    updateFormState();
  });
}

/* ── Form-dependent listeners (guarded for test pages) ── */

function _bind(sel, evt, fn) {
  var el = $(sel);
  if (el) el.addEventListener(evt, fn);
}

_bind("#browse-output", "click", async () => {
  var btn = $("#browse-output");
  btn.disabled = true;
  try {
    var res = await fetch("/api/browse-folder", { method: "POST" });
    var data = await res.json();
    if (data.path) {
      $("#cfg-output").value = data.path;
      await fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ output_dir: data.path }),
      });
    }
  } catch (e) {}
  btn.disabled = false;
});

_bind("#save-output", "click", async () => {
  const dir = $("#cfg-output").value.trim();
  if (!dir) return;
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_dir: dir }),
  });
});

_bind("#cfg-output", "blur", async () => {
  const dir = $("#cfg-output").value.trim();
  if (!dir) return;
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_dir: dir }),
  });
});

_bind("#design-model", "change", async () => {
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ design_model: $("#design-model").value }),
  });
});

_bind("#generate-model", "change", async () => {
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generate_model: $("#generate-model").value }),
  });
});

_bind("#cfg-revision-cycles", "change", async () => {
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ max_revision_cycles: parseInt($("#cfg-revision-cycles").value) }),
  });
});

_bind("#cfg-effort", "change", async () => {
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ effort: $("#cfg-effort").value }),
  });
});

_bind("#cfg-diagrams", "change", async () => {
  var sel = $("#cfg-diagrams");
  var mode = sel.value;

  if (mode === "excalidraw") {
    sel.disabled = true;
    setDiagramsStatus("setting up\u2026", "loading");
    try {
      var res = await fetch("/api/diagrams/setup", { method: "POST" });
      var data = await res.json();
      if (data.ok) {
        setDiagramsStatus("ready", "ok");
        await fetch("/api/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ diagram_mode: "excalidraw" }),
        });
      } else {
        sel.value = "ascii";
        setDiagramsStatus(data.message || "setup failed", "error");
      }
    } catch (e) {
      sel.value = "ascii";
      setDiagramsStatus("setup failed", "error");
    }
    sel.disabled = false;
  } else {
    setDiagramsStatus("", "");
    await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ diagram_mode: "ascii" }),
    });
  }
});

/* ── Auto-grow textarea + live form validation ────── */

_bind("#url-input", "input", updateFormState);
_bind("#level-input", "input", () => {
  var el = $("#level-input");
  el.style.height = "auto";
  el.style.height = el.scrollHeight + "px";
  updateFormState();
});

document.querySelectorAll("textarea:not(#level-input)").forEach((el) => {
  el.addEventListener("input", () => {
    el.style.height = "auto";
    el.style.height = el.scrollHeight + "px";
  });
});

/* ── Refs ─────────────────────────────────────────── */

_bind("#add-ref", "click", () => {
  const row = document.createElement("div");
  row.className = "ref-row";
  row.innerHTML =
    '<input type="url" name="ref" placeholder="https://...">' +
    '<input type="text" name="ref_annotation" class="ref-annotation" placeholder="role — e.g. peer reviews, reference impl...">' +
    '<button type="button" class="ref-remove" aria-label="remove reference">&times;</button>';
  row.querySelector(".ref-remove").addEventListener("click", () => row.remove());
  $("#refs").appendChild(row);
});

/* ── Courses ──────────────────────────────────────── */

function loadCourses() {
  fetch("/api/courses")
    .then((r) => r.json())
    .then((data) => {
      var section = $("#courses-section");
      if (!data.courses || !data.courses.length) {
        section.classList.add("empty");
        return;
      }
      section.classList.remove("empty");
      coursesEl.innerHTML = data.courses
        .map(
          (c) =>
            '<a class="course-item" href="/reader.html?course=' + encodeURIComponent(c.name) + '">' +
            '<div class="course-name">' +
            esc(c.title || c.name) +
            "</div>" +
            '<div class="course-meta">' +
            (c.modules ? c.modules + " modules" : "") +
            (c.file_count ? " &middot; " + c.file_count + " files" : "") +
            "</div>" +
            '<div class="course-path">' +
            esc(c.path) +
            "</div>" +
            "</a>"
        )
        .join("");
    })
    .catch(() => {});
}
loadCourses();

/* ── Progress bar ────────────────────────────────── */

var phaseWeights = {
  preprocess: { start: 0, end: 10 },
  analyze: { start: 10, end: 25 },
  design: { start: 25, end: 40 },
  generate: { start: 40, end: 85 },
  review: { start: 85, end: 100 },
  done: { start: 100, end: 100 },
};

var phaseLabels = {
  preprocess: "preprocessing",
  analyze: "analyzing",
  design: "designing curriculum",
  generate: "generating modules",
  review: "reviewing",
  done: "complete",
};

function updateProgressBar(phase, detail) {
  var w = phaseWeights[phase];
  if (!w) return;

  var fill = $("#phase-fill");
  var label = $("#phase-label");
  var det = $("#phase-detail");

  fill.style.width = w.start + "%";
  label.textContent = phaseLabels[phase] || phase;
  det.textContent = detail || "";

  // Stripe animation when actively working
  if (phase === "done") {
    fill.classList.remove("active");
    stopElapsedTimer();
  } else {
    fill.classList.add("active");
  }
}

function updateModuleProgress() {
  var fill = $("#phase-fill");
  var det = $("#phase-detail");
  var completed = completedModuleSet.size;
  var w = phaseWeights.generate;
  var pct = w.start + ((w.end - w.start) * Math.min(completed, totalModules) / Math.max(totalModules, 1));
  fill.style.width = pct + "%";
  det.textContent = Math.min(completed, totalModules) + "/" + totalModules;
}

function startElapsedTimer() {
  stopElapsedTimer();
  var el = $("#phase-elapsed");
  if (!el) return;
  function tick() {
    if (!genStart) return;
    var secs = Math.floor((Date.now() - genStart) / 1000);
    var mm = String(Math.floor(secs / 60)).padStart(2, "0");
    var ss = String(secs % 60).padStart(2, "0");
    el.textContent = mm + ":" + ss;
  }
  tick();
  elapsedTimer = setInterval(tick, 1000);
}

function stopElapsedTimer() {
  if (elapsedTimer) {
    clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
}

function updateLiveCost(costData) {
  if (!costData) return;
  if (costData.total_cost_usd != null) liveCostUsd = costData.total_cost_usd;
  if (costData.input_tokens != null) liveTokensIn = costData.input_tokens;
  if (costData.output_tokens != null) liveTokensOut = costData.output_tokens;
  var el = $("#phase-cost");
  if (el && liveCostUsd > 0) {
    el.textContent = "$" + liveCostUsd.toFixed(4);
  }
}

/* ── beforeunload guard ──────────────────────────── */

window.addEventListener("beforeunload", function (e) {
  if (generating) {
    e.preventDefault();
    e.returnValue = "";
  }
});

/* ── Generate ─────────────────────────────────────── */

if (form) form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const fd = new FormData(form);
  const params = {
    url: fd.get("url"),
    level: fd.get("level"),
    design_model: fd.get("design_model"),
    generate_model: fd.get("generate_model"),
    refs: [],
    ref_annotations: [],
  };
  // Collect refs + annotations together so filtering keeps them aligned
  document.querySelectorAll('.ref-row').forEach((row) => {
    const url = (row.querySelector('input[name="ref"]') || {}).value || "";
    const ann = (row.querySelector('input[name="ref_annotation"]') || {}).value || "";
    if (url.trim()) {
      params.refs.push(url.trim());
      params.ref_annotations.push(ann.trim());
    }
  });

  generating = true;
  goBtn.disabled = true;
  goBtn.textContent = "generating...";
  goBtn.classList.add("generating");
  logEl.innerHTML = "";
  progressEl.classList.add("active");
  // Expand log by default at start of generation
  var logSection = $("#log-section");
  if (logSection) logSection.classList.remove("collapsed");
  var logBox = $("#log");
  if (logBox) logBox.classList.remove("log-collapsed");
  genStart = Date.now();
  flowNodes = {};
  flowEdges = {};
  flowEdgeMeta = [];
  flowActive = false;
  totalModules = 0;
  completedModuleSet = new Set();
  activeModuleSet = new Set();
  liveCostUsd = 0;
  liveTokensIn = 0;
  liveTokensOut = 0;
  if ($("#phase-elapsed")) $("#phase-elapsed").textContent = "";
  if ($("#phase-cost")) $("#phase-cost").textContent = "";
  startElapsedTimer();
  updateProgressBar("preprocess", "");

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      appendLog(err.error || "failed to start generation", "error");
      resetBtn();
      return;
    }

    const { job_id } = await res.json();
    connectSSE(job_id);
  } catch (err) {
    appendLog("connection failed: " + err.message, "error");
    resetBtn();
  }
});

function connectSSE(jobId) {
  const src = new EventSource("/api/events/" + jobId);

  src.onmessage = (e) => {
    const ev = JSON.parse(e.data);

    if (ev.type === "heartbeat") {
      return; // keep-alive, ignore
    } else if (ev.type === "phase") {
      updateProgressBar(ev.phase, "");
      appendPhase(ev.phase);
      // When entering generate phase, collapse log to foreground the DAG
      if (ev.phase === "generate") {
        var ls = $("#log-section");
        if (ls) ls.classList.add("collapsed");
        var lb = $("#log");
        if (lb) lb.classList.add("log-collapsed");
      }
    } else if (ev.type === "log") {
      appendLog(ev.message, ev.level || "");
      // Update live cost if present
      if (ev.cost) updateLiveCost(ev.cost);
    } else if (ev.type === "curriculum") {
      totalModules = ev.data.modules ? ev.data.modules.length : 0;
      renderCurriculumFlow(ev.data);
    } else if (ev.type === "module_start") {
      // Mark module as actively generating
      activeModuleSet.add(ev.module_index);
      markFlowNodeActive(ev.module_index);
      appendLog(
        "module " + ev.module_index + " (" + (ev.title || "") + ") started",
        "dim"
      );
    } else if (ev.type === "module_complete") {
      activeModuleSet.delete(ev.module_index);
      completedModuleSet.add(ev.module_index);
      updateModuleProgress();
      activateFlowNode(ev.module_index);
      updateEdgeStates();
      appendLog(
        "module " + ev.module_index + " (" + ev.title + ") generated",
        "ok"
      );
    } else if (ev.type === "cost") {
      updateLiveCost(ev);
    } else if (ev.type === "complete") {
      updateProgressBar("done", "");
      showResult(ev.result);
      src.close();
      resetBtn();
      loadCourses();
    } else if (ev.type === "error") {
      appendLog(ev.message, "error");
      src.close();
      resetBtn();
    }

    logEl.scrollTop = logEl.scrollHeight;
  };

  src.onerror = () => {
    src.close();
    if (generating) {
      appendLog("connection lost — reconnecting in 3s...", "warn");
      setTimeout(function () { connectSSE(jobId); }, 3000);
    } else {
      resetBtn();
    }
  };
}

/* ── Reconnect to active job on page load ──────────── */

fetch("/api/jobs/active")
  .then(function (r) { return r.json(); })
  .then(function (data) {
    if (data.job_id) {
      generating = true;
      goBtn.disabled = true;
      goBtn.classList.add("generating");
      goBtn.textContent = "generating...";
      progressEl.classList.add("active");
      appendLog("reconnected to running generation", "ok");
      startElapsedTimer();
      connectSSE(data.job_id);
    }
  })
  .catch(function () {});

/* ── Log rendering ────────────────────────────────── */

function appendPhase(phase) {
  const el = document.createElement("div");
  el.className = "phase-label";
  el.textContent = "\u2500\u2500 " + (phaseLabels[phase] || phase) + " \u2500\u2500";
  logEl.appendChild(el);
}

function appendLog(message, level) {
  const el = document.createElement("div");
  el.className = "log-entry" + (level ? " " + level : "");
  el.title = message;

  const secs = genStart ? Math.floor((Date.now() - genStart) / 1000) : 0;
  const mm = String(Math.floor(secs / 60)).padStart(2, "0");
  const ss = String(secs % 60).padStart(2, "0");

  el.innerHTML =
    '<span class="ts">[' + mm + ":" + ss + "]</span> " + esc(message);
  logEl.appendChild(el);
}

function showResult(result) {
  genStart = null;

  if (flowActive) {
    finalizeFlow(result);
  }

  // Build cost summary
  var costParts = [];
  if (result.total_cost_usd != null) {
    costParts.push("$" + result.total_cost_usd.toFixed(4));
  }
  if (result.usage) {
    var u = result.usage;
    if (u.input_tokens || u.output_tokens) {
      costParts.push(
        (u.input_tokens || 0).toLocaleString() + " in / " +
        (u.output_tokens || 0).toLocaleString() + " out"
      );
    }
    if (u.api_calls) {
      costParts.push(u.api_calls + " calls");
    }
  }

  // Remove previous banner if any
  var oldBanner = progressEl.querySelector(".result-banner");
  if (oldBanner) oldBanner.remove();

  // Show success banner
  var banner = document.createElement("div");
  banner.className = "result-banner";
  banner.innerHTML =
    '<div class="result-banner-title">course generated</div>' +
    (result.course_dir
      ? '<div class="result-banner-path">' + esc(result.course_dir) + '</div>'
      : '') +
    (costParts.length
      ? '<div class="result-banner-cost">' + esc(costParts.join(" \u00b7 ")) + '</div>'
      : '');
  progressEl.appendChild(banner);
  banner.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ── Course flow visualization (Brilliant-style) ── */

var ICON_SIZE = 76;

function renderCurriculumFlow(data) {
  // Remove previous DAG if any
  var old = document.getElementById("course-result");
  if (old) old.remove();

  flowNodes = {};
  flowEdges = {};
  flowEdgeMeta = [];
  flowActive = true;

  var modules = data.modules;
  var totalEx = modules.reduce(function (s, m) {
    return s + m.exercise_count;
  }, 0);

  var el = document.createElement("div");
  el.className = "course-result";
  el.id = "course-result";
  el.innerHTML =
    '<div class="course-result-header">' +
    '<div class="course-result-title">' + esc(data.title) + "</div>" +
    '<div class="course-result-meta">' +
    modules.length + " modules \u00b7 " + totalEx + " exercises</div>" +
    '<div class="course-result-dir" id="flow-dir"></div>' +
    "</div>" +
    '<div class="course-flow"></div>';

  progressEl.appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "start" });

  var flow = el.querySelector(".course-flow");

  requestAnimationFrame(function () {
    var W = flow.offsetWidth;
    if (!W) return;

    var byIdx = {};
    modules.forEach(function (m) { byIdx[m.index] = m; });

    var depth = {};
    function getDepth(idx) {
      if (depth[idx] !== undefined) return depth[idx];
      var m = byIdx[idx];
      if (!m || !m.depends_on || m.depends_on.length === 0) {
        depth[idx] = 0;
        return 0;
      }
      var maxParent = 0;
      m.depends_on.forEach(function (p) {
        var d = getDepth(p);
        if (d + 1 > maxParent) maxParent = d + 1;
      });
      depth[idx] = maxParent;
      return maxParent;
    }
    modules.forEach(function (m) { getDepth(m.index); });

    var layers = {};
    var maxLayer = 0;
    modules.forEach(function (m) {
      var d = depth[m.index];
      if (!layers[d]) layers[d] = [];
      layers[d].push(m);
      if (d > maxLayer) maxLayer = d;
    });

    var layerSpacing = 140;
    var startY = 24;
    var startX = W * 0.5;
    var totalH = startY + (maxLayer + 2) * layerSpacing;
    flow.style.height = totalH + "px";

    var positions = {};
    for (var layer = 0; layer <= maxLayer; layer++) {
      var group = layers[layer] || [];
      var count = group.length;
      var y = startY + (layer + 1) * layerSpacing;
      group.forEach(function (m, i) {
        var x;
        if (count === 1) {
          x = layer % 2 === 0 ? W * 0.35 : W * 0.65;
        } else {
          var margin = W * 0.15;
          var usable = W - 2 * margin;
          x = margin + (usable * (i + 0.5)) / count;
        }
        positions[m.index] = { x: x, y: y };
      });
    }

    var NS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", W);
    svg.setAttribute("height", totalH);
    svg.style.cssText =
      "position:absolute;top:0;left:0;pointer-events:none;";

    // Build edge list with module index references for state tracking
    var edgeList = [];
    var edgeSeen = {};
    function addEdge(from, to, fromDot, fromIdx, toIdx, targetLayer) {
      var key = Math.round(from.x) + "," + Math.round(from.y) +
                "-" + Math.round(to.x) + "," + Math.round(to.y);
      if (edgeSeen[key]) return;
      edgeSeen[key] = true;
      edgeList.push({ from: from, to: to, fromDot: fromDot, fromIdx: fromIdx, toIdx: toIdx, layer: targetLayer });
    }
    modules.forEach(function (m) {
      if (!m.depends_on || m.depends_on.length === 0) {
        addEdge({ x: startX, y: startY }, positions[m.index], true, null, m.index, depth[m.index]);
      }
    });
    modules.forEach(function (m) {
      if (m.depends_on) {
        m.depends_on.forEach(function (parentIdx) {
          if (positions[parentIdx]) {
            addEdge(positions[parentIdx], positions[m.index], false, parentIdx, m.index, depth[m.index]);
          }
        });
      }
    });

    // Draw edges with consistent S-curves
    edgeList.forEach(function (edge) {
      var fy = edge.fromDot ? edge.from.y + 10 : edge.from.y + 15;
      var ty = edge.to.y - (ICON_SIZE / 2 + 8);
      var fx = edge.from.x;
      var tx = edge.to.x;

      // Vertical exit from parent, then curve to child entry
      var ctrlLen = Math.abs(ty - fy) * 0.45;
      var d = "M " + fx + " " + fy +
        " C " + fx + " " + (fy + ctrlLen) +
        ", " + tx + " " + (ty - ctrlLen) +
        ", " + tx + " " + ty;

      var pathEl = document.createElementNS(NS, "path");
      pathEl.setAttribute("d", d);
      pathEl.setAttribute("stroke", isDarkMode() ? "#444" : "#ddd");
      pathEl.setAttribute("stroke-width", "2.5");
      pathEl.setAttribute("fill", "none");
      pathEl.setAttribute("stroke-linecap", "round");
      pathEl.classList.add("flow-edge");
      svg.appendChild(pathEl);

      // Store for state tracking
      var edgeKey = (edge.fromIdx != null ? edge.fromIdx : "root") + "-" + edge.toIdx;
      flowEdges[edgeKey] = pathEl;
      flowEdgeMeta.push({ pathEl: pathEl, fromIdx: edge.fromIdx, toIdx: edge.toIdx, layer: edge.layer });
    });
    flow.appendChild(svg);

    // Sequenced edge animation: draw layer by layer
    var allPaths = svg.querySelectorAll("path");
    allPaths.forEach(function (p) {
      var len = p.getTotalLength();
      p.style.strokeDasharray = len;
      p.style.strokeDashoffset = len;
    });

    // Group edges by target layer for sequenced animation
    var edgesByLayer = {};
    flowEdgeMeta.forEach(function (em) {
      if (!edgesByLayer[em.layer]) edgesByLayer[em.layer] = [];
      edgesByLayer[em.layer].push(em.pathEl);
    });

    var baseDelay = 100;
    var layerAnimDur = 800; // ms per layer
    for (var lyr = 0; lyr <= maxLayer; lyr++) {
      (function (layer) {
        var paths = edgesByLayer[layer] || [];
        var delay = baseDelay + layer * layerAnimDur;
        setTimeout(function () {
          paths.forEach(function (p) {
            p.style.transition = "stroke-dashoffset " + (layerAnimDur / 1000) + "s ease-out";
            p.style.strokeDashoffset = "0";
          });
        }, delay);
      })(lyr);
    }

    // Root dot
    var dot = document.createElement("div");
    dot.className = "flow-node";
    dot.style.left = startX + "px";
    dot.style.top = startY + "px";
    dot.innerHTML = '<div class="flow-dot"></div>';
    flow.appendChild(dot);
    setTimeout(function () { dot.classList.add("visible"); }, 150);

    // Render module nodes, timed to appear after their layer's edges draw
    for (var ly = 0; ly <= maxLayer; ly++) {
      (function (layer) {
        var group = layers[layer] || [];
        var nodeDelay = baseDelay + layer * layerAnimDur + layerAnimDur * 0.6;
        group.forEach(function (m) {
          var pos = positions[m.index];
          var node = document.createElement("div");
          node.className = "flow-node pending";
          node.setAttribute("role", "group");
          node.setAttribute("aria-label", m.title + " \u2014 pending");
          node.style.left = pos.x + "px";
          node.style.top = pos.y + "px";
          var tipHtml = "";
          if (m.description) {
            tipHtml += '<div class="tip-desc">' + esc(m.description) + "</div>";
          }
          if (m.exercises && m.exercises.length) {
            tipHtml += m.exercises.map(function (t) {
              return '<div class="tip-ex">' + esc(t) + "</div>";
            }).join("");
          }

          node.innerHTML =
            '<div class="flow-icon">' + isoIcon(ICON_SIZE, ISO_PALETTES.pending) + "</div>" +
            '<div class="flow-label">' + esc(m.title) + "</div>" +
            '<div class="flow-sub">' + m.exercise_count + " exercises</div>" +
            (tipHtml ? '<div class="flow-tooltip">' + tipHtml + "</div>" : "");
          flow.appendChild(node);
          flowNodes[m.index] = node;

          setTimeout(function () { node.classList.add("visible"); }, nodeDelay);
        });
      })(ly);
    }
  });
}

function markFlowNodeActive(moduleIndex) {
  var node = flowNodes[moduleIndex];
  if (node && node.classList.contains("pending")) {
    node.classList.remove("pending");
    node.classList.add("active");
    node.setAttribute("aria-label", node.getAttribute("aria-label").replace("pending", "generating"));
    var icon = node.querySelector(".flow-icon");
    if (icon) icon.innerHTML = isoIcon(ICON_SIZE, ISO_PALETTES.generated);
  }
}

function activateFlowNode(moduleIndex) {
  var node = flowNodes[moduleIndex];
  if (node) {
    node.classList.remove("pending", "active");
    node.classList.add("generated");
    node.setAttribute("aria-label",
      node.getAttribute("aria-label").replace(/pending|generating/, "generated"));
    var icon = node.querySelector(".flow-icon");
    if (icon) icon.innerHTML = isoIcon(ICON_SIZE, ISO_PALETTES.generated);
    // Add checkmark
    var sub = node.querySelector(".flow-sub");
    if (sub && !node.querySelector(".flow-check")) {
      var check = document.createElement("div");
      check.className = "flow-check";
      check.textContent = "\u2713";
      sub.after(check);
    }
  }
}

function updateEdgeStates() {
  flowEdgeMeta.forEach(function (em) {
    // An edge is "completed" when both its source and target are generated
    var fromDone = em.fromIdx == null || completedModuleSet.has(em.fromIdx); // root is always done
    var toDone = completedModuleSet.has(em.toIdx);
    if (fromDone && toDone) {
      em.pathEl.classList.add("completed");
    }
  });
}

function finalizeFlow(result) {
  var dirEl = document.getElementById("flow-dir");
  if (dirEl && result.course_dir) {
    dirEl.textContent = result.course_dir;
  }
}

/* ── Helpers ──────────────────────────────────────── */

function resetBtn() {
  generating = false;
  goBtn.disabled = false;
  goBtn.textContent = "generate";
  goBtn.classList.remove("generating");
  stopElapsedTimer();
  updateFormState();
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/* ── Form auto-save to localStorage ──────────────── */

var FORM_KEY = "distill_form";

function getFormState() {
  return {
    url: $("#url-input").value,
    level: $("#level-input").value,
    design_model: $("#design-model").value,
    generate_model: $("#generate-model").value,
    refs: [...document.querySelectorAll('input[name="ref"]')]
      .map(function (el) { return el.value; }),
    ref_annotations: [...document.querySelectorAll('input[name="ref_annotation"]')]
      .map(function (el) { return el.value; }),
  };
}

function restoreFormState(state) {
  if (!state) return;
  if (state.url) $("#url-input").value = state.url;
  if (state.level) {
    $("#level-input").value = state.level;
    // Trigger auto-grow
    var el = $("#level-input");
    el.style.height = "auto";
    el.style.height = el.scrollHeight + "px";
  }
  if (state.design_model && $("#design-model")) {
    var ds = $("#design-model");
    if (!ds.querySelector('option[value="' + state.design_model + '"]')) {
      ds.innerHTML += '<option value="' + esc(state.design_model) + '">' + esc(state.design_model) + '</option>';
    }
    ds.value = state.design_model;
  }
  if (state.generate_model && $("#generate-model")) {
    var gs = $("#generate-model");
    if (!gs.querySelector('option[value="' + state.generate_model + '"]')) {
      gs.innerHTML += '<option value="' + esc(state.generate_model) + '">' + esc(state.generate_model) + '</option>';
    }
    gs.value = state.generate_model;
  }
  if (state.refs && state.refs.length) {
    state.refs.forEach(function (url, i) {
      if (!url) return;
      var ann = (state.ref_annotations && state.ref_annotations[i]) || "";
      var row = document.createElement("div");
      row.className = "ref-row";
      row.innerHTML =
        '<input type="url" name="ref" placeholder="https://..." value="' + esc(url) + '">' +
        '<input type="text" name="ref_annotation" class="ref-annotation" placeholder="role — e.g. peer reviews, reference impl..." value="' + esc(ann) + '">' +
        '<button type="button" class="ref-remove" aria-label="remove reference">&times;</button>';
      row.querySelector(".ref-remove").addEventListener("click", function () {
        row.remove();
        saveFormState();
      });
      row.querySelectorAll("input").forEach(function (inp) {
        inp.addEventListener("input", saveFormState);
      });
      $("#refs").appendChild(row);
    });
  }
  updateFormState();
}

function saveFormState() {
  try {
    localStorage.setItem(FORM_KEY, JSON.stringify(getFormState()));
  } catch (e) {}
}

function clearForm() {
  $("#url-input").value = "";
  $("#level-input").value = "";
  $("#level-input").style.height = "auto";
  $("#refs").innerHTML = "";
  // Reset models to defaults
  var provider = $("#setup-provider") ? $("#setup-provider").value : "anthropic";
  populateModelDropdowns(provider);
  // Clear stored state
  try { localStorage.removeItem(FORM_KEY); } catch (e) {}
  $("#preset-select").value = "";
  updateFormState();
}

// Restore on page load (after a small delay so model dropdowns are populated)
setTimeout(function () {
  try {
    var saved = JSON.parse(localStorage.getItem(FORM_KEY));
    if (saved) restoreFormState(saved);
  } catch (e) {}
}, 200);

// Auto-save on all form inputs
_bind("#url-input", "input", saveFormState);
_bind("#level-input", "input", saveFormState);
_bind("#design-model", "change", saveFormState);
_bind("#generate-model", "change", saveFormState);

// Clear button
_bind("#form-clear", "click", clearForm);

// Also save when refs change — patch the add-ref handler
_bind("#add-ref", "click", function () {
  // The ref row was just added by the original handler.
  // Attach auto-save to the new row's inputs and remove button.
  var rows = document.querySelectorAll(".ref-row");
  var lastRow = rows[rows.length - 1];
  if (lastRow) {
    lastRow.querySelectorAll("input").forEach(function (inp) {
      inp.addEventListener("input", saveFormState);
    });
    var removeBtn = lastRow.querySelector(".ref-remove");
    if (removeBtn) removeBtn.addEventListener("click", function () { saveFormState(); });
  }
});

/* ── Background Profiles ─────────────────────────── */

var profiles = []; // [{label, description}]

function loadProfiles() {
  fetch("/api/config").then(function (r) { return r.json(); }).then(function (cfg) {
    profiles = cfg.profiles || [];
    renderProfileDropdown();
  }).catch(function () {});
}

function renderProfileDropdown() {
  var sel = $("#profile-select");
  if (!sel) return;
  sel.innerHTML = '<option value="">profiles...</option>';
  profiles.forEach(function (p, i) {
    sel.innerHTML += '<option value="' + i + '">' + esc(p.label) + '</option>';
  });
}

function saveProfiles() {
  return fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profiles: profiles }),
  });
}

_bind("#profile-select", "change", function () {
  var idx = parseInt(this.value);
  if (isNaN(idx) || !profiles[idx]) return;
  $("#level-input").value = profiles[idx].description;
  var el = $("#level-input");
  el.style.height = "auto";
  el.style.height = el.scrollHeight + "px";
  saveFormState();
  updateFormState();
});

_bind("#profile-save", "click", function () {
  var desc = $("#level-input").value.trim();
  if (!desc) return;
  var label = prompt("profile name (short label):");
  if (!label || !label.trim()) return;
  label = label.trim();
  var existing = profiles.findIndex(function (p) { return p.label === label; });
  if (existing >= 0) {
    profiles[existing].description = desc;
  } else {
    profiles.push({ label: label, description: desc });
  }
  saveProfiles().then(function () { renderProfileDropdown(); });
});

loadProfiles();

/* ── Run Presets ─────────────────────────────────── */

var presets = []; // [{name, url, refs, ref_annotations, level, design_model, generate_model}]

function loadPresets() {
  fetch("/api/config").then(function (r) { return r.json(); }).then(function (cfg) {
    presets = cfg.presets || [];
    renderPresetDropdown();
  }).catch(function () {});
}

function renderPresetDropdown() {
  var sel = $("#preset-select");
  if (!sel) return;
  sel.innerHTML = '<option value="">load preset...</option>';
  presets.forEach(function (p, i) {
    sel.innerHTML += '<option value="' + i + '">' + esc(p.name) + '</option>';
  });
  var del = $("#preset-delete");
  if (del) del.style.display = presets.length ? "" : "none";
}

function savePresets() {
  return fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ presets: presets }),
  });
}

_bind("#preset-select", "change", function () {
  var idx = parseInt(this.value);
  if (isNaN(idx) || !presets[idx]) return;
  var p = presets[idx];
  var refs = $("#refs");
  if (refs) refs.innerHTML = "";
  restoreFormState({
    url: p.url || "",
    level: p.level || "",
    design_model: p.design_model || "",
    generate_model: p.generate_model || "",
    refs: p.refs || [],
    ref_annotations: p.ref_annotations || [],
  });
  saveFormState();
});

_bind("#preset-save", "click", function () {
  var state = getFormState();
  if (!state.url) { alert("fill in at least the source url first"); return; }
  var name = prompt("preset name:");
  if (!name || !name.trim()) return;
  name = name.trim();
  var existing = presets.findIndex(function (p) { return p.name === name; });
  var preset = {
    name: name,
    url: state.url,
    refs: state.refs.filter(Boolean),
    ref_annotations: state.ref_annotations || [],
    level: state.level,
    design_model: state.design_model,
    generate_model: state.generate_model,
  };
  if (existing >= 0) {
    presets[existing] = preset;
  } else {
    presets.push(preset);
  }
  savePresets().then(function () {
    renderPresetDropdown();
    var sel = $("#preset-select");
    if (sel) sel.value = "" + (existing >= 0 ? existing : presets.length - 1);
  });
});

_bind("#preset-delete", "click", function () {
  var sel = $("#preset-select");
  if (!sel) return;
  var idx = parseInt(sel.value);
  if (isNaN(idx) || !presets[idx]) return;
  var name = presets[idx].name;
  if (!confirm('delete preset "' + name + '"?')) return;
  presets.splice(idx, 1);
  savePresets().then(function () {
    renderPresetDropdown();
    var s = $("#preset-select");
    if (s) s.value = "";
  });
});

loadPresets();
