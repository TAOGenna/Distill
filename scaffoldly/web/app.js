/* scaffoldly web ui */

const $ = (s) => document.querySelector(s);
const form = $("#form");
const logEl = $("#log");
const progressEl = $("#progress");
const coursesEl = $("#courses");
const goBtn = $("#go");

let genStart = null;
let flowNodes = {};   // module_index → DOM node
let flowActive = false;
let generating = false;
let totalModules = 0;
let completedModuleSet = new Set();

/* ── Isometric icon builder ───────────────────────── */

var ISO_PALETTES = {
  generated: {
    bottom: { left: "#2e2e2e", right: "#444444", top: "#5a5a5a" },
    top:    { left: "#a5a5a5", right: "#666666", top: "#888888" },
  },
  pending: {
    bottom: { left: "#bebebe", right: "#bebebe", top: "#f0f0f0" },
    top:    { left: "#bebebe", right: "#bebebe", top: "#f0f0f0" },
  },
};

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
  if (row) row.style.display = provider === "ollama" ? "none" : "";
}

function updateFormState() {
  if (generating) {
    goBtn.disabled = true;
    return;
  }
  var provider = $("#setup-provider") ? $("#setup-provider").value : "anthropic";
  var hasAuth = apiKeySet || provider === "ollama";
  var hasUrl = $("#url-input").value.trim().length > 0;
  var hasLevel = $("#level-input").value.trim().length > 0;

  goBtn.disabled = !(hasAuth && hasUrl && hasLevel);
}

function markConfigured() {
  apiKeySet = true;
  updateFormState();
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
    $("#setup-key").placeholder = key.slice(0, 8) + "..." + key.slice(-4);
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

$("#browse-output").addEventListener("click", async () => {
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

$("#save-output").addEventListener("click", async () => {
  const dir = $("#cfg-output").value.trim();
  if (!dir) return;
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_dir: dir }),
  });
});

// Auto-save output dir on blur too (not just the save button)
$("#cfg-output").addEventListener("blur", async () => {
  const dir = $("#cfg-output").value.trim();
  if (!dir) return;
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_dir: dir }),
  });
});

// Persist model selections on change
$("#design-model").addEventListener("change", async () => {
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ design_model: $("#design-model").value }),
  });
});

$("#generate-model").addEventListener("change", async () => {
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ generate_model: $("#generate-model").value }),
  });
});

$("#cfg-revision-cycles").addEventListener("change", async () => {
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ max_revision_cycles: parseInt($("#cfg-revision-cycles").value) }),
  });
});

/* ── Auto-grow textarea + live form validation ────── */

$("#url-input").addEventListener("input", updateFormState);
$("#level-input").addEventListener("input", () => {
  // Auto-grow
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

$("#add-ref").addEventListener("click", () => {
  const row = document.createElement("div");
  row.className = "ref-row";
  row.innerHTML =
    '<input type="url" name="ref" placeholder="https://...">' +
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
            '<div class="course-item">' +
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
            "</div>"
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

/* ── beforeunload guard ──────────────────────────── */

window.addEventListener("beforeunload", function (e) {
  if (generating) {
    e.preventDefault();
    e.returnValue = "";
  }
});

/* ── Generate ─────────────────────────────────────── */

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const fd = new FormData(form);
  const params = {
    url: fd.get("url"),
    level: fd.get("level"),
    design_model: fd.get("design_model"),
    generate_model: fd.get("generate_model"),
    series: $("#series").checked,
    refs: [...document.querySelectorAll('input[name="ref"]')]
      .map((el) => el.value.trim())
      .filter(Boolean),
  };

  generating = true;
  goBtn.disabled = true;
  goBtn.textContent = "generating...";
  goBtn.classList.add("generating");
  logEl.innerHTML = "";
  progressEl.classList.add("active");
  genStart = Date.now();
  flowNodes = {};
  flowActive = false;
  totalModules = 0;
  completedModuleSet = new Set();
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

    if (ev.type === "phase") {
      updateProgressBar(ev.phase, "");
      appendPhase(ev.phase);
    } else if (ev.type === "log") {
      appendLog(ev.message, ev.level || "");
    } else if (ev.type === "curriculum") {
      totalModules = ev.data.modules ? ev.data.modules.length : 0;
      renderCurriculumFlow(ev.data);
    } else if (ev.type === "module_complete") {
      completedModuleSet.add(ev.module_index);
      updateModuleProgress();
      activateFlowNode(ev.module_index);
      appendLog(
        "module " + ev.module_index + " (" + ev.title + ") generated",
        "ok"
      );
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
    if (generating) {
      appendLog("connection lost — generation may still be running on the server", "warn");
    }
    src.close();
    resetBtn();
  };
}

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

function renderCurriculumFlow(data) {
  flowNodes = {};
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

    var layerSpacing = 120;
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
          var margin = W * 0.2;
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

    var edges = [];
    var edgeSeen = {};
    function addEdge(from, to, fromDot) {
      var key = Math.round(from.x) + "," + Math.round(from.y) +
                "-" + Math.round(to.x) + "," + Math.round(to.y);
      if (edgeSeen[key]) return;
      edgeSeen[key] = true;
      edges.push({ from: from, to: to, fromDot: fromDot });
    }
    modules.forEach(function (m) {
      if (!m.depends_on || m.depends_on.length === 0) {
        addEdge({ x: startX, y: startY }, positions[m.index], true);
      }
    });
    modules.forEach(function (m) {
      if (m.depends_on) {
        m.depends_on.forEach(function (parentIdx) {
          if (positions[parentIdx]) {
            addEdge(positions[parentIdx], positions[m.index], false);
          }
        });
      }
    });

    function bowDir(edge) {
      var ox = 0, n = 0;
      edges.forEach(function (e) {
        if (e === edge) return;
        if (Math.abs(e.from.x - edge.from.x) < 5 && Math.abs(e.from.y - edge.from.y) < 5) {
          ox += e.to.x - edge.from.x; n++;
        }
        if (Math.abs(e.to.x - edge.to.x) < 5 && Math.abs(e.to.y - edge.to.y) < 5) {
          ox += e.from.x - edge.to.x; n++;
        }
      });
      return (n > 0 && ox > 0) ? -45 : 45;
    }

    edges.forEach(function (edge) {
      var fy = edge.fromDot ? edge.from.y + 10 : edge.from.y + 5;
      var ty = edge.to.y - 38;
      var fx = edge.from.x;
      var tx = edge.to.x;
      var midY = (fy + ty) / 2;

      var d;
      if (Math.abs(tx - fx) < 20) {
        var bow = bowDir(edge);
        d = "M " + fx + " " + fy +
          " C " + (fx + bow) + " " + midY +
          ", " + (tx + bow) + " " + midY +
          ", " + tx + " " + ty;
      } else {
        d = "M " + fx + " " + fy +
          " C " + fx + " " + midY +
          ", " + tx + " " + midY +
          ", " + tx + " " + ty;
      }

      var pathEl = document.createElementNS(NS, "path");
      pathEl.setAttribute("d", d);
      pathEl.setAttribute("stroke", "#ddd");
      pathEl.setAttribute("stroke-width", "2.5");
      pathEl.setAttribute("fill", "none");
      pathEl.setAttribute("stroke-linecap", "round");
      svg.appendChild(pathEl);
    });
    flow.appendChild(svg);

    var paths = svg.querySelectorAll("path");
    paths.forEach(function (p) {
      var len = p.getTotalLength();
      p.style.strokeDasharray = len;
      p.style.strokeDashoffset = len;
    });
    setTimeout(function () {
      paths.forEach(function (p) {
        p.style.transition = "stroke-dashoffset 1.2s ease-out";
        p.style.strokeDashoffset = "0";
      });
    }, 100);

    var dot = document.createElement("div");
    dot.className = "flow-node";
    dot.style.left = startX + "px";
    dot.style.top = startY + "px";
    dot.innerHTML = '<div class="flow-dot"></div>';
    flow.appendChild(dot);
    setTimeout(function () { dot.classList.add("visible"); }, 150);

    var animDur = 1.2;
    for (var ly = 0; ly <= maxLayer; ly++) {
      (function (layer) {
        var group = layers[layer] || [];
        group.forEach(function (m) {
          var pos = positions[m.index];
          var node = document.createElement("div");
          node.className = "flow-node pending";
          node.setAttribute("role", "img");
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
            '<div class="flow-icon">' + isoIcon(56, ISO_PALETTES.pending) + "</div>" +
            '<div class="flow-label">' + esc(m.title) + "</div>" +
            '<div class="flow-sub">' + m.exercise_count + " exercises</div>" +
            (tipHtml ? '<div class="flow-tooltip">' + tipHtml + "</div>" : "");
          flow.appendChild(node);
          flowNodes[m.index] = node;

          var delay = (animDur * (layer + 1)) / (maxLayer + 2);
          setTimeout(function () { node.classList.add("visible"); }, delay * 1000 + 100);
        });
      })(ly);
    }
  });
}

function activateFlowNode(moduleIndex) {
  var node = flowNodes[moduleIndex];
  if (node) {
    node.classList.remove("pending");
    node.classList.add("generated");
    node.setAttribute("aria-label", node.getAttribute("aria-label").replace("pending", "generated"));
    var icon = node.querySelector(".flow-icon");
    if (icon) icon.innerHTML = isoIcon(56, ISO_PALETTES.generated);
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
  updateFormState();
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
