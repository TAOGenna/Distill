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

/* ── Config ───────────────────────────────────────── */

fetch("/api/config")
  .then((r) => r.json())
  .then((cfg) => {
    if (!cfg.api_key_set) $("#api-warning").classList.add("visible");
    if (cfg.output_dir) $("#cfg-output").value = cfg.output_dir;
    if (cfg.api_key_masked) {
      $("#cfg-key").placeholder = cfg.api_key_masked;
    } else if (cfg.auth_method === "claude_code") {
      $("#cfg-key").placeholder = "using Claude Code auth";
    }
  })
  .catch(() => {});

$("#save-key").addEventListener("click", async () => {
  const key = $("#cfg-key").value.trim();
  if (!key) return;
  await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: key }),
  });
  $("#cfg-key").value = "";
  $("#cfg-key").placeholder = key.slice(0, 8) + "..." + key.slice(-4);
  $("#api-warning").classList.remove("visible");
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

/* ── Auto-grow textarea ───────────────────────────── */

document.querySelectorAll("textarea").forEach((el) => {
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
    '<button type="button" class="ref-remove">&times;</button>';
  row.querySelector(".ref-remove").addEventListener("click", () => row.remove());
  $("#refs").appendChild(row);
});

/* ── Courses ──────────────────────────────────────── */

function loadCourses() {
  fetch("/api/courses")
    .then((r) => r.json())
    .then((data) => {
      if (!data.courses || !data.courses.length) {
        coursesEl.innerHTML = '<p class="dim">no courses generated yet</p>';
        return;
      }
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

/* ── Generate ─────────────────────────────────────── */

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const fd = new FormData(form);
  const params = {
    url: fd.get("url"),
    level: fd.get("level"),
    model: fd.get("model"),
    generate_model: fd.get("generate_model"),
    series: $("#series").checked,
    refs: [...document.querySelectorAll('input[name="ref"]')]
      .map((el) => el.value.trim())
      .filter(Boolean),
  };

  goBtn.disabled = true;
  goBtn.textContent = "generating...";
  logEl.innerHTML = "";
  progressEl.classList.add("active");
  genStart = Date.now();
  flowNodes = {};
  flowActive = false;

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
      appendPhase(ev.phase);
    } else if (ev.type === "log") {
      appendLog(ev.message, ev.level || "");
    } else if (ev.type === "curriculum") {
      renderCurriculumFlow(ev.data);
    } else if (ev.type === "module_complete") {
      activateFlowNode(ev.module_index);
      appendLog(
        "module " + ev.module_index + " (" + ev.title + ") generated",
        "ok"
      );
    } else if (ev.type === "complete") {
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
    resetBtn();
  };
}

/* ── Log rendering ────────────────────────────────── */

const phaseNames = {
  preprocess: "preprocessing",
  agent: "analyzing & designing",
  analyze: "analyzing",
  design: "designing curriculum",
  generate: "generating modules",
  review: "reviewing",
  done: "complete",
};

function appendPhase(phase) {
  const el = document.createElement("div");
  el.className = "phase-label";
  el.textContent = "\u2500\u2500 " + (phaseNames[phase] || phase) + " \u2500\u2500";
  logEl.appendChild(el);
}

function appendLog(message, level) {
  const el = document.createElement("div");
  el.className = "log-entry" + (level ? " " + level : "");

  const secs = genStart ? Math.floor((Date.now() - genStart) / 1000) : 0;
  const mm = String(Math.floor(secs / 60)).padStart(2, "0");
  const ss = String(secs % 60).padStart(2, "0");

  el.innerHTML =
    '<span class="ts">[' + mm + ":" + ss + "]</span> " + esc(message);
  logEl.appendChild(el);
}

function showResult(result) {
  genStart = null;

  // If the flow was already rendered progressively, just finalize it
  if (flowActive) {
    finalizeFlow(result);
    return;
  }

  // Fallback: curriculum came with the complete event (no progressive render)
  if (
    result.curriculum &&
    result.curriculum.modules &&
    result.curriculum.modules.length > 0
  ) {
    renderCurriculumFlow(result.curriculum);
    // Immediately activate all nodes
    result.curriculum.modules.forEach(function (m) {
      activateFlowNode(m.index);
    });
    finalizeFlow(result);
  } else {
    var el = document.createElement("div");
    el.className = "result-box";
    el.innerHTML =
      "<strong>course generated</strong>" +
      '<div class="stat">path <span>' +
      esc(result.course_dir || "\u2014") +
      "</span></div>";
    logEl.appendChild(el);
  }
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

    // ── DAG layout ───────────────────────────────────
    var byIdx = {};
    modules.forEach(function (m) { byIdx[m.index] = m; });

    // Assign layers: longest path from any root
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

    // Group by layer
    var layers = {};
    var maxLayer = 0;
    modules.forEach(function (m) {
      var d = depth[m.index];
      if (!layers[d]) layers[d] = [];
      layers[d].push(m);
      if (d > maxLayer) maxLayer = d;
    });

    // Position nodes
    var layerSpacing = 120;
    var startY = 24;
    var startX = W * 0.5;
    var totalH = startY + (maxLayer + 2) * layerSpacing;
    flow.style.height = totalH + "px";

    var positions = {}; // index → {x, y}
    for (var layer = 0; layer <= maxLayer; layer++) {
      var group = layers[layer] || [];
      var count = group.length;
      var y = startY + (layer + 1) * layerSpacing;
      group.forEach(function (m, i) {
        // Spread nodes across the width for this layer
        var x;
        if (count === 1) {
          // Single node: alternate sides by layer for visual interest
          x = layer % 2 === 0 ? W * 0.35 : W * 0.65;
        } else {
          // Multiple nodes: distribute evenly
          var margin = W * 0.2;
          var usable = W - 2 * margin;
          x = margin + (usable * (i + 0.5)) / count;
        }
        positions[m.index] = { x: x, y: y };
      });
    }

    // ── SVG edges ────────────────────────────────────
    var NS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", W);
    svg.setAttribute("height", totalH);
    svg.style.cssText =
      "position:absolute;top:0;left:0;pointer-events:none;";

    // Arrowhead marker
    var defs = document.createElementNS(NS, "defs");
    var marker = document.createElementNS(NS, "marker");
    marker.setAttribute("id", "arrow");
    marker.setAttribute("markerWidth", "8");
    marker.setAttribute("markerHeight", "6");
    marker.setAttribute("refX", "7");
    marker.setAttribute("refY", "3");
    marker.setAttribute("orient", "auto");
    var tri = document.createElementNS(NS, "polygon");
    tri.setAttribute("points", "0 0.5, 7 3, 0 5.5");
    tri.setAttribute("fill", "#ddd");
    marker.appendChild(tri);
    defs.appendChild(marker);
    svg.appendChild(defs);

    // Build edges: start→roots, then parent→child (deduplicated)
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

    // Draw edges as beziers with arrowheads (offset to stop at node edges)
    edges.forEach(function (edge) {
      var fy = edge.fromDot ? edge.from.y + 10 : edge.from.y + 5;
      var ty = edge.to.y - 38;
      var midY = (fy + ty) / 2;
      var d =
        "M " + edge.from.x + " " + fy +
        " C " + edge.from.x + " " + midY +
        ", " + edge.to.x + " " + midY +
        ", " + edge.to.x + " " + ty;

      var pathEl = document.createElementNS(NS, "path");
      pathEl.setAttribute("d", d);
      pathEl.setAttribute("stroke", "#ddd");
      pathEl.setAttribute("stroke-width", "2");
      pathEl.setAttribute("fill", "none");
      pathEl.setAttribute("marker-end", "url(#arrow)");
      svg.appendChild(pathEl);
    });
    flow.appendChild(svg);

    // Animate all paths drawing in
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

    // ── Nodes ────────────────────────────────────────

    // Start dot
    var dot = document.createElement("div");
    dot.className = "flow-node";
    dot.style.left = startX + "px";
    dot.style.top = startY + "px";
    dot.innerHTML = '<div class="flow-dot"></div>';
    flow.appendChild(dot);
    setTimeout(function () { dot.classList.add("visible"); }, 150);

    // Module nodes by layer — appear in pending state
    var animDur = 1.2;
    for (var ly = 0; ly <= maxLayer; ly++) {
      (function (layer) {
        var group = layers[layer] || [];
        group.forEach(function (m) {
          var pos = positions[m.index];
          var node = document.createElement("div");
          node.className = "flow-node pending";
          node.style.left = pos.x + "px";
          node.style.top = pos.y + "px";
          node.innerHTML =
            '<div class="flow-icon">' +
            String(m.index).padStart(2, "0") +
            "</div>" +
            '<div class="flow-label">' + esc(m.title) + "</div>" +
            '<div class="flow-sub">' + m.exercise_count + " exercises</div>";
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
  goBtn.disabled = false;
  goBtn.textContent = "generate";
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
