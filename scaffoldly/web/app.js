/* scaffoldly web ui */

const $ = (s) => document.querySelector(s);
const form = $("#form");
const logEl = $("#log");
const progressEl = $("#progress");
const coursesEl = $("#courses");
const goBtn = $("#go");

let genStart = null;

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
    } else if (ev.type === "module_complete") {
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
  const el = document.createElement("div");
  el.className = "result-box";

  let html = "<strong>course generated</strong>";
  if (result.course_dir)
    html += '<div class="stat">path <span>' + esc(result.course_dir) + "</span></div>";
  if (result.total_cost_usd != null)
    html +=
      '<div class="stat">cost <span>$' +
      result.total_cost_usd.toFixed(4) +
      "</span></div>";

  el.innerHTML = html;
  logEl.appendChild(el);
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
