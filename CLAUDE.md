# CLAUDE.md

## Project Structure

```
scaffoldly/
├── __main__.py       # python -m scaffoldly
├── cli.py            # CLI argument parsing + web UI launcher
├── server.py         # Local Starlette web server + SSE progress streaming
├── fetch.py          # Source preprocessing — URL → local artifacts (no LLM)
├── agent.py          # Claude Agent SDK orchestrator + sub-agent definitions
├── tools.py          # Custom @tool definitions (MCP server)
├── schemas.py        # Pydantic models for structured output
├── system_prompt.py  # CS231n pedagogy + workflow instructions
└── web/              # Static frontend (no build step, no node_modules)
    ├── index.html    # Generation form + progress + course list + settings
    ├── style.css     # JetBrains Mono, monochrome aesthetic
    ├── app.js        # Vanilla JS — SSE, form handling, DAG visualization
    └── test_dag.html # Standalone DAG test page with mock data presets
```

## Architecture

Powered by the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python). Two interfaces, same pipeline:

- **Web UI** (default): `scaffoldly` → opens browser at localhost:8420
- **CLI**: `scaffoldly generate <url> --level "..."` → headless generation

Three-phase architecture:

```
scaffoldly generate <url> [--ref ...] [--series] --level "..."
        │
        ▼
┌──────────────────────────────────────┐
│  Preprocessing (fetch.py, no LLM)    │
│                                      │
│  URL → detect type → handler:        │
│  arxiv  → TeX source tarball         │
│  blog   → Jina markdown + images     │
│  pdf    → download + Jina text       │
│  github → git clone --depth 1        │
│                                      │
│  Output: _sources/ + manifest.json   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 1: Main Agent (Opus)          │
│                                      │
│  1. Consume preprocessed sources     │
│  2. Analyze + triage concepts        │
│     → submit_analysis                │
│  3. Design + coverage check          │
│     → submit_curriculum              │
│     → emits `curriculum` event       │
│       (DAG appears in web UI)        │
│  3b. Re-read quantitative claims     │
│  4. Create root README + dirs        │
│     → STOP                           │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 2: Orchestrator (Python)      │
│                                      │
│  Parallel dispatch via query():      │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │module 0 │ │module 1 │ │module N│ │
│  │(Sonnet) │ │(Sonnet) │ │(Sonnet)│ │
│  └─────────┘ └─────────┘ └────────┘ │
│  Each emits `module_complete` event  │
│  (node lights up in web UI DAG)      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 3: Main Agent (Opus)          │
│                                      │
│  5. Review (adversarial QA)          │
│     → reviewer sub-agent (Sonnet)    │
│  6. Fix & resubmit if needed         │
└──────────────────────────────────────┘
```

## Sub-Agents & Dispatch

- **module_generator** — dispatched programmatically by the orchestrator (not by the LLM). Uses standalone `query()` per module, all running in parallel via `anyio.create_task_group()`. Each gets a self-contained system prompt with full pedagogy guidelines.
- **reviewer** — dispatched by the main agent in Phase 3. Adversarial quality check against 10 criteria (structure, scaffolding, docs, milestones, progressive difficulty, syntax, realism, questions, outcomes, organization). Returns PASS or REVISE.

## Custom Tools (MCP)

| Tool | Purpose |
|------|---------|
| `submit_analysis` | Structured analysis with Pydantic validation. Returns triage summary. |
| `submit_curriculum` | Curriculum design + coverage check against essential concepts. |

The agent also uses Claude Code built-in tools (Bash, Read, Write, Edit) to create all course files directly.

## Web UI

### Launch Banner
Orange-themed two-panel box with a pixel art cactus mascot (green with black Mario Bros-style eyes, orange pot). Left panel: mascot + clickable URL. Right panel: tips + recent courses + auth/output info. Detects WSL to skip browser auto-open.

### DAG Visualization
After Phase 1 completes, the curriculum structure is emitted as a `curriculum` event. The web UI renders a Brilliant-style DAG:

- **Layout**: proper topological layering using `depends_on` fields — not just linear. Modules at the same depth layer are positioned side-by-side. Single modules per layer zigzag left/right.
- **Edges**: SVG cubic bezier curves with arrowheads, following actual dependency relationships. Deduplicated by coordinate key.
- **Progressive**: nodes start in pending state (outlined, dim). As each module finishes generating in parallel, its node transitions to generated state (filled, 3D box-shadow depth effect).
- **Animation**: path draws in via stroke-dashoffset, nodes appear with staggered fade+scale by layer.

Test the DAG without running a generation at `/test_dag.html` — presets for linear, diamond, fan-out, and complex graphs.

### Log Box
Scrollable (200px max-height) with a top fade mask. New entries auto-scroll into view. The DAG lives outside the log box in the main page flow.

### Settings & Auth
Config persists to `~/.config/scaffoldly/config.json`. Claude Code auth is auto-detected (checks for `claude` CLI in PATH). Falls back to `ANTHROPIC_API_KEY` from config or env.

## Key Design Decisions

### Source Preprocessing
URLs are preprocessed into local artifacts before the agent starts (`fetch.py`). This saves LLM tokens and gives the agent richer input — especially for arXiv papers (native LaTeX) and blogs (markdown + downloaded figures). The agent reads local files from `_sources/` instead of curling raw HTML. Jina Reader provides clean markdown; images are extracted from the markdown and downloaded directly (no browser dependency).

### Concept Triage
Every concept gets a priority (essential/supporting/contextual) with a rationale during analysis. The `submit_curriculum` tool checks that all essential concepts have exercises before generation begins. Contextual concepts go in the "What's Next" section, not exercises.

### Analytical Question Rubric
Module READMEs require Level 3+ questions (analysis/synthesis), not recall. The system prompt includes a 4-level rubric with gold-standard exemplars.

### Multi-Source Support
- **Reference mode**: focus URL gets deep analysis, refs get minimal skim for supplementary concepts
- **Series mode**: all sources fetched thoroughly, curriculum spans the full arc

### Content-Type Pedagogy
The `content_type` field (systems_engineering, ml_research, tutorial, library_walkthrough) drives milestone style, scaffolding strategy, and progression pattern. See `system_prompt.py` for details.

### No Test Frameworks
Observable milestones replace tests. Each exercise ends with a `__main__` block that prints measurements, comparisons, or visualizations. The output IS the validation.

### Event Emission
`agent.py` uses a `ContextVar`-based event sink so the web server can receive real-time progress without changing the existing logging to stderr. Event types: `log`, `phase`, `curriculum` (full DAG structure), `module_complete`. When no sink is registered (CLI mode), events are silently dropped.
