# CLAUDE.md

## Project Structure

```
distill/
├── __main__.py           # python -m distill
├── cli.py                # Server launcher (web UI only)
├── server.py             # Local Starlette web server + SSE progress streaming
├── fetch.py              # Source preprocessing — URL → local artifacts (no LLM)
├── pipeline.py           # Course generation pipeline — multi-turn conversations (LiteLLM)
├── claude_pipeline.py    # Course generation pipeline — Claude Agent SDK (standalone)
├── diagrams.py           # Excalidraw MCP lifecycle + pure Python SVG renderer
├── llm.py                # LLM client (LiteLLM + Instructor) — provider abstraction
├── mock.py               # Mock LLM client for zero-cost end-to-end testing
├── sources.py            # Source budget management — read + truncate/summarize
├── prompts.py            # System prompts + turn templates for conversational flow
├── schemas.py            # Pydantic models (Blueprint schemas + review schemas)
└── web/                  # Static frontend (no build step, no node_modules)
    ├── index.html        # Generation form + progress + course list + settings
    ├── style.css         # JetBrains Mono, monochrome, CSS custom properties + dark theme
    ├── app.js            # Vanilla JS — SSE, form handling, DAG visualization
    ├── test_dag.html     # Standalone DAG test page with mock data presets
    └── test_iso.html     # Isometric icon tuning page
```

## Goal

Transform a technical blog post, paper, or repo into a hands-on course that walks the student through **reproducing the author's results as faithfully as possible**. Each module's progression builds toward that reproduction — the final module is the capstone by design.

## Quality Standard

The target is MIT 6.102-level course material, not README summaries. Each module produces:

**Lesson document (README.md)** — 3,000-10,000 words:
- Local table of contents + explicit learning objectives
- Running example that evolves through the lesson
- Inline code snippets showing concept → code translation
- Embedded comprehension checks at points of friction
- Formula translation: math → plain language → code (step by step)
- Analytical questions at Level 3+ depth (analysis/synthesis, not recall)
- Synthesis section reconnecting to the course goal

**Exercise files (.py, .c, .rs)** — separate runnable files:
- ~65% provided code (imports, classes, helpers, __main__ test harness)
- ~35% TODO blocks with line count hints (`# YOUR CODE HERE - 8-12 lines`)
- Numpy-style docstrings (purpose, parameters with types/shapes, returns)
- `__main__` block: 20-50 lines, always fully provided, never scaffolded
- Solution versions in `_solutions/` — must run and produce correct output
- Real dependencies only (numpy, torch — never placeholder packages)
- Baked-in realistic domain-appropriate data (not foo/bar/42)

**Explanatory diagrams (diagrams/*.svg)** — 2-4 per module (Claude Code route):
- Excalidraw JSON source (`.excalidraw`) preserved alongside rendered SVG
- Color-coded regions, heavy annotations, spatial layouts
- Referenced inline in README.md where the concept is explained
- Style reference: Aleksa Gordic's matmul blog, Simon Boehm's CUDA-MMM
- Model uses MCP tools (`describe_scene`) for spatial feedback during creation

## Architecture

Two pipeline implementations share the same phases, schemas, and prompts:

- **LiteLLM pipeline** (`pipeline.py`): Multi-provider via LiteLLM + Instructor. Supports Anthropic, OpenAI, Google, Ollama, OpenRouter. Full cost tracking via `litellm.completion_cost()`.
- **Claude Code pipeline** (`claude_pipeline.py`): Standalone mode using Claude Agent SDK. No external API keys — uses Claude Code CLI for auth. Cost tracking via `AssistantMessage.usage` token accumulation with `ResultMessage.total_cost_usd` as primary source (fallback to token-based calculation when SDK returns None).

Web UI is the sole interface.

```
Browser → http://localhost:8420
        │
        ▼
┌──────────────────────────────────────┐
│  Phase 0: Preprocessing              │
│  (fetch.py, no LLM)                 │
│                                      │
│  URL → detect type → handler:        │
│  arxiv/blog/pdf/github               │
│  Output: _sources/ + manifest.json   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 1a: Analyze (1 API call)      │
│  design model, structured output     │
│                                      │
│  Input: full source material         │
│  Output: Analysis (Pydantic)         │
│  → concepts, prerequisites, type     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 1b: Blueprint (1 API call)    │
│  design model, structured output     │
│                                      │
│  Input: analysis + full source       │
│  Output: CurriculumDesign (Pydantic) │
│  → scaffold contracts per exercise   │
│  → key_excerpts (verbatim formulas)  │
│  → root README + requirements        │
│  Python: coverage check, write files │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 2: Generate (N parallel)      │
│  generate model, multi-turn convos   │
│                                      │
│  Per module — conversational chain:  │
│  Turn 1: Write lesson (raw markdown) │
│  Turn 2: Write ex1 scaffold (code)   │
│  Turn 3: Write ex1 solution (code)   │
│     → Python executes, captures out  │
│  Turn 4: Write ex2 scaffold          │
│     → sees ex1's execution output    │
│  ...repeat per exercise...           │
│  Fix turn: syntax errors corrected   │
│                                      │
│  Modules parallel, turns sequential  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 3: Review                     │
│                                      │
│  3a: Pre-flight (Python, no LLM)    │
│      syntax, TODOs, file length,     │
│      expected output patterns        │
│  3b: Quality review (LLM per module)│
│      contract compliance, realism    │
│  Re-generate failed modules          │
└──────────────────────────────────────┘
```

### Why multi-turn conversations (not single-shot)

Single-shot structured output produces 200-word README summaries and hollow exercise shells. The conversational approach restores the quality drivers that made the old agent architecture produce good content:

| Capability | How it's restored |
|---|---|
| Full source access | Full source material in every turn (no truncation) |
| One file at a time | Each conversation turn produces exactly one file |
| Write→Run→Fix loop | Python executes solutions between turns, syntax errors trigger fix turns |
| Iteration budget | ~11 turns per module (lesson + 2 per exercise + fixes) |
| Lesson-first ordering | Model deeply processes source while writing 5,000+ word lesson, exercises flow from that understanding |
| Cross-exercise references | Execution output from ex1 fed into ex2's prompt — real numbers, not imagined |

### Why lesson-first matters

The model writes a 3,000-10,000 word lesson BEFORE any exercise code. During that process, it:
- Translates the source material's formulas step by step
- Develops a running example
- Articulates the key insights in prose

By the time it writes exercises, it has deeply processed the material. This is the opposite of the old approach (README written last as an afterthought) and dramatically better than single-shot (no deep processing at all).

### Diagram generation (Claude Code route only)

The module agent creates Excalidraw diagrams using MCP tools from `yctimlin/mcp_excalidraw`. The Express canvas server is auto-started before Phase 2 and stopped after. Workflow per diagram:
1. Agent calls `batch_create_elements` to build the diagram
2. Agent calls `describe_scene` for spatial feedback (no browser needed)
3. Agent refines with `update_element`, `align_elements`, `distribute_elements`
4. Agent calls `export_scene` to save as `diagrams/<name>.excalidraw`
5. After the agent finishes, `diagrams.py` renders each `.excalidraw` → `.svg` via a pure Python renderer (feTurbulence wobble filter, Virgil font family)
6. README references: `![Description](diagrams/name.svg)`

If the MCP server is unavailable, the agent falls back to writing `.excalidraw` JSON directly via Write tool. The `.excalidraw` source files are always preserved for full-fidelity viewing at excalidraw.com.

## Blueprint Schema

Phase 1b produces a rich contract that constrains Phase 2:

| Field | Purpose |
|---|---|
| `what_is_provided` | What working code the student receives (~65%) |
| `what_student_writes` | What the student implements (~35%) with line counts |
| `key_insight` | The single most important thing the exercise teaches |
| `common_mistakes` | What students typically get wrong |
| `expected_output_pattern` | String that should appear in stdout when correct |
| `key_excerpts` (per module) | Verbatim formulas/algorithms from the source |
| `shared_definitions` | Language, dependencies, naming conventions |

## LLM Client (llm.py)

`LLMClient` wraps LiteLLM + Instructor. Supports:
- **Structured output** (Phase 1): Instructor with provider-specific modes (JSON_SCHEMA for OpenAI, TOOLS for Anthropic)
- **Raw completions** (Phase 2): Free-form text for lessons and code — no JSON constraints
- **Cumulative cost tracking**: input/output tokens and cost across all calls
- **Quota detection**: `QuotaExhaustedError` aborts immediately (no wasted retries)

Providers: anthropic, openai, google, ollama, openrouter.

## Web UI

- **Settings panel**: collapsible (auto-collapses when API key is set), provider, API key, output dir, review rounds
- **Form**: URL input (visually prominent), refs, series mode, background profiles, model selection, sticky generate button
- **Presets**: save/load full form configs, background profiles
- **Progress**: phase bar (8px, striped animation, elapsed timer, live cost), collapsible log (auto-hides during generation)
- **DAG**: Brilliant-style topological layout, 76px isometric icons, sequenced edge animation per-layer, three node states (pending/active/generated), edges darken on completion
- **Theme**: dark/light via `prefers-color-scheme` + manual toggle, all colors as CSS custom properties
- **Events**: `log`, `phase`, `curriculum`, `module_start`, `module_complete`, `cost`
- Config persists to `~/.config/distill/config.json` (chmod 600)

## Key Design Decisions

### Source Preprocessing
URLs are preprocessed into local artifacts before the pipeline starts (`fetch.py`). Jina Reader provides clean markdown; images are extracted and downloaded directly. The full preprocessed content is passed to ALL phases — no truncation.

### Concept Triage
Every concept gets a priority (essential/supporting/contextual) with a rationale. Coverage check verifies essential concepts have exercises before generation begins.

### No Test Frameworks
Observable milestones replace tests. Each exercise ends with a `__main__` block that prints measurements, comparisons, or visualizations. The solution is executed between turns to validate it works.

### Security
- Config file: chmod 600 (owner-only)
- API keys: never stored on LLMClient instance, only in os.environ
- Error sanitization: raw exceptions never forwarded to browser
- Host header validation: blocks DNS rebinding attacks
- Key masking: only last 4 chars shown in UI

### Event Emission
Both pipelines use a `ContextVar`-based event sink for real-time progress. Event types: `log`, `phase`, `curriculum`, `module_start`, `module_complete`, `cost`.

### Cost Tracking
- **LiteLLM path**: `litellm.completion_cost()` per call, accumulated across all phases.
- **Claude Code path**: Token accumulation from `AssistantMessage.usage` on every SDK message. Primary cost from `ResultMessage.total_cost_usd` (SDK has exact ephemeral cache tier pricing); fallback to token-based calculation with hardcoded rates when SDK returns None. Pricing table covers opus/sonnet/haiku with input, output, cache_creation, cache_read rates per million tokens.

## Quality Reference Courses

When auditing generated course quality, compare against:

```
/home/kenyi/kenyi/projects/ai-notebooks-implementations/courses/
├── mit-6.172                          # C/performance engineering (12 modules)
├── Reinforcement-Learning-Stanford-S24 # RL with neural nets (3 assignments)
├── cornell-cs5780-intro-to-ml          # Intro ML
└── csc412-probabilistic-ml-UToronto    # Probabilistic ML (Julia/Python)
```

Also reference MIT 6.102 Software Construction (https://web.mit.edu/6.102/www/sp26/) for lesson document quality — readings are 4,500-15,000 words with inline exercises, running examples, and embedded comprehension checks.
