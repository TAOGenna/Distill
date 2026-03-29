# CLAUDE.md

## Project Structure

```
scaffoldly/
├── __main__.py       # python -m scaffoldly
├── cli.py            # Server launcher (web UI only, no CLI generation)
├── server.py         # Local Starlette web server + SSE progress streaming
├── fetch.py          # Source preprocessing — URL → local artifacts (no LLM)
├── pipeline.py       # Course generation pipeline — direct API calls
├── llm.py            # LLM client (LiteLLM + Instructor) — provider abstraction
├── sources.py        # Source budget management — read + truncate/summarize
├── prompts.py        # System prompts for each pipeline phase
├── schemas.py        # Pydantic models for structured output
├── tools.py          # Pure validation helpers (coverage check, etc.)
└── web/              # Static frontend (no build step, no node_modules)
    ├── index.html    # Generation form + progress + course list + settings
    ├── style.css     # JetBrains Mono, monochrome aesthetic
    ├── app.js        # Vanilla JS — SSE, form handling, DAG visualization
    └── test_dag.html # Standalone DAG test page with mock data presets
```

## Architecture

Agent-agnostic pipeline using direct LLM API calls via LiteLLM + Instructor. Supports multiple providers (Anthropic, OpenAI, Google, Ollama, OpenRouter). Web UI is the sole interface.

```
Browser → http://localhost:8420
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
│  Phase 1a: Analyze (1 API call)      │
│  design model                        │
│                                      │
│  Input: system prompt + sources      │
│         (token-budget managed)       │
│  Output: Analysis (Pydantic)         │
│  Python: validate → _analysis.json   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 1b: Design (1 API call)       │
│  design model                        │
│                                      │
│  Input: analysis + source excerpts   │
│  Output: CurriculumDesign (Pydantic) │
│          curriculum + root README    │
│  Python: coverage check, write files │
│          emit `curriculum` event     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 2: Generate (N parallel)      │
│  generate model                      │
│                                      │
│  Per module: 1 API call              │
│  Input: module spec + source excerpt │
│  Output: ModuleOutput (all files)    │
│  Python: write files, validate       │
│          syntax, emit events         │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 3: Review                     │
│                                      │
│  3a: Pre-flight (Python, no LLM)    │
│      syntax, structure checks        │
│  3b: Quality review (LLM per module)│
│      pedagogical quality, realism    │
│  Re-generate failed modules          │
└──────────────────────────────────────┘
```

## LLM Client (llm.py)

`LLMClient` wraps LiteLLM + Instructor. Supports:
- **Structured output**: pass `response_model=SomePydanticModel` to get validated objects back
- **Retry with feedback**: Instructor automatically retries on Pydantic validation failure
- **Provider routing**: maps `(provider, model)` to LiteLLM model strings

Providers: anthropic, openai, google, ollama, openrouter. Model defaults per provider in `PROVIDER_DEFAULTS`.

## Source Budget Management (sources.py)

Reads fetch.py artifacts and manages token limits. Strategy:
1. Sources fit budget → return as-is
2. Moderate overflow → truncate by sections (LaTeX/markdown headers)
3. Large overflow → summarize with cheap model, return summary + excerpts

## Schemas (schemas.py)

| Schema | Phase | Purpose |
|--------|-------|---------|
| `Analysis` | 1a | Concept extraction, triage, content type |
| `CurriculumDesign` | 1b | Curriculum + root README + requirements |
| `ModuleOutput` | 2 | All files for a module (README + exercises) |
| `GeneratedFile` | 2 | Single file with path, content, language |
| `ReviewResult` | 3b | Per-module pass/fail with issues |

## Web UI

### Launch Banner
Orange-themed two-panel box with a pixel art cactus mascot. Detects WSL to skip browser auto-open.

### DAG Visualization
After Phase 1b completes, the curriculum structure is emitted as a `curriculum` event. The web UI renders a Brilliant-style DAG with topological layering, SVG bezier edges, and progressive node activation.

### Settings
Config persists to `~/.config/scaffoldly/config.json`. Settings include:
- **Provider**: Anthropic, OpenAI, Google, Ollama, OpenRouter
- **API key**: stored per-provider
- **Design/generate model**: populated from provider defaults
- **Output directory**
- **Max revision cycles**: 0-3 (default 1)

## Key Design Decisions

### Source Preprocessing
URLs are preprocessed into local artifacts before the pipeline starts (`fetch.py`). Jina Reader provides clean markdown; images are extracted and downloaded directly.

### Concept Triage
Every concept gets a priority (essential/supporting/contextual) with a rationale. Coverage check verifies essential concepts have exercises.

### Analytical Question Rubric
Module READMEs require Level 3+ questions (analysis/synthesis, not recall).

### Content-Type Pedagogy
The `content_type` field drives milestone style, scaffolding strategy, and progression pattern. See `prompts.py` for details.

### No Test Frameworks
Observable milestones replace tests. Each exercise ends with a `__main__` block that prints measurements, comparisons, or visualizations.

### Event Emission
`pipeline.py` uses a `ContextVar`-based event sink so the web server can receive real-time progress. Event types: `log`, `phase`, `curriculum`, `module_complete`.

## Quality Reference Courses

When auditing generated course quality, compare against the reference courses at:

```
/home/kenyi/kenyi/projects/ai-notebooks-implementations/courses/
├── mit-6.172                          # C/performance engineering (12 modules)
├── Reinforcement-Learning-Stanford-S24 # RL with neural nets (3 assignments)
├── cornell-cs5780-intro-to-ml          # Intro ML
└── csc412-probabilistic-ml-UToronto    # Probabilistic ML (Julia/Python)
```

These were generated by the old architecture (Opus+Sonnet agent loop) and represent the quality target. Key measurable patterns:
- **40-200 lines** per exercise file
- **~65% provided code**, ~35% TODO blocks
- **3-5 TODO blocks** per exercise with line count hints (`YOUR CODE HERE - 8-12 lines`)
- **100% docstring coverage** (numpy-style: purpose, parameters with shapes, returns)
- **`__main__` block: 20-50 lines** with full test harness, always fully provided
- **Real dependencies** (numpy, torch, gym — never placeholder packages)
- **Baked-in realistic data** (domain-appropriate fixtures, not foo/bar/42)
