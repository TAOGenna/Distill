# Plan: Make Scaffoldly Agent-Agnostic

## Problem

Scaffoldly is locked into the Claude Agent SDK (`claude-agent-sdk`), which means:
- Users must have Claude Code installed or an `ANTHROPIC_API_KEY`
- No option to use cheaper models (GPT-4o, Gemini, local models)
- Each lesson costs **$7-11** — prohibitive for most users

## Root Cause: Architecture, Not Just Model Choice

The cost isn't primarily about which model is used. It's about **running structured content generation through a coding agent loop**.

Current architecture uses Claude Agent SDK's `ClaudeSDKClient` with tools (`Bash`, `Read`, `Write`, `Edit`). Every file write is a tool-call round trip where the model sees the entire conversation history again. This multiplies token cost ~3-5x.

### What scaffoldly actually does vs what it pays for

| What it does | What it pays for |
|---|---|
| Read preprocessed sources | Agent reading files via `Read` tool (round trip) |
| Produce structured analysis | Multi-turn tool use with `submit_analysis` |
| Design curriculum | Multi-turn tool use with `submit_curriculum` |
| Write lesson files from scratch | Agent calling `Write` tool per file (round trip each) |
| Review generated content | Agent reading all files back via `Read` tool |

None of this requires an autonomous coding agent. It's **structured text generation + file I/O**.

## Proposed Architecture

Replace the agent loop with direct API calls. Python handles all file operations. The CLI is removed — the web UI is the sole interface.

```
Browser → http://localhost:8420
        │
        ▼
┌──────────────────────────────────────┐
│  Web UI: generate form + settings    │
│  Provider, model, API key config     │
│  SSE progress stream + DAG viz       │
└──────────────┬───────────────────────┘
               │ POST /api/generate
               ▼
┌──────────────────────────────────────┐
│  Phase 0: Preprocessing             │
│  (fetch.py — no change)             │
│  URL → _sources/ + manifest.json    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 1a: Analyze                   │
│  1 API call (design model)           │
│                                      │
│  Input: system prompt + sources      │
│         (token-budget managed)       │
│  Output: Analysis (Pydantic)         │
│  Python: validate → _analysis.json   │
│  On failure: retry with error (2x)   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 1b: Design Curriculum         │
│  1 API call (design model)           │
│                                      │
│  Input: system prompt + analysis     │
│         + source excerpts            │
│  Output: Curriculum + root README    │
│          (Pydantic)                  │
│  Python: validate + coverage check   │
│          → _curriculum.json          │
│          create dirs, write README   │
│          emit `curriculum` event     │
│  On failure: retry with error (2x)   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 2: Generate Modules           │
│  N parallel API calls (gen model)    │
│                                      │
│  Per module:                         │
│    Input: module prompt + spec       │
│           + source excerpts          │
│    Output: ModuleOutput (all files)  │
│    Python: write files, validate     │
│            syntax, emit event        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 3: Review                     │
│                                      │
│  3a: Pre-flight (Python, no LLM)    │
│      Syntax check, structural check  │
│      → modules that fail go back to  │
│        Phase 2 with error context    │
│                                      │
│  3b: Quality review (LLM)           │
│      Per-module: pedagogical quality,│
│      progressive difficulty,         │
│      realism, question depth         │
│      → PASS or revision instructions │
│      → failed modules re-generated   │
└──────────────────────────────────────┘
```

### Why Phase 1 is two calls, not one

Analysis and curriculum design are distinct cognitive tasks with different failure modes:

- **Analyze** can fail by misclassifying concepts, missing key ideas, or getting the content_type wrong. Validating the analysis before designing curriculum catches these early.
- **Design** can fail with coverage gaps, bad dependency graphs, or exercises that don't match the student level. The coverage check (essential concepts → exercises) acts as a gate.

If merged into one call, a bad analysis poisons the curriculum with no recovery. Two calls costs marginally more in prompt tokens (the source content is likely cached on the second call) but saves substantially on retries and produces more reliable output.

### Why review uses LLM judgment, not just deterministic checks

Some checks ARE deterministic (syntax, structural completeness) — Python runs these as a pre-flight gate so the LLM doesn't waste tokens reviewing a module that doesn't even parse.

But the checks that matter most for educational quality require critical thinking:
- Does difficulty actually progress within and across modules?
- Is baked-in data realistic for the domain, or placeholder?
- Do analytical questions force synthesis (Level 3+), or are they recall?
- Do exercises connect to the source material's specific insights?
- Would a student know where to start and feel motivated to continue?

These cannot be reduced to regex or AST checks. The LLM review stays as a default part of the pipeline.

### Cost impact estimate

For a 4-module course:

| Scenario | Source size | Phase 1 | Phase 2 | Phase 3 | Total |
|---|---|---|---|---|---|
| Short blog (~10K tokens) | small | ~$0.50 | ~$1.00 | ~$0.50 | **~$2.00** |
| arXiv paper (~50K tokens) | large | ~$2.00 | ~$1.50 | ~$0.75 | **~$4.25** |
| Multi-source series (~80K) | xlarge | ~$3.00 | ~$2.00 | ~$1.00 | **~$6.00** |

**With same models (Opus + Sonnet): ~$2-6** (down from $7-11, ~50% reduction)

**With cheaper models (GPT-4o + 4o-mini): ~$0.25-0.75** (down from $7-11, ~95% reduction)

The savings come from:
1. Eliminating the tool-call round-trip tax (~3-5x context resend)
2. Single-shot structured output instead of multi-turn conversations
3. Python doing all file I/O (zero LLM tokens for writes)

## Implementation Plan

### Step 1: Provider abstraction layer

Use **LiteLLM** + **Instructor** as the backend. LiteLLM routes to 100+ providers. Instructor gives validated Pydantic structured output across all of them. This avoids writing per-provider code.

```python
# scaffoldly/llm.py

import instructor
import litellm

@dataclass
class CompletionResult:
    content: str
    structured: BaseModel | None
    usage: Usage

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cost_usd: float | None

class LLMClient:
    """Thin wrapper around LiteLLM + Instructor for structured output."""

    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self.client = instructor.from_litellm(litellm.acompletion)

    async def complete(
        self,
        messages: list[dict],
        model: str,
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 16384,
        max_retries: int = 2,
    ) -> CompletionResult:
        """Single API call with optional structured output.

        If response_model is given, Instructor handles validation
        and retries automatically — failed Pydantic validation gets
        fed back to the model as an error message.
        """
        ...
```

Provider-specific model names are handled by LiteLLM's routing:
- `anthropic/claude-sonnet-4-6` → Anthropic API
- `gpt-4o` → OpenAI API
- `gemini/gemini-2.5-flash` → Google API
- `ollama/llama3` → local Ollama

### Step 2: Source budget management

Before any LLM call, Python must read and manage the source content to fit within token limits. Large sources (arXiv papers, repos, multi-article series) can easily blow up context.

```python
# scaffoldly/sources.py

MAX_SOURCE_TOKENS = 40_000  # leaves room for system prompt + output

def prepare_sources(manifest: dict, sources_dir: str) -> str:
    """Read and budget-manage source content for LLM consumption."""
    content = read_all_sources(manifest, sources_dir)
    token_count = estimate_tokens(content)

    if token_count <= MAX_SOURCE_TOKENS:
        return content

    # Strategy 1: multi-source — truncate refs, keep focus source full
    # Strategy 2: single large source — extract key sections
    #             (LaTeX \section headers, markdown ## headers)
    # Strategy 3: very large — summarize with cheap model first,
    #             then pass summary + key excerpts to design model
    return budget_managed_content
```

This replaces the agent's multi-turn `Read` tool calls with a single, deterministic source preparation step.

### Step 3: Refactor Phase 2 (module generation) — do this first

Module generation is the highest-cost phase (N modules × ~30 agent turns each). It's also the most mechanical — given a spec, produce files. This makes it the best candidate for first migration.

Current (`agent.py`):
- `query()` per module with `ClaudeAgentOptions`, max 30 turns
- Agent uses `Write` tool per file, `Bash` for syntax check
- Each turn re-sends the full system prompt + conversation history

Proposed:
- Same `anyio.create_task_group()` parallel dispatch
- Each module: **1 API call** with structured output containing all files
- Python writes files to disk, validates syntax (ast.parse, compiler)
- On syntax failure: retry that module with the error appended

```python
# New schemas for module generation output

class GeneratedFile(BaseModel):
    relative_path: str        # e.g. "ex01_basic.py", "data/sample.csv"
    content: str
    language: str             # for syntax validation routing

class ModuleOutput(BaseModel):
    readme: str               # module README.md content
    files: list[GeneratedFile]
```

The module generator prompt is adapted from `MODULE_GENERATOR_SYSTEM_PROMPT` — same pedagogy guidelines, but instead of "use Write to create each file", it says "return all files as structured output."

### Step 4: Refactor Phase 1 (analyze & design)

Current (`agent.py`):
- `ClaudeSDKClient` multi-turn conversation
- Agent reads sources via `Read`, calls `submit_analysis` MCP tool, calls `submit_curriculum` MCP tool, uses `Write` for root README

Proposed — two sequential API calls:

**Phase 1a: Analyze**
- Python reads all sources into a budget-managed string (Step 2)
- 1 API call: system prompt + sources → `Analysis` (structured output)
- Pydantic validates, saves `_analysis.json`
- Reuses existing `Analysis` schema from `schemas.py` as-is

**Phase 1b: Design Curriculum**
- 1 API call: system prompt + analysis + source excerpts → `CurriculumDesign` (structured output)
- Pydantic validates, coverage check runs (essential concepts → exercises)
- Python creates directory structure, writes root README
- Emits `curriculum` event for web UI DAG

```python
# Extended schema for Phase 1b output

class CurriculumDesign(BaseModel):
    curriculum: Curriculum      # reuse existing schema
    root_readme: str            # full README.md content
    requirements: str           # requirements.txt or equivalent setup
```

The coverage check (currently in `submit_curriculum` tool) moves to pure Python validation after the API call returns. Same logic, just no MCP wrapper.

### Step 5: Refactor Phase 3 (review)

Current (`agent.py`):
- Main agent dispatches `reviewer` sub-agent (Sonnet)
- Reviewer reads all files via `Read` tool, checks 10 criteria
- If REVISE, main agent fixes issues and re-dispatches reviewer

Proposed:

**Phase 3a: Pre-flight validation (Python, zero LLM cost)**

Catches structural issues before spending LLM tokens on review:
- Syntax validation: `ast.parse()` for Python, compiler for C/Rust
- `__main__` / `main()` block present in every exercise file
- TODO markers present in scaffolded files
- README.md exists per module
- Files exist where curriculum spec says they should

Modules that fail pre-flight go back to Phase 2 for re-generation with the error context appended.

**Phase 3b: Quality review (LLM, per-module)**

For modules that pass pre-flight, the LLM evaluates what Python can't:
- Progressive difficulty (within and across modules)
- Realism of baked-in data
- Analytical question depth (Level 3+ per rubric)
- Connection to source material's specific insights
- Scaffolding quality (not just presence, but helpfulness)
- Overall student experience

```python
class ModuleReview(BaseModel):
    module_index: int
    verdict: Literal["pass", "revise"]
    issues: list[ReviewIssue]

class ReviewIssue(BaseModel):
    criterion: str            # which rubric item
    description: str          # what's wrong
    file_path: str | None     # which file, if specific
    suggested_fix: str        # actionable revision instruction

class ReviewResult(BaseModel):
    modules: list[ModuleReview]
    overall_verdict: Literal["pass", "revise"]
```

Failed modules get re-generated (Phase 2) with revision instructions appended to the prompt. Number of revision cycles is configurable in settings (default: 1).

### Step 6: Drop CLI, simplify entry point

The CLI (`generate`, `review` subcommands) is removed. The web UI is the sole interface.

**What changes:**
- `cli.py` reduces to server launcher only (port config, WSL detection, browser open)
- `__main__.py` just calls server start
- `run_agent_sync()` removed — server is already async
- `run_review_sync()` removed — review is part of the pipeline, triggered from UI
- All configuration (provider, model, API keys, output dir) lives in web UI settings → `~/.config/scaffoldly/config.json`

**What stays:**
- `server.py` — Starlette server, SSE streaming, REST API
- `web/` — static frontend (generate form, progress, DAG, settings)
- Event emission (`curriculum`, `module_complete`, `phase`, `log`)

**Web UI updates:**
- Add provider selector to settings page (Anthropic, OpenAI, Google, Ollama, etc.)
- Add API key fields per provider
- Remove Claude Code CLI detection (no longer relevant)
- Keep existing model dropdowns (design model, generate model)

### Step 7: Remove claude-agent-sdk dependency

Only after all phases are migrated:

- Remove `claude-agent-sdk` from `pyproject.toml`
- Delete all SDK imports from `agent.py`
- `tools.py` becomes pure Pydantic validation helpers (no MCP server, no `@tool` decorator)
- `create_scaffoldly_server()` goes away
- `REVIEWER_AGENT` definition goes away (replaced by review prompt + structured output)
- Add `litellm` and `instructor` to dependencies

## Retry-with-Feedback (replaces multi-turn agent loop)

The agent loop's main value was error recovery — if a tool call failed validation, the agent could read the error and retry. We preserve this with a much cheaper pattern:

```python
# Instructor handles this natively via max_retries parameter.
# On Pydantic validation failure, it feeds the error back to the
# model and retries the call. Each retry only sends:
#   original messages + failed attempt + validation error
# vs the agent loop which resends the ENTIRE conversation history.
```

This gives the same "fix and retry" behavior at ~1/5 the token cost.

## What We Lose (and Mitigations)

| Lost capability | Impact | Mitigation |
|---|---|---|
| Adaptive mid-generation behavior | Model can't read what it wrote and adjust | Review phase catches issues; re-generate targeted modules |
| Agent decides file organization on the fly | None — curriculum design already defines structure | Python creates structure deterministically from curriculum |
| Code execution to verify exercises | Low — scaffoldly's design is "no test frameworks" | Pre-flight syntax validation in Python (ast.parse, compiler) |
| Effort parameter | Anthropic-specific | Drop it; use temperature or reasoning tokens where supported |
| Built-in tools (Bash, Read, Write, Edit) | None of these are needed for content generation | Python does all file I/O |
| CLI interface | Intentional removal | Web UI is the sole interface |

## Migration Order

1. **Provider layer** (`llm.py`) + source budget management (`sources.py`) — can be built and tested independently
2. **Phase 2 first** — module generation is the highest-cost phase, most mechanical, easiest to validate by diffing outputs
3. **Phase 1** — split into analyze + design, structured output with validation checkpoints
4. **Phase 3** — pre-flight Python checks + LLM quality review
5. **Drop CLI** — simplify `cli.py` to server-only launcher
6. **Remove SDK dependency** — only after all phases migrated and tested
7. **Web UI updates** — provider selector, API key management, remove Claude Code detection

Each step can be shipped incrementally. During migration, unconverted phases can still use the SDK while converted phases use the new provider layer.

## Decided Questions

1. **Instructor for structured output.** Use Instructor (via LiteLLM) for all structured output calls. It handles cross-provider JSON schema differences, Pydantic validation, and retry-with-feedback natively. Avoids reinventing this for each provider.
2. **Phase 1b generates root README in the same call.** The model already has full context (analysis + sources) when designing the curriculum — perfect moment to also produce the README. Saves a round trip. The `CurriculumDesign` schema includes `root_readme` and `requirements` fields.
3. **Max revision cycles: configurable, default 1.** Exposed in web UI settings. 1 cycle (review → fix → ship) is the default. Users who want higher quality can increase it; cost scales linearly per cycle.
4. **Source summarization is transparent.** If sources exceed token budget, Python quietly summarizes with a cheap model and passes summary + key excerpts to the design model. No extra UI phase — just a log line ("Source material exceeds token budget, summarizing..."). The user cares about course quality, not our token budget.
