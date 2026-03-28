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

Replace the agent loop with direct API calls. Python handles all file operations.

```
scaffoldly generate <url> --level "..." [--provider anthropic|openai|google|...]
        │
        ▼
┌──────────────────────────────────────┐
│  Preprocessing (fetch.py — no change)│
│  URL → _sources/ + manifest.json     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 1: Analyze & Design           │
│  Single API call (structured output) │
│                                      │
│  Input: system prompt + sources      │
│  Output: Analysis + Curriculum JSON  │
│  Python: validates with Pydantic,    │
│          creates dirs, writes README │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 2: Generate Modules           │
│  N parallel API calls                │
│                                      │
│  Input: system prompt + curriculum   │
│         + relevant source excerpts   │
│  Output: module content (structured) │
│  Python: writes all files to disk    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Phase 3: Review                     │
│  1-2 API calls                       │
│                                      │
│  Input: generated content + rubric   │
│  Output: PASS or structured revisions│
│  Python: applies fixes or re-calls   │
│          Phase 2 for failed modules  │
└──────────────────────────────────────┘
```

### Cost impact estimate

| Phase | Current (agent loop) | Proposed (direct API) |
|---|---|---|
| Phase 1 | Multi-turn Opus, ~5-10 tool-call round trips | 1 API call, structured output |
| Phase 2 | N × multi-turn Sonnet conversations | N × 1 API call each |
| Phase 3 | Multi-turn Opus review | 1-2 API calls |
| **Token multiplier** | ~3-5x (context resent each turn) | ~1x |
| **Estimated cost** | $7-11 per lesson | **~$2-4 per lesson** (same models) |

With cheaper models (GPT-4o, Gemini), costs could drop further to $1-2.

## Implementation Plan

### Step 1: Provider abstraction layer

Create a thin provider interface. No heavy framework — just a common call signature.

```python
# scaffoldly/providers.py

class Provider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        system: str,
        model: str,
        response_format: type[BaseModel] | None = None,  # structured output
        max_tokens: int = 16384,
    ) -> CompletionResult: ...

@dataclass
class CompletionResult:
    content: str
    structured: BaseModel | None  # parsed if response_format given
    usage: Usage

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
```

Implementations:
- `AnthropicProvider` — uses `anthropic` SDK directly (messages API, not Agent SDK)
- `OpenAIProvider` — uses `openai` SDK (works with OpenAI, Azure, OpenRouter, local)
- `GoogleProvider` — uses `google-genai` SDK

Or: use **LiteLLM** as the single backend (routes to 100+ providers). This avoids writing per-provider code but adds a dependency.

### Step 2: Refactor Phase 1 (analyze & design)

Current (`agent.py`):
- Creates `ClaudeSDKClient` with options
- Multi-turn conversation: agent reads sources, calls `submit_analysis`, calls `submit_curriculum`, writes files

Proposed:
- Python reads source files from `_sources/` into a string
- Single API call with system prompt + sources, requesting structured JSON output
- Pydantic validates the response (reuse existing `Analysis` and `Curriculum` schemas)
- Python creates directory structure and writes root README
- Emits `curriculum` event for web UI DAG (no change to frontend)

### Step 3: Refactor Phase 2 (module generation)

Current (`agent.py`):
- Parallel `query()` calls via `anyio.create_task_group()`
- Each module generator is a multi-turn Sonnet conversation that writes files via tools

Proposed:
- Same parallel dispatch pattern (`anyio.create_task_group()`)
- Each module: single API call with module-specific prompt + source excerpts
- Response is structured output containing all file contents for that module
- Python writes files, emits `module_complete` event

New schema needed:
```python
class ModuleOutput(BaseModel):
    readme_content: str
    exercise_files: list[ExerciseFile]

class ExerciseFile(BaseModel):
    path: str  # relative to module dir
    content: str
```

### Step 4: Refactor Phase 3 (review)

Current:
- `ClaudeSDKClient` with `reviewer` sub-agent
- Multi-turn Opus conversation

Proposed:
- Single API call: pass all generated content + review rubric
- Structured output: list of pass/fail per module with revision instructions
- For failed modules: re-call Phase 2 for just those modules with the revision feedback appended

### Step 5: Update CLI and web UI

- Add `--provider` flag (anthropic, openai, google, litellm, etc.)
- Add `--design-model` and `--generate-model` flags (already exist, just wire to new provider)
- Update settings page to show provider selection
- Keep `ANTHROPIC_API_KEY` support, add `OPENAI_API_KEY`, `GOOGLE_API_KEY`, etc.
- Remove Claude Code CLI detection (no longer needed)

### Step 6: Remove claude-agent-sdk dependency

- Remove `claude-agent-sdk` from `pyproject.toml`
- Delete all SDK imports from `agent.py` and `tools.py`
- `tools.py` becomes pure Pydantic validation (no MCP server needed)
- The `@tool` decorator and `create_sdk_mcp_server` go away

## What We Lose (and Mitigations)

| Lost capability | Impact | Mitigation |
|---|---|---|
| Adaptive mid-generation behavior | Model can't read what it wrote and adjust | Review phase catches issues; re-generate targeted modules |
| Agent decides file organization on the fly | None — curriculum design already defines structure | Python creates structure deterministically from curriculum |
| Code execution to verify exercises | Low — scaffoldly's design is "no test frameworks" | Observable milestones philosophy unchanged |
| Effort parameter | Anthropic-specific | Drop it; use temperature or reasoning tokens where supported |
| Built-in tools (Bash, Read, Write, Edit) | None of these are needed | Python does all file I/O |

## Migration Order

1. **Provider layer** (`providers.py`) — can be built and tested independently
2. **Phase 2 first** — module generation is the simplest phase, highest parallelism, easiest to validate
3. **Phase 1** — analysis & design, structured output
4. **Phase 3** — review, least complex
5. **Remove SDK dependency** — only after all phases migrated
6. **CLI/web UI updates** — provider selection

Each step can be shipped incrementally. Phases can be migrated one at a time while the others still use the SDK.
