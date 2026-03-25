# CLAUDE.md

## Project Structure

```
scaffoldly/
├── __main__.py       # python -m scaffoldly
├── cli.py            # CLI argument parsing
├── agent.py          # Claude Agent SDK orchestrator + sub-agent definitions
├── tools.py          # Custom @tool definitions (MCP server)
├── schemas.py        # Pydantic models for structured output
└── system_prompt.py  # CS231n pedagogy + workflow instructions
```

## Architecture

Powered by the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python). Three-phase architecture:

```
scaffoldly generate <url> [--ref ...] [--series] --level "..."
        │
        ▼
┌──────────────────────────────────────┐
│  Phase 1: Main Agent (Opus)          │
│                                      │
│  1. Fetch source(s)                  │
│     (focus: deep read, refs: skim)   │
│  2. Analyze + triage concepts        │
│     → submit_analysis                │
│  3. Design + coverage check          │
│     → submit_curriculum              │
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

## Key Design Decisions

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
