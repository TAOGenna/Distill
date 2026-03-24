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

Powered by the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python). The main agent runs Claude Code with a system prompt encoding CS231n pedagogy.

```
scaffoldly generate <url> [--ref ...] [--series] --level "..."
        │
        ▼
┌──────────────────────────────────────┐
│  Main Agent (Claude Code)            │
│                                      │
│  1. Fetch source(s)                  │
│     (focus: deep read, refs: skim)   │
│  2. Analyze + triage concepts        │
│     → submit_analysis                │
│  3. Design + coverage check          │
│     → submit_curriculum              │
│  3b. Re-read quantitative claims     │
│  4. Generate → Write files           │
│  5. Review (adversarial QA)          │
│  6. Fix & resubmit if needed         │
└────────┬───────────┬─────────────────┘
         │           │
    ┌────▼────┐ ┌────▼─────┐
    │module   │ │reviewer  │
    │generator│ │(Sonnet)  │
    │(parallel│ │Audits 10 │
    │ per mod)│ │quality   │
    └─────────┘ │criteria  │
                └──────────┘
```

## Sub-Agents

- **module_generator** — generates source files for a single module. Can run in parallel.
- **reviewer** — adversarial quality check against 10 criteria (structure, scaffolding, docs, milestones, progressive difficulty, syntax, realism, questions, outcomes, organization). Returns PASS or REVISE.

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
