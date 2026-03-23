# Scaffoldly

Turn technical blog posts and repos into progressive, hands-on coursework.

Scaffoldly takes expert-level content — deep blog posts, GitHub repos, papers — and generates [CS231n](https://cs231n.stanford.edu/)-style projects with scaffolded exercises, observable milestones, and inline conceptual questions. The goal: make it possible for mid/junior engineers to actually *learn from* the incredible content that senior engineers publish, instead of just reading and nodding along.

## The Problem

There are brilliant technical blogs out there — [Andrew Chan](https://andrewkchan.dev/) on GPU optimization and LLM inference, [Wilson Lin](https://blog.wilsonl.in/) on search engines and vector databases — but they're written expert-to-expert. A junior/mid engineer reading them faces:

- **Assumed prerequisites** that aren't explained
- **No exercises** — you read but don't build muscle memory
- **No feedback** — no way to know if you actually understood
- **Integrated multi-domain knowledge** with no clear learning path

Meanwhile, structured courses like Stanford CS231n are incredibly effective because they provide scaffolded code with `TODO` markers, progressive difficulty, observable results, and conceptual questions.

**Scaffoldly bridges this gap.**

## Quick Start

```bash
# Generate a course (requires Claude Code)
uv run scaffoldly generate \
  "https://andrewkchan.dev/posts/yalm.html" \
  --level "mid-level Python developer, new to systems programming"
```

## Usage

```bash
uv run scaffoldly generate <url> \
  --level "describe your current proficiency" \
  [--model claude-opus-4-6] \
  [--effort high] \
  [--output ./output] \
  [--max-turns 50]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--level` | *required* | Free-text description of the student's current level |
| `--model` | `claude-opus-4-6` | Claude model to use |
| `--effort` | `high` | Agent effort level: `low`, `medium`, `high`, `max` |
| `--output` | `./output` | Output directory for generated course |
| `--max-turns` | `50` | Maximum agent turns before stopping |

### Examples

```bash
# Minimal
uv run scaffoldly generate "https://blog.example.com/post" --level "junior Python dev"

# Detailed level, max effort
uv run scaffoldly generate "https://github.com/user/repo" \
  --level "senior backend engineer with 5 years of Go, but zero ML experience" \
  --effort max

# Custom output directory
uv run scaffoldly generate "https://blog.example.com/post" \
  --level "CS undergrad, knows basic Python and linear algebra" \
  --output ~/my-courses
```

## What the Output Looks Like

The agent generates a **real project** with proper file structure. The language and organization match the source material (Python, C, Rust, etc.):

```
output/billion_page_crawler/
├── README.md                        # Course overview, setup, module order
├── requirements.txt                 # Dependencies
├── _analysis.json                   # Extracted concepts & prerequisites
├── _curriculum.json                 # Course design
├── module_01_fetching/
│   ├── README.md                    # Module intro & exercise guide
│   ├── ex01_sync_fetcher.py         # Scaffolded exercise with TODOs + milestone
│   ├── ex02_async_fetcher.py        # Builds on exercise 01
│   └── ex03_worker_scaling.py       # Builds on 01 and 02
├── module_02_politeness/
│   └── ...
└── module_03_frontier/
    └── ...
```

Exercises use scaffolded code with TODO markers and end with an **observable milestone** — a `__main__` block that runs the student's code and prints output that teaches something:

```python
def fetch_pages(urls, max_workers=1):
    """Fetch URLs concurrently with asyncio.

    The approach:
    1. Create an aiohttp ClientSession
    2. Use asyncio.gather with a semaphore to limit concurrency
    3. Return list of (url, status, content_length)
    """
    # ======================================================================
    # TODO: Implement async fetching
    #
    # Hint: Use aiohttp.ClientSession with asyncio.gather
    # ======================================================================
    raise NotImplementedError("Implement this function")
    # ======================================================================

if __name__ == "__main__":
    start = time.time()
    results = asyncio.run(fetch_pages(SEED_URLS, max_workers=10))
    elapsed = time.time() - start
    ok = [r for r in results if r[1] == 200]
    throughput = len(ok) / elapsed
    print(f"Fetched {len(ok)}/{len(SEED_URLS)} pages in {elapsed:.1f}s")
    print(f"Throughput: {throughput:.1f} pages/sec")
    print()
    print(f">> The blog needed 11,574 pages/sec to crawl 1B in 24hrs.")
    print(f">> Next exercise: what happens when you scale to 100 workers?")
```

The student runs `python ex02_async_fetcher.py`, sees the numbers, and discovers the same insights the blog author discovered. No test framework needed — the output *is* the feedback.

Works with any language — C, Rust, Go, etc. The agent decides the right structure for the domain.

## How It Works

Scaffoldly is powered by the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python), which runs Claude Code as an autonomous agent with full tool access.

### Architecture

```
scaffoldly generate <url> --level "..."
        │
        ▼
┌─────────────────────────────────┐
│  Main Agent (Claude Code)       │
│  System prompt: CS231n pedagogy │
│                                 │
│  1. Fetch source material       │
│  2. Analyze → submit_analysis   │
│  3. Design → submit_curriculum  │
│  4. Generate → Write files       │
│  5. Review (adversarial QA)     │
│  6. Fix & resubmit if needed    │
└────────┬───────────┬────────────┘
         │           │
    ┌────▼────┐ ┌────▼─────┐
    │module   │ │reviewer  │
    │generator│ │(Sonnet)  │
    │(parallel│ │Audits 10 │
    │ per mod)│ │quality   │
    └─────────┘ │criteria  │
                └──────────┘
```

### Sub-Agents

- **module_generator** — generates source files for a single module. The main agent can dispatch multiple in parallel for speed.
- **reviewer** — adversarial reviewer that audits generated files against 10 quality criteria (project structure, scaffolding, documentation, milestones, progressive difficulty, compilation/syntax, conceptual questions, etc.). Returns PASS or REVISE.

### Custom Tools

| Tool | Purpose |
|------|---------|
| `submit_analysis` | Structured analysis with Pydantic validation |
| `submit_curriculum` | Curriculum design, creates course directory |

The agent uses Claude Code's built-in tools (Bash, Read, Write, Edit) to create all course files directly — source code, READMEs, Makefiles, etc.

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

## Requirements

- Python 3.10+
- [Claude Code](https://claude.ai/code) (bundled with the Agent SDK)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Acknowledgments

Inspired by [karpathify](https://github.com/nuwandavek/karpathify) and Stanford's [CS231n](https://cs231n.stanford.edu/) assignments.
