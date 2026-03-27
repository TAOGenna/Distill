# Scaffoldly

Turn technical blog posts and repos into progressive, hands-on coursework.

Scaffoldly takes expert-level content — deep blog posts, GitHub repos, papers — and generates [CS231n](https://cs231n.stanford.edu/)-style projects with scaffolded exercises, observable milestones, and analytical questions. It adapts its pedagogy to the source material: systems engineering blogs get measurement-driven milestones, ML papers get atom-first exercises with visualization milestones, tutorials get enhanced scaffolding.

The goal: make it possible for mid/junior engineers to actually *learn from* the incredible content that senior engineers and researchers publish, instead of just reading and nodding along.

## The Problem

There are brilliant technical blogs out there — [Andrew Chan](https://andrewkchan.dev/) on GPU optimization and LLM inference, [Wilson Lin](https://blog.wilsonl.in/) on search engines and vector databases — but they're written expert-to-expert. A junior/mid engineer reading them faces:

- **Assumed prerequisites** that aren't explained
- **No exercises** — you read but don't build muscle memory
- **No feedback** — no way to know if you actually understood
- **Integrated multi-domain knowledge** with no clear learning path

**Scaffoldly bridges this gap.**

## Quick Start

```bash
pip install scaffoldly
# or: uv pip install scaffoldly

# Launch the web UI
scaffoldly
# → opens http://localhost:8420 in your browser
# → paste a URL, pick your level, hit generate
```

Or use the CLI directly:

```bash
# Single source
scaffoldly generate \
  "https://andrewkchan.dev/posts/yalm.html" \
  --level "mid-level Python developer, new to systems programming"

# Blog series (Part 1 → Part 2 → Part 3)
scaffoldly generate "https://blog.example.com/crawler-part1" \
  --ref "https://blog.example.com/crawler-part2" \
  --ref "https://blog.example.com/crawler-part3" \
  --series \
  --level "junior Python dev"

# Focus source + supplementary references
scaffoldly generate "https://arxiv.org/abs/main-paper" \
  --ref "https://arxiv.org/abs/background-paper" \
  --ref "https://blog.example.com/practical-take" \
  --level "ML engineer, familiar with transformers"
```

## Usage

### Web UI (recommended)

```bash
scaffoldly                    # launch at localhost:8420
scaffoldly --port 3000        # custom port
scaffoldly --no-open          # don't auto-open browser
```

Paste a URL, describe your level, and hit generate. Progress streams in real time. Generated courses appear in the history and are saved to the output directory.

API key: if you have [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed, auth is automatic. Otherwise, add your `ANTHROPIC_API_KEY` in the settings section of the web UI.

### CLI

```bash
scaffoldly generate <url> \
  --level "describe your current proficiency" \
  [--ref <url>] \
  [--series] \
  [--model claude-opus-4-6] \
  [--effort high] \
  [--output ./output] \
  [--max-turns 50]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--level` | *required* | Free-text description of the student's current level |
| `--ref` | — | Additional reference URL (repeatable). Skimmed for supplementary concepts |
| `--series` | `false` | Treat all URLs as an ordered series (e.g. Part 1 → Part 2 → Part 3) |
| `--model` | `claude-opus-4-6` | Model for analysis and curriculum design |
| `--generate-model` | `sonnet` | Model for module generation sub-agents |
| `--effort` | `high` | Agent effort level: `low`, `medium`, `high`, `max` |
| `--output` | `./output` | Output directory for generated course |
| `--max-turns` | `30` | Maximum agent turns per phase |

**Reference mode** (default with `--ref`): The primary URL drives the curriculum. Refs are skimmed with minimal effort for supplementary concepts only.

**Series mode** (`--series`): All sources form an ordered progression. The curriculum spans the full arc.

## What You Get

A real, cloneable project that a student works through module by module:

```
output/billion_page_crawler/
├── README.md                        # Setup, learning path, what's next
├── requirements.txt
├── module_01_fetching/
│   ├── README.md                    # Exercises + analytical questions
│   ├── ex01_sync_fetcher.py
│   ├── ex02_async_fetcher.py
│   └── ex03_worker_scaling.py
├── module_02_politeness/
│   └── ...
└── module_03_frontier/
    └── ...
```

Each exercise is scaffolded code with TODO markers and an **observable milestone** — run it, see the numbers, discover the same insights the author discovered:

```python
if __name__ == "__main__":
    results = asyncio.run(fetch_pages(SEED_URLS, max_workers=10))
    throughput = len(ok) / elapsed
    print(f"Throughput: {throughput:.1f} pages/sec")
    print(f">> The blog needed 11,574 pages/sec to crawl 1B in 24hrs.")
    print(f">> Next exercise: what happens at 100 workers?")
```

No test framework — the output *is* the feedback. Works with any language (Python, C, Rust, Go).

The course README includes a **Learning Path** (module dependencies) and a **What's Next** section bridging to advanced topics not covered in exercises.

Module READMEs include **analytical questions** that push beyond recall into tradeoff reasoning:

- *"At 950 pages/sec with 250KB pages, what's your worst-case write bandwidth?"*
- *"At what concurrency level does throughput stop increasing? What's the bottleneck?"*
- *"The author chose a bloom filter over a hash set. At what scale does this pay off?"*

## Requirements

- Python 3.10+
- [Claude Code](https://claude.ai/code) (provides the Agent SDK)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Acknowledgments

Inspired by [karpathify](https://github.com/nuwandavek/karpathify) and Stanford's [CS231n](https://cs231n.stanford.edu/) assignments.
