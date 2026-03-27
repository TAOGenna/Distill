# Scaffoldly

Turn technical blog posts, papers, and repos into progressive, hands-on coursework.

Scaffoldly takes expert-level content — deep blog posts, GitHub repos, papers — and generates [CS231n](https://cs231n.stanford.edu/)-style projects with scaffolded exercises, observable milestones, and analytical questions. It adapts its pedagogy to the source material: systems engineering blogs get measurement-driven milestones, ML papers get atom-first exercises with visualization milestones, tutorials get enhanced scaffolding.

The goal: make it possible for anyone to actually *learn from* the incredible content that senior engineers and researchers publish, instead of just reading and nodding along.

## Quick Start

```bash
pip install scaffoldly
# or: uv pip install scaffoldly

# Launch the web UI
scaffoldly
# → opens http://localhost:8420
# → paste a URL, pick your level, hit generate
```

That's it. A local web UI opens where you paste a URL, describe your level, and generate a full course. If you have [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed, auth is automatic — no API key needed.

## Usage

### Web UI (recommended)

```bash
scaffoldly                    # launch at localhost:8420
scaffoldly --port 3000        # custom port
scaffoldly --no-open          # don't auto-open browser
```

**What you see:**

1. Paste a source URL, add optional references, describe your level
2. Hit generate — logs stream in real time in a scrollable box
3. Once the curriculum is designed, a **DAG visualization** appears showing the module dependency graph (Brilliant-style flowing path with animated nodes)
4. Modules generate in parallel — each node lights up as its code is ready
5. The finished course appears in your output directory and in the course history

**Auth:** Claude Code is auto-detected. Otherwise, add your `ANTHROPIC_API_KEY` in the settings section at the bottom of the page.

### CLI

```bash
scaffoldly generate <url> \
  --level "describe your current proficiency" \
  [--ref <url>] \
  [--series] \
  [--model claude-opus-4-6] \
  [--generate-model sonnet] \
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

No test framework — the output *is* the feedback.

Module READMEs include **analytical questions** that push beyond recall into tradeoff reasoning:

- *"At 950 pages/sec with 250KB pages, what's your worst-case write bandwidth?"*
- *"At what concurrency level does throughput stop increasing? What's the bottleneck?"*
- *"The author chose a bloom filter over a hash set. At what scale does this pay off?"*

## How It Works

1. **Preprocess** — URLs are fetched into local artifacts (arXiv → TeX source, blogs → markdown + images, GitHub → shallow clone). No LLM tokens spent here.
2. **Analyze & Design** — An Opus agent identifies key concepts, triages them (essential/supporting/contextual), and designs a curriculum with module dependencies. The web UI shows the dependency DAG at this point.
3. **Generate** — Module generators run in parallel (Sonnet by default). Each module gets scaffolded exercises, observable milestones, and analytical questions. Nodes in the DAG light up as each module finishes.
4. **Review** — An adversarial reviewer checks 10 quality criteria. If issues are found, the agent fixes and re-reviews.

## Requirements

- Python 3.10+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (provides the Agent SDK + automatic auth) or an `ANTHROPIC_API_KEY`
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Development

```bash
git clone https://github.com/TAOGenna/scaffoldly.git
cd scaffoldly
uv sync
uv run scaffoldly
```

Test the DAG visualization without running a generation:
```
http://localhost:8420/test_dag.html
```

## Acknowledgments

Inspired by [karpathify](https://github.com/nuwandavek/karpathify) and Stanford's [CS231n](https://cs231n.stanford.edu/) assignments.
