# Distill

**Paste a URL. Pick your level. Get a course.**

Distill takes expert-level content — deep blog posts, arXiv papers, GitHub repos — and generates progressive courses with lesson documents, scaffolded exercises, and observable milestones. Each course builds toward reproducing the author's results.

<div align="center">

https://github.com/user-attachments/assets/53a5ba44-a4a9-434b-914b-2be105dc6901

</div>

> **Lesson documents, not summaries** — 3,000–10,000 word teaching documents with running examples, inline code, comprehension checks, and formula walkthroughs.

https://github.com/user-attachments/assets/4e1466f1-d883-4351-bbf4-c586f51415bd

> **Exercises you can actually run** — scaffolded `.py` files with TODO blocks, docstrings, and a `__main__` test harness. Solutions included.

https://github.com/user-attachments/assets/ec7a532d-f290-4788-9c03-1e4564110d75

## Install

```bash
git clone https://github.com/TAOGenna/Distill.git
cd Distill
uv sync
uv run python -m distill
# → opens http://localhost:8420
```

**macOS desktop app** (optional):

```bash
./build_app.sh
cp -R dist/Distill.app /Applications/
```

## How It Works

**Phase 1 — Blueprint.** Reads the full source material and produces a curriculum: module dependencies, scaffold contracts per exercise (what's provided vs what the student writes), key excerpts, and validation criteria.

**Phase 2 — Generate.** Each module gets a multi-turn conversation with the full source. The model writes the lesson first (deep processing), then exercises one at a time. Solutions are executed between turns — real output from exercise 1 feeds into exercise 2's prompt. Modules generate in parallel.

**Phase 3 — Review.** Pre-flight checks (syntax, TODOs, output patterns) catch structural issues. LLM review checks pedagogical quality and contract compliance. Failed modules are re-generated.

## Providers

| Provider | Design model | Generate model |
|---|---|---|
| Anthropic | claude-opus-4-6 | claude-sonnet-4-6 |
| OpenAI | gpt-5.4 | gpt-5.4 |
| Google | gemini-2.5-pro | gemini-2.5-flash |
| Ollama | llama3 | llama3 |
| OpenRouter | claude-opus-4-6 | claude-sonnet-4-6 |
| Claude Code | opus | sonnet |

## Acknowledgments

Inspired by [karpathify](https://github.com/nuwandavek/karpathify), Stanford's [CS231n](https://cs231n.stanford.edu/) assignments, and [MIT 6.102](https://web.mit.edu/6.102/www/sp26/) course readings.
