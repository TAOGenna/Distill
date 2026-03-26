# Source Preprocessing Pipeline

## Problem

Scaffoldly generates coursework from URLs. The agent previously fetched content
with `curl` inside Bash, which returns raw HTML. This works for text-heavy
sources but produces a **degraded representation** of visual-heavy content:

- **Images and diagrams** are completely lost (the agent never sees them)
- **Rendered equations** survive only if the HTML contains LaTeX source
  (MathJax/KaTeX). If equations are rendered to SVG or images, they're gone
- **Layout and visual structure** is lost — the spatial relationship between
  figures, code, and text (which authors use deliberately) disappears

This matters because scaffoldly's best sources — Lilian Weng, Distill, ML
papers, systems engineering blogs — are exactly the kind of content that relies
heavily on figures, equations, and diagrams to communicate ideas.

## Solution

**Preprocess every URL into local artifacts before the agent starts.** A new
`fetch.py` module runs as deterministic Python code (no LLM tokens spent)
before Phase 1 of the agent pipeline. It classifies each URL by pattern,
routes to the appropriate handler, and writes artifacts to `_sources/`.

The agent then reads local files instead of curling URLs.

## Architecture

```
scaffoldly generate <url> --level "..."
        │
        ▼
┌──────────────────────────────────────┐
│  Preprocessing (fetch.py, no LLM)    │
│                                      │
│  URL pattern match → handler:        │
│  ┌─────────────────────────────────┐ │
│  │ arxiv.org/…  → TeX source      │ │
│  │ github.com/… → git clone       │ │
│  │ *.pdf        → download + Jina │ │
│  │ everything   → Playwright +    │ │
│  │   else         Jina Reader     │ │
│  └─────────────────────────────────┘ │
│                                      │
│  Output: _sources/ + manifest.json   │
└──────────────┬───────────────────────┘
               │
               ▼
        Existing 3-phase pipeline
        (agent reads local files)
```

## Source Type Detection

Pure URL pattern matching in `detect_source_type()`:

| Pattern | Type | Handler |
|---------|------|---------|
| `arxiv.org/{abs,pdf,html,src}/<id>` | arxiv | Download TeX source tarball |
| `github.com/<user>/<repo>` | github | `git clone --depth 1` |
| URL ending in `.pdf` | pdf | Download PDF + Jina text extraction |
| Everything else | blog | Playwright (PDF + images) + Jina (markdown) |

## Handlers

### Blog/Article (the main path)

This is the most complex handler and the one that solves the core problem.
Uses Playwright (headless Chromium) as primary, Jina Reader as supplementary.

**Playwright pipeline** (`_playwright_render`):

1. Launch headless Chromium (1280x900 viewport)
2. Navigate to URL, wait for `networkidle`
3. Scroll the entire page incrementally (100ms per viewport step) to trigger
   lazy-loaded images
4. Wait for `networkidle` again after scrolling
5. Wait for MathJax/KaTeX rendering (check for absence of `.MathJax_Processing`)
6. Remove cookie banners/overlays via CSS selector heuristics (`[class*="cookie"]`,
   `[id*="consent"]`, `[class*="gdpr"]`, etc.)
7. Generate PDF with `screen` media (not `print`), A4 format, `print_background=True`
8. Extract images:
   - Query all `<img>` elements
   - Skip images smaller than 50x50px (icons, tracking pixels)
   - Use `element.screenshot()` to capture rendered pixels (handles responsive
     images, CSS transforms, inline SVGs)
   - Deduplicate by SHA-256 hash of screenshot bytes
   - Extract alt text, src URL, and figcaption from parent elements
   - Write to `images/` directory with `manifest.json`
9. Close browser

**Jina Reader** (always runs, regardless of Playwright):

- `GET https://r.jina.ai/{url}` with `Accept: text/markdown`
- Returns clean markdown with headings, code blocks, LaTeX equations preserved
- Saved as `source.md`

**Output:**
```
_sources/focus/
├── source.pdf          # visual rendering (Playwright)
├── source.md           # clean markdown (Jina)
├── images/
│   ├── manifest.json   # [{filename, original_url, alt_text, caption, dimensions}]
│   ├── fig_01.png
│   ├── fig_02.png
│   └── ...
└── meta.json
```

**Fallback chain:**
- Playwright unavailable → skip PDF/images, Jina markdown only
- Playwright fails → log error, continue with Jina markdown
- Jina fails → log error, agent falls back to curl (existing behavior)
- Both fail → agent curls directly (degraded but functional)

### arXiv Papers

arXiv papers have TeX Source available for download. This is the **richest
possible input** — native LaTeX equations, structured sections, figure files
included in the archive. No browser rendering needed.

**Pipeline:**

1. Extract paper ID from URL (`arxiv.org/{abs,pdf,html,src}/<id>`)
2. Download `https://arxiv.org/src/<paper_id>` (single HTTP GET)
3. Response is usually a gzipped tarball, sometimes a single gzipped `.tex` file
4. Extract to `source/` preserving directory structure
5. Flatten if tarball had a single top-level directory (common pattern)
6. Find main `.tex` file by scanning for `\documentclass`
7. Catalog `.tex` files and figure files in `meta.json`

**Tarball extraction safety:** filters out absolute paths and `..` traversal.
Uses Python 3.12+ `filter="data"` when available.

**Fallback:** if `/src` returns non-200 (rare — PDF-only submissions), download
the PDF from `arxiv.org/pdf/<id>` instead.

**Output:**
```
_sources/focus/
├── source/             # extracted tarball, preserving structure
│   ├── main.tex
│   ├── methods.tex
│   ├── figures/
│   │   ├── fig1.pdf
│   │   └── fig2.png
│   └── references.bib
└── meta.json           # {tex_files, figure_files, main_tex, ...}
```

### Direct PDF

1. Download the PDF
2. Try Jina Reader for text extraction (pass PDF URL to Jina)
3. Keep original PDF as-is

**Output:**
```
_sources/focus/
├── source.pdf
├── source.md           # text extraction (if Jina succeeded)
└── meta.json
```

### GitHub Repo

1. `git clone --depth 1`
2. Agent navigates the repo with Read and Bash as before

**Output:**
```
_sources/focus/
├── repo/               # shallow clone
└── meta.json
```

## Multi-Source Handling

Different treatment based on mode and role:

| Source | Mode | Preprocessing |
|--------|------|---------------|
| Focus URL | always | Full pipeline (Playwright + Jina for blogs, TeX for arxiv) |
| Ref URL | reference mode | Markdown only — no Playwright (agent just skims) |
| Ref URL | series mode | Full pipeline (all sources are important) |

Directory layout:
```
_sources/
├── focus/              # primary source
├── series_02/          # series mode: Part 2
├── series_03/          # series mode: Part 3
├── ref_01/             # reference mode: supplementary
├── ref_02/
└── manifest.json       # top-level index
```

## Manifest

Top-level `manifest.json` tells the agent what's available:

```json
{
    "focus": {
        "url": "https://...",
        "dir": "focus",
        "type": "blog",
        "artifacts": ["source.pdf", "source.md", "images/"]
    },
    "series": [],
    "refs": [
        {
            "url": "https://...",
            "dir": "ref_01",
            "type": "blog",
            "artifacts": ["source.md"]
        }
    ]
}
```

For arxiv sources, the manifest also includes `main_tex`, `tex_files`, and
`figure_files` so the agent knows exactly what to read.

## Caching

Before preprocessing a URL, `_is_cached()` checks if `meta.json` exists
for that URL and has artifacts. If so, skip preprocessing and reuse. No
TTL/expiry — user deletes `_sources/` to force re-fetch.

## Integration Points

| File | Change |
|------|--------|
| **`fetch.py`** (new) | The preprocessing module. All source fetching logic. |
| **`pyproject.toml`** | Added `httpx` and `playwright` dependencies. |
| **`cli.py`** | Calls `preprocess_sources()` before `run_agent_sync()`. Passes `sources_dir` and `render` flag. Added `--no-render` CLI flag. |
| **`agent.py`** | Added `sources_dir` parameter. When present, `phase1_prompt` references local files + manifest instead of raw URLs. Falls back to URL-based prompt if manifest is missing. |
| **`system_prompt.py`** | Step 1 changed from FETCH to CONSUME. Teaches agent about each source type and when to use each artifact (markdown for text, PDF for visual context, images for close study). |
| **`tools.py`** | Unchanged. |
| **`schemas.py`** | Unchanged. |

## Dependencies

| Package | Purpose | Install size |
|---------|---------|-------------|
| `httpx` | HTTP client for Jina API, arxiv downloads, PDF downloads | Lightweight |
| `playwright` | Headless Chromium for page rendering, PDF generation, image extraction | Python pkg is small; Chromium binary ~200MB (auto-installed on first run) |

### Chromium Auto-Install

`_ensure_chromium()` checks if Chromium is installed before each Playwright
run. If missing, runs `playwright install chromium` automatically with a
user-visible message:

```
Chromium not found — installing (one-time ~200MB download)...
Chromium installed
```

### `--no-render` Flag

Skips Playwright entirely. Blog sources get Jina markdown only (no PDF, no
images). Useful for CI, minimal containers, or when visual content isn't needed.

```bash
scaffoldly generate https://example.com/post --level "..." --no-render
```

## What the Agent Sees

The system prompt's CONSUME step teaches the agent how to use each source type:

- **ArXiv**: Read `.tex` files directly. Equations are native LaTeX. Follow
  `\input{}` references. Read figure files when studying diagrams.
- **Blog**: Read `source.md` for text/equations (primary). Consult `source.pdf`
  for visual context when needed. Read individual images from `images/` for
  close study of diagrams.
- **PDF**: Read `source.pdf` visually with pages parameter. Use `source.md`
  if available for text.
- **GitHub**: Navigate `repo/` with Read and Bash as before.

## Not Implemented

These items from the original plan were deferred:

- **Figure-to-PDF-page mapping**: computing which PDF page each figure lands on
  and writing `figure_pages` / `pdf_page` fields into the manifest. The agent
  can find figures by reading the PDF visually or checking `images/manifest.json`.
- **`markdownify` fallback**: local HTML-to-markdown conversion when both Jina
  and Playwright fail. Current fallback is the agent curling directly (existing
  behavior).
- **Preprocessing timeout tuning**: the plan discussed 30s page load + 10s
  post-scroll. Currently using Playwright's default `networkidle` with 30s
  `goto` timeout and 10s MathJax wait.
