"""Source preprocessing — fetch URLs into local artifacts before the agent starts.

Routes each URL by pattern (arxiv, github, pdf, blog) and produces a
_sources/ directory with artifacts the agent can consume locally.

Blog/article sources get the full Playwright pipeline: browser rendering →
PDF (screen media), image extraction via element.screenshot(), plus Jina
Reader for clean markdown. ArXiv papers get TeX source directly.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import httpx

try:
    from playwright.sync_api import sync_playwright

    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False

SourceType = Literal["arxiv", "github", "pdf", "blog"]

_FIGURE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg", ".gif"})
_MIN_IMAGE_DIM = 50  # skip images smaller than 50x50 (icons, tracking pixels)


# ── Source type detection ────────────────────────────────────────────────────


def detect_source_type(url: str) -> tuple[SourceType, dict[str, str]]:
    """Classify a URL by pattern matching."""
    # arXiv: abs, pdf, html, src URLs
    m = re.match(
        r"https?://(?:export\.)?arxiv\.org/(?:abs|pdf|html|src)/(\d{4}\.\d{4,5}(?:v\d+)?)",
        url,
    )
    if m:
        return "arxiv", {"paper_id": m.group(1)}

    # GitHub repo
    m = re.match(r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?(?:/|$)", url)
    if m:
        return "github", {"repo": m.group(1)}

    # Direct PDF link
    if url.lower().rstrip("/").endswith(".pdf"):
        return "pdf", {}

    return "blog", {}


# ── Handlers ─────────────────────────────────────────────────────────────────


def _fetch_arxiv(paper_id: str, dest: Path, log: Callable) -> dict[str, Any]:
    """Download arXiv TeX source, extract preserving structure."""
    src_url = f"https://arxiv.org/src/{paper_id}"
    meta: dict[str, Any] = {
        "url": f"https://arxiv.org/abs/{paper_id}",
        "type": "arxiv",
        "paper_id": paper_id,
        "fetched_at": _now(),
        "artifacts": [],
        "errors": [],
    }

    source_dir = dest / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    log(f"Downloading TeX source for {paper_id}...")
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        resp = client.get(src_url)

    if resp.status_code != 200:
        meta["errors"].append(f"arxiv /src returned {resp.status_code}")
        log("TeX source unavailable, falling back to PDF...")
        return _fallback_arxiv_pdf(paper_id, dest, meta, log)

    content = resp.content

    # Try tarball first (most common)
    extracted = False
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            members = [
                m
                for m in tar.getmembers()
                if not m.name.startswith("/") and ".." not in m.name.split("/")
            ]
            _safe_extractall(tar, source_dir, members)
        extracted = True
        log(f"Extracted {len(members)} files from tarball")
    except (tarfile.TarError, gzip.BadGzipFile, EOFError):
        pass

    if not extracted:
        # Single gzipped .tex file
        try:
            text = gzip.decompress(content).decode("utf-8", errors="replace")
            (source_dir / "main.tex").write_text(text)
            extracted = True
            log("Extracted single .tex file")
        except Exception:
            meta["errors"].append("Could not extract source")
            return _fallback_arxiv_pdf(paper_id, dest, meta, log)

    # Flatten if tarball had a single top-level directory
    _flatten_single_subdir(source_dir)

    # Catalog what we got
    tex_files = sorted(source_dir.rglob("*.tex"))
    figure_files = sorted(
        f
        for f in source_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in _FIGURE_EXTS
    )

    meta["artifacts"].append("source/")
    meta["tex_files"] = [str(f.relative_to(source_dir)) for f in tex_files]
    meta["figure_files"] = [str(f.relative_to(source_dir)) for f in figure_files]

    # Find main .tex file
    main_tex = _find_main_tex(tex_files)
    if main_tex:
        meta["main_tex"] = str(main_tex.relative_to(source_dir))

    log(f"Found {len(tex_files)} .tex files, {len(figure_files)} figures")
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _fetch_blog(
    url: str, dest: Path, log: Callable, *, render: bool = True,
) -> dict[str, Any]:
    """Fetch blog content: Playwright (PDF + images) + Jina (markdown).

    When render=True and Playwright is available, launches headless Chromium
    to capture the page as the reader sees it. Falls back to Jina-only if
    Playwright is unavailable or fails.
    """
    meta: dict[str, Any] = {
        "url": url,
        "type": "blog",
        "fetched_at": _now(),
        "artifacts": [],
        "errors": [],
    }

    # ── Playwright: PDF + images ─────────────────────────────────────────
    if render and _HAS_PLAYWRIGHT:
        try:
            _playwright_render(url, dest, meta, log)
        except Exception as e:
            meta["errors"].append(f"Playwright failed: {e}")
            log(f"Playwright failed: {e}")
    elif render and not _HAS_PLAYWRIGHT:
        meta["errors"].append("Playwright not installed, skipping render")
        log("Playwright not installed — run: pip install playwright && playwright install chromium")

    # ── Jina Reader: markdown ────────────────────────────────────────────
    log("Fetching markdown via Jina Reader...")
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        try:
            resp = client.get(
                f"https://r.jina.ai/{url}",
                headers={"Accept": "text/markdown"},
            )
            if resp.status_code == 200 and len(resp.text.strip()) > 100:
                (dest / "source.md").write_text(resp.text)
                meta["artifacts"].append("source.md")
                meta["markdown_source"] = "jina"
                meta["markdown_length_chars"] = len(resp.text)
                log(f"Got {len(resp.text)} chars of markdown")
            else:
                meta["errors"].append(
                    f"Jina returned {resp.status_code} or too short"
                )
                log("Jina failed, agent will curl directly")
        except httpx.RequestError as e:
            meta["errors"].append(f"Jina request failed: {e}")
            log("Jina unreachable, agent will curl directly")

    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _playwright_render(
    url: str, dest: Path, meta: dict, log: Callable,
) -> None:
    """Use Playwright to render the page, generate PDF, and extract images."""
    _ensure_chromium(log)

    log("Launching browser...")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        # Navigate and wait for content
        log("Loading page...")
        page.goto(url, wait_until="networkidle", timeout=30_000)

        # Scroll to trigger lazy-loaded images
        page.evaluate("""
            async () => {
                const delay = ms => new Promise(r => setTimeout(r, ms));
                const height = document.body.scrollHeight;
                const step = window.innerHeight;
                for (let y = 0; y < height; y += step) {
                    window.scrollTo(0, y);
                    await delay(100);
                }
                window.scrollTo(0, 0);
            }
        """)
        page.wait_for_load_state("networkidle")

        # Wait for MathJax/KaTeX rendering
        page.wait_for_function(
            "() => !document.querySelector('.MathJax_Processing')",
            timeout=10_000,
        )

        # Remove cookie banners / overlays
        page.evaluate("""
            () => {
                const selectors = [
                    '[class*="cookie"]', '[id*="cookie"]',
                    '[class*="consent"]', '[id*="consent"]',
                    '[class*="gdpr"]',
                ];
                for (const sel of selectors) {
                    document.querySelectorAll(sel).forEach(el => el.remove());
                }
            }
        """)

        # ── Generate PDF (screen media, not print) ──────────────────────
        pdf_path = dest / "source.pdf"
        page.emulate_media(media="screen")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "0.5in", "right": "0.5in",
                    "bottom": "0.5in", "left": "0.5in"},
        )
        meta["artifacts"].append("source.pdf")
        log(f"PDF saved ({pdf_path.stat().st_size // 1024}KB)")

        # ── Extract images ──────────────────────────────────────────────
        images_dir = dest / "images"
        images_dir.mkdir(exist_ok=True)

        img_elements = page.query_selector_all("img")
        image_manifest = []
        seen_hashes: set[str] = set()
        fig_num = 0

        for img in img_elements:
            bbox = img.bounding_box()
            if not bbox:
                continue
            if bbox["width"] < _MIN_IMAGE_DIM or bbox["height"] < _MIN_IMAGE_DIM:
                continue

            # Screenshot the element as rendered
            screenshot_bytes = img.screenshot()
            if not screenshot_bytes:
                continue

            # Deduplicate by content hash
            img_hash = hashlib.sha256(screenshot_bytes).hexdigest()[:16]
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            fig_num += 1
            filename = f"fig_{fig_num:02d}.png"
            (images_dir / filename).write_bytes(screenshot_bytes)

            # Extract context
            alt = img.get_attribute("alt") or ""
            src = img.get_attribute("src") or ""
            caption = img.evaluate("""
                el => {
                    const parent = el.closest('figure, .figure, p, section, article') || el.parentElement;
                    const cap = parent?.querySelector('figcaption')?.textContent?.trim() || '';
                    return cap;
                }
            """)

            image_manifest.append({
                "filename": filename,
                "original_url": src,
                "alt_text": alt,
                "caption": caption,
                "dimensions": {
                    "width": int(bbox["width"]),
                    "height": int(bbox["height"]),
                },
            })

        if image_manifest:
            (images_dir / "manifest.json").write_text(
                json.dumps(image_manifest, indent=2)
            )
            meta["artifacts"].append("images/")
            meta["images_extracted"] = len(image_manifest)
            log(f"Extracted {len(image_manifest)} images")
        else:
            log("No content images found")

        browser.close()


def _fetch_pdf(url: str, dest: Path, log: Callable) -> dict[str, Any]:
    """Download a PDF and optionally extract text via Jina."""
    meta: dict[str, Any] = {
        "url": url,
        "type": "pdf",
        "fetched_at": _now(),
        "artifacts": [],
        "errors": [],
    }

    log("Downloading PDF...")
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        resp = client.get(url)
        if resp.status_code == 200:
            (dest / "source.pdf").write_bytes(resp.content)
            meta["artifacts"].append("source.pdf")
            log(f"Downloaded {len(resp.content) // 1024}KB")

            # Try Jina for text extraction
            try:
                jina_resp = client.get(
                    f"https://r.jina.ai/{url}",
                    headers={"Accept": "text/markdown"},
                    timeout=30,
                )
                if jina_resp.status_code == 200 and len(jina_resp.text.strip()) > 100:
                    (dest / "source.md").write_text(jina_resp.text)
                    meta["artifacts"].append("source.md")
            except httpx.RequestError:
                pass  # PDF alone is fine
        else:
            meta["errors"].append(f"Download failed: {resp.status_code}")

    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _fetch_github(repo: str, dest: Path, log: Callable) -> dict[str, Any]:
    """Shallow clone a GitHub repo."""
    meta: dict[str, Any] = {
        "url": f"https://github.com/{repo}",
        "type": "github",
        "repo": repo,
        "fetched_at": _now(),
        "artifacts": [],
        "errors": [],
    }

    repo_dir = dest / "repo"
    log(f"Cloning {repo}...")
    result = subprocess.run(
        [
            "git", "clone", "--depth", "1",
            f"https://github.com/{repo}.git", str(repo_dir),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode == 0:
        meta["artifacts"].append("repo/")
        log("Cloned successfully")
    else:
        meta["errors"].append(f"git clone failed: {result.stderr.strip()}")
        log(f"Clone failed: {result.stderr.strip()[:100]}")

    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ── Orchestrator ─────────────────────────────────────────────────────────────


def preprocess_sources(
    focus_url: str,
    refs: list[str] | None = None,
    series: bool = False,
    output_dir: str = "./output",
    log: Callable | None = None,
    render: bool = True,
) -> Path:
    """Preprocess source URLs into local artifacts.

    Returns the _sources/ directory path.
    """
    if log is None:
        log = lambda msg: print(f"  {msg}", file=sys.stderr)

    sources_dir = Path(output_dir) / "_sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"focus": None, "series": [], "refs": []}

    # Focus URL
    focus_dest = sources_dir / "focus"
    if _is_cached(focus_dest, focus_url):
        log(f"Using cached source for {focus_url}")
        cached_meta = json.loads((focus_dest / "meta.json").read_text())
        manifest["focus"] = _manifest_entry(focus_url, "focus", cached_meta)
    else:
        focus_dest.mkdir(parents=True, exist_ok=True)
        focus_meta = _fetch_source(focus_url, focus_dest, log, render=render)
        manifest["focus"] = _manifest_entry(focus_url, "focus", focus_meta)

    # Additional sources
    if refs:
        for i, ref_url in enumerate(refs):
            dir_name = f"series_{i + 2:02d}" if series else f"ref_{i + 1:02d}"
            ref_dest = sources_dir / dir_name

            if _is_cached(ref_dest, ref_url):
                log(f"Using cached source for {ref_url}")
                cached_meta = json.loads((ref_dest / "meta.json").read_text())
                entry = _manifest_entry(ref_url, dir_name, cached_meta)
            else:
                ref_dest.mkdir(parents=True, exist_ok=True)
                # In reference mode, only render the focus source fully;
                # refs get markdown-only (agent just skims them)
                ref_render = render if series else False
                ref_meta = _fetch_source(ref_url, ref_dest, log, render=ref_render)
                entry = _manifest_entry(ref_url, dir_name, ref_meta)

            if series:
                manifest["series"].append(entry)
            else:
                manifest["refs"].append(entry)

    (sources_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return sources_dir


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fetch_source(
    url: str, dest: Path, log: Callable, *, render: bool = True,
) -> dict[str, Any]:
    """Route to the appropriate handler."""
    source_type, metadata = detect_source_type(url)
    log(f"Detected source type: {source_type}")

    if source_type == "arxiv":
        return _fetch_arxiv(metadata["paper_id"], dest, log)
    elif source_type == "github":
        return _fetch_github(metadata["repo"], dest, log)
    elif source_type == "pdf":
        return _fetch_pdf(url, dest, log)
    else:
        return _fetch_blog(url, dest, log, render=render)


def _find_main_tex(tex_files: list[Path]) -> Path | None:
    """Find the main .tex file (contains \\documentclass)."""
    for f in tex_files:
        try:
            if r"\documentclass" in f.read_text(errors="replace"):
                return f
        except OSError:
            continue
    return max(tex_files, key=lambda f: f.stat().st_size) if tex_files else None


def _flatten_single_subdir(directory: Path) -> None:
    """If directory contains a single subdirectory, move its contents up."""
    contents = list(directory.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        subdir = contents[0]
        for item in list(subdir.iterdir()):
            item.rename(directory / item.name)
        subdir.rmdir()


def _fallback_arxiv_pdf(
    paper_id: str, dest: Path, meta: dict, log: Callable,
) -> dict:
    """Download the PDF when TeX source isn't available."""
    pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        resp = client.get(pdf_url)
    if resp.status_code == 200:
        (dest / "source.pdf").write_bytes(resp.content)
        meta["artifacts"].append("source.pdf")
        meta["fallback"] = "pdf"
        log(f"Downloaded PDF ({len(resp.content) // 1024}KB)")
    else:
        meta["errors"].append(f"PDF download failed: {resp.status_code}")
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _safe_extractall(
    tar: tarfile.TarFile, dest: Path, members: list[tarfile.TarInfo],
) -> None:
    """Extract tarball members safely."""
    kwargs: dict[str, Any] = {}
    if sys.version_info >= (3, 12):
        kwargs["filter"] = "data"
    tar.extractall(dest, members=members, **kwargs)


def _is_cached(dest: Path, url: str) -> bool:
    """Check if we already have artifacts for this URL."""
    meta_path = dest / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        return meta.get("url") == url and bool(meta.get("artifacts"))
    except (json.JSONDecodeError, OSError):
        return False


def _manifest_entry(url: str, dir_name: str, meta: dict) -> dict:
    """Build a manifest entry from source metadata."""
    entry: dict[str, Any] = {
        "url": url,
        "dir": dir_name,
        "type": meta["type"],
        "artifacts": meta.get("artifacts", []),
    }
    if "main_tex" in meta:
        entry["main_tex"] = meta["main_tex"]
    if "tex_files" in meta:
        entry["tex_files"] = meta["tex_files"]
    if "figure_files" in meta:
        entry["figure_files"] = meta["figure_files"]
    return entry


def _ensure_chromium(log: Callable) -> None:
    """Install Chromium browser if not present."""
    try:
        from playwright._impl._driver import compute_driver_executable  # noqa: F401

        # Check if chromium is already installed by trying to find the executable
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True, text=True, timeout=10,
        )
        # If --dry-run isn't supported, just check if launch works
        if result.returncode != 0 and "already installed" not in result.stdout.lower():
            raise FileNotFoundError
    except Exception:
        log("Chromium not found — installing (one-time ~200MB download)...")
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            timeout=300, check=True,
        )
        log("Chromium installed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
