"""Source preprocessing — fetch URLs into local artifacts before the agent starts.

Routes each URL by pattern (arxiv, github, pdf, blog) and produces a
_sources/ directory with artifacts the agent can consume locally.

ArXiv papers get TeX source directly. Blogs get clean markdown via Jina
Reader plus images downloaded from the markdown. PDFs are downloaded.
GitHub repos are shallow-cloned.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urljoin, urlparse

import httpx

SourceType = Literal["arxiv", "github", "github_file", "pdf", "blog"]

_FIGURE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg", ".gif"})
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"})
_MIN_IMAGE_BYTES = 2048  # skip tiny images (icons, tracking pixels, spacers)


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

    # GitHub single file — /blob/{branch}/{path}
    m = re.match(
        r"https?://github\.com/([^/]+/[^/]+?)/blob/([^/]+)/(.+)", url
    )
    if m:
        return "github_file", {
            "repo": m.group(1),
            "branch": m.group(2),
            "path": m.group(3),
        }

    # GitHub repo (or /tree/ directory view)
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
        log("TeX source unavailable, falling back to PDF...", "warn")
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
        log(f"Extracted {len(members)} files from tarball", "ok")
    except (tarfile.TarError, gzip.BadGzipFile, EOFError):
        pass

    if not extracted:
        # Single gzipped .tex file
        try:
            text = gzip.decompress(content).decode("utf-8", errors="replace")
            (source_dir / "main.tex").write_text(text)
            extracted = True
            log("Extracted single .tex file", "ok")
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

    log(f"Found {len(tex_files)} .tex files, {len(figure_files)} figures", "ok")
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _fetch_blog(url: str, dest: Path, log: Callable) -> dict[str, Any]:
    """Fetch blog content: Jina Reader for markdown, then download images from it."""
    meta: dict[str, Any] = {
        "url": url,
        "type": "blog",
        "fetched_at": _now(),
        "artifacts": [],
        "errors": [],
    }

    # ── Jina Reader: markdown ────────────────────────────────────────────
    markdown = None
    log("Fetching markdown via Jina Reader...")
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        try:
            resp = client.get(
                f"https://r.jina.ai/{url}",
                headers={"Accept": "text/markdown"},
            )
            if resp.status_code == 200 and len(resp.text.strip()) > 100:
                markdown = resp.text
                (dest / "source.md").write_text(markdown)
                meta["artifacts"].append("source.md")
                meta["markdown_length_chars"] = len(markdown)
                log(f"Got {len(markdown)} chars of markdown", "ok")
            else:
                meta["errors"].append(
                    f"Jina returned {resp.status_code} or too short"
                )
                log("Jina failed, agent will curl directly", "warn")
        except httpx.RequestError as e:
            meta["errors"].append(f"Jina request failed: {e}")
            log("Jina unreachable, agent will curl directly", "error")

    # ── Extract and download images from markdown ────────────────────────
    if markdown:
        images = _download_images_from_markdown(markdown, url, dest, log)
        if images:
            meta["artifacts"].append("images/")
            meta["images_downloaded"] = len(images)

    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _download_images_from_markdown(
    markdown: str, base_url: str, dest: Path, log: Callable,
) -> list[dict]:
    """Parse image URLs from markdown, download them, return manifest entries."""
    # Match markdown image syntax: ![alt](url)
    img_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    matches = img_pattern.findall(markdown)
    if not matches:
        return []

    images_dir = dest / "images"
    images_dir.mkdir(exist_ok=True)
    manifest = []
    seen_urls: set[str] = set()
    fig_num = 0

    with httpx.Client(follow_redirects=True, timeout=15) as client:
        for alt_text, img_url in matches:
            # Resolve relative URLs
            resolved = urljoin(base_url, img_url)

            # Skip duplicates and data URIs
            if resolved in seen_urls or resolved.startswith("data:"):
                continue
            seen_urls.add(resolved)

            # Skip non-image URLs
            parsed = urlparse(resolved)
            ext = Path(parsed.path).suffix.lower()
            if ext and ext not in _IMAGE_EXTS:
                continue

            try:
                resp = client.get(resolved)
                if resp.status_code != 200:
                    continue

                # Skip tiny images (icons, spacers, tracking pixels)
                if len(resp.content) < _MIN_IMAGE_BYTES:
                    continue

                # Determine extension from content-type if URL didn't have one
                if not ext:
                    ct = resp.headers.get("content-type", "")
                    ext = _ext_from_content_type(ct) or ".png"

                fig_num += 1
                filename = f"fig_{fig_num:02d}{ext}"
                (images_dir / filename).write_bytes(resp.content)

                manifest.append({
                    "filename": filename,
                    "original_url": resolved,
                    "alt_text": alt_text,
                })
            except (httpx.RequestError, httpx.HTTPStatusError):
                continue

    if manifest:
        (images_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log(f"Downloaded {len(manifest)} images", "ok")
    else:
        log("No content images found", "warn")

    return manifest


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
            log(f"Downloaded {len(resp.content) // 1024}KB", "ok")

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
            log(f"Download failed: {resp.status_code}", "error")

    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _fetch_github_file(
    repo: str, branch: str, path: str, dest: Path, log: Callable,
) -> dict[str, Any]:
    """Fetch a single file from a GitHub repo via raw.githubusercontent.com."""
    raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    meta: dict[str, Any] = {
        "url": f"https://github.com/{repo}/blob/{branch}/{path}",
        "type": "github_file",
        "repo": repo,
        "branch": branch,
        "path": path,
        "fetched_at": _now(),
        "artifacts": [],
        "errors": [],
    }

    filename = Path(path).name
    log(f"Fetching {repo}/{path}...")

    with httpx.Client(follow_redirects=True, timeout=30) as client:
        resp = client.get(raw_url)

    if resp.status_code != 200:
        meta["errors"].append(f"Fetch failed: {resp.status_code}")
        log(f"Failed to fetch {path}: {resp.status_code}", "error")
        (dest / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    content = resp.text
    (dest / filename).write_text(content)
    meta["artifacts"].append(filename)
    meta["filename"] = filename
    log(f"Got {len(content)} chars from {filename}", "ok")

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
        log("Cloned successfully", "ok")
    else:
        meta["errors"].append(f"git clone failed: {result.stderr.strip()}")
        log(f"Clone failed: {result.stderr.strip()[:100]}", "error")

    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ── Orchestrator ─────────────────────────────────────────────────────────────


def preprocess_sources(
    focus_url: str,
    refs: list[str] | None = None,
    output_dir: str = "./output",
    log: Callable | None = None,
    ref_annotations: list[str] | None = None,
) -> Path:
    """Preprocess source URLs into local artifacts.

    Returns the _sources/ directory path.
    """
    if log is None:
        log = lambda msg, level="info": print(f"  {msg}", file=sys.stderr)

    sources_dir = Path(output_dir) / "_sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"focus": None, "refs": []}

    # Focus URL
    focus_dest = sources_dir / "focus"
    if _is_cached(focus_dest, focus_url):
        log(f"Using cached source for {focus_url}", "ok")
        cached_meta = json.loads((focus_dest / "meta.json").read_text())
        manifest["focus"] = _manifest_entry(focus_url, "focus", cached_meta)
    else:
        focus_dest.mkdir(parents=True, exist_ok=True)
        focus_meta = _fetch_source(focus_url, focus_dest, log)
        manifest["focus"] = _manifest_entry(focus_url, "focus", focus_meta)

    # Additional sources
    if refs:
        for i, ref_url in enumerate(refs):
            dir_name = f"ref_{i + 1:02d}"
            ref_dest = sources_dir / dir_name

            if _is_cached(ref_dest, ref_url):
                log(f"Using cached source for {ref_url}", "ok")
                cached_meta = json.loads((ref_dest / "meta.json").read_text())
                entry = _manifest_entry(ref_url, dir_name, cached_meta)
            else:
                ref_dest.mkdir(parents=True, exist_ok=True)
                ref_meta = _fetch_source(ref_url, ref_dest, log)
                entry = _manifest_entry(ref_url, dir_name, ref_meta)

            # Attach user-provided annotation if present
            if ref_annotations and i < len(ref_annotations) and ref_annotations[i].strip():
                entry["annotation"] = ref_annotations[i].strip()

            manifest["refs"].append(entry)

    (sources_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return sources_dir


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fetch_source(url: str, dest: Path, log: Callable) -> dict[str, Any]:
    """Route to the appropriate handler."""
    source_type, metadata = detect_source_type(url)
    log(f"Detected source type: {source_type}")

    if source_type == "arxiv":
        return _fetch_arxiv(metadata["paper_id"], dest, log)
    elif source_type == "github_file":
        return _fetch_github_file(
            metadata["repo"], metadata["branch"], metadata["path"], dest, log,
        )
    elif source_type == "github":
        return _fetch_github(metadata["repo"], dest, log)
    elif source_type == "pdf":
        return _fetch_pdf(url, dest, log)
    else:
        return _fetch_blog(url, dest, log)


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
        log(f"Downloaded PDF ({len(resp.content) // 1024}KB)", "ok")
    else:
        meta["errors"].append(f"PDF download failed: {resp.status_code}")
        log(f"PDF download failed: {resp.status_code}", "error")
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
    if "filename" in meta:
        entry["filename"] = meta["filename"]
    return entry


def _ext_from_content_type(ct: str) -> str | None:
    """Map content-type to file extension."""
    ct = ct.split(";")[0].strip().lower()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
        "image/webp": ".webp",
    }.get(ct)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
