"""Source budget management — read preprocessed sources and manage token limits.

Reads artifacts produced by fetch.py (_sources/ directory) and prepares them
for LLM consumption. Handles token budget management for large sources:
truncation, section extraction, and (if needed) summarization via a cheap model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient

# Rough estimate: 1 token ≈ 4 characters for English text / code.
CHARS_PER_TOKEN = 4
MAX_SOURCE_TOKENS = 40_000  # leaves room for system prompt + output
MAX_SOURCE_CHARS = MAX_SOURCE_TOKENS * CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Errs on the side of overestimating."""
    return len(text) // CHARS_PER_TOKEN + 1


# ── Source reading ───────────────────────────────────────────────────────────


def _read_text_file(path: Path) -> str:
    """Read a text file, returning empty string on failure."""
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _read_source_entry(entry: dict, sources_dir: Path) -> str:
    """Read all text artifacts for a single manifest entry.

    Returns a formatted string with the source content, labeled by type.
    """
    src_dir = sources_dir / entry["dir"]
    source_type = entry.get("type", "blog")
    url = entry.get("url", "")
    parts: list[str] = []

    parts.append(f"--- Source: {url} (type: {source_type}) ---\n")

    if source_type == "arxiv":
        # Read main TeX file first, then any additional tex files
        main_tex = entry.get("main_tex")
        if main_tex:
            content = _read_text_file(src_dir / "source" / main_tex)
            if content:
                parts.append(f"[Main TeX: {main_tex}]\n{content}\n")

        # Additional tex files (inputs)
        for tex_file in entry.get("tex_files", []):
            if tex_file != main_tex:
                content = _read_text_file(src_dir / "source" / tex_file)
                if content:
                    parts.append(f"[TeX: {tex_file}]\n{content}\n")

    elif source_type == "blog":
        # Read markdown source
        md_path = src_dir / "source.md"
        if md_path.exists():
            parts.append(_read_text_file(md_path))

    elif source_type == "pdf":
        # Prefer markdown extraction, fall back to noting PDF exists
        md_path = src_dir / "source.md"
        if md_path.exists():
            parts.append(_read_text_file(md_path))
        else:
            parts.append("[PDF source — text extraction not available]\n")

    elif source_type == "github":
        # Read key source files from cloned repo
        repo_dir = src_dir / "repo"
        if repo_dir.exists():
            parts.append(_read_repo_key_files(repo_dir))

    return "\n".join(parts)


def _read_repo_key_files(repo_dir: Path, max_files: int = 20) -> str:
    """Read the most important files from a cloned repo."""
    # Prioritize: README, main source files, configs
    priority_patterns = [
        "README*", "readme*",
        "*.py", "*.rs", "*.go", "*.c", "*.h", "*.cpp",
        "*.ts", "*.js",
        "Cargo.toml", "pyproject.toml", "package.json", "Makefile",
    ]

    files_found: list[Path] = []
    for pattern in priority_patterns:
        for f in repo_dir.rglob(pattern):
            if f.is_file() and not any(
                part.startswith(".") or part == "node_modules" or part == "__pycache__"
                for part in f.parts
            ):
                files_found.append(f)

    # Deduplicate and limit
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files_found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    unique = unique[:max_files]

    parts: list[str] = []
    for f in unique:
        rel = f.relative_to(repo_dir)
        content = _read_text_file(f)
        if content:
            parts.append(f"[{rel}]\n{content}\n")

    return "\n".join(parts) if parts else "[No readable source files found]\n"


# ── Budget management ────────────────────────────────────────────────────────


def _truncate_by_sections(content: str, max_chars: int) -> str:
    """Truncate content at section boundaries to fit budget.

    For LaTeX: splits on \\section, \\subsection
    For Markdown: splits on ## headers
    Keeps the beginning (abstract/intro) and truncates from the end.
    """
    if len(content) <= max_chars:
        return content

    # Try splitting on section markers
    import re
    section_pattern = re.compile(
        r"(?=\\(?:sub)?section\{)|(?=^##+ )", re.MULTILINE
    )
    sections = section_pattern.split(content)

    if len(sections) <= 1:
        # No sections found — hard truncate
        return content[:max_chars] + "\n\n[... truncated to fit token budget ...]\n"

    # Keep sections from the start until we hit the budget
    result: list[str] = []
    total = 0
    for section in sections:
        if total + len(section) > max_chars and result:
            break
        result.append(section)
        total += len(section)

    truncated = "".join(result)
    remaining = len(sections) - len(result)
    if remaining > 0:
        truncated += f"\n\n[... {remaining} sections truncated to fit token budget ...]\n"
    return truncated


def _budget_multi_source(
    focus_content: str,
    ref_contents: list[tuple[str, str]],  # (label, content)
    max_chars: int,
) -> str:
    """Budget-manage multiple sources: keep focus full, truncate refs."""
    # Reserve 70% for focus, 30% for refs
    focus_budget = int(max_chars * 0.7)
    ref_budget = max_chars - focus_budget

    focus_managed = _truncate_by_sections(focus_content, focus_budget)

    if not ref_contents:
        return focus_managed

    # Split ref budget evenly
    per_ref = ref_budget // max(len(ref_contents), 1)
    ref_parts: list[str] = []
    for label, content in ref_contents:
        truncated = _truncate_by_sections(content, per_ref)
        ref_parts.append(truncated)

    return focus_managed + "\n\n" + "\n\n".join(ref_parts)


# ── Public API ───────────────────────────────────────────────────────────────


def prepare_sources(
    sources_dir: str | Path,
    max_tokens: int = MAX_SOURCE_TOKENS,
) -> str:
    """Read all preprocessed sources and return budget-managed content.

    This is the single entry point: give it the _sources/ directory
    produced by fetch.py, get back a string ready to paste into an LLM prompt.
    """
    sources_dir = Path(sources_dir)
    manifest_path = sources_dir / "manifest.json"

    if not manifest_path.exists():
        return "[No manifest.json found in sources directory]\n"

    manifest = json.loads(manifest_path.read_text())
    max_chars = max_tokens * CHARS_PER_TOKEN

    # Read focus source
    focus_entry = manifest.get("focus")
    focus_content = ""
    if focus_entry:
        focus_content = _read_source_entry(focus_entry, sources_dir)

    # Read additional sources (series or refs)
    ref_contents: list[tuple[str, str]] = []
    for entry in manifest.get("series", []):
        content = _read_source_entry(entry, sources_dir)
        ref_contents.append((entry.get("url", ""), content))
    for entry in manifest.get("refs", []):
        content = _read_source_entry(entry, sources_dir)
        ref_contents.append((entry.get("url", ""), content))

    # Total size check
    total_content = focus_content + "\n\n".join(c for _, c in ref_contents)
    if estimate_tokens(total_content) <= max_tokens:
        return total_content

    # Budget management needed
    return _budget_multi_source(focus_content, ref_contents, max_chars)


async def prepare_sources_with_summary(
    sources_dir: str | Path,
    llm_client: LLMClient,
    summary_model: str,
    max_tokens: int = MAX_SOURCE_TOKENS,
) -> str:
    """Read sources, summarizing with a cheap model if they exceed budget.

    Strategy:
    1. If sources fit in budget → return as-is
    2. If sources are 1-2x over budget → truncate by sections
    3. If sources are >2x over budget → summarize with cheap model, then
       return summary + key excerpts

    The summarization is transparent to the user — just a log line.
    """
    sources_dir = Path(sources_dir)
    full_content = prepare_sources(sources_dir, max_tokens=max_tokens * 3)
    token_count = estimate_tokens(full_content)

    if token_count <= max_tokens:
        return full_content

    if token_count <= max_tokens * 2:
        # Moderate overflow — section truncation is enough
        return prepare_sources(sources_dir, max_tokens=max_tokens)

    # Large overflow — summarize with cheap model
    # Split into chunks that fit the cheap model's context
    chunk_size = max_tokens * CHARS_PER_TOKEN * 2  # cheap models have large contexts
    chunks = [
        full_content[i : i + chunk_size]
        for i in range(0, len(full_content), chunk_size)
    ]

    summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        result = await llm_client.complete(
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this source material (part {i + 1}/{len(chunks)}) "
                    f"for a technical educator who needs to design coursework from it. "
                    f"Preserve: key concepts, algorithms, measurements, benchmarks, "
                    f"code patterns, and quantitative claims. Be thorough.\n\n{chunk}"
                ),
            }],
            model=summary_model,
            max_tokens=4096,
        )
        summaries.append(result.content)

    summary = "\n\n".join(summaries)

    # Return summary + key excerpts from the original (first and last sections)
    excerpt_budget = (max_tokens * CHARS_PER_TOKEN) // 3
    excerpt = full_content[:excerpt_budget]

    return (
        f"[Source material summarized to fit token budget — "
        f"original was ~{token_count} tokens]\n\n"
        f"## Summary\n\n{summary}\n\n"
        f"## Key Excerpts (beginning of source)\n\n{excerpt}\n"
    )
