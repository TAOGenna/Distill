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

    annotation = entry.get("annotation", "")
    parts.append(f"--- Source: {url} (type: {source_type}) ---\n")
    if annotation:
        parts.append(f"[Role: {annotation}]\n")

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


def _read_repo_key_files(repo_dir: Path, budget_chars: int = 300_000) -> str:
    """Read a repo intelligently: tree + READMEs + ranked source files.

    Instead of grabbing 20 files by extension, this:
    1. Generates the filtered directory tree (structural context)
    2. Reads all READMEs and config files (author's own explanation)
    3. Parses imports to find hub modules (most-imported = most important)
    4. Packs ranked files into a token budget
    5. Includes first-line stubs for overflow files
    """
    import re as _re

    # ── Excluded directories and files ────────────────────────────────
    _SKIP_DIRS = frozenset({
        ".git", ".github", ".vscode", ".idea", ".mypy_cache", ".pytest_cache",
        "__pycache__", "node_modules", "vendor", "third_party", "dist", "build",
        "target", ".tox", ".eggs", "venv", ".venv", "env", ".env",
        "cmake-build-debug", "cmake-build-release",
    })
    _SKIP_SUFFIXES = frozenset({
        ".lock", ".sum", ".min.js", ".min.css", ".map", ".pyc", ".pyo",
        ".so", ".dylib", ".dll", ".o", ".a", ".class", ".jar",
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
        ".woff", ".woff2", ".ttf", ".eot",
        ".pdf", ".zip", ".tar", ".gz", ".bz2",
        ".pb", ".onnx", ".pt", ".bin", ".npy", ".npz", ".h5",
    })
    _SOURCE_EXTS = frozenset({
        ".py", ".rs", ".go", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
        ".cu", ".cuh",
        ".ts", ".tsx", ".js", ".jsx", ".java", ".kt", ".scala",
        ".rb", ".jl", ".lua", ".zig", ".nim", ".ex", ".exs",
        ".sh", ".bash", ".zsh", ".fish",
        ".sql", ".proto", ".thrift", ".graphql",
        ".toml", ".yaml", ".yml", ".json", ".xml",
        ".md", ".rst", ".txt",
        ".cmake", ".mak",
    })

    def _should_skip_dir(name: str) -> bool:
        return name in _SKIP_DIRS or name.startswith(".")

    def _should_skip_file(path: Path) -> bool:
        return path.suffix.lower() in _SKIP_SUFFIXES

    def _is_source(path: Path) -> bool:
        s = path.suffix.lower()
        return s in _SOURCE_EXTS or path.name in {
            "Makefile", "Dockerfile", "Vagrantfile", "Rakefile",
            "CMakeLists.txt", "BUILD", "WORKSPACE",
        }

    # ── 1. Directory tree ─────────────────────────────────────────────

    def _build_tree(root: Path, prefix: str = "", max_depth: int = 5, depth: int = 0) -> list[str]:
        if depth > max_depth:
            return [prefix + "..."]
        lines: list[str] = []
        try:
            entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return lines
        dirs = [e for e in entries if e.is_dir() and not _should_skip_dir(e.name)]
        files = [e for e in entries if e.is_file() and not _should_skip_file(e)]
        items = dirs + files
        for i, item in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            child_prefix = "    " if i == len(items) - 1 else "│   "
            if item.is_dir():
                lines.append(prefix + connector + item.name + "/")
                lines.extend(_build_tree(item, prefix + child_prefix, max_depth, depth + 1))
            else:
                lines.append(prefix + connector + item.name)
        return lines

    tree_lines = [repo_dir.name + "/"] + _build_tree(repo_dir)
    tree_text = "\n".join(tree_lines)

    # ── 2. Collect all source files ───────────────────────────────────

    all_files: list[Path] = []
    for f in repo_dir.rglob("*"):
        if not f.is_file():
            continue
        if any(_should_skip_dir(part) for part in f.relative_to(repo_dir).parts):
            continue
        if _should_skip_file(f):
            continue
        if _is_source(f):
            all_files.append(f)

    # ── 3. Categorize: READMEs, configs, source code ──────────────────

    readmes: list[Path] = []
    configs: list[Path] = []
    source_code: list[Path] = []

    _CONFIG_NAMES = frozenset({
        "pyproject.toml", "setup.py", "setup.cfg",
        "Cargo.toml", "package.json",
        "Makefile", "CMakeLists.txt", "BUILD",
        "go.mod", "Gemfile", "mix.exs", "build.zig",
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    })

    for f in all_files:
        name_lower = f.name.lower()
        if name_lower.startswith("readme"):
            readmes.append(f)
        elif f.name in _CONFIG_NAMES:
            configs.append(f)
        elif f.suffix.lower() in {".md", ".rst", ".txt", ".toml", ".yaml", ".yml",
                                    ".json", ".xml", ".proto", ".graphql", ".sql"}:
            # Non-code text files — lower priority, skip for ranking
            pass
        else:
            source_code.append(f)

    # ── 4. Import graph → centrality ranking ──────────────────────────

    # Map relative paths to file paths for import resolution
    path_index: dict[str, Path] = {}
    for f in source_code:
        rel = f.relative_to(repo_dir)
        # Index by multiple keys for fuzzy matching
        path_index[str(rel)] = f                           # src/runner.cu
        path_index[str(rel.with_suffix(""))] = f           # src/runner
        path_index[rel.name] = f                           # runner.cu
        path_index[rel.stem] = f                           # runner
        # Python module style: a/b/c.py → a.b.c
        module_key = str(rel.with_suffix("")).replace("/", ".").replace("\\", ".")
        path_index[module_key] = f

    # Import patterns per language
    _IMPORT_PATTERNS: dict[str, list[_re.Pattern]] = {
        ".py": [
            _re.compile(r"^\s*from\s+([\w.]+)\s+import", _re.MULTILINE),
            _re.compile(r"^\s*import\s+([\w.]+)", _re.MULTILINE),
        ],
        ".rs": [
            _re.compile(r"^\s*use\s+crate::([\w:]+)", _re.MULTILINE),
            _re.compile(r"^\s*mod\s+(\w+)\s*;", _re.MULTILINE),
        ],
        ".go": [
            _re.compile(r'"[^"]*?/([^"/]+)"', _re.MULTILINE),
        ],
        ".js": [
            _re.compile(r"""(?:import|require)\s*\(?['"]\.\/([^'"]+)['"]""", _re.MULTILINE),
            _re.compile(r"""from\s+['"]\.\/([^'"]+)['"]""", _re.MULTILINE),
        ],
        ".ts": [
            _re.compile(r"""(?:import|require)\s*\(?['"]\.\/([^'"]+)['"]""", _re.MULTILINE),
            _re.compile(r"""from\s+['"]\.\/([^'"]+)['"]""", _re.MULTILINE),
        ],
        ".c": [_re.compile(r'#include\s+"([^"]+)"', _re.MULTILINE)],
        ".h": [_re.compile(r'#include\s+"([^"]+)"', _re.MULTILINE)],
        ".cpp": [_re.compile(r'#include\s+"([^"]+)"', _re.MULTILINE)],
        ".hpp": [_re.compile(r'#include\s+"([^"]+)"', _re.MULTILINE)],
        ".cc": [_re.compile(r'#include\s+"([^"]+)"', _re.MULTILINE)],
    }
    # Alias tsx/jsx to ts/js patterns, cu/cuh to c/cpp
    _IMPORT_PATTERNS[".tsx"] = _IMPORT_PATTERNS[".ts"]
    _IMPORT_PATTERNS[".jsx"] = _IMPORT_PATTERNS[".js"]
    _IMPORT_PATTERNS[".cu"] = _IMPORT_PATTERNS[".c"]
    _IMPORT_PATTERNS[".cuh"] = _IMPORT_PATTERNS[".c"]

    # Count how many files import each file (in-degree)
    in_degree: dict[Path, int] = {f: 0 for f in source_code}
    readme_text_lower = ""
    for r in readmes:
        readme_text_lower += _read_text_file(r).lower() + "\n"

    for f in source_code:
        ext = f.suffix.lower()
        patterns = _IMPORT_PATTERNS.get(ext, [])
        if not patterns:
            continue
        try:
            content = f.read_text(errors="replace")
        except OSError:
            continue
        for pat in patterns:
            for match in pat.finditer(content):
                target = match.group(1)
                # Try to resolve the import target to a file
                # Clean up: crate::foo::bar → foo/bar, a.b.c → a/b/c
                cleaned = target.replace("::", "/").replace(".", "/")
                leaf = target.rsplit(".", 1)[-1].rsplit("::", 1)[-1].rsplit("/", 1)[-1]
                candidates = [target, cleaned, leaf]
                # For C/C++: target already has extension (e.g. "llama.h")
                # For Python: add stem without dots (e.g. "pipeline")
                if "." in target and not target.endswith((".py", ".h", ".hpp", ".cuh")):
                    candidates.append(target.replace(".", "/"))
                for candidate_key in candidates:
                    resolved = path_index.get(candidate_key)
                    if resolved and resolved != f:
                        in_degree[resolved] = in_degree.get(resolved, 0) + 1
                        break

    # ── 5. Score each source file ─────────────────────────────────────

    _ENTRY_NAMES = frozenset({
        "main", "lib", "__main__", "__init__", "index", "app", "mod",
        "cli", "server", "core", "engine",
    })
    _IMPORTANT_NAMES = frozenset({
        "core", "engine", "model", "models", "pipeline", "parser",
        "compiler", "runtime", "scheduler", "worker", "handler",
    })

    scored: list[tuple[float, Path]] = []
    for f in source_code:
        rel = f.relative_to(repo_dir)
        depth = len(rel.parts) - 1
        score = 0.0

        # Import centrality (strongest signal)
        score += in_degree.get(f, 0) * 10.0

        # Mentioned in README
        stem_lower = f.stem.lower()
        if stem_lower in readme_text_lower or f.name.lower() in readme_text_lower:
            score += 8.0

        # Entry point
        if stem_lower in _ENTRY_NAMES:
            score += 6.0

        # Important-sounding name
        if stem_lower in _IMPORTANT_NAMES:
            score += 3.0

        # Shallow depth (root files are more important)
        score += max(0, 4.0 - depth)

        # File size: prefer medium-sized files (not tiny, not huge)
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        if 200 < size < 50_000:
            score += 2.0
        elif size >= 50_000:
            score += 0.5  # still include, just lower priority

        scored.append((score, f))

    scored.sort(key=lambda x: -x[0])

    # ── 6. Pack into budget ───────────────────────────────────────────

    parts: list[str] = []
    used = 0

    # Budget allocation: tree 5%, READMEs 15%, configs 5%, source 75%
    tree_cap = max(budget_chars // 20, 5_000)
    readme_cap = max(budget_chars // 7, 20_000)
    config_cap = max(budget_chars // 20, 5_000)

    # Directory tree (capped)
    tree_section = f"[Directory Structure]\n{tree_text}\n"
    if len(tree_section) > tree_cap:
        tree_section = tree_section[:tree_cap] + "\n[... tree truncated ...]\n"
    parts.append(tree_section)
    used += len(tree_section)

    # READMEs — root README gets the lion's share, sub-READMEs get stubs
    readme_used = 0
    for r in sorted(readmes, key=lambda p: len(p.relative_to(repo_dir).parts)):
        rel = r.relative_to(repo_dir)
        content = _read_text_file(r)
        if not content:
            continue
        remaining = readme_cap - readme_used
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[:remaining] + "\n[... truncated ...]\n"
        section = f"[{rel}]\n{content}\n"
        parts.append(section)
        used += len(section)
        readme_used += len(section)

    # Config files (capped)
    config_used = 0
    for c in configs:
        rel = c.relative_to(repo_dir)
        content = _read_text_file(c)
        if not content:
            continue
        section = f"[{rel}]\n{content}\n"
        if config_used + len(section) > config_cap:
            break
        parts.append(section)
        used += len(section)
        config_used += len(section)

    # Fill remaining budget with ranked source files
    included_paths: set[Path] = set(readmes) | set(configs)
    overflow_stubs: list[str] = []

    for _score, f in scored:
        if f in included_paths:
            continue
        rel = f.relative_to(repo_dir)
        content = _read_text_file(f)
        if not content:
            continue

        section = f"[{rel}]\n{content}\n"
        if used + len(section) <= budget_chars:
            parts.append(section)
            used += len(section)
            included_paths.add(f)
        else:
            # Over budget — record stub (first 3 lines)
            first_lines = "\n".join(content.split("\n")[:3])
            overflow_stubs.append(f"  {rel}: {first_lines.strip()[:120]}")

    # Append overflow summary (capped to not blow budget)
    if overflow_stubs:
        stub_cap = max(budget_chars - used, 0)
        stub_header = (
            f"\n[{len(overflow_stubs)} additional source files not shown — "
            f"first lines listed for context]\n"
        )
        stub_lines: list[str] = []
        stub_used = len(stub_header)
        for stub in overflow_stubs:
            if stub_used + len(stub) + 1 > stub_cap:
                break
            stub_lines.append(stub)
            stub_used += len(stub) + 1
        parts.append(stub_header + "\n".join(stub_lines) + "\n")

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

    # Read additional sources
    ref_contents: list[tuple[str, str]] = []
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
