"""Build a static site from generated courses and push to GitHub Pages."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WEB_DIR = Path(__file__).parent / "web"


def publish(
    output_dir: str | None = None,
    to: str | None = None,
    courses: list[str] | None = None,
) -> None:
    """Build static site and deploy to a local directory (auto-commits if git repo)."""
    from .server import _load_config, _save_config

    config = _load_config()
    output_dir = output_dir or config.get("output_dir", "./output")

    # Remember --to for next time
    if to:
        config["publish_to"] = to
        _save_config(config)
    else:
        to = config.get("publish_to")

    print(f"  Building site from {output_dir}")
    site = build_site(output_dir, courses)

    if to:
        deploy_site(site, to)
    else:
        print(f"  Site built \u2192 {site}")
        print("  No --to specified. Re-run with --to /path/to/your-site/courses")


def build_site(output_dir: str, courses: list[str] | None = None) -> Path:
    """Build a static site from courses in output_dir. Returns site path."""
    out = Path(output_dir).resolve()
    site = out / "_site"

    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)

    # Shared assets
    shutil.copy(WEB_DIR / "reader.css", site / "reader.css")
    shutil.copy(WEB_DIR / "reader.js", site / "reader.js")
    favicon = WEB_DIR / "favicon.png"
    if favicon.exists():
        shutil.copy(favicon, site / "favicon.png")

    # Discover courses
    course_dirs = []
    for d in sorted(
        out.iterdir(),
        key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
        reverse=True,
    ):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if not (d / "_curriculum.json").exists():
            continue
        if courses and d.name not in courses:
            continue
        course_dirs.append(d)

    if not course_dirs:
        print("  No courses found.")
        _build_index([], site)
        return site

    # Build each course
    entries: list[dict] = []
    for cd in course_dirs:
        print(f"  · {cd.name}")
        entry = _build_course(cd, site)
        if entry:
            entries.append(entry)

    _build_index(entries, site)
    print(f"  {len(entries)} course(s) ready")
    return site


# ── Course page builder ──────────────────────────────────────────────────────


def _build_course(course_dir: Path, site: Path) -> dict | None:
    """Build a single course's static page. Returns an index entry dict."""
    try:
        curriculum = json.loads((course_dir / "_curriculum.json").read_text())
    except (json.JSONDecodeError, OSError):
        return None

    slug = course_dir.name
    course_site = site / slug
    course_site.mkdir(parents=True, exist_ok=True)
    files_dir = course_site / "files"

    modules_detail: list[dict] = []
    modules_data: dict[str, dict] = {}

    for m in curriculum.get("modules", []):
        idx = m["module_index"]
        title = m["title"]

        # Find module directory on disk
        dirs = sorted(
            d for d in course_dir.iterdir()
            if d.is_dir() and d.name.startswith(f"module_{idx:02d}")
        )
        dir_name = dirs[0].name if dirs else None
        exercise_names: list[str] = []

        if dir_name:
            module_dir = course_dir / dir_name

            # Exercise filenames for detail
            exercise_names = sorted(
                f.name for f in module_dir.iterdir()
                if f.is_file() and f.name.startswith("ex") and f.suffix == ".py"
            )

            # Read lesson content
            readme = module_dir / "README.md"
            content = readme.read_text() if readme.exists() else ""

            # Read exercise file contents
            exercises = []
            for f in sorted(module_dir.iterdir()):
                if f.is_file() and f.suffix == ".py" and f.name.startswith("ex"):
                    try:
                        exercises.append({
                            "filename": f.name,
                            "content": f.read_text(),
                        })
                    except OSError:
                        pass

            modules_data[dir_name] = {
                "content": content,
                "exercises": exercises,
            }

            # Copy static assets (images, diagrams)
            _copy_assets(module_dir, files_dir / dir_name)

        modules_detail.append({
            "index": idx,
            "title": title,
            "description": m.get("description", ""),
            "dir_name": dir_name,
            "exercises": exercise_names,
            "depends_on": m.get("depends_on", []),
        })

    detail = {
        "title": curriculum.get("course_title", slug),
        "description": curriculum.get("course_description", ""),
        "modules": modules_detail,
    }

    # Embed all data as JSON
    embedded = json.dumps(
        {"courseName": slug, "detail": detail, "modules": modules_data},
        ensure_ascii=False,
    )

    html = _reader_page(detail["title"], embedded)
    (course_site / "index.html").write_text(html)

    # Clean up empty files dir
    if files_dir.exists() and not any(files_dir.rglob("*")):
        shutil.rmtree(files_dir)

    desc = curriculum.get("course_description", "")
    if len(desc) > 200:
        desc = desc[:197] + "\u2026"

    return {
        "slug": slug,
        "title": detail["title"],
        "description": desc,
        "module_count": len(modules_detail),
        "exercise_count": sum(len(md.get("exercises", [])) for md in modules_detail),
    }


def _copy_assets(module_dir: Path, dest: Path) -> None:
    """Copy non-content files (images, diagrams) from a module directory."""
    for f in module_dir.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(module_dir)
        # Skip solutions, lesson content, exercise files, metadata
        if rel.parts and rel.parts[0] == "_solutions":
            continue
        if f.name == "README.md":
            continue
        if f.name.startswith("_"):
            continue
        if f.name.startswith("ex") and f.suffix == ".py":
            continue

        dest_file = dest / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest_file)


# ── HTML templates ───────────────────────────────────────────────────────────


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _reader_page(title: str, embedded_json: str) -> str:
    # Prevent </script> in JSON from breaking out of the script tag
    safe_json = embedded_json.replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{_esc(title)} \u2014 Distill</title>
    <link rel="icon" type="image/png" href="../favicon.png">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
    <link rel="stylesheet" href="../reader.css">
</head>
<body>
    <header>
        <nav class="reader-nav">
            <a href="../" class="back-link">&larr; courses</a>
            <span class="course-title" id="course-title"></span>
            <select id="module-select" class="module-select"></select>
            <button type="button" class="theme-toggle" id="theme-toggle" title="toggle theme" aria-label="toggle theme"></button>
        </nav>
    </header>

    <nav class="module-sidebar" id="module-sidebar"></nav>

    <article id="article">
        <p class="loading">Loading&hellip;</p>
    </article>

    <nav class="module-pager" id="module-pager"></nav>

    <script id="course-data" type="application/json">{safe_json}</script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked@15.0.4/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/highlight.min.js"></script>
    <script src="../reader.js"></script>
</body>
</html>"""


def _build_index(courses: list[dict], site: Path) -> None:
    """Build the course listing page."""
    entries_html = ""
    for c in courses:
        meta_parts = []
        n = c["module_count"]
        if n:
            meta_parts.append(f'{n} module{"s" if n != 1 else ""}')
        n = c["exercise_count"]
        if n:
            meta_parts.append(f'{n} exercise{"s" if n != 1 else ""}')
        meta = " \u00b7 ".join(meta_parts)

        entries_html += f"""
            <a href="{c['slug']}/" class="course-entry">
                <h2>{_esc(c['title'])}</h2>
                <p class="description">{_esc(c['description'])}</p>
                <p class="meta">{meta}</p>
            </a>"""

    if not entries_html:
        entries_html = '<p class="empty">No courses published yet.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Distill \u2014 Courses</title>
    <link rel="icon" type="image/png" href="favicon.png">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        :root {{
            --bg: #fffff8; --fg: #111; --fg-dim: #555;
            --fg-muted: #888; --border: #ddd; --accent: #a00000;
            --hover-bg: rgba(0,0,0,0.02);
        }}

        html {{ font-size: 15px; -webkit-font-smoothing: antialiased; }}

        body {{
            font-family: Georgia, 'Palatino Linotype', 'Book Antiqua', Palatino, serif;
            background: var(--bg); color: var(--fg);
            max-width: 740px; margin: 0 auto; padding: 4rem 2rem 6rem;
        }}

        ::selection {{ background: rgba(160, 0, 0, 0.15); }}

        header {{ margin-bottom: 3rem; }}

        .site-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem; font-weight: 700;
            letter-spacing: 0.05em; color: var(--accent);
        }}

        h1 {{
            font-weight: 400; font-size: 2.4rem;
            line-height: 1.15; margin-top: 0.5rem;
        }}

        .course-list {{ list-style: none; }}

        .course-entry {{
            display: block; padding: 1.5rem 0;
            border-bottom: 1px solid var(--border);
            text-decoration: none; color: inherit;
        }}
        .course-entry:first-child {{ border-top: 1px solid var(--border); }}
        .course-entry:hover h2 {{ text-decoration: underline; }}

        .course-entry h2 {{
            font-weight: 400; font-size: 1.5rem;
            line-height: 1.3; color: var(--accent);
        }}
        .course-entry .description {{
            color: var(--fg-dim); font-size: 1.1rem;
            line-height: 1.6; margin-top: 0.4rem;
        }}
        .course-entry .meta {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem; color: var(--fg-muted); margin-top: 0.5rem;
        }}

        .empty {{ color: var(--fg-muted); font-style: italic; margin-top: 2rem; }}

        footer {{
            margin-top: 4rem; padding-top: 1rem;
            border-top: 1px solid var(--border);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem; color: var(--fg-muted);
        }}

        .theme-toggle {{
            position: fixed; top: 1rem; right: 1rem;
            background: none; border: 1px solid var(--border);
            border-radius: 3px; width: 28px; height: 28px;
            cursor: pointer; display: flex; align-items: center;
            justify-content: center; color: var(--fg-dim); font-size: 1rem;
        }}
        .theme-toggle:hover {{ color: var(--fg); border-color: var(--fg-dim); }}
        .theme-toggle::before {{ content: '\\263E'; }}
        [data-theme="dark"] .theme-toggle::before {{ content: '\\2600'; }}

        :root[data-theme="dark"] {{
            --bg: #1a1a1a; --fg: #d4d4d4; --fg-dim: #999;
            --fg-muted: #666; --border: #333; --accent: #f87171;
            --hover-bg: rgba(255,255,255,0.03);
        }}

        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) {{
                --bg: #1a1a1a; --fg: #d4d4d4; --fg-dim: #999;
                --fg-muted: #666; --border: #333; --accent: #f87171;
                --hover-bg: rgba(255,255,255,0.03);
            }}
        }}

        [data-theme="dark"] ::selection {{ background: rgba(248, 113, 113, 0.2); }}

        @media (max-width: 600px) {{
            body {{ padding: 2rem 1.2rem 4rem; }}
            h1 {{ font-size: 1.8rem; }}
        }}
    </style>
</head>
<body>
    <button type="button" class="theme-toggle" id="theme-toggle"
            title="toggle theme" aria-label="toggle theme"></button>

    <header>
        <div class="site-title">distill</div>
        <h1>Courses</h1>
    </header>

    <main>
        <div class="course-list">{entries_html}
        </div>
    </main>

    <footer>generated with distill</footer>

    <script>
    (function() {{
        var s = localStorage.getItem('distill_theme');
        if (s) document.documentElement.setAttribute('data-theme', s);
        document.getElementById('theme-toggle').addEventListener('click', function() {{
            var c = document.documentElement.getAttribute('data-theme');
            var d = c === 'dark' || (!c && window.matchMedia('(prefers-color-scheme: dark)').matches);
            var n = d ? 'light' : 'dark';
            localStorage.setItem('distill_theme', n);
            document.documentElement.setAttribute('data-theme', n);
        }});
    }})();
    </script>
</body>
</html>"""

    (site / "index.html").write_text(html)


# ── Deploy to local directory ────────────────────────────────────────────────


def deploy_site(site: Path, dest: str) -> None:
    """Copy site contents to dest. Auto-commits and pushes if inside a git repo."""
    dest_path = Path(dest).resolve()
    print(f"  Deploying to {dest_path}")

    # Clean destination contents (keep the directory itself)
    if dest_path.exists():
        for item in dest_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        dest_path.mkdir(parents=True)

    # Copy site contents (skip internal _site/.git)
    for item in site.iterdir():
        if item.name == ".git":
            continue
        target = dest_path / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    # If dest is inside a git repo, commit and push
    git_root = _find_git_root(dest_path)
    if not git_root:
        print(f"  Done. (not a git repo \u2014 push manually)")
        return

    rel = dest_path.relative_to(git_root)
    _git(["add", str(rel)], git_root)

    # Check if there are staged changes
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=git_root,
        capture_output=True,
    )
    if result.returncode == 0:
        print("  No changes to publish.")
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _git(["commit", "-m", f"Update courses \u2014 {now}"], git_root)
    _git(["push"], git_root)
    print(f"  Pushed to {git_root.name}")


def _find_git_root(path: Path) -> Path | None:
    """Walk up from path to find the nearest git root."""
    current = path
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    cmd = ["git"] + args
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  git error: {' '.join(args)}", file=sys.stderr)
        if result.stderr:
            print(f"  {result.stderr.strip()}", file=sys.stderr)
        raise SystemExit(1)
    return result
