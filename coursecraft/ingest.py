"""Content ingestion — fetch and parse blogs, repos, and web content."""

import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def detect_source_type(url: str) -> str:
    """Detect whether a URL points to a blog post, blog index, or GitHub repo."""
    parsed = urlparse(url)
    if "github.com" in parsed.netloc:
        # Check if it's a repo (has exactly /user/repo path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and not any(
            p in parts for p in ["blob", "tree", "issues", "pulls"]
        ):
            return "repo"
    return "blog"


def fetch_html(url: str) -> str:
    """Fetch raw HTML from a URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CourseCraft/0.1; educational-tool)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_blog_content(html: str, url: str) -> dict:
    """Extract structured content from a blog post HTML page.

    Returns dict with title, content (cleaned text), code_blocks, and images.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer, header elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Try to find the main content area
    article = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(r"post|article|content|entry", re.I))
    )
    if not article:
        article = soup.find("body") or soup

    # Extract title
    title = ""
    title_tag = soup.find("h1") or soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Extract code blocks
    code_blocks = []
    for pre in article.find_all("pre"):
        code_tag = pre.find("code")
        code_text = code_tag.get_text() if code_tag else pre.get_text()
        # Try to detect language from class
        lang = ""
        classes = (code_tag or pre).get("class", [])
        for cls in classes:
            if cls.startswith("language-") or cls.startswith("lang-"):
                lang = cls.split("-", 1)[1]
                break
        code_blocks.append({"language": lang, "code": code_text.strip()})

    # Convert to clean text preserving structure
    content_parts = []
    for element in article.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "ul", "ol", "blockquote"]
    ):
        if element.name and element.name.startswith("h"):
            level = int(element.name[1])
            content_parts.append(f"\n{'#' * level} {element.get_text(strip=True)}\n")
        elif element.name == "pre":
            code_tag = element.find("code")
            code = code_tag.get_text() if code_tag else element.get_text()
            content_parts.append(f"\n```\n{code.strip()}\n```\n")
        elif element.name == "p":
            text = element.get_text(strip=True)
            if text:
                content_parts.append(text)
        elif element.name in ("ul", "ol"):
            for li in element.find_all("li", recursive=False):
                content_parts.append(f"- {li.get_text(strip=True)}")
        elif element.name == "blockquote":
            content_parts.append(f"> {element.get_text(strip=True)}")

    content = "\n\n".join(content_parts)

    return {
        "type": "blog",
        "title": title,
        "url": url,
        "content": content,
        "code_blocks": code_blocks,
    }


def extract_blog_post_links(html: str, base_url: str) -> list[dict]:
    """Extract links to individual blog posts from a blog index page."""
    soup = BeautifulSoup(html, "html.parser")

    # Find all links
    links = []
    seen_urls = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        text = a_tag.get_text(strip=True)

        # Skip empty links, anchors, external links to other domains
        if not text or href.startswith("#") or href.startswith("mailto:"):
            continue

        # Keep links that look like blog posts
        parsed_base = urlparse(base_url)
        parsed_link = urlparse(full_url)

        # Must be same domain
        if parsed_link.netloc != parsed_base.netloc:
            continue

        # Must have a path deeper than the base
        if parsed_link.path in ("/", "", parsed_base.path):
            continue

        # Skip common non-post paths
        skip_patterns = [
            "/about",
            "/contact",
            "/rss",
            "/feed",
            "/tag",
            "/category",
            "/archive",
            "/page/",
        ]
        if any(p in parsed_link.path.lower() for p in skip_patterns):
            continue

        if full_url not in seen_urls:
            seen_urls.add(full_url)
            links.append({"title": text, "url": full_url})

    return links


def fetch_repo_content(url: str) -> dict:
    """Clone a GitHub repo and extract key source files."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    repo_name = parts[1] if len(parts) >= 2 else "repo"

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_path = Path(tmpdir) / repo_name
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(clone_path)],
            capture_output=True,
            timeout=60,
        )

        # Read important source files
        files = {}
        extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".rs",
            ".go",
            ".c",
            ".cpp",
            ".h",
            ".java",
        }
        skip_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            ".cache",
            "dist",
            "build",
            "venv",
            ".venv",
        }

        for fpath in clone_path.rglob("*"):
            if fpath.is_file() and fpath.suffix in extensions:
                # Skip files in excluded directories
                if any(d in fpath.parts for d in skip_dirs):
                    continue
                try:
                    rel_path = str(fpath.relative_to(clone_path))
                    content = fpath.read_text(errors="ignore")
                    # Skip very large files
                    if len(content) < 50_000:
                        files[rel_path] = content
                except Exception:
                    continue

        # Build summary
        content_parts = [f"# Repository: {repo_name}\n"]
        for fpath, fcontent in sorted(files.items()):
            content_parts.append(f"\n## File: {fpath}\n```\n{fcontent}\n```\n")

        return {
            "type": "repo",
            "title": repo_name,
            "url": url,
            "content": "\n".join(content_parts),
            "files": files,
        }


def ingest(url: str) -> dict:
    """Main entry point: detect source type and fetch content."""
    source_type = detect_source_type(url)

    if source_type == "repo":
        return fetch_repo_content(url)
    else:
        html = fetch_html(url)
        return extract_blog_content(html, url)


def ingest_blog_index(url: str) -> list[dict]:
    """Fetch a blog index and return list of post metadata."""
    html = fetch_html(url)
    return extract_blog_post_links(html, url)
