"""Excalidraw diagram support — MCP server lifecycle + SVG renderer.

Two responsibilities:
1. Manage the yctimlin/mcp_excalidraw Express canvas server (start/stop/clear)
2. Render .excalidraw JSON files to SVG for embedding in READMEs

The MCP server gives the model spatial feedback via describe_scene so it can
iteratively refine diagrams. The Python SVG renderer converts the final
.excalidraw files to SVG for display in markdown.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import httpx


# ── MCP Canvas Server Lifecycle ─────────────────────────────────────────────

# Path to the mcp_excalidraw package — installed as a git submodule or
# cloned alongside the project. Adjust if your layout differs.
MCP_PACKAGE_DIR = Path(__file__).parent / "excalidraw"

_CANVAS_PORT = 18420  # Uncommon port — avoids 3000 collisions with React/Next.js
_CANVAS_HOST = "localhost"
_CANVAS_URL = f"http://{_CANVAS_HOST}:{_CANVAS_PORT}"
_STARTUP_TIMEOUT = 15  # seconds to wait for canvas server


def _port_open(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _is_canvas_server(host: str, port: int) -> bool:
    """Verify the process on this port is actually our Excalidraw canvas server.

    Hits GET /health and checks for the expected response shape.
    Uses urllib (stdlib) to stay synchronous — called from start_canvas_server.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/health", method="GET"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "healthy"
    except (urllib.error.URLError, json.JSONDecodeError, OSError, KeyError):
        return False


def find_mcp_package() -> Path | None:
    """Locate the mcp_excalidraw package directory.

    Returns the path if found and built (dist/ exists), else None.
    """
    pkg = MCP_PACKAGE_DIR
    if pkg.exists() and (pkg / "dist" / "server.js").exists():
        return pkg
    return None


def start_canvas_server() -> subprocess.Popen | None:
    """Start the Express canvas server as a background subprocess.

    Returns the Popen handle, or None if the server couldn't start.
    The caller is responsible for calling stop_canvas_server() when done.
    """
    # If port is already in use, verify it's actually our canvas server
    if _port_open(_CANVAS_HOST, _CANVAS_PORT):
        if _is_canvas_server(_CANVAS_HOST, _CANVAS_PORT):
            print(
                f"  Canvas server already running on port {_CANVAS_PORT}",
                file=sys.stderr,
            )
            return None  # We didn't start it — don't stop it
        else:
            print(
                f"  Warning: port {_CANVAS_PORT} is in use by another process. "
                f"Cannot start canvas server.",
                file=sys.stderr,
            )
            return None

    pkg = find_mcp_package()
    if pkg is None:
        print(
            "  Warning: mcp_excalidraw not found or not built at "
            f"{MCP_PACKAGE_DIR}. Diagrams will use blind generation.",
            file=sys.stderr,
        )
        return None

    env = {**os.environ, "PORT": str(_CANVAS_PORT)}
    try:
        proc = subprocess.Popen(
            ["node", str(pkg / "dist" / "server.js")],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=str(pkg),
        )
    except FileNotFoundError:
        print(
            "  Warning: 'node' not found in PATH. "
            "Diagrams will use blind generation.",
            file=sys.stderr,
        )
        return None

    # Wait for port to become available
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            print(
                f"  Warning: Canvas server exited early: {stderr[:200]}",
                file=sys.stderr,
            )
            return None
        if _port_open(_CANVAS_HOST, _CANVAS_PORT):
            return proc
        time.sleep(0.3)

    # Timed out
    proc.terminate()
    print(
        f"  Warning: Canvas server did not start within {_STARTUP_TIMEOUT}s",
        file=sys.stderr,
    )
    return None


def stop_canvas_server(proc: subprocess.Popen | None) -> None:
    """Stop the Express canvas server."""
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


async def clear_canvas() -> None:
    """Clear all elements from the canvas (reset between modules)."""
    try:
        async with httpx.AsyncClient() as client:
            await client.delete(
                f"{_CANVAS_URL}/api/elements/clear", timeout=5.0
            )
    except Exception:
        pass  # Best effort — if server isn't running, skip silently


def get_mcp_server_config() -> dict | None:
    """Return MCP server config if the canvas server is running and verified.

    Returns None if the package isn't built OR the Express canvas server
    isn't reachable (prevents handing the agent tools that point at a
    wrong server on the same port).
    """
    pkg = find_mcp_package()
    if pkg is None:
        return None

    if not _is_canvas_server(_CANVAS_HOST, _CANVAS_PORT):
        return None

    return {
        "excalidraw": {
            "command": "node",
            "args": [str(pkg / "dist" / "index.js")],
            "env": {
                "EXPRESS_SERVER_URL": _CANVAS_URL,
                "ENABLE_CANVAS_SYNC": "true",
            },
        }
    }


def mcp_tools_available() -> bool:
    """Check if the MCP Excalidraw server can be used."""
    return find_mcp_package() is not None


# ── Pure Python SVG Renderer ────────────────────────────────────────────────
#
# Converts .excalidraw JSON to SVG. This is a lightweight renderer that
# handles the common element types (rectangle, ellipse, diamond, arrow,
# line, text). For full-fidelity hand-drawn rendering, open the .excalidraw
# source at https://excalidraw.com.

_PADDING = 40
_MAX_DISPLAY_WIDTH = 800

_FONT_FAMILIES = {
    1: "Virgil, Segoe Print, Comic Sans MS, cursive",
    2: "Helvetica, Arial, sans-serif",
    3: "Cascadia Code, Fira Code, monospace",
}

_FILTER_SVG = (
    '    <filter id="handdrawn" x="-5%" y="-5%" width="110%" height="110%">\n'
    '      <feTurbulence type="turbulence" baseFrequency="0.02" numOctaves="3"\n'
    '                    result="noise" seed="2"/>\n'
    '      <feDisplacementMap in="SourceGraphic" in2="noise" scale="1.2"\n'
    '                         xChannelSelector="R" yChannelSelector="G"/>\n'
    "    </filter>"
)


def _compute_bounds(
    elements: list[dict],
) -> tuple[float, float, float, float]:
    """Compute bounding box. Returns (min_x, min_y, max_x, max_y)."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for el in elements:
        t = el.get("type", "")
        x, y = float(el.get("x", 0)), float(el.get("y", 0))
        w, h = float(el.get("width", 0)), float(el.get("height", 0))

        if t in ("arrow", "line"):
            for pt in el.get("points", [[0, 0]]):
                px, py = x + pt[0], y + pt[1]
                min_x, min_y = min(min_x, px), min(min_y, py)
                max_x, max_y = max(max_x, px), max(max_y, py)
        elif t == "text":
            text = el.get("text", "")
            fs = el.get("fontSize", 16)
            lines = text.split("\n")
            tw = max((len(ln) for ln in lines), default=0) * fs * 0.6
            th = len(lines) * fs * 1.4
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + tw), max(max_y, y + th)
        else:
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + w), max(max_y, y + h)

    if min_x == float("inf"):
        return 0, 0, 400, 300
    return min_x, min_y, max_x, max_y


def _fill(el: dict) -> str:
    bg = el.get("backgroundColor", "transparent")
    return "none" if bg in ("transparent", "") else bg


def _stroke_dash(el: dict) -> str:
    s = el.get("strokeStyle", "solid")
    if s == "dashed":
        return ' stroke-dasharray="8 4"'
    if s == "dotted":
        return ' stroke-dasharray="2 4"'
    return ""


def _label_svg(el: dict, cx: float, cy: float) -> str:
    """Render a label centered inside a shape."""
    label = el.get("label")
    if not label or not label.get("text"):
        return ""
    text = label["text"]
    fs = label.get("fontSize", 16)
    ff = _FONT_FAMILIES.get(label.get("fontFamily", 1), _FONT_FAMILIES[1])
    color = label.get("strokeColor", "#1e1e1e")
    lines = text.split("\n")
    total_h = len(lines) * fs * 1.2
    start_y = cy - total_h / 2 + fs * 0.8
    parts = []
    for i, line in enumerate(lines):
        ly = start_y + i * fs * 1.2
        parts.append(
            f'    <text x="{cx}" y="{ly}" font-size="{fs}" '
            f'font-family="{ff}" fill="{color}" '
            f'text-anchor="middle">{xml_escape(line)}</text>'
        )
    return "\n".join(parts)


def _render_rectangle(el: dict) -> str:
    x, y = el.get("x", 0), el.get("y", 0)
    w, h = el.get("width", 100), el.get("height", 60)
    rnd = el.get("roundness")
    rx = 8 if rnd and rnd.get("type") == 3 else 0
    stroke = el.get("strokeColor", "#1e1e1e")
    sw = el.get("strokeWidth", 2)
    op = el.get("opacity", 100) / 100.0
    dash = _stroke_dash(el)

    svg = (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="{rx}" fill="{_fill(el)}" stroke="{stroke}" '
        f'stroke-width="{sw}" opacity="{op}"{dash} '
        f'filter="url(#handdrawn)"/>'
    )
    lbl = _label_svg(el, x + w / 2, y + h / 2)
    return svg + ("\n" + lbl if lbl else "")


def _render_ellipse(el: dict) -> str:
    x, y = el.get("x", 0), el.get("y", 0)
    w, h = el.get("width", 100), el.get("height", 60)
    cx, cy = x + w / 2, y + h / 2
    stroke = el.get("strokeColor", "#1e1e1e")
    sw = el.get("strokeWidth", 2)
    dash = _stroke_dash(el)

    svg = (
        f'  <ellipse cx="{cx}" cy="{cy}" rx="{w / 2}" ry="{h / 2}" '
        f'fill="{_fill(el)}" stroke="{stroke}" '
        f'stroke-width="{sw}"{dash} filter="url(#handdrawn)"/>'
    )
    lbl = _label_svg(el, cx, cy)
    return svg + ("\n" + lbl if lbl else "")


def _render_diamond(el: dict) -> str:
    x, y = el.get("x", 0), el.get("y", 0)
    w, h = el.get("width", 100), el.get("height", 60)
    cx, cy = x + w / 2, y + h / 2
    pts = f"{cx},{y} {x + w},{cy} {cx},{y + h} {x},{cy}"
    stroke = el.get("strokeColor", "#1e1e1e")
    sw = el.get("strokeWidth", 2)
    dash = _stroke_dash(el)

    svg = (
        f'  <polygon points="{pts}" fill="{_fill(el)}" '
        f'stroke="{stroke}" stroke-width="{sw}"{dash} '
        f'filter="url(#handdrawn)"/>'
    )
    lbl = _label_svg(el, cx, cy)
    return svg + ("\n" + lbl if lbl else "")


def _render_arrow(el: dict, markers: set[str]) -> str:
    x, y = float(el.get("x", 0)), float(el.get("y", 0))
    points = el.get("points", [[0, 0], [100, 0]])
    color = el.get("strokeColor", "#1e1e1e")
    sw = el.get("strokeWidth", 2)
    dash = _stroke_dash(el)

    if not points:
        return ""

    d_parts = [f"M {x + points[0][0]:.1f} {y + points[0][1]:.1f}"]
    for pt in points[1:]:
        d_parts.append(f"L {x + pt[0]:.1f} {y + pt[1]:.1f}")
    d = " ".join(d_parts)

    # Arrowhead markers
    marker_end = ""
    end_ah = el.get("endArrowhead", "arrow")
    if end_ah and end_ah != "none":
        mid = f"ah-{color.lstrip('#')}"
        markers.add(mid)
        marker_end = f' marker-end="url(#{mid})"'

    marker_start = ""
    start_ah = el.get("startArrowhead")
    if start_ah and start_ah != "none":
        mid = f"ah-{color.lstrip('#')}"
        markers.add(mid)
        marker_start = f' marker-start="url(#{mid})"'

    svg = (
        f'  <path d="{d}" fill="none" stroke="{color}" '
        f'stroke-width="{sw}"{dash}{marker_end}{marker_start} '
        f'filter="url(#handdrawn)"/>'
    )

    # Arrow label
    label = el.get("label")
    if label and label.get("text") and len(points) >= 2:
        mid_idx = len(points) // 2
        mx = x + (points[mid_idx - 1][0] + points[mid_idx][0]) / 2
        my = y + (points[mid_idx - 1][1] + points[mid_idx][1]) / 2
        fs = label.get("fontSize", 14)
        text = xml_escape(label["text"])
        lw = len(label["text"]) * fs * 0.6
        svg += (
            f'\n  <rect x="{mx - lw / 2:.1f}" y="{my - fs - 2:.1f}" '
            f'width="{lw:.1f}" height="{fs + 4}" '
            f'fill="white" stroke="none" opacity="0.85"/>'
            f'\n  <text x="{mx:.1f}" y="{my - 2:.1f}" font-size="{fs}" '
            f'font-family="{_FONT_FAMILIES[1]}" fill="{color}" '
            f'text-anchor="middle">{text}</text>'
        )

    return svg


def _render_line(el: dict, markers: set[str]) -> str:
    """Render a line (arrow without arrowheads by default)."""
    # Lines use the same format as arrows but default to no arrowheads
    el_copy = {**el}
    el_copy.setdefault("endArrowhead", None)
    el_copy.setdefault("startArrowhead", None)
    return _render_arrow(el_copy, markers)


def _render_text(el: dict) -> str:
    x, y = el.get("x", 0), el.get("y", 0)
    text = el.get("text", "")
    fs = el.get("fontSize", 16)
    ff = _FONT_FAMILIES.get(el.get("fontFamily", 1), _FONT_FAMILIES[1])
    color = el.get("strokeColor", "#1e1e1e")
    align = el.get("textAlign", "left")
    anchor = {"left": "start", "center": "middle", "right": "end"}.get(
        align, "start"
    )

    lines = text.split("\n")
    parts = []
    for i, line in enumerate(lines):
        ly = y + (i + 1) * fs * 1.2
        parts.append(
            f'  <text x="{x}" y="{ly:.1f}" font-size="{fs}" '
            f'font-family="{ff}" fill="{color}" '
            f'text-anchor="{anchor}">{xml_escape(line)}</text>'
        )
    return "\n".join(parts)


def _build_markers_svg(markers: set[str]) -> str:
    """Generate SVG arrowhead marker definitions."""
    parts = []
    for mid in sorted(markers):
        color = "#" + mid.split("-", 1)[1] if "-" in mid else "#1e1e1e"
        parts.append(
            f'    <marker id="{mid}" viewBox="0 0 10 10" refX="10" refY="5" '
            f'markerWidth="8" markerHeight="8" orient="auto-start-reverse">\n'
            f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/>\n'
            f"    </marker>"
        )
    return "\n".join(parts)


def render_excalidraw_to_svg(
    excalidraw_path: Path,
    svg_path: Path | None = None,
) -> Path:
    """Render an .excalidraw JSON file to SVG.

    Returns path to the generated SVG file.
    """
    raw = excalidraw_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    elements = data if isinstance(data, list) else data.get("elements", [])

    if not elements:
        raise ValueError(f"No elements in {excalidraw_path}")

    min_x, min_y, max_x, max_y = _compute_bounds(elements)
    vb_x = min_x - _PADDING
    vb_y = min_y - _PADDING
    vb_w = (max_x - min_x) + 2 * _PADDING
    vb_h = (max_y - min_y) + 2 * _PADDING

    disp_w = min(vb_w, _MAX_DISPLAY_WIDTH)
    disp_h = vb_h * (disp_w / vb_w) if vb_w > 0 else 300

    # Render elements, collecting arrowhead markers
    markers: set[str] = set()
    rendered: list[str] = []

    for el in elements:
        t = el.get("type", "")
        if t == "rectangle":
            rendered.append(_render_rectangle(el))
        elif t == "ellipse":
            rendered.append(_render_ellipse(el))
        elif t == "diamond":
            rendered.append(_render_diamond(el))
        elif t == "arrow":
            rendered.append(_render_arrow(el, markers))
        elif t == "line":
            rendered.append(_render_line(el, markers))
        elif t == "text":
            rendered.append(_render_text(el))
        elif t:
            print(
                f"  Warning: skipping unsupported element type '{t}' in "
                f"{excalidraw_path.name}",
                file=sys.stderr,
            )

    # Build defs
    markers_svg = _build_markers_svg(markers)
    defs_content = _FILTER_SVG
    if markers_svg:
        defs_content += "\n" + markers_svg

    svg = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{vb_x:.0f} {vb_y:.0f} {vb_w:.0f} {vb_h:.0f}" '
        f'width="{disp_w:.0f}" height="{disp_h:.0f}" '
        f'style="background: white;">\n'
        f"  <defs>\n{defs_content}\n  </defs>\n"
        + "\n".join(rendered)
        + "\n</svg>\n"
    )

    if svg_path is None:
        svg_path = excalidraw_path.with_suffix(".svg")
    svg_path.write_text(svg, encoding="utf-8")
    return svg_path


def render_module_diagrams(module_dir: Path) -> int:
    """Render all .excalidraw files in module's diagrams/ to SVG.

    Returns count of successfully rendered diagrams. Logs warnings on
    failure but never raises — graceful degradation.
    """
    diagrams_dir = module_dir / "diagrams"
    if not diagrams_dir.exists():
        return 0

    rendered = 0
    for f in sorted(diagrams_dir.glob("*.excalidraw")):
        try:
            render_excalidraw_to_svg(f)
            rendered += 1
        except Exception as e:
            print(f"  Warning: diagram {f.name}: {e}", file=sys.stderr)

    return rendered
