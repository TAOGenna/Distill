"""Local web server for Scaffoldly."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import time
import uuid
import webbrowser
from pathlib import Path
from threading import Thread
from typing import Any

import anyio
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

WEB_DIR = Path(__file__).parent / "web"
CONFIG_DIR = Path.home() / ".config" / "scaffoldly"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Single-user job storage
_jobs: dict[str, dict[str, Any]] = {}


# ── Config ────────────────────────────────────────────────────────────────────


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def _apply_config() -> None:
    """Load saved API key into env if not already set."""
    config = _load_config()
    if config.get("api_key") and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = config["api_key"]


def _get_output_dir() -> str:
    return _load_config().get("output_dir", "./output")


def _has_claude_code() -> bool:
    """Check if Claude Code CLI is installed (provides auth for the Agent SDK)."""
    return shutil.which("claude") is not None


# ── API endpoints ─────────────────────────────────────────────────────────────


async def _config_endpoint(request: Request) -> JSONResponse:
    if request.method == "GET":
        config = _load_config()
        has_key = bool(
            config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        )
        has_claude_code = _has_claude_code()
        result: dict[str, Any] = {
            "output_dir": config.get("output_dir", "./output"),
            "api_key_set": has_key or has_claude_code,
            "auth_method": (
                "claude_code" if has_claude_code and not has_key
                else "api_key" if has_key
                else "none"
            ),
        }
        key = config.get("api_key", "")
        if key and len(key) > 12:
            result["api_key_masked"] = key[:8] + "..." + key[-4:]
        return JSONResponse(result)

    body = await request.json()
    config = _load_config()

    if "api_key" in body and body["api_key"]:
        config["api_key"] = body["api_key"]
        os.environ["ANTHROPIC_API_KEY"] = body["api_key"]
    if "output_dir" in body and body["output_dir"]:
        config["output_dir"] = body["output_dir"]

    _save_config(config)
    return JSONResponse({"ok": True})


async def _generate_endpoint(request: Request) -> JSONResponse:
    body = await request.json()

    url = body.get("url", "").strip()
    level = body.get("level", "").strip()
    if not url or not level:
        return JSONResponse(
            {"error": "url and level are required"}, status_code=400
        )

    # Check auth — Claude Code provides auth automatically, otherwise need a key
    if not os.environ.get("ANTHROPIC_API_KEY") and not _has_claude_code():
        return JSONResponse(
            {"error": "no API key configured — set it in settings or install Claude Code"},
            status_code=400,
        )

    job_id = uuid.uuid4().hex[:8]
    q: queue.Queue[dict | None] = queue.Queue()
    _jobs[job_id] = {"queue": q, "status": "running", "params": body}

    asyncio.get_event_loop().create_task(
        _run_generation(job_id, body, q)
    )

    return JSONResponse({"job_id": job_id})


async def _run_generation(
    job_id: str,
    params: dict,
    event_queue: queue.Queue,
) -> None:
    """Run the full generation pipeline, pushing events to the queue."""
    from .agent import run_agent
    from .fetch import preprocess_sources

    url = params["url"]
    refs = [r for r in params.get("refs", []) if r.strip()]
    series = params.get("series", False)
    level = params["level"]
    model = params.get("model", "claude-opus-4-6")
    generate_model = params.get("generate_model", "sonnet")
    output_dir = params.get("output_dir") or _get_output_dir()

    def emit(event: dict) -> None:
        event_queue.put(event)

    try:
        # ── Preprocess ────────────────────────────────────────────────
        emit({"type": "phase", "phase": "preprocess"})

        def preprocess_log(msg: str, log_level: str = "info") -> None:
            emit({"type": "log", "message": msg, "level": log_level})

        sources_dir = await anyio.to_thread.run_sync(
            lambda: preprocess_sources(
                focus_url=url,
                refs=refs,
                series=series,
                output_dir=output_dir,
                log=preprocess_log,
            )
        )
        emit({"type": "log", "message": "sources ready", "level": "ok"})

        # ── Agent ─────────────────────────────────────────────────────
        emit({"type": "phase", "phase": "agent"})

        result = await run_agent(
            url=url,
            user_level=level,
            refs=refs,
            series=series,
            output_dir=output_dir,
            model=model,
            generate_model=generate_model,
            sources_dir=str(sources_dir),
            on_event=emit,
        )

        # Cleanup preprocessed sources
        shutil.rmtree(sources_dir, ignore_errors=True)

        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["result"] = result

        emit(
            {
                "type": "complete",
                "result": {
                    "course_dir": result.get("course_dir"),
                    "total_cost_usd": result.get("total_cost_usd"),
                    "usage": result.get("usage"),
                },
            }
        )

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        emit({"type": "error", "message": str(e)})

    finally:
        event_queue.put(None)  # sentinel — close the SSE stream


async def _events_endpoint(request: Request) -> StreamingResponse | JSONResponse:
    job_id = request.path_params["job_id"]
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)

    event_queue: queue.Queue = job["queue"]

    async def stream():
        while True:
            try:
                event = event_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.15)
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _courses_endpoint(request: Request) -> JSONResponse:
    output_dir = Path(_get_output_dir())
    courses: list[dict] = []

    if output_dir.exists():
        for d in sorted(
            output_dir.iterdir(),
            key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
            reverse=True,
        ):
            if not d.is_dir() or d.name.startswith("_"):
                continue

            course: dict[str, Any] = {
                "name": d.name,
                "path": str(d.resolve()),
            }

            curriculum = d / "_curriculum.json"
            if curriculum.exists():
                try:
                    curr = json.loads(curriculum.read_text())
                    course["modules"] = len(curr.get("modules", []))
                    course["title"] = curr.get("course_title", d.name)
                except (json.JSONDecodeError, OSError):
                    pass

            readme = d / "README.md"
            if readme.exists():
                try:
                    for line in readme.read_text().splitlines()[1:]:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            course["description"] = line[:200]
                            break
                except OSError:
                    pass

            files = [
                f
                for f in d.rglob("*")
                if f.is_file() and not f.name.startswith("_")
            ]
            course["file_count"] = len(files)

            courses.append(course)

    return JSONResponse({"courses": courses})


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> Starlette:
    _apply_config()

    routes = [
        Route("/api/config", _config_endpoint, methods=["GET", "PUT"]),
        Route("/api/generate", _generate_endpoint, methods=["POST"]),
        Route("/api/events/{job_id}", _events_endpoint),
        Route("/api/courses", _courses_endpoint),
        Mount("/", StaticFiles(directory=str(WEB_DIR), html=True)),
    ]

    return Starlette(routes=routes)


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def serve(host: str = "127.0.0.1", port: int = 8420, open_browser: bool = True) -> None:
    """Start the local web server."""
    import uvicorn

    from . import __version__

    _apply_config()

    app = create_app()

    if open_browser and not _is_wsl():
        Thread(
            target=lambda: (time.sleep(1), webbrowser.open(f"http://{host}:{port}")),
            daemon=True,
        ).start()

    url = f"http://{host}:{port}"
    link = f"\033]8;;{url}\033\\{url}\033]8;;\033\\"

    # ── Colors ────────────────────────────────────────────────────
    o  = "\033[38;5;208m"       # orange — accent
    ob = "\033[1;38;5;208m"     # orange bold
    g  = "\033[32m"             # green (cactus)
    w  = "\033[97m"             # bright white (eyes)
    B  = "\033[1m"              # bold
    D  = "\033[2m"              # dim
    R  = "\033[0m"              # reset

    LW, RW = 28, 30
    TW = LW + RW + 5

    def row(l: str = "", r: str = "", lv: int = 0, rv: int = 0) -> None:
        print(
            f"  {o}│{R} {l}{' ' * (LW - lv)} {o}│{R} {r}{' ' * (RW - rv)} {o}│{R}"
        )

    # ── Gather context ────────────────────────────────────────────
    config = _load_config()
    has_key = bool(config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"))
    auth = "claude code" if _has_claude_code() else ("api key" if has_key else "not set")
    output = config.get("output_dir", "./output")
    if len(output) > 20:
        output = output[:17] + "..."

    courses: list[str] = []
    op = Path(output)
    if op.exists():
        for d in sorted(
            (x for x in op.iterdir() if x.is_dir() and not x.name.startswith("_")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:3]:
            courses.append(d.name)

    # ── Cactus pixel art ──────────────────────────────────────────
    _P = " " * 8  # centering offset
    cactus = [
        (f"{_P}     {g}██{R}",                          15),
        (f"{_P}    {g}████{R}",                          16),
        (f"{_P}{g}█{R}  {g}█{R} {g}██{R} {g}█{R}  {g}█{R}",  20),
        (f"{_P}{g}█{R}  {g}██████{R}  {g}█{R}",          20),
        (f"{_P}    {g}████{R}",                          16),
        (f"{_P}   {o}██████{R}",                         17),
    ]

    # ── Render ────────────────────────────────────────────────────
    title = f" scaffoldly v{__version__} "
    print()
    print(f"  {o}╭──{ob}{title}{R}{o}{'─' * (TW - 2 - len(title))}╮{R}")
    row()
    row(cactus[0][0], f"{ob}getting started{R}",       cactus[0][1], 15)
    row(cactus[1][0], "paste a url. pick your",        cactus[1][1], 22)
    row(cactus[2][0], "level. hit generate.",           cactus[2][1], 20)
    row(cactus[3][0], "",                               cactus[3][1], 0)
    row(cactus[4][0], f"{ob}recent courses{R}",        cactus[4][1], 14)

    if courses:
        cn = courses[0][: RW - 1]
        row(cactus[5][0], f"{D}{cn}{R}",               cactus[5][1], len(cn))
    else:
        row(cactus[5][0], f"{D}none yet{R}",            cactus[5][1], 8)

    row()
    row(
        f" {o}▸{R} {B}{link}{R}",
        f"{D}auth    {R}{auth}",
        len(url) + 3,
        8 + len(auth),
    )
    row(
        "",
        f"{D}output  {R}{output}",
        0,
        8 + len(output),
    )
    row()
    print(f"  {o}╰{'─' * TW}╯{R}")
    print(f"  {D}ctrl+c to stop{R}")
    print()

    uvicorn.run(app, host=host, port=port, log_level="warning")
