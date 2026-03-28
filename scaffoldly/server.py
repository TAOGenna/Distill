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
    os.chmod(CONFIG_DIR, 0o700)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    os.chmod(CONFIG_FILE, 0o600)


def _apply_config() -> None:
    """Load saved API keys into env if not already set."""
    config = _load_config()
    # Support both legacy single api_key and per-provider keys
    from .llm import PROVIDER_ENV_VARS

    for provider, env_var in PROVIDER_ENV_VARS.items():
        key = config.get(f"{provider}_api_key") or config.get("api_key", "")
        if key and not os.environ.get(env_var):
            os.environ[env_var] = key


def _get_output_dir() -> str:
    return _load_config().get("output_dir", "./output")


def _get_provider() -> str:
    return _load_config().get("provider", "anthropic")


# ── API endpoints ─────────────────────────────────────────────────────────────


async def _browse_folder_endpoint(request: Request) -> JSONResponse:
    """Open a native OS folder picker dialog and return the selected path."""
    current = _get_output_dir()

    def _pick() -> str | None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            # Raise the dialog above other windows
            root.attributes("-topmost", True)
            path = filedialog.askdirectory(
                title="Choose output folder",
                initialdir=current,
            )
            root.destroy()
            return path or None
        except Exception:
            return None

    selected = await anyio.to_thread.run_sync(_pick)

    if selected:
        return JSONResponse({"path": selected})
    return JSONResponse({"path": None})


async def _config_endpoint(request: Request) -> JSONResponse:
    from .llm import PROVIDER_DEFAULTS, PROVIDER_ENV_VARS, PROVIDER_PREFIXES

    if request.method == "GET":
        config = _load_config()
        provider = config.get("provider", "anthropic")

        # Check if the current provider has an API key set
        env_var = PROVIDER_ENV_VARS.get(provider, "")
        provider_key = config.get(f"{provider}_api_key", "")
        has_key = bool(provider_key or (env_var and os.environ.get(env_var)))

        result: dict[str, Any] = {
            "output_dir": config.get("output_dir", "./output"),
            "provider": provider,
            "providers": list(PROVIDER_PREFIXES.keys()),
            "provider_defaults": PROVIDER_DEFAULTS,
            "api_key_set": has_key,
            "design_model": config.get("design_model", ""),
            "generate_model": config.get("generate_model", ""),
            "max_revision_cycles": config.get("max_revision_cycles", 1),
            "profiles": config.get("profiles", []),
            "presets": config.get("presets", []),
        }

        if provider_key and len(provider_key) > 8:
            result["api_key_masked"] = "****" + provider_key[-4:]

        return JSONResponse(result)

    body = await request.json()
    config = _load_config()

    if "provider" in body:
        config["provider"] = body["provider"]
    if "api_key" in body and body["api_key"]:
        provider = body.get("provider", config.get("provider", "anthropic"))
        config[f"{provider}_api_key"] = body["api_key"]
        # Also set in env immediately
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var:
            os.environ[env_var] = body["api_key"]
    if "output_dir" in body and body["output_dir"]:
        config["output_dir"] = body["output_dir"]
    if "design_model" in body:
        config["design_model"] = body["design_model"]
    if "generate_model" in body:
        config["generate_model"] = body["generate_model"]
    if "max_revision_cycles" in body:
        config["max_revision_cycles"] = body["max_revision_cycles"]
    if "profiles" in body:
        config["profiles"] = body["profiles"]
    if "presets" in body:
        config["presets"] = body["presets"]

    _save_config(config)
    return JSONResponse({"ok": True})


async def _generate_endpoint(request: Request) -> JSONResponse:
    from .llm import PROVIDER_ENV_VARS

    body = await request.json()

    url = body.get("url", "").strip()
    level = body.get("level", "").strip()
    if not url or not level:
        return JSONResponse(
            {"error": "url and level are required"}, status_code=400
        )

    # Check auth for the configured provider
    config = _load_config()
    provider = body.get("provider") or config.get("provider", "anthropic")
    env_var = PROVIDER_ENV_VARS.get(provider, "")
    provider_key = config.get(f"{provider}_api_key", "")
    has_key = bool(provider_key or (env_var and os.environ.get(env_var)))

    # Ollama doesn't need a key
    if provider != "ollama" and not has_key:
        return JSONResponse(
            {"error": f"no API key configured for {provider} — set it in settings"},
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
    from .fetch import preprocess_sources
    from .pipeline import run_pipeline

    config = _load_config()
    url = params["url"]
    refs = [r for r in params.get("refs", []) if r.strip()]
    series = params.get("series", False)
    level = params["level"]
    provider = params.get("provider") or config.get("provider", "anthropic")
    design_model = params.get("design_model") or config.get("design_model") or None
    generate_model = params.get("generate_model") or config.get("generate_model") or None
    output_dir = params.get("output_dir") or _get_output_dir()
    max_revision_cycles = config.get("max_revision_cycles", 1)

    # Resolve API key for the provider
    from .llm import PROVIDER_ENV_VARS
    env_var = PROVIDER_ENV_VARS.get(provider, "")
    api_key = config.get(f"{provider}_api_key") or (os.environ.get(env_var) if env_var else None)

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

        # ── Pipeline ──────────────────────────────────────────────────
        result = await run_pipeline(
            url=url,
            user_level=level,
            refs=refs,
            series=series,
            output_dir=output_dir,
            provider=provider,
            api_key=api_key,
            design_model=design_model,
            generate_model=generate_model,
            max_revision_cycles=max_revision_cycles,
            sources_dir=str(sources_dir),
            on_event=emit,
        )

        # Cleanup preprocessed sources
        shutil.rmtree(sources_dir, ignore_errors=True)

        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["result"] = result

        emit({
            "type": "complete",
            "result": {
                "course_dir": result.get("course_dir"),
                "total_cost_usd": result.get("total_cost_usd"),
                "usage": result.get("usage"),
            },
        })

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        # Sanitize — don't forward raw exceptions (may contain API keys/tokens)
        import sys
        print(f"  Generation error: {e}", file=sys.stderr)
        safe_msg = f"Generation failed ({type(e).__name__}). Check server logs for details."
        emit({"type": "error", "message": safe_msg})

    finally:
        event_queue.put(None)  # sentinel — close the SSE stream
        # Cleanup job params after completion (don't retain request data)
        if job_id in _jobs:
            _jobs[job_id].pop("params", None)


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
        Route("/api/browse-folder", _browse_folder_endpoint, methods=["POST"]),
        Route("/api/generate", _generate_endpoint, methods=["POST"]),
        Route("/api/events/{job_id}", _events_endpoint),
        Route("/api/courses", _courses_endpoint),
        Mount("/", StaticFiles(directory=str(WEB_DIR), html=True)),
    ]

    app = Starlette(routes=routes)

    # Disable browser caching for all responses — this is a local app,
    # files are tiny, and stale CSS/JS causes confusing UI bugs.
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    # Reject requests from non-localhost origins (DNS rebinding protection)
    ALLOWED_HOSTS = {"localhost", "127.0.0.1"}

    class SecurityMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # Host header check — block DNS rebinding attacks
            host = request.headers.get("host", "").split(":")[0]
            if host not in ALLOWED_HOSTS:
                return JSONResponse(
                    {"error": "forbidden"}, status_code=403
                )
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response

    app.add_middleware(SecurityMiddleware)

    return app


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
    provider = config.get("provider", "anthropic")
    from .llm import PROVIDER_ENV_VARS
    env_var = PROVIDER_ENV_VARS.get(provider, "")
    has_key = bool(config.get(f"{provider}_api_key") or (env_var and os.environ.get(env_var)))
    auth = f"{provider}" + (" ✓" if has_key else " (no key)")
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
