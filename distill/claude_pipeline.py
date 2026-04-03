"""Claude Code pipeline — runs the full course generation through Claude Code SDK.

No LiteLLM, no external API keys. Uses Claude Code CLI for auth and execution.
Leverages the same Blueprint schema, quality prompts, pre-flight validation,
and event emission as the LiteLLM pipeline.

Phase 1: Blueprint design via query() + Pydantic validation
Phase 2: Module generation via query() per module (parallel), Blueprint-constrained
Phase 3: Pre-flight validation + Claude Code fix agent
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)


# ── Cost tracking ───────────────────────────────────────────────────────────

# Pricing per million tokens (USD)
_PRICING: dict[str, dict[str, float]] = {
    "opus": {"input": 15.0, "output": 75.0, "cache_creation": 18.75, "cache_read": 1.5},
    "sonnet": {"input": 3.0, "output": 15.0, "cache_creation": 3.75, "cache_read": 0.3},
    "haiku": {"input": 0.25, "output": 1.25, "cache_creation": 0.3, "cache_read": 0.03},
}


def _get_pricing(model: str) -> dict[str, float]:
    """Get pricing tier for a model name."""
    m = model.lower()
    if "opus" in m:
        return _PRICING["opus"]
    if "haiku" in m:
        return _PRICING["haiku"]
    return _PRICING["sonnet"]


def _cost_from_usage(usage: dict, model: str) -> float:
    """Calculate cost in USD from a usage dict and model name."""
    if not usage:
        return 0.0
    p = _get_pricing(model)
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    cw = usage.get("cache_creation_input_tokens", 0)
    cr = usage.get("cache_read_input_tokens", 0)
    return (
        (inp / 1e6) * p["input"]
        + (out / 1e6) * p["output"]
        + (cw / 1e6) * p["cache_creation"]
        + (cr / 1e6) * p["cache_read"]
    )

from .prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    ASCII_DIAGRAM_GUIDE,
    CURRICULUM_DESIGN_SYSTEM_PROMPT,
    EXCALIDRAW_DIAGRAM_GUIDE,
    MODULE_CONVERSATION_SYSTEM_PROMPT,
    LESSON_TURN_TEMPLATE,
    REVIEW_SYSTEM_PROMPT,
)
from .diagrams import (
    clear_canvas,
    get_mcp_server_config,
    mcp_tools_available,
    render_module_diagrams,
    start_canvas_server,
    stop_canvas_server,
)
from .schemas import Analysis, CurriculumDesign, ModuleReview, slugify


# ── ANSI colors ──────────────────────────────────────────────────────────────


class _C:
    _enabled = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    RESET = "\033[0m" if _enabled else ""
    BOLD = "\033[1m" if _enabled else ""
    DIM = "\033[2m" if _enabled else ""
    RED = "\033[31m" if _enabled else ""
    GREEN = "\033[32m" if _enabled else ""
    YELLOW = "\033[33m" if _enabled else ""
    BLUE = "\033[34m" if _enabled else ""
    MAGENTA = "\033[35m" if _enabled else ""
    CYAN = "\033[36m" if _enabled else ""


# ── Event emission ───────────────────────────────────────────────────────────

_event_sink: ContextVar[Any] = ContextVar("_event_sink", default=None)
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _emit(event: dict) -> None:
    cb = _event_sink.get(None)
    if cb is not None:
        cb(event)


_start_time = 0.0


def _log(msg: str, color: str = "") -> None:
    elapsed = time.time() - _start_time if _start_time else 0
    mins, secs = divmod(int(elapsed), 60)
    ts = f"{_C.DIM}[{mins:02d}:{secs:02d}]{_C.RESET}"
    c = color or _C.RESET
    print(f"  {ts} {c}{msg}{_C.RESET}", file=sys.stderr, flush=True)
    _emit({"type": "log", "message": _ANSI_RE.sub("", msg)})


def _log_step(msg: str) -> None:
    print(file=sys.stderr, flush=True)
    _log(f"{_C.BOLD}{msg}", _C.CYAN)


_slugify = slugify  # local alias used throughout this file


# ── SDK helpers ──────────────────────────────────────────────────────────────


def _extract_text(messages: list) -> str:
    """Extract all text content from SDK message stream."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
    return "\n".join(parts)


def _extract_result(messages: list) -> ResultMessage | None:
    """Find the ResultMessage in an SDK message stream."""
    for msg in messages:
        if isinstance(msg, ResultMessage):
            return msg
    return None


def _extract_json(text: str) -> dict | None:
    """Extract JSON object from text that may contain markdown fences or prose."""
    # Try to find JSON in code fences first
    m = re.search(r'```(?:json)?\s*\n(\{.*?\})\s*\n```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find the first { ... } block
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


async def _query_sdk(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    effort: str | None = "high",
    max_turns: int = 50,
    cwd: str | Path | None = None,
    allowed_tools: list[str] | None = None,
    add_dirs: list[str | Path] | None = None,
    mcp_servers: dict | None = None,
) -> tuple[list, ResultMessage | None, dict]:
    """Run a Claude Agent SDK query and collect all messages.

    Returns (messages, result, usage) where usage is an aggregated dict with
    input_tokens, output_tokens, cache_creation_input_tokens,
    cache_read_input_tokens, cost_usd.
    """
    extra: dict[str, Any] = {}
    if effort:
        extra["effort"] = effort
    kwargs: dict[str, Any] = dict(
        system_prompt=system,
        model=model,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        cwd=str(cwd) if cwd else None,
        allowed_tools=allowed_tools or ["Bash", "Read", "Write", "Edit"],
        add_dirs=[str(d) for d in add_dirs] if add_dirs else [],
        extra_args=extra,
        env={"DISTILL_VERIFY": "1"},
    )
    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers
    options = ClaudeAgentOptions(**kwargs)

    messages: list = []
    agg = {"input_tokens": 0, "output_tokens": 0,
           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    try:
        async for msg in query(prompt=prompt, options=options):
            messages.append(msg)

            if isinstance(msg, AssistantMessage):
                # Accumulate tokens from each assistant turn
                if msg.usage:
                    agg["input_tokens"] += msg.usage.get("input_tokens", 0)
                    agg["output_tokens"] += msg.usage.get("output_tokens", 0)
                    agg["cache_creation_input_tokens"] += msg.usage.get("cache_creation_input_tokens", 0)
                    agg["cache_read_input_tokens"] += msg.usage.get("cache_read_input_tokens", 0)

                # Log tool use for progress tracking
                for block in msg.content:
                    if isinstance(block, TextBlock) and len(block.text) > 10:
                        _log(block.text[:120], _C.DIM)
                    elif isinstance(block, ToolUseBlock):
                        tool_args = ", ".join(
                            f"{k}={str(v)[:50]}" for k, v in (block.input or {}).items()
                        )
                        _log(f"{block.name}({tool_args})", _C.BLUE)
    except Exception as e:
        # SDK can throw on tool failures (exit code != 0) — recover gracefully
        # since we may already have enough messages/output to proceed
        _log(f"SDK query ended with error: {e}", _C.YELLOW)

    result = _extract_result(messages)
    model_name = model or "sonnet"

    # SDK has exact pricing (incl. ephemeral cache tiers) — prefer it
    if result and result.total_cost_usd:
        agg["cost_usd"] = result.total_cost_usd
    else:
        agg["cost_usd"] = _cost_from_usage(agg, model_name)

    return messages, result, agg


# ── Phase 1a: Analyze ────────────────────────────────────────────────────────


async def _phase_analyze(
    source_content: str,
    url: str,
    model: str,
) -> tuple[Analysis, dict]:
    """Analyze source material via Claude Code → structured Analysis."""
    _log_step("Phase 1a: Analyzing source material...")
    _emit({"type": "phase", "phase": "analyze"})

    prompt = (
        f"Analyze the following source material and respond with a JSON object "
        f"matching this exact schema. Output ONLY valid JSON, no markdown fences, "
        f"no prose before or after.\n\n"
        f"Schema:\n{json.dumps(Analysis.model_json_schema(), indent=2)}\n\n"
        f"Source URL: {url}\n\n"
        f"Source content:\n\n{source_content}"
    )

    messages, result, usage = await _query_sdk(
        prompt=prompt,
        system=ANALYSIS_SYSTEM_PROMPT,
        model=model,
        effort="high",
        max_turns=5,
        allowed_tools=[],  # no tools needed — just produce JSON
    )

    text = _extract_text(messages)
    data = _extract_json(text)
    if data is None:
        raise ValueError("Phase 1a: Claude Code did not return valid JSON for Analysis")

    analysis = Analysis(**data)

    essential = [c.name for c in analysis.key_concepts if c.priority == "essential"]
    supporting = [c.name for c in analysis.key_concepts if c.priority == "supporting"]
    contextual = [c.name for c in analysis.key_concepts if c.priority == "contextual"]

    _log(
        f"Analysis: {len(analysis.key_concepts)} concepts "
        f"({len(essential)} essential, {len(supporting)} supporting, "
        f"{len(contextual)} contextual), type={analysis.content_type} "
        f"(${usage['cost_usd']:.4f})",
        _C.GREEN,
    )

    return analysis, usage


# ── Phase 1b: Blueprint Design ───────────────────────────────────────────────


async def _phase_design(
    analysis: Analysis,
    source_content: str,
    url: str,
    student_level: str,
    model: str,
    output_dir: str | Path = "./output",
) -> tuple[CurriculumDesign, dict]:
    """Design Blueprint via Claude Code → structured CurriculumDesign."""
    _log_step("Phase 1b: Designing Blueprint...")
    _emit({"type": "phase", "phase": "design"})

    analysis_json = analysis.model_dump_json(indent=2)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Write the JSON to a file — the schema is too large for inline text output
    abs_output = Path(output_dir).resolve()
    abs_output.mkdir(parents=True, exist_ok=True)
    design_json_path = abs_output / "_blueprint.json"

    prompt = (
        f"Design a progressive course Blueprint matching this exact JSON schema.\n\n"
        f"Schema:\n{json.dumps(CurriculumDesign.model_json_schema(), indent=2)}\n\n"
        f"Source URL: {url}\n"
        f"Student level: {student_level}\n"
        f"Date: {today}\n\n"
        f"Analysis:\n{analysis_json}\n\n"
        f"Source material:\n\n{source_content}\n\n"
        f"IMPORTANT: The output JSON is large. Write the complete valid JSON object "
        f"to this file using the Write tool:\n"
        f"  {design_json_path}\n\n"
        f"Write ONLY valid JSON to the file — no markdown fences, no prose. "
        f"Make sure the JSON is complete and not truncated."
    )

    messages, result, usage = await _query_sdk(
        prompt=prompt,
        system=CURRICULUM_DESIGN_SYSTEM_PROMPT,
        model=model,
        effort="high",  # Blueprint design needs deep thinking
        max_turns=15,
        cwd=abs_output,
        allowed_tools=["Write", "Read", "Bash", "WebFetch", "ToolSearch"],
    )

    # Primary: read from the file the model wrote
    data = None
    if design_json_path.exists():
        try:
            raw = design_json_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError:
            # File exists but isn't valid JSON — try extracting
            data = _extract_json(raw)

    # Fallback: try extracting from text output
    if data is None:
        text = _extract_text(messages)
        data = _extract_json(text)

    if data is None:
        raise ValueError("Phase 1b: Claude Code did not return valid JSON for CurriculumDesign")

    design = CurriculumDesign(**data)
    curriculum = design.curriculum

    # Coverage check
    essential_names = {
        c.name for c in analysis.key_concepts if c.priority == "essential"
    }
    covered_names: set[str] = set()
    for m in curriculum.modules:
        covered_names.update(m.concepts_covered)
    uncovered = essential_names - covered_names
    if uncovered:
        _log(f"Coverage gap: {', '.join(sorted(uncovered))}", _C.YELLOW)

    _log(
        f"Blueprint: {len(curriculum.modules)} modules, "
        f"'{curriculum.course_title}' (${usage['cost_usd']:.4f})",
        _C.GREEN,
    )

    return design, usage


# ── Phase 2: Module Generation ───────────────────────────────────────────────


def _build_module_prompt(
    module_spec: dict,
    course_context: str,
    source_content: str,
    student_level: str,
    sources_dir: str | None,
    module_dir: Path,
    excalidraw_enabled: bool = False,
) -> str:
    """Build the comprehensive prompt for a module generation agent."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    exercises = module_spec.get("exercises", [])
    key_excerpts = module_spec.get("key_excerpts", [])

    # Build file manifest — explicit ordering
    file_list = [f"1. README.md — lesson document (5,000-10,000 words)"]
    file_num = 2
    if excalidraw_enabled:
        file_list.append(
            f"{file_num}. diagrams/*.excalidraw — 2-4 explanatory diagrams\n"
            f"   Use MCP tools (or Write) to create. Reference as ![desc](diagrams/name.svg) in README."
        )
        file_num += 1
    for i, ex in enumerate(exercises):
        ex_title = ex.get("title", f"exercise_{i+1}")
        ex_slug = _slugify(ex_title)
        ex_format = ex.get("format", "single_file")
        validate_cmd = ex.get("validate_command", "")

        if ex_format == "project":
            dirname = f"ex{i+1:02d}_{ex_slug}"
            file_list.append(
                f"{file_num}. {dirname}/ — project directory\n"
                f"   Create infrastructure files + stub files with TODOs\n"
                f"   {dirname}/_solutions/ — completed stub files only"
            )
            file_num += 1
            if validate_cmd:
                file_list.append(
                    f"   Validate: cd {dirname} && {validate_cmd}"
                )
        else:
            filename = f"ex{i+1:02d}_{ex_slug}.py"
            file_list.append(
                f"{file_num}. {filename} — scaffold (student version with TODOs)"
            )
            file_num += 1
            file_list.append(
                f"{file_num}. _solutions/{filename} — solution (complete, runnable)\n"
                f"   After writing, run: python _solutions/{filename}\n"
                f"   Verify output contains: \"{ex.get('expected_output_pattern', '')}\""
            )
            file_num += 1

    file_manifest = "\n".join(file_list)

    # Build exercise specs
    exercise_specs = []
    for i, ex in enumerate(exercises):
        ex_title = ex.get("title", f"exercise_{i+1}")
        ex_slug = _slugify(ex_title)
        ex_format = ex.get("format", "single_file")
        validate_cmd = ex.get("validate_command", "")
        provided_files = ex.get("provided_files", [])

        if ex_format == "project":
            dirname = f"ex{i+1:02d}_{ex_slug}"
            provided_str = ", ".join(provided_files) if provided_files else "(none specified)"
            exercise_specs.append(
                f"Exercise {i+1}: \"{ex_title}\" → {dirname}/ [PROJECT]\n"
                f"  Type: {ex.get('type', 'implement')}\n"
                f"  Format: project (multi-file directory)\n"
                f"  Infrastructure (provided, student does NOT modify): {provided_str}\n"
                f"  What student writes: {ex.get('what_student_writes', '')}\n"
                f"  Validate command: {validate_cmd}\n"
                f"  Key insight: {ex.get('key_insight', '')}\n"
                f"  Common mistakes: {ex.get('common_mistakes', '')}\n"
                f"  Milestone: {ex.get('milestone', '')}\n"
                f"  Expected output pattern: {ex.get('expected_output_pattern', '')}"
            )
        else:
            filename = f"ex{i+1:02d}_{ex_slug}.py"
            exercise_specs.append(
                f"Exercise {i+1}: \"{ex_title}\" → {filename}\n"
                f"  Type: {ex.get('type', 'implement')}\n"
                f"  Scaffolding: {ex.get('scaffolding_level', 'heavy')}\n"
                f"  What is provided (~65%): {ex.get('what_is_provided', '')}\n"
                f"  What student writes (~35%): {ex.get('what_student_writes', '')}\n"
                f"  Key insight: {ex.get('key_insight', '')}\n"
                f"  Common mistakes: {ex.get('common_mistakes', '')}\n"
                f"  Milestone: {ex.get('milestone', '')}\n"
                f"  Expected output pattern: {ex.get('expected_output_pattern', '')}"
            )
    exercise_block = "\n\n".join(exercise_specs)

    # Key excerpts
    excerpts_block = ""
    if key_excerpts:
        excerpts_block = (
            "KEY EXCERPTS FROM SOURCE (ground truth — translate directly to code):\n"
            + "\n".join(f"  [{i+1}] {exc}" for i, exc in enumerate(key_excerpts))
        )

    # Source access
    source_access = ""
    if sources_dir:
        source_access = (
            f"\nFull source material is available at: {sources_dir}\n"
            f"Use Read to access specific sections when needed.\n"
        )

    if excalidraw_enabled:
        diagram_bullet = "\n- 2-4 inline diagrams: ![Description](diagrams/name.svg) placed with explanations"
    else:
        diagram_bullet = "\n- 2-4 ASCII diagrams in fenced code blocks, placed inline with explanations"

    return f"""\
Execute this Blueprint for Module {idx}: "{title}"

Working directory: {module_dir}
Create all files in this directory.

{course_context}
Student level: {student_level}

═══════════════════════════════════════════════════════════════════════════
FILES TO CREATE (in this EXACT order — lesson FIRST)
═══════════════════════════════════════════════════════════════════════════

{file_manifest}

═══════════════════════════════════════════════════════════════════════════
LESSON DOCUMENT (README.md) — write this FIRST
═══════════════════════════════════════════════════════════════════════════

This is the primary teaching content. 5,000-10,000 words. NOT a summary.
- Table of contents + learning objectives
- Running example that evolves through the lesson
- Inline code snippets showing concept → code translation
- Embedded comprehension checks: "What would happen if...?"
- Formula translation: math → plain language → code (step by step)
- 2-4 analytical questions at Level 3+ depth (each as ### Question N — Title)
- Synthesis section reconnecting to the course goal{diagram_bullet}

═══════════════════════════════════════════════════════════════════════════
EXERCISE CONTRACTS (follow these EXACTLY)
═══════════════════════════════════════════════════════════════════════════

{exercise_block}

═══════════════════════════════════════════════════════════════════════════
SCAFFOLD PATTERN — students must write substantial code
═══════════════════════════════════════════════════════════════════════════

FOR SINGLE-FILE EXERCISES:

Each exercise MUST have 3-5 TODO blocks. Each TODO block must require
MINIMUM 5 lines, target 8-15 lines. No 2-3 line warm-up blocks — those
teach nothing. The student should write 30-60 lines TOTAL per exercise.

GOOD scaffold (~65% provided, ~35% student writes):
- Imports, class __init__, helper utilities: PROVIDED
- Core algorithm functions (2-3 functions): TODO blocks of 8-15 lines each
- __main__ test harness: PROVIDED (20-50 lines)

BAD scaffold (~95% provided, ~5% student writes):
- Everything provided except one 3-line function body
- Student writes 5 lines total — too easy, learns nothing

```python
def function_name(param):
    \"\"\"Docstring with Parameters, Returns, types.\"\"\"
    ###########################################################
    # YOUR CODE HERE - 8-12 lines                             #
    #                                                         #
    # Hint: specific hint about the approach                  #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################
```

FOR PROJECT EXERCISES:

Create directory ex{{NN}}_{{slug}}/ containing:
- Infrastructure files (from provided_files): complete, working, untouched by student
- Stub files: the files the student modifies, with TODO markers
- _solutions/: completed versions of ONLY the stub files
- A brief README.md explaining what to implement and how to validate

The infrastructure must actually work — build system, test harness, supporting
libraries. When solution files replace stubs, validate_command must exit 0.

═══════════════════════════════════════════════════════════════════════════
{excerpts_block}
═══════════════════════════════════════════════════════════════════════════
{source_access}
═══════════════════════════════════════════════════════════════════════════
MANDATORY: VALIDATE EVERY EXERCISE — NO EXCEPTIONS
═══════════════════════════════════════════════════════════════════════════

For EACH exercise, after writing the solution you MUST validate it:

  single_file: Run the solution script directly.
  project: Copy solution files over stubs, run validate_command.

Steps:
1. Run/validate the exercise
2. If it errors, fix and re-run until it succeeds
3. Copy the actual output — you need it for the next exercise

DO NOT skip validation for any exercise. DO NOT move to the next exercise
until the current one passes. The output feeds into subsequent exercises.
"""


async def _generate_module_claude(
    module_spec: dict,
    course_context: str,
    source_content: str,
    student_level: str,
    model: str,
    effort: str,
    course_dir: Path,
    sources_dir: str | None,
    mcp_config: dict | None = None,
) -> tuple[int, dict]:
    """Generate a module via Claude Code agent."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    _emit({"type": "module_start", "module_index": idx, "title": title})
    module_slug = f"module_{idx:02d}_{_slugify(title)}"
    module_dir = course_dir / module_slug
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "_solutions").mkdir(exist_ok=True)
    if mcp_config:
        (module_dir / "diagrams").mkdir(exist_ok=True)

    prompt = _build_module_prompt(
        module_spec=module_spec,
        course_context=course_context,
        source_content=source_content,
        student_level=student_level,
        sources_dir=sources_dir,
        module_dir=module_dir,
        excalidraw_enabled=bool(mcp_config),
    )

    _log(f"Module {idx}: launching Claude Code agent...", _C.BLUE)

    # Build tool list + diagram guide
    allowed = ["Bash", "Read", "Write", "Edit"]
    system = MODULE_CONVERSATION_SYSTEM_PROMPT
    if mcp_config:
        allowed.append("mcp__excalidraw__*")
        system += "\n\n" + EXCALIDRAW_DIAGRAM_GUIDE
    else:
        system += "\n\n" + ASCII_DIAGRAM_GUIDE

    messages, result, usage = await _query_sdk(
        prompt=prompt,
        system=system,
        model=model,
        effort=effort,
        max_turns=50,  # 30 was cutting modules short — agents need room for write→run→fix loops
        cwd=module_dir,
        allowed_tools=allowed,
        add_dirs=[sources_dir] if sources_dir else [],
        mcp_servers=mcp_config,
    )

    # Strip Pandoc-style heading anchors like {#anchor-id} from README
    readme_path = module_dir / "README.md"
    if readme_path.exists():
        import re
        content = readme_path.read_text(encoding="utf-8")
        cleaned = re.sub(r'\s*\{#[^}]+\}', '', content)
        if cleaned != content:
            readme_path.write_text(cleaned, encoding="utf-8")

    # Render Excalidraw diagrams to SVG
    diagram_count = render_module_diagrams(module_dir)
    if diagram_count:
        _log(f"Module {idx}: rendered {diagram_count} diagram(s) to SVG", _C.DIM)

    # Clear canvas for next module
    if mcp_config:
        await clear_canvas()

    # Clean up empty diagrams directory
    diagrams_dir = module_dir / "diagrams"
    if diagrams_dir.exists() and not any(diagrams_dir.iterdir()):
        diagrams_dir.rmdir()

    cost = usage["cost_usd"]
    turns = result.num_turns if result else 0

    # Count generated files
    files = list(module_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file() and not f.name.startswith("_"))

    _log(
        f"Module {idx} ({title}) generated "
        f"({file_count} files, {turns} turns, ${cost:.4f})",
        _C.GREEN,
    )
    _emit({"type": "module_complete", "module_index": idx, "title": title})

    return idx, {
        "files_written": file_count,
        "turns": turns,
        "cost": cost,
        "usage": usage,
    }


def _module_is_complete(course_dir: Path, module_spec: dict) -> bool:
    """Check if a module directory has a README and at least one exercise."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    slug = f"module_{idx:02d}_{_slugify(title)}"
    module_dir = course_dir / slug

    if not module_dir.is_dir():
        return False

    readme = module_dir / "README.md"
    if not readme.exists() or readme.stat().st_size < 500:
        return False

    # Check for at least one exercise file
    exercises = [f for f in module_dir.iterdir()
                 if f.is_file() and f.name.startswith("ex") and f.suffix == ".py"]
    return len(exercises) > 0


async def _phase_generate(
    design: CurriculumDesign,
    analysis: Analysis,
    source_content: str,
    student_level: str,
    model: str,
    effort: str,
    course_dir: Path,
    sources_dir: str | None,
    mcp_config: dict | None = None,
) -> dict[int, dict]:
    """Generate modules sequentially to avoid rate limits.

    Claude Code sessions are heavy — each spawns a CLI subprocess that makes
    multiple API calls. Running 5+ in parallel triggers rate limiting, causing
    agents to die with 0 turns and 0 files. Sequential execution is slower
    but reliable.

    Supports resume: modules with existing README + exercises are skipped.
    """
    curriculum = design.curriculum
    modules = curriculum.modules
    _log_step(f"Phase 2: Generating {len(modules)} modules sequentially...")
    _emit({"type": "phase", "phase": "generate"})

    concept_lines = "\n".join(
        f"  - {c.name} ({c.priority}): {c.description}"
        for c in analysis.key_concepts
    )
    module_map_lines = "\n".join(
        f"  Module {m.module_index}: {m.title}"
        for m in modules
    )
    course_context = (
        f"Course: {curriculum.course_title}\n"
        f"Description: {curriculum.course_description}\n"
        f"Content type: {analysis.content_type}\n\n"
        f"Course modules (use ONLY these indices for cross-references):\n"
        f"{module_map_lines}\n\n"
        f"Key concepts:\n{concept_lines}"
    )

    results: dict[int, dict] = {}

    for module in modules:
        module_spec = module.model_dump()
        idx = module_spec["module_index"]
        title = module_spec["title"]

        # Resume: skip modules that are already complete on disk
        if _module_is_complete(course_dir, module_spec):
            _log(f"Module {idx} ({title}): already complete — skipping", _C.GREEN)
            _emit({"type": "module_start", "module_index": idx, "title": title})
            _emit({"type": "module_complete", "module_index": idx, "title": title})
            slug = f"module_{idx:02d}_{_slugify(title)}"
            module_dir = course_dir / slug
            file_count = sum(1 for f in module_dir.rglob("*")
                             if f.is_file() and not f.name.startswith("_"))
            results[idx] = {"files_written": file_count, "turns": 0,
                            "cost": 0.0, "usage": {}, "resumed": True}
            continue

        try:
            _, summary = await _generate_module_claude(
                module_spec=module_spec,
                course_context=course_context,
                source_content=source_content,
                student_level=student_level,
                model=model,
                effort=effort,
                course_dir=course_dir,
                sources_dir=sources_dir,
                mcp_config=mcp_config,
            )

            # Check if the agent actually produced files
            if summary.get("files_written", 0) == 0:
                _log(f"Module {idx} ({title}): 0 files — retrying...", _C.YELLOW)
                _, summary = await _generate_module_claude(
                    module_spec=module_spec,
                    course_context=course_context,
                    source_content=source_content,
                    student_level=student_level,
                    model=model,
                    effort=effort,
                    course_dir=course_dir,
                    sources_dir=sources_dir,
                    mcp_config=mcp_config,
                )

            results[idx] = summary
        except Exception as e:
            print(f"  Module {idx} error: {e}", file=sys.stderr)
            _log(f"Module {idx} ({title}) failed ({type(e).__name__})", _C.RED)

    generated = len(results)
    total = len(modules)
    if generated == total:
        _log(f"All {total} modules generated", _C.GREEN)
    elif generated > 0:
        _log(f"{generated}/{total} modules generated", _C.YELLOW)
    else:
        _log(f"No modules generated", _C.RED)

    return results


# ── Phase 3: Review + Fix ────────────────────────────────────────────────────


async def _fix_module_claude(
    module_dir: Path,
    module_spec: dict,
    issues: list[str],
    sources_dir: str | None,
    model: str,
) -> None:
    """Fix issues in a module using Claude Code agent with Read/Edit/Bash."""
    idx = module_spec.get("module_index", 0)
    issues_text = "\n".join(f"  - {issue}" for issue in issues)

    prompt = f"""\
Fix quality issues in this course module.

Module directory: {module_dir}
Module: {module_spec.get('title', '')}

ISSUES FOUND:
{issues_text}

Instructions:
1. Read the files that have issues
2. Make TARGETED edits to fix each issue — do NOT rewrite files from scratch
3. After fixing solution files, run them with Bash to verify they work
4. Verify scaffold files parse without errors: python -c "import ast; ast.parse(open('filename').read())"
"""

    _log(f"Module {idx}: fixing {len(issues)} issues...", _C.CYAN)

    messages, result, usage = await _query_sdk(
        prompt=prompt,
        model=model,
        max_turns=20,
        cwd=module_dir,
        allowed_tools=["Bash", "Read", "Edit", "Write"],
        add_dirs=[sources_dir] if sources_dir else [],
    )

    turns = result.num_turns if result else 0
    _log(f"Module {idx}: fixes applied ({turns} turns, ${usage['cost_usd']:.4f})", _C.GREEN)
    return usage


async def _phase_review(
    design: CurriculumDesign,
    course_dir: Path,
    sources_dir: str | None,
    model: str,
    max_revision_cycles: int = 1,
) -> float:
    """Phase 3: pre-flight validation + Claude Code fix agent."""
    from .pipeline import _preflight_module

    curriculum = design.curriculum
    _log_step("Phase 3: Reviewing generated modules...")
    _emit({"type": "phase", "phase": "review"})

    review_cost = 0.0

    for cycle in range(max_revision_cycles + 1):
        if cycle > 0:
            _log(f"Revision cycle {cycle}/{max_revision_cycles}", _C.CYAN)

        # Pre-flight validation
        modules_with_issues: dict[int, list[str]] = {}
        for module in curriculum.modules:
            idx = module.module_index
            module_slug = f"module_{idx:02d}_{_slugify(module.title)}"
            module_dir = course_dir / module_slug
            if module_dir.exists():
                errors = _preflight_module(module_dir, module.model_dump())
                if errors:
                    modules_with_issues[idx] = errors
                    _log(f"Module {idx}: {len(errors)} pre-flight issues", _C.YELLOW)
                else:
                    _log(f"Module {idx}: pre-flight OK", _C.GREEN)

        if not modules_with_issues:
            _log("All modules passed pre-flight", _C.GREEN)
            return review_cost

        if cycle >= max_revision_cycles:
            _log(
                f"{len(modules_with_issues)} modules still have issues (max cycles reached)",
                _C.YELLOW,
            )
            return review_cost

        # Fix with Claude Code agents (sequential to avoid rate limits)
        module_map = {m.module_index: m for m in curriculum.modules}

        for idx, issues in modules_with_issues.items():
            mod = module_map.get(idx)
            if not mod:
                continue
            module_slug = f"module_{idx:02d}_{_slugify(mod.title)}"
            module_dir_path = course_dir / module_slug
            try:
                fix_usage = await _fix_module_claude(
                    module_dir=module_dir_path,
                    module_spec=mod.model_dump(),
                    issues=issues,
                    sources_dir=sources_dir,
                    model=model,
                )
                if fix_usage:
                    review_cost += fix_usage.get("cost_usd", 0.0)
            except Exception as e:
                print(f"  Module {idx} fix error: {e}", file=sys.stderr)
                _log(f"Module {idx}: fix failed ({type(e).__name__})", _C.RED)

    return review_cost


# ── Main pipeline ────────────────────────────────────────────────────────────


async def run_claude_pipeline(
    url: str,
    user_level: str,
    refs: list[str] | None = None,
    output_dir: str = "./output",
    design_model: str = "opus",
    generate_model: str = "sonnet",
    effort: str = "high",
    max_revision_cycles: int = 1,
    sources_dir: str | None = None,
    on_event: Any = None,
    diagram_mode: str = "ascii",
) -> dict:
    """Run the full course generation pipeline via Claude Code SDK.

    No LiteLLM, no external API keys. Claude Code CLI handles auth.
    """
    token = _event_sink.set(on_event) if on_event is not None else None

    global _start_time
    _start_time = time.time()
    total_cost = 0.0

    # ── Phase 0: Read preprocessed sources ────────────────────────────────
    _log_step("Reading preprocessed sources...")
    from .sources import prepare_sources
    if sources_dir:
        source_content = prepare_sources(sources_dir)
    else:
        source_content = f"[No preprocessed sources — original URL: {url}]"
    _log(f"Source content: ~{len(source_content) // 4} tokens", _C.DIM)

    # Aggregate usage across all phases
    total_usage = {"input_tokens": 0, "output_tokens": 0,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    def _accum(u: dict) -> None:
        for k in total_usage:
            total_usage[k] += u.get(k, 0)

    abs_output_dir = Path(output_dir).resolve()
    abs_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1a: Analyze (skip if cached) ────────────────────────────────
    analysis_path = abs_output_dir / "_analysis.json"
    if analysis_path.exists():
        _log_step("Phase 1a: Using cached analysis")
        _emit({"type": "phase", "phase": "analyze"})
        analysis = Analysis(**json.loads(analysis_path.read_text(encoding="utf-8")))
        _log(f"Analysis: {len(analysis.key_concepts)} concepts (cached)", _C.DIM)
    else:
        analysis, analyze_usage = await _phase_analyze(
            source_content=source_content,
            url=url,
            model=design_model,
        )
        total_cost += analyze_usage["cost_usd"]
        _accum(analyze_usage)
        analysis_path.write_text(
            analysis.model_dump_json(indent=2), encoding="utf-8"
        )

    # ── Phase 1b: Blueprint Design (skip if cached) ───────────────────────
    blueprint_path = abs_output_dir / "_blueprint.json"
    if blueprint_path.exists():
        _log_step("Phase 1b: Using cached Blueprint")
        _emit({"type": "phase", "phase": "design"})
        raw = json.loads(blueprint_path.read_text(encoding="utf-8"))
        design = CurriculumDesign(**raw)
        _log(f"Blueprint: {len(design.curriculum.modules)} modules (cached)", _C.DIM)
    else:
        design, design_usage = await _phase_design(
            analysis=analysis,
            source_content=source_content,
            url=url,
            student_level=user_level,
            model=design_model,
            output_dir=output_dir,
        )
        total_cost += design_usage["cost_usd"]
        _accum(design_usage)

    curriculum = design.curriculum
    course_slug = _slugify(curriculum.course_title)
    course_dir = abs_output_dir / course_slug
    course_dir.mkdir(parents=True, exist_ok=True)

    (course_dir / "_curriculum.json").write_text(
        curriculum.model_dump_json(indent=2), encoding="utf-8"
    )
    (course_dir / "README.md").write_text(design.root_readme, encoding="utf-8")

    req_content = design.requirements.strip()
    if req_content:
        (course_dir / "requirements.txt").write_text(req_content + "\n", encoding="utf-8")

    _log(f"Course directory: {course_dir}", _C.GREEN)

    # Emit curriculum event for DAG
    _emit({
        "type": "curriculum",
        "data": {
            "title": curriculum.course_title,
            "course_dir": str(course_dir),
            "modules": [
                {
                    "index": m.module_index,
                    "title": m.title,
                    "description": m.description,
                    "exercise_count": len(m.exercises),
                    "exercises": [e.title for e in m.exercises],
                    "depends_on": m.depends_on,
                }
                for m in curriculum.modules
            ],
        },
    })

    # ── Phase 2: Generate Modules ─────────────────────────────────────────
    # Start Excalidraw MCP canvas server if user chose excalidraw mode
    mcp_config = None
    canvas_proc = None
    if diagram_mode == "excalidraw" and mcp_tools_available():
        _log("Starting Excalidraw canvas server...", _C.DIM)
        canvas_proc = start_canvas_server()
        mcp_config = get_mcp_server_config()
        if mcp_config:
            _log("Excalidraw MCP enabled for diagram generation", _C.CYAN)
        else:
            _log("Excalidraw MCP unavailable — falling back to ASCII diagrams", _C.YELLOW)
    elif diagram_mode == "excalidraw":
        _log("Excalidraw not built — falling back to ASCII diagrams", _C.YELLOW)
    else:
        _log("Using ASCII diagrams", _C.DIM)

    try:
        module_outputs = await _phase_generate(
            design=design,
            analysis=analysis,
            source_content=source_content,
            student_level=user_level,
            model=generate_model,
            effort=effort,
            course_dir=course_dir,
            sources_dir=sources_dir,
            mcp_config=mcp_config,
        )
    finally:
        stop_canvas_server(canvas_proc)

    # Accumulate costs from module generation
    for summary in module_outputs.values():
        total_cost += summary.get("cost", 0.0)
        if "usage" in summary and summary["usage"]:
            _accum(summary["usage"])

    # ── Phase 3: Review + Fix ─────────────────────────────────────────────
    if not module_outputs:
        _log("Skipping review — no modules were generated", _C.YELLOW)
    else:
        review_cost = await _phase_review(
            design=design,
            course_dir=course_dir,
            sources_dir=sources_dir,
            model=generate_model,
            max_revision_cycles=max_revision_cycles,
        )
        total_cost += review_cost

    # ── Done ──────────────────────────────────────────────────────────────
    _emit({"type": "phase", "phase": "done"})
    total_elapsed = time.time() - _start_time
    mins, secs = divmod(int(total_elapsed), 60)

    generated_files = [
        f for f in course_dir.rglob("*")
        if f.is_file() and not f.name.startswith("_")
    ]
    dir_count = sum(1 for f in course_dir.rglob("*") if f.is_dir())

    print(file=sys.stderr)
    _log(
        f"{_C.BOLD}Done. {len(generated_files)} files in {dir_count} directories. "
        f"Time: {mins}m {secs}s",
        _C.GREEN,
    )
    _log(f"Course: {course_dir}", _C.GREEN)
    _log(f"Cost: ${total_cost:.4f}", _C.DIM)

    if token is not None:
        _event_sink.reset(token)

    return {
        "course_dir": str(course_dir),
        "total_cost_usd": total_cost,
        "usage": {
            "input_tokens": total_usage["input_tokens"],
            "output_tokens": total_usage["output_tokens"],
            "cache_creation_input_tokens": total_usage["cache_creation_input_tokens"],
            "cache_read_input_tokens": total_usage["cache_read_input_tokens"],
        },
    }
