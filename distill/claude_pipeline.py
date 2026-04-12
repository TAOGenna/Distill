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


from .prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    ASCII_DIAGRAM_GUIDE,
    CURRICULUM_DESIGN_SYSTEM_PROMPT,
    EXCALIDRAW_DIAGRAM_GUIDE,
    MODULE_CONVERSATION_SYSTEM_PROMPT,
    LESSON_TURN_TEMPLATE,
    REVIEW_SYSTEM_PROMPT,
    SOURCE_IMAGE_GUIDE,
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
) -> tuple[list, ResultMessage | None]:
    """Run a Claude Agent SDK query and collect all messages.

    Returns (messages, result).
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

    try:
        async for msg in query(prompt=prompt, options=options):
            messages.append(msg)

            if isinstance(msg, AssistantMessage):
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

    return messages, result


# ── Phase 1a: Analyze ────────────────────────────────────────────────────────


async def _phase_analyze(
    source_content: str,
    url: str,
    model: str,
) -> Analysis:
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

    messages, result = await _query_sdk(
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
        f"{len(contextual)} contextual), type={analysis.content_type}",
        _C.GREEN,
    )

    return analysis


# ── Phase 1b: Blueprint Design ───────────────────────────────────────────────


async def _phase_design(
    analysis: Analysis,
    source_content: str,
    url: str,
    student_level: str,
    model: str,
    output_dir: str | Path = "./output",
) -> CurriculumDesign:
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

    messages, result = await _query_sdk(
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
        f"'{curriculum.course_title}'",
        _C.GREEN,
    )

    return design


# ── Phase 2: Module Generation ───────────────────────────────────────────────


def _build_exercise_specs(exercises: list[dict]) -> str:
    """Build the exercise specs block shared by lesson and exercise prompts."""
    specs = []
    for i, ex in enumerate(exercises):
        ex_title = ex.get("title", f"exercise_{i+1}")
        ex_slug = _slugify(ex_title)
        ex_format = ex.get("format", "single_file")
        validate_cmd = ex.get("validate_command", "")
        provided_files = ex.get("provided_files", [])

        if ex_format == "project":
            dirname = f"ex{i+1:02d}_{ex_slug}"
            provided_str = ", ".join(provided_files) if provided_files else "(none specified)"
            specs.append(
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
            specs.append(
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
    return "\n\n".join(specs)


def _build_lesson_prompt(
    module_spec: dict,
    course_context: str,
    source_content: str,
    student_level: str,
    sources_dir: str | None,
    module_dir: Path,
    excalidraw_enabled: bool = False,
    source_images: list[dict] | None = None,
) -> str:
    """Build prompt for lesson-only generation (README + diagrams)."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    exercises = module_spec.get("exercises", [])
    key_excerpts = module_spec.get("key_excerpts", [])

    # File manifest: only lesson + diagrams
    file_list = ["1. README.md — lesson document (5,000-10,000 words)"]
    if excalidraw_enabled:
        file_list.append(
            "2. diagrams/*.excalidraw — 2-4 explanatory diagrams\n"
            "   Use MCP tools (or Write) to create. Reference as ![desc](diagrams/name.svg) in README."
        )

    exercise_block = _build_exercise_specs(exercises)

    # Key excerpts
    excerpts_block = ""
    if key_excerpts:
        excerpts_block = (
            "KEY EXCERPTS FROM SOURCE (ground truth — use in the lesson):\n"
            + "\n".join(f"  [{i+1}] {exc}" for i, exc in enumerate(key_excerpts))
        )

    # Source access
    source_access = ""
    if sources_dir:
        source_access = (
            f"\nFull source material is available at: {sources_dir}\n"
            f"Use Read to access specific sections when needed.\n"
        )

    # Source images catalog
    images_block = ""
    if source_images:
        img_lines = []
        for img in source_images:
            alt = img.get("alt_text", "")
            desc = f' — "{alt}"' if alt else ""
            img_lines.append(f"  • {img['path']}{desc}")
        images_block = (
            "\n═══════════════════════════════════════════════════════════════════════════\n"
            "AVAILABLE SOURCE IMAGES (from the original material)\n"
            "═══════════════════════════════════════════════════════════════════════════\n\n"
            f"{len(source_images)} images available. View them with Read to decide which "
            f"to include in the lesson. Copy selected images to ./images/ and reference "
            f"in README.md.\n\n"
            + "\n".join(img_lines)
            + "\n"
        )

    if excalidraw_enabled:
        diagram_bullet = "\n- 2-4 inline diagrams: ![Description](diagrams/name.svg) placed with explanations"
    else:
        diagram_bullet = "\n- 2-4 ASCII diagrams in fenced code blocks, placed inline with explanations"

    return f"""\
Write the lesson for Module {idx}: "{title}"

Working directory: {module_dir}

{course_context}
Student level: {student_level}

═══════════════════════════════════════════════════════════════════════════
FILES TO CREATE
═══════════════════════════════════════════════════════════════════════════

{chr(10).join(file_list)}

Write ONLY the lesson README (and diagrams if applicable).
Do NOT create any exercise files (.py) or _solutions/ directory.
Those will be created in a subsequent step.

═══════════════════════════════════════════════════════════════════════════
LESSON DOCUMENT (README.md)
═══════════════════════════════════════════════════════════════════════════

This is the primary teaching content. 5,000-10,000 words. NOT a summary.
- Start with # Module N: Title, then go DIRECTLY to ## Table of Contents and ## Learning Objectives. No blockquotes, course name, module sequence links, or epigraphs between title and TOC
- TOC anchors must be full heading text slugified: lowercase, spaces→hyphens, strip non-alphanumeric. Never abbreviated slugs
- Running example that evolves through the lesson
- Inline code snippets showing concept → code translation
- Embedded comprehension checks: "What would happen if...?"
- Formula translation: math → plain language → code (step by step)
- 2-4 analytical questions at Level 3+ depth (each as ### Question N — Title)
- Synthesis section reconnecting to the course goal{diagram_bullet}

═══════════════════════════════════════════════════════════════════════════
UPCOMING EXERCISES (reference these in the lesson for foreshadowing)
═══════════════════════════════════════════════════════════════════════════

{exercise_block}

═══════════════════════════════════════════════════════════════════════════
{excerpts_block}
═══════════════════════════════════════════════════════════════════════════
{source_access}
{images_block}"""


def _build_exercises_prompt(
    module_spec: dict,
    course_context: str,
    student_level: str,
    module_dir: Path,
    sources_dir: str | None = None,
) -> str:
    """Build prompt for exercise-only generation (scaffolds + solutions)."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    exercises = module_spec.get("exercises", [])
    key_excerpts = module_spec.get("key_excerpts", [])

    # File manifest: only exercises
    file_list = []
    file_num = 1
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
                file_list.append(f"   Validate: cd {dirname} && {validate_cmd}")
        else:
            filename = f"ex{i+1:02d}_{ex_slug}.py"
            file_list.append(f"{file_num}. {filename} — scaffold (student version with TODOs)")
            file_num += 1
            file_list.append(
                f"{file_num}. _solutions/{filename} — solution (complete, runnable)\n"
                f"   After writing, run: python _solutions/{filename}\n"
                f"   Verify output contains: \"{ex.get('expected_output_pattern', '')}\""
            )
            file_num += 1

    exercise_block = _build_exercise_specs(exercises)

    excerpts_block = ""
    if key_excerpts:
        excerpts_block = (
            "KEY EXCERPTS FROM SOURCE (ground truth — translate directly to code):\n"
            + "\n".join(f"  [{i+1}] {exc}" for i, exc in enumerate(key_excerpts))
        )

    source_access = ""
    if sources_dir:
        source_access = (
            f"\nFull source material is available at: {sources_dir}\n"
            f"Use Read to access specific sections when needed.\n"
        )

    return f"""\
Generate exercises for Module {idx}: "{title}"

Working directory: {module_dir}

FIRST: Read ./README.md to understand the lesson context. The exercises
must be consistent with the lesson's running example, terminology, and
progression.

{course_context}
Student level: {student_level}

═══════════════════════════════════════════════════════════════════════════
FILES TO CREATE (in this EXACT order)
═══════════════════════════════════════════════════════════════════════════

{chr(10).join(file_list)}

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


async def _generate_lesson_claude(
    module_spec: dict,
    course_context: str,
    source_content: str,
    student_level: str,
    model: str,
    effort: str,
    course_dir: Path,
    sources_dir: str | None,
    mcp_config: dict | None = None,
    source_images: list[dict] | None = None,
) -> tuple[int, dict]:
    """Generate only the lesson README (+ diagrams) via Claude Code agent."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    _emit({"type": "module_start", "module_index": idx, "title": title})
    module_slug = f"module_{idx:02d}_{_slugify(title)}"
    module_dir = course_dir / module_slug
    module_dir.mkdir(parents=True, exist_ok=True)
    if mcp_config:
        (module_dir / "diagrams").mkdir(exist_ok=True)

    prompt = _build_lesson_prompt(
        module_spec=module_spec,
        course_context=course_context,
        source_content=source_content,
        student_level=student_level,
        sources_dir=sources_dir,
        module_dir=module_dir,
        excalidraw_enabled=bool(mcp_config),
        source_images=source_images,
    )

    _log(f"Module {idx}: writing lesson...", _C.BLUE)

    # Build tool list + guides
    allowed = ["Bash", "Read", "Write", "Edit"]
    system = MODULE_CONVERSATION_SYSTEM_PROMPT
    if source_images:
        system += "\n\n" + SOURCE_IMAGE_GUIDE
    if mcp_config:
        allowed.append("mcp__excalidraw__*")
        system += "\n\n" + EXCALIDRAW_DIAGRAM_GUIDE
    else:
        system += "\n\n" + ASCII_DIAGRAM_GUIDE

    messages, result = await _query_sdk(
        prompt=prompt,
        system=system,
        model=model,
        effort=effort,
        max_turns=25,  # lesson + diagrams only — fewer turns needed
        cwd=module_dir,
        allowed_tools=allowed,
        add_dirs=[sources_dir] if sources_dir else [],
        mcp_servers=mcp_config,
    )

    # Strip Pandoc-style heading anchors from README
    readme_path = module_dir / "README.md"
    if readme_path.exists():
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

    turns = result.num_turns if result else 0

    readme_ok = _lesson_is_complete(course_dir, module_spec)
    if readme_ok:
        _log(f"Module {idx} ({title}) lesson ready ({turns} turns)", _C.GREEN)
        _emit({
            "type": "lesson_ready",
            "module_index": idx,
            "title": title,
            "dir_name": module_slug,
        })
    else:
        _log(f"Module {idx} ({title}) lesson FAILED — no README written ({turns} turns)", _C.RED)

    return idx, {
        "lesson_turns": turns,
    }


async def _generate_exercises_claude(
    module_spec: dict,
    course_context: str,
    student_level: str,
    model: str,
    effort: str,
    course_dir: Path,
    sources_dir: str | None,
) -> tuple[int, dict]:
    """Generate exercises for a module via Claude Code agent.

    The agent reads the already-written README.md for lesson context.
    """
    idx = module_spec["module_index"]
    title = module_spec["title"]
    module_slug = f"module_{idx:02d}_{_slugify(title)}"
    module_dir = course_dir / module_slug
    (module_dir / "_solutions").mkdir(exist_ok=True)

    prompt = _build_exercises_prompt(
        module_spec=module_spec,
        course_context=course_context,
        student_level=student_level,
        module_dir=module_dir,
        sources_dir=sources_dir,
    )

    _log(f"Module {idx}: building exercises...", _C.BLUE)

    messages, result = await _query_sdk(
        prompt=prompt,
        system=MODULE_CONVERSATION_SYSTEM_PROMPT,
        model=model,
        effort=effort,
        max_turns=40,  # exercises need room for write→run→fix loops
        cwd=module_dir,
        allowed_tools=["Bash", "Read", "Write", "Edit"],
        add_dirs=[sources_dir] if sources_dir else [],
    )

    turns = result.num_turns if result else 0

    # Count exercise files
    files = list(module_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file() and not f.name.startswith("_")
                     and f.name != "README.md" and f.parent.name != "diagrams")

    _log(
        f"Module {idx} ({title}) exercises done "
        f"({file_count} files, {turns} turns)",
        _C.GREEN,
    )
    _emit({"type": "module_complete", "module_index": idx, "title": title})

    return idx, {
        "exercise_files": file_count,
        "exercise_turns": turns,
    }


def _lesson_is_complete(course_dir: Path, module_spec: dict) -> bool:
    """Check if a module has a substantive README."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    slug = f"module_{idx:02d}_{_slugify(title)}"
    module_dir = course_dir / slug

    if not module_dir.is_dir():
        return False

    readme = module_dir / "README.md"
    return readme.exists() and readme.stat().st_size >= 500


def _exercises_are_complete(course_dir: Path, module_spec: dict) -> bool:
    """Check if a module has at least one exercise file."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    slug = f"module_{idx:02d}_{_slugify(title)}"
    module_dir = course_dir / slug

    if not module_dir.is_dir():
        return False

    exercises = [f for f in module_dir.iterdir()
                 if f.is_file() and f.name.startswith("ex") and f.suffix == ".py"]
    return len(exercises) > 0


def _build_course_context(design: CurriculumDesign, analysis: Analysis) -> str:
    """Build shared course context string for module generation."""
    curriculum = design.curriculum
    concept_lines = "\n".join(
        f"  - {c.name} ({c.priority}): {c.description}"
        for c in analysis.key_concepts
    )
    module_map_lines = "\n".join(
        f"  Module {m.module_index}: {m.title}"
        for m in curriculum.modules
    )
    return (
        f"Course: {curriculum.course_title}\n"
        f"Description: {curriculum.course_description}\n"
        f"Content type: {analysis.content_type}\n\n"
        f"Course modules (use ONLY these indices for cross-references):\n"
        f"{module_map_lines}\n\n"
        f"Key concepts:\n{concept_lines}"
    )


async def _phase_generate_lessons(
    design: CurriculumDesign,
    analysis: Analysis,
    source_content: str,
    student_level: str,
    model: str,
    effort: str,
    course_dir: Path,
    sources_dir: str | None,
    mcp_config: dict | None = None,
    source_images: list[dict] | None = None,
) -> dict[int, dict]:
    """Phase 2a: Generate all lesson READMEs sequentially.

    Returns dict of module_index → summary_dict.
    Supports resume: modules with existing README are skipped.
    """
    curriculum = design.curriculum
    modules = curriculum.modules
    _log_step(f"Phase 2a: Writing {len(modules)} lessons sequentially...")
    _emit({"type": "phase", "phase": "generate_lessons"})

    course_context = _build_course_context(design, analysis)
    results: dict[int, dict] = {}

    for module in modules:
        module_spec = module.model_dump()
        idx = module_spec["module_index"]
        title = module_spec["title"]

        if _lesson_is_complete(course_dir, module_spec):
            _log(f"Module {idx} ({title}): lesson already complete — skipping", _C.GREEN)
            slug = f"module_{idx:02d}_{_slugify(title)}"
            _emit({"type": "module_start", "module_index": idx, "title": title})
            _emit({"type": "lesson_ready", "module_index": idx, "title": title, "dir_name": slug})
            results[idx] = {"lesson_turns": 0, "resumed": True}
            continue

        try:
            _, summary = await _generate_lesson_claude(
                module_spec=module_spec,
                course_context=course_context,
                source_content=source_content,
                student_level=student_level,
                model=model,
                effort=effort,
                course_dir=course_dir,
                sources_dir=sources_dir,
                mcp_config=mcp_config,
                source_images=source_images,
            )
            # Retry once if MCP crash left no README (transient SDK error)
            if not _lesson_is_complete(course_dir, module_spec):
                _log(f"Module {idx} ({title}): retrying lesson...", _C.YELLOW)
                _, summary = await _generate_lesson_claude(
                    module_spec=module_spec,
                    course_context=course_context,
                    source_content=source_content,
                    student_level=student_level,
                    model=model,
                    effort=effort,
                    course_dir=course_dir,
                    sources_dir=sources_dir,
                    mcp_config=mcp_config,
                    source_images=source_images,
                )
            results[idx] = summary
        except Exception as e:
            print(f"  Module {idx} lesson error: {e}", file=sys.stderr)
            _log(f"Module {idx} ({title}) lesson failed ({type(e).__name__})", _C.RED)

    _log(f"{len(results)}/{len(modules)} lessons written", _C.GREEN if len(results) == len(modules) else _C.YELLOW)
    return results


async def _phase_generate_exercises(
    design: CurriculumDesign,
    analysis: Analysis,
    student_level: str,
    model: str,
    effort: str,
    course_dir: Path,
    sources_dir: str | None,
) -> dict[int, dict]:
    """Phase 2b: Generate exercises for all modules sequentially.

    Supports resume: modules with existing exercises are skipped.
    """
    curriculum = design.curriculum
    modules = curriculum.modules
    _log_step(f"Phase 2b: Building exercises for {len(modules)} modules...")
    _emit({"type": "phase", "phase": "generate_exercises"})

    course_context = _build_course_context(design, analysis)
    results: dict[int, dict] = {}

    for module in modules:
        module_spec = module.model_dump()
        idx = module_spec["module_index"]
        title = module_spec["title"]

        if _exercises_are_complete(course_dir, module_spec):
            _log(f"Module {idx} ({title}): exercises already complete — skipping", _C.GREEN)
            _emit({"type": "module_complete", "module_index": idx, "title": title})
            results[idx] = {"exercise_files": 0, "exercise_turns": 0, "resumed": True}
            continue

        if not _lesson_is_complete(course_dir, module_spec):
            _log(f"Module {idx} ({title}): no lesson — skipping exercises", _C.YELLOW)
            continue

        try:
            _, summary = await _generate_exercises_claude(
                module_spec=module_spec,
                course_context=course_context,
                student_level=student_level,
                model=model,
                effort=effort,
                course_dir=course_dir,
                sources_dir=sources_dir,
            )
            results[idx] = summary
        except Exception as e:
            print(f"  Module {idx} exercise error: {e}", file=sys.stderr)
            _log(f"Module {idx} ({title}) exercises failed ({type(e).__name__})", _C.RED)

    _log(f"{len(results)}/{len(modules)} modules' exercises generated", _C.GREEN if len(results) == len(modules) else _C.YELLOW)
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

    messages, result = await _query_sdk(
        prompt=prompt,
        model=model,
        max_turns=20,
        cwd=module_dir,
        allowed_tools=["Bash", "Read", "Edit", "Write"],
        add_dirs=[sources_dir] if sources_dir else [],
    )

    turns = result.num_turns if result else 0
    _log(f"Module {idx}: fixes applied ({turns} turns)", _C.GREEN)


async def _phase_review(
    design: CurriculumDesign,
    course_dir: Path,
    sources_dir: str | None,
    model: str,
    max_revision_cycles: int = 1,
) -> None:
    """Phase 3: pre-flight validation + Claude Code fix agent."""
    from .pipeline import _preflight_module

    curriculum = design.curriculum
    _log_step("Phase 3: Reviewing generated modules...")
    _emit({"type": "phase", "phase": "review"})

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
            return

        if cycle >= max_revision_cycles:
            _log(
                f"{len(modules_with_issues)} modules still have issues (max cycles reached)",
                _C.YELLOW,
            )
            return

        # Fix with Claude Code agents (sequential to avoid rate limits)
        module_map = {m.module_index: m for m in curriculum.modules}

        for idx, issues in modules_with_issues.items():
            mod = module_map.get(idx)
            if not mod:
                continue
            module_slug = f"module_{idx:02d}_{_slugify(mod.title)}"
            module_dir_path = course_dir / module_slug
            try:
                await _fix_module_claude(
                    module_dir=module_dir_path,
                    module_spec=mod.model_dump(),
                    issues=issues,
                    sources_dir=sources_dir,
                    model=model,
                )
            except Exception as e:
                print(f"  Module {idx} fix error: {e}", file=sys.stderr)
                _log(f"Module {idx}: fix failed ({type(e).__name__})", _C.RED)


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

    # ── Phase 0: Read preprocessed sources ────────────────────────────────
    _log_step("Reading preprocessed sources...")
    from .sources import get_source_images, prepare_sources
    if sources_dir:
        source_content = prepare_sources(sources_dir)
    else:
        source_content = f"[No preprocessed sources — original URL: {url}]"
    _log(f"Source content: ~{len(source_content) // 4} tokens", _C.DIM)

    # Load source images catalog (figures from papers, blog posts, etc.)
    source_images: list[dict] = []
    if sources_dir:
        source_images = get_source_images(sources_dir)
        if source_images:
            _log(f"Found {len(source_images)} source images available for modules", _C.CYAN)

    abs_output_dir = Path(output_dir).resolve()
    abs_output_dir.mkdir(parents=True, exist_ok=True)

    # Cache key: hash URL + refs so different source sets get separate caches
    import hashlib
    _cache_parts = url + "\n" + "\n".join(sorted(refs or []))
    _cache_key = hashlib.sha256(_cache_parts.encode()).hexdigest()[:12]

    # ── Phase 1a: Analyze (skip if cached) ────────────────────────────────
    analysis_path = abs_output_dir / f"_analysis_{_cache_key}.json"
    # Migrate old URL-only cache key from before refs were included
    if not analysis_path.exists():
        _old_key = hashlib.sha256(url.encode()).hexdigest()[:12]
        _old_path = abs_output_dir / f"_analysis_{_old_key}.json"
        if _old_path.exists():
            _old_path.rename(analysis_path)
    if analysis_path.exists():
        _log_step("Phase 1a: Using cached analysis")
        _emit({"type": "phase", "phase": "analyze"})
        analysis = Analysis(**json.loads(analysis_path.read_text(encoding="utf-8")))
        _log(f"Analysis: {len(analysis.key_concepts)} concepts (cached)", _C.DIM)
    else:
        analysis = await _phase_analyze(
            source_content=source_content,
            url=url,
            model=design_model,
        )
        analysis_path.write_text(
            analysis.model_dump_json(indent=2), encoding="utf-8"
        )

    # ── Phase 1b: Blueprint Design (skip if cached) ───────────────────────
    blueprint_path = abs_output_dir / f"_blueprint_{_cache_key}.json"
    # Also check unhashed _blueprint.json (written by older runs / the agent)
    blueprint_fallback = abs_output_dir / "_blueprint.json"
    if not blueprint_path.exists() and blueprint_fallback.exists():
        blueprint_fallback.rename(blueprint_path)
    if blueprint_path.exists():
        _log_step("Phase 1b: Using cached Blueprint")
        _emit({"type": "phase", "phase": "design"})
        raw = json.loads(blueprint_path.read_text(encoding="utf-8"))
        design = CurriculumDesign(**raw)
        _log(f"Blueprint: {len(design.curriculum.modules)} modules (cached)", _C.DIM)
    else:
        design = await _phase_design(
            analysis=analysis,
            source_content=source_content,
            url=url,
            student_level=user_level,
            model=design_model,
            output_dir=output_dir,
        )
        blueprint_path.write_text(
            design.model_dump_json(indent=2), encoding="utf-8"
        )

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

    # ── Phase 2a: Generate Lessons ──────────────────────────────────────
    # Start Excalidraw MCP canvas server for lesson diagrams
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
        lesson_outputs = await _phase_generate_lessons(
            design=design,
            analysis=analysis,
            source_content=source_content,
            student_level=user_level,
            model=generate_model,
            effort=effort,
            course_dir=course_dir,
            sources_dir=sources_dir,
            mcp_config=mcp_config,
            source_images=source_images or None,
        )
    finally:
        stop_canvas_server(canvas_proc)

    # ── Phase 2b: Generate Exercises ─────────────────────────────────
    exercise_outputs = await _phase_generate_exercises(
        design=design,
        analysis=analysis,
        student_level=user_level,
        model=generate_model,
        effort=effort,
        course_dir=course_dir,
        sources_dir=sources_dir,
    )

    # Merge into module_outputs for Phase 3
    module_outputs: dict[int, dict] = {}
    for idx in set(lesson_outputs) | set(exercise_outputs):
        module_outputs[idx] = {
            **lesson_outputs.get(idx, {}),
            **exercise_outputs.get(idx, {}),
        }

    # ── Phase 3: Review + Fix ─────────────────────────────────────────────
    if not module_outputs:
        _log("Skipping review — no modules were generated", _C.YELLOW)
    else:
        await _phase_review(
            design=design,
            course_dir=course_dir,
            sources_dir=sources_dir,
            model=generate_model,
            max_revision_cycles=max_revision_cycles,
        )

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

    if token is not None:
        _event_sink.reset(token)

    # Check if every module has both a lesson and exercises
    all_complete = all(
        _lesson_is_complete(course_dir, m.model_dump())
        and _exercises_are_complete(course_dir, m.model_dump())
        for m in curriculum.modules
    )

    return {
        "course_dir": str(course_dir),
        "all_complete": all_complete,
    }
