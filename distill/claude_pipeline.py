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
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from .prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    CURRICULUM_DESIGN_SYSTEM_PROMPT,
    MODULE_CONVERSATION_SYSTEM_PROMPT,
    LESSON_TURN_TEMPLATE,
    REVIEW_SYSTEM_PROMPT,
)
from .schemas import Analysis, CurriculumDesign, ModuleReview


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


def _slugify(title: str) -> str:
    slug = title.lower()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = slug.strip().replace(" ", "_")
    return slug[:50].rstrip("_")


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
) -> tuple[list, ResultMessage | None]:
    """Run a Claude Code query and collect all messages."""
    extra: dict[str, str | None] = {}
    if effort:
        extra["effort"] = effort

    options = ClaudeCodeOptions(
        system_prompt=system,
        model=model,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        cwd=str(cwd) if cwd else None,
        allowed_tools=allowed_tools or ["Bash", "Read", "Write", "Edit"],
        add_dirs=[str(d) for d in add_dirs] if add_dirs else [],
        extra_args=extra,
    )

    messages: list = []
    async for msg in query(prompt=prompt, options=options):
        messages.append(msg)

        # Log tool use for progress tracking
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and len(block.text) > 10:
                    _log(block.text[:120], _C.DIM)
                elif isinstance(block, ToolUseBlock):
                    tool_args = ", ".join(
                        f"{k}={str(v)[:50]}" for k, v in (block.input or {}).items()
                    )
                    _log(f"{block.name}({tool_args})", _C.BLUE)

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
) -> CurriculumDesign:
    """Design Blueprint via Claude Code → structured CurriculumDesign."""
    _log_step("Phase 1b: Designing Blueprint...")
    _emit({"type": "phase", "phase": "design"})

    analysis_json = analysis.model_dump_json(indent=2)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = (
        f"Design a progressive course Blueprint and respond with a JSON object "
        f"matching this exact schema. Output ONLY valid JSON.\n\n"
        f"Schema:\n{json.dumps(CurriculumDesign.model_json_schema(), indent=2)}\n\n"
        f"Source URL: {url}\n"
        f"Student level: {student_level}\n"
        f"Date: {today}\n\n"
        f"Analysis:\n{analysis_json}\n\n"
        f"Source material:\n\n{source_content}"
    )

    messages, result = await _query_sdk(
        prompt=prompt,
        system=CURRICULUM_DESIGN_SYSTEM_PROMPT,
        model=model,
        effort="high",  # Blueprint design needs deep thinking
        max_turns=5,
        allowed_tools=[],
    )

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


def _build_module_prompt(
    module_spec: dict,
    course_context: str,
    source_content: str,
    student_level: str,
    sources_dir: str | None,
    module_dir: Path,
) -> str:
    """Build the comprehensive prompt for a module generation agent."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    exercises = module_spec.get("exercises", [])
    key_excerpts = module_spec.get("key_excerpts", [])

    # Build file manifest — explicit ordering
    file_list = [f"1. README.md — lesson document (3,000-10,000 words)"]
    file_num = 2
    for i, ex in enumerate(exercises):
        ex_title = ex.get("title", f"exercise_{i+1}")
        ex_slug = _slugify(ex_title)
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

This is the primary teaching content. 3,000-10,000 words. NOT a summary.
- Table of contents + learning objectives
- Running example that evolves through the lesson
- Inline code snippets showing concept → code translation
- Embedded comprehension checks: "What would happen if...?"
- Formula translation: math → plain language → code (step by step)
- 2-4 analytical questions at Level 3+ depth
- Synthesis section reconnecting to the course goal

═══════════════════════════════════════════════════════════════════════════
EXERCISE CONTRACTS (follow these EXACTLY)
═══════════════════════════════════════════════════════════════════════════

{exercise_block}

═══════════════════════════════════════════════════════════════════════════
SCAFFOLD PATTERN
═══════════════════════════════════════════════════════════════════════════

~65% provided code (imports, classes, helpers, __main__ harness)
~35% TODO blocks with line count hints

```python
def function_name(param):
    \"\"\"Numpy-style docstring with Parameters, Returns, types.\"\"\"
    ###########################################################
    # YOUR CODE HERE - 8-12 lines                             #
    #                                                         #
    # Hint: specific hint about the approach                  #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################
```

The __main__ block must be 20-50 lines, ALWAYS fully provided (never scaffolded).
It is the test harness that validates the student's work.

═══════════════════════════════════════════════════════════════════════════
{excerpts_block}
═══════════════════════════════════════════════════════════════════════════
{source_access}
AFTER writing each solution file, run it with Bash to verify it works.
If it fails, fix it before moving to the next exercise.

When writing exercise N+1, reference the actual output from exercise N
in the docstrings and comments ("In exercise 1 you saw accuracy of 31.2%...")
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
) -> tuple[int, dict]:
    """Generate a module via Claude Code agent."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    module_slug = f"module_{idx:02d}_{_slugify(title)}"
    module_dir = course_dir / module_slug
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "_solutions").mkdir(exist_ok=True)

    prompt = _build_module_prompt(
        module_spec=module_spec,
        course_context=course_context,
        source_content=source_content,
        student_level=student_level,
        sources_dir=sources_dir,
        module_dir=module_dir,
    )

    _log(f"Module {idx}: launching Claude Code agent...", _C.BLUE)

    messages, result = await _query_sdk(
        prompt=prompt,
        system=MODULE_CONVERSATION_SYSTEM_PROMPT,
        model=model,
        effort=effort,
        max_turns=30,
        cwd=module_dir,
        allowed_tools=["Bash", "Read", "Write", "Edit"],
        add_dirs=[sources_dir] if sources_dir else [],
    )

    cost = result.total_cost_usd if result else 0.0
    turns = result.num_turns if result else 0

    # Count generated files
    files = list(module_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file() and not f.name.startswith("_"))

    _log(f"Module {idx} ({title}) generated ({file_count} files, {turns} turns)", _C.GREEN)
    _emit({"type": "module_complete", "module_index": idx, "title": title})

    return idx, {
        "files_written": file_count,
        "turns": turns,
        "cost": cost,
    }


async def _phase_generate(
    design: CurriculumDesign,
    analysis: Analysis,
    source_content: str,
    student_level: str,
    model: str,
    effort: str,
    course_dir: Path,
    sources_dir: str | None,
) -> dict[int, dict]:
    """Generate all modules in parallel via Claude Code agents."""
    curriculum = design.curriculum
    modules = curriculum.modules
    _log_step(f"Phase 2: Generating {len(modules)} modules in parallel...")
    _emit({"type": "phase", "phase": "generate"})

    concept_lines = "\n".join(
        f"  - {c.name} ({c.priority}): {c.description}"
        for c in analysis.key_concepts
    )
    course_context = (
        f"Course: {curriculum.course_title}\n"
        f"Description: {curriculum.course_description}\n"
        f"Content type: {analysis.content_type}\n\n"
        f"Key concepts:\n{concept_lines}"
    )

    results: dict[int, dict] = {}

    async def _run(module_spec: dict) -> None:
        idx = module_spec["module_index"]
        title = module_spec["title"]
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
            )
            results[idx] = summary
        except Exception as e:
            print(f"  Module {idx} error: {e}", file=sys.stderr)
            _log(f"Module {idx} ({title}) failed ({type(e).__name__})", _C.RED)

    async with anyio.create_task_group() as tg:
        for module in modules:
            tg.start_soon(_run, module.model_dump())

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

        # Fix with Claude Code agents
        module_map = {m.module_index: m for m in curriculum.modules}

        async def _fix(idx: int, issues: list[str]) -> None:
            mod = module_map.get(idx)
            if not mod:
                return
            module_slug = f"module_{idx:02d}_{_slugify(mod.title)}"
            module_dir = course_dir / module_slug
            await _fix_module_claude(
                module_dir=module_dir,
                module_spec=mod.model_dump(),
                issues=issues,
                sources_dir=sources_dir,
                model=model,
            )

        async with anyio.create_task_group() as tg:
            for idx, issues in modules_with_issues.items():
                tg.start_soon(_fix, idx, issues)


# ── Main pipeline ────────────────────────────────────────────────────────────


async def run_claude_pipeline(
    url: str,
    user_level: str,
    refs: list[str] | None = None,
    series: bool = False,
    output_dir: str = "./output",
    design_model: str = "opus",
    generate_model: str = "sonnet",
    effort: str = "high",
    max_revision_cycles: int = 1,
    sources_dir: str | None = None,
    on_event: Any = None,
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

    # ── Phase 1a: Analyze ─────────────────────────────────────────────────
    analysis = await _phase_analyze(
        source_content=source_content,
        url=url,
        model=design_model,
    )

    abs_output_dir = Path(output_dir).resolve()
    abs_output_dir.mkdir(parents=True, exist_ok=True)
    (abs_output_dir / "_analysis.json").write_text(
        analysis.model_dump_json(indent=2), encoding="utf-8"
    )

    # ── Phase 1b: Blueprint Design ────────────────────────────────────────
    design = await _phase_design(
        analysis=analysis,
        source_content=source_content,
        url=url,
        student_level=user_level,
        model=design_model,
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

    # ── Phase 2: Generate Modules ─────────────────────────────────────────
    module_outputs = await _phase_generate(
        design=design,
        analysis=analysis,
        source_content=source_content,
        student_level=user_level,
        model=generate_model,
        effort=effort,
        course_dir=course_dir,
        sources_dir=sources_dir,
    )

    # Accumulate costs from module generation
    for summary in module_outputs.values():
        total_cost += summary.get("cost", 0.0)

    # ── Phase 3: Review + Fix ─────────────────────────────────────────────
    if not module_outputs:
        _log("Skipping review — no modules were generated", _C.YELLOW)
    else:
        await _phase_review(
            design=design,
            course_dir=course_dir,
            sources_dir=sources_dir,
            model=generate_model,  # use cheaper model for fixes
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
    _log(f"Cost: ${total_cost:.4f}", _C.DIM)

    if token is not None:
        _event_sink.reset(token)

    return {
        "course_dir": str(course_dir),
        "total_cost_usd": total_cost,
        "usage": None,
    }
