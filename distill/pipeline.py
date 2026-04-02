"""Course generation pipeline — replaces the agent loop with direct API calls.

Three-phase architecture using structured output:
  Phase 1a: Analyze source material → Analysis
  Phase 1b: Design curriculum → CurriculumDesign (curriculum + root files)
  Phase 2:  Generate modules (parallel) → multi-turn conversations
  Phase 3:  Pre-flight validation + LLM quality review
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio

from .llm import LLMClient, QuotaExhaustedError
from .prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    CURRICULUM_DESIGN_SYSTEM_PROMPT,
    LESSON_TURN_TEMPLATE,
    MODULE_CONVERSATION_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    SCAFFOLD_TURN_TEMPLATE,
    SOLUTION_TURN_TEMPLATE,
)
from .schemas import (
    Analysis,
    CurriculumDesign,
    ModuleReview,
)
from .sources import prepare_sources_with_summary


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


# ── Logging ──────────────────────────────────────────────────────────────────

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


# ── Helpers ──────────────────────────────────────────────────────────────────


from .schemas import slugify

_slugify = slugify  # local alias used throughout this file


# ── Phase 1a: Analyze ────────────────────────────────────────────────────────


async def _phase_analyze(
    llm: LLMClient,
    source_content: str,
    url: str,
    model: str,
) -> Analysis:
    """Analyze source material → structured Analysis."""
    _log_step("Phase 1a: Analyzing source material...")
    _emit({"type": "phase", "phase": "analyze"})

    messages = [{
        "role": "user",
        "content": (
            f"Analyze the following source material from: {url}\n\n"
            f"Source content:\n\n{source_content}"
        ),
    }]

    result = await llm.complete(
        messages=messages,
        model=model,
        system=ANALYSIS_SYSTEM_PROMPT,
        response_model=Analysis,
        max_tokens=8192,
    )

    analysis = result.structured
    assert isinstance(analysis, Analysis)

    essential = [c.name for c in analysis.key_concepts if c.priority == "essential"]
    supporting = [c.name for c in analysis.key_concepts if c.priority == "supporting"]
    contextual = [c.name for c in analysis.key_concepts if c.priority == "contextual"]

    _log(
        f"Analysis complete: {len(analysis.key_concepts)} concepts "
        f"({len(essential)} essential, {len(supporting)} supporting, "
        f"{len(contextual)} contextual), content_type={analysis.content_type}",
        _C.GREEN,
    )

    return analysis


# ── Phase 1b: Design Curriculum ──────────────────────────────────────────────


async def _phase_design(
    llm: LLMClient,
    analysis: Analysis,
    source_content: str,
    url: str,
    student_level: str,
    model: str,
) -> CurriculumDesign:
    """Design curriculum + root files → structured CurriculumDesign."""
    _log_step("Phase 1b: Designing curriculum...")
    _emit({"type": "phase", "phase": "design"})

    analysis_json = analysis.model_dump_json(indent=2)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    messages = [{
        "role": "user",
        "content": (
            f"Design a progressive CS231n-style curriculum based on this analysis.\n\n"
            f"Source URL: {url}\n"
            f"Student level: {student_level}\n"
            f"Date: {today}\n\n"
            f"Analysis:\n{analysis_json}\n\n"
            f"Source material excerpts (for reference when writing the README):\n\n"
            f"{source_content}"
        ),
    }]

    result = await llm.complete(
        messages=messages,
        model=model,
        system=CURRICULUM_DESIGN_SYSTEM_PROMPT,
        response_model=CurriculumDesign,
        max_tokens=16384,
        max_retries=3,
    )

    design = result.structured
    assert isinstance(design, CurriculumDesign)

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
        _log(
            f"Coverage gap: essential concepts not covered: {', '.join(sorted(uncovered))}",
            _C.YELLOW,
        )

    _log(
        f"Curriculum: {len(curriculum.modules)} modules, "
        f"'{curriculum.course_title}'",
        _C.GREEN,
    )

    return design


# ── Phase 2: Generate Modules ────────────────────────────────────────────────


def _validate_syntax_str(content: str, path: str, language: str) -> str | None:
    """Validate syntax for a file content string. Returns error or None."""
    if language == "python":
        try:
            ast.parse(content)
            return None
        except SyntaxError as e:
            return f"{path}: SyntaxError at line {e.lineno}: {e.msg}"
    return None


# ── Code execution helper ────────────────────────────────────────────────────


def _execute_python(file_path: Path, timeout: int = 10) -> str:
    """Execute a Python file and capture output. Returns stdout+stderr."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(file_path.parent),
            env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = ""
        if result.stdout:
            output += result.stdout[:2000]
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr[:1000])
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"
    except Exception as e:
        return f"(execution failed: {type(e).__name__})"


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped its output."""
    text = text.strip()
    # Remove ```python ... ``` or ```markdown ... ``` wrapping
    import re
    m = re.match(r'^```\w*\s*\n(.*?)```\s*$', text, re.DOTALL)
    if m:
        text = m.group(1)
    # Strip Pandoc-style heading anchors like {#anchor-id}
    text = re.sub(r'\s*\{#[^}]+\}', '', text)
    return text


# ── Conversational module generation ─────────────────────────────────────────


async def _generate_module_conversational(
    llm: LLMClient,
    module_spec: dict,
    course_context: str,
    source_content: str,
    student_level: str,
    model: str,
    course_dir: Path,
    shared_defs: dict | None = None,
) -> tuple[int, dict]:
    """Generate a module via multi-turn conversation.

    Returns (module_index, summary_dict) with files written to disk.
    """
    idx = module_spec["module_index"]
    title = module_spec["title"]
    _emit({"type": "module_start", "module_index": idx, "title": title})
    module_slug = f"module_{idx:02d}_{_slugify(title)}"
    module_dir = course_dir / module_slug
    module_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir = module_dir / "_solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)

    exercises = module_spec.get("exercises", [])
    key_excerpts = module_spec.get("key_excerpts", [])
    errors: list[str] = []

    # Build conversation messages
    messages: list[dict] = []

    def _add_user(content: str) -> None:
        messages.append({"role": "user", "content": content})

    def _add_assistant(content: str) -> None:
        messages.append({"role": "assistant", "content": content})

    # ── Turn 1: Generate the lesson (README) ─────────────────────────

    objectives = "\n".join(
        f"  - {obj}" for obj in module_spec.get("learning_objectives", [])
    )
    excerpts_text = "\n".join(
        f"  [{i+1}] {exc}" for i, exc in enumerate(key_excerpts)
    ) or "  (none provided)"

    exercise_summaries = "\n".join(
        f"  Exercise {i+1}: \"{ex.get('title', '')}\" ({ex.get('type', '')}) "
        f"[{ex.get('format', 'single_file')}]\n"
        f"    Student writes: {ex.get('what_student_writes', '')}\n"
        f"    Milestone: {ex.get('milestone', '')}"
        for i, ex in enumerate(exercises)
    )

    lesson_prompt = LESSON_TURN_TEMPLATE.format(
        module_title=title,
        module_description=module_spec.get("description", ""),
        objectives=objectives,
        key_excerpts=excerpts_text,
        exercise_summaries=exercise_summaries,
        student_level=student_level,
        source_content=source_content,
    )

    _add_user(lesson_prompt)
    _log(f"Module {idx}: writing lesson...", _C.BLUE)

    result = await llm.complete(
        messages=messages,
        model=model,
        system=MODULE_CONVERSATION_SYSTEM_PROMPT,
        max_tokens=16384,
    )
    lesson_content = _strip_code_fences(result.content)
    _add_assistant(lesson_content)

    # Save README
    (module_dir / "README.md").write_text(lesson_content, encoding="utf-8")
    lesson_words = len(lesson_content.split())
    _log(f"Module {idx}: lesson written ({lesson_words} words)", _C.DIM)

    # ── Turns 2-N: Generate exercises (scaffold then solution) ───────

    prev_execution_output: str | None = None

    for ex_i, ex in enumerate(exercises):
        ex_num = ex_i + 1
        ex_title = ex.get("title", f"exercise_{ex_num}")
        ex_format = ex.get("format", "single_file")

        # Project exercises need agent tool access (Bash/Write/Edit) — skip in LiteLLM
        if ex_format == "project":
            _log(
                f"Module {idx}: skipping ex{ex_num:02d} (project format — "
                f"use Claude Code provider for project exercises)",
                _C.YELLOW,
            )
            continue

        ex_slug = _slugify(ex_title)
        filename = f"ex{ex_num:02d}_{ex_slug}.py"

        # Build predecessor context
        predecessor_context = ""
        if prev_execution_output:
            predecessor_context = (
                f"\nPrevious exercise execution output:\n"
                f"{prev_execution_output}\n"
                f"Reference these actual results in this exercise's narrative."
            )

        # ── Scaffold turn ────────────────────────────────────────────

        scaffold_prompt = SCAFFOLD_TURN_TEMPLATE.format(
            ex_index=ex_num,
            ex_title=ex_title,
            ex_type=ex.get("type", "implement"),
            what_is_provided=ex.get("what_is_provided", ""),
            what_student_writes=ex.get("what_student_writes", ""),
            key_insight=ex.get("key_insight", ""),
            common_mistakes=ex.get("common_mistakes", ""),
            milestone=ex.get("milestone", ""),
            predecessor_context=predecessor_context,
        )
        _add_user(scaffold_prompt)

        result = await llm.complete(
            messages=messages,
            model=model,
            system=MODULE_CONVERSATION_SYSTEM_PROMPT,
            max_tokens=8192,
        )
        scaffold_code = _strip_code_fences(result.content)
        _add_assistant(scaffold_code)

        # Save scaffold + validate
        scaffold_path = module_dir / filename
        scaffold_path.write_text(scaffold_code, encoding="utf-8")

        err = _validate_syntax_str(scaffold_code, filename, "python")
        if err:
            errors.append(f"scaffold {err}")
            _add_user(f"Syntax error in scaffold: {err}. Please fix and rewrite the complete file.")
            fix_result = await llm.complete(
                messages=messages,
                model=model,
                system=MODULE_CONVERSATION_SYSTEM_PROMPT,
                max_tokens=8192,
            )
            scaffold_code = _strip_code_fences(fix_result.content)
            _add_assistant(scaffold_code)
            scaffold_path.write_text(scaffold_code, encoding="utf-8")

        # ── Solution turn ────────────────────────────────────────────

        solution_prompt = SOLUTION_TURN_TEMPLATE.format(
            milestone=ex.get("milestone", ""),
            expected_pattern=ex.get("expected_output_pattern", ""),
        )
        _add_user(solution_prompt)

        result = await llm.complete(
            messages=messages,
            model=model,
            system=MODULE_CONVERSATION_SYSTEM_PROMPT,
            max_tokens=8192,
        )
        solution_code = _strip_code_fences(result.content)
        _add_assistant(solution_code)

        # Save solution + validate
        solution_path = solutions_dir / filename
        solution_path.write_text(solution_code, encoding="utf-8")

        err = _validate_syntax_str(solution_code, filename, "python")
        if err:
            errors.append(f"solution {err}")
            _add_user(f"Syntax error in solution: {err}. Please fix and rewrite the complete file.")
            fix_result = await llm.complete(
                messages=messages,
                model=model,
                system=MODULE_CONVERSATION_SYSTEM_PROMPT,
                max_tokens=8192,
            )
            solution_code = _strip_code_fences(fix_result.content)
            _add_assistant(solution_code)
            solution_path.write_text(solution_code, encoding="utf-8")

        # Execute solution and capture output for next exercise
        exec_output = _execute_python(solution_path)
        prev_execution_output = exec_output

        # Feed execution result back into conversation
        _add_user(
            f"Exercise {ex_num} solution executed.\n"
            f"Output:\n{exec_output}\n\n"
            f"Acknowledged. Ready for next exercise."
        )
        _add_assistant("Understood. Ready for the next exercise.")

        _log(
            f"Module {idx}: ex{ex_num:02d} done ({len(scaffold_code.splitlines())} "
            f"scaffold / {len(solution_code.splitlines())} solution lines)",
            _C.DIM,
        )

    _log(f"Module {idx} ({title}) generated", _C.GREEN)
    _emit({"type": "module_complete", "module_index": idx, "title": title})

    return idx, {
        "files_written": 1 + len(exercises) * 2,  # README + (scaffold + solution) per exercise
        "lesson_words": lesson_words,
        "errors": errors,
    }


async def _phase_generate(
    llm: LLMClient,
    design: CurriculumDesign,
    analysis: Analysis,
    source_content: str,
    student_level: str,
    model: str,
    course_dir: Path,
) -> dict[int, dict]:
    """Generate all modules in parallel via multi-turn conversations.

    Returns dict of module_index → summary dict.
    """
    curriculum = design.curriculum
    modules = curriculum.modules
    _log_step(f"Phase 2: Generating {len(modules)} modules in parallel...")
    _emit({"type": "phase", "phase": "generate"})

    # Build shared course context
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

    async def _run(module_spec: dict) -> None:
        idx = module_spec["module_index"]
        title = module_spec["title"]
        try:
            _, summary = await _generate_module_conversational(
                llm=llm,
                module_spec=module_spec,
                course_context=course_context,
                source_content=source_content,
                student_level=student_level,
                model=model,
                course_dir=course_dir,
            )
            results[idx] = summary

        except QuotaExhaustedError:
            _log(f"Module {idx} ({title}): API quota exhausted — aborting", _C.RED)
            raise
        except Exception as e:
            print(f"  Module {idx} error: {e}", file=sys.stderr)
            _log(f"Module {idx} ({title}) failed ({type(e).__name__})", _C.RED)

    try:
        async with anyio.create_task_group() as tg:
            for module in modules:
                tg.start_soon(_run, module.model_dump())
    except BaseException as e:
        if any(isinstance(exc, QuotaExhaustedError) for exc in getattr(e, 'exceptions', [e])):
            _log("Generation aborted: API quota exhausted. Check your billing.", _C.RED)

    generated = len(results)
    total = len(modules)
    if generated == total:
        _log(f"All {total} modules generated", _C.GREEN)
    elif generated > 0:
        _log(f"{generated}/{total} modules generated ({total - generated} failed)", _C.YELLOW)
    else:
        _log(f"No modules generated — all {total} failed", _C.RED)

    return results


# ── Phase 3a: Pre-flight validation ──────────────────────────────────────────


def _preflight_module(module_dir: Path, module_spec: dict | None = None) -> list[str]:
    """Run deterministic + contract-aware checks. Returns list of errors."""
    import shutil
    import subprocess
    import tempfile

    from .schemas import slugify

    errors: list[str] = []
    exercises = (module_spec or {}).get("exercises", [])

    # Check README exists and length
    readme_path = module_dir / "README.md"
    if not readme_path.exists():
        errors.append("Missing README.md")
    else:
        word_count = len(readme_path.read_text(errors="replace").split())
        if word_count < 4000:
            errors.append(f"README.md: only ~{word_count} words (expected 5,000+)")

    # ── Single-file exercise checks ──────────────────────────────────────
    for py_file in module_dir.glob("*.py"):
        content = py_file.read_text(errors="replace")

        # Syntax check
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            errors.append(f"{py_file.name}: SyntaxError at line {e.lineno}: {e.msg}")
            continue

        # Check for __main__ block
        has_main = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            ):
                has_main = True
                break
        if not has_main:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == "main":
                        has_main = True
                        break
        if not has_main:
            errors.append(f"{py_file.name}: missing __main__ block or main() call")

        # Contract-aware: check TODO markers exist (scaffold should have them)
        if "YOUR CODE HERE" not in content and "TODO" not in content:
            errors.append(f"{py_file.name}: no TODO/YOUR CODE HERE markers found")

        # Contract-aware: check file isn't trivially short (likely placeholder)
        lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]
        if len(lines) < 15:
            errors.append(f"{py_file.name}: only {len(lines)} non-comment lines (expected 40+)")

    # ── Solution verification for single-file exercises ──────────────────
    solutions_dir = module_dir / "_solutions"
    if solutions_dir.exists():
        for sol_file in sorted(solutions_dir.glob("*.py")):
            sol_content = sol_file.read_text(errors="replace")

            # Dry-run import check: skip if deps are unavailable
            try:
                sol_tree = ast.parse(sol_content)
            except SyntaxError:
                continue
            skip = False
            for node in ast.walk(sol_tree):
                mod_name = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod_name = alias.name.split(".")[0]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mod_name = node.module.split(".")[0]
                if mod_name and mod_name not in sys.stdlib_module_names:
                    try:
                        __import__(mod_name)
                    except ImportError:
                        skip = True
                        break
            if skip:
                continue

            try:
                result = subprocess.run(
                    [sys.executable, str(sol_file)],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(module_dir),
                )
                if result.returncode != 0:
                    stderr_snippet = result.stderr.strip().splitlines()[-3:]
                    errors.append(
                        f"_solutions/{sol_file.name}: execution failed — "
                        + "; ".join(stderr_snippet)
                    )
            except subprocess.TimeoutExpired:
                errors.append(
                    f"_solutions/{sol_file.name}: execution timed out (>60s)"
                )
            except Exception as e:
                errors.append(
                    f"_solutions/{sol_file.name}: could not run ({type(e).__name__})"
                )

    # ── Project exercise checks ──────────────────────────────────────────
    for i, ex in enumerate(exercises):
        if ex.get("format") != "project":
            continue
        idx = i + 1
        ex_slug = slugify(ex.get("title", ""))
        ex_dir = module_dir / f"ex{idx:02d}_{ex_slug}"

        if not ex_dir.exists():
            errors.append(f"ex{idx:02d}_{ex_slug}/: project directory missing")
            continue

        # Check solution files exist
        sol_dir = ex_dir / "_solutions"
        if not sol_dir.exists() or not any(sol_dir.iterdir()):
            errors.append(f"ex{idx:02d}_{ex_slug}/_solutions/: missing or empty")

        # Check stub files have TODO markers
        for stub in ex_dir.iterdir():
            if stub.is_file() and stub.suffix in (".py", ".go", ".rs", ".c", ".java", ".js", ".ts"):
                if stub.name.startswith("_"):
                    continue
                content = stub.read_text(errors="replace")
                if "YOUR CODE HERE" not in content and "TODO" not in content:
                    errors.append(f"{ex_dir.name}/{stub.name}: no TODO markers found")

        # Validate with validate_command — swap in solutions first
        validate_cmd = ex.get("validate_command", "")
        if validate_cmd and sol_dir.exists():
            # Copy project to temp dir, overlay solutions, run validation there
            tmp_dir = None
            try:
                tmp_dir = Path(tempfile.mkdtemp(prefix="distill_validate_"))
                shutil.copytree(ex_dir, tmp_dir / "project", dirs_exist_ok=True)
                tmp_project = tmp_dir / "project"
                # Overlay solution files onto stubs
                for sol_file in sol_dir.iterdir():
                    if sol_file.is_file():
                        shutil.copy2(sol_file, tmp_project / sol_file.name)
                result = subprocess.run(
                    validate_cmd, shell=True,
                    capture_output=True, text=True, timeout=120,
                    cwd=str(tmp_project),
                )
                if result.returncode != 0:
                    stderr_snippet = result.stderr.strip().splitlines()[-3:]
                    errors.append(
                        f"ex{idx:02d}_{ex_slug}/: validate_command failed — "
                        + "; ".join(stderr_snippet)
                    )
            except subprocess.TimeoutExpired:
                errors.append(
                    f"ex{idx:02d}_{ex_slug}/: validate_command timed out (>120s)"
                )
            except Exception as e:
                errors.append(
                    f"ex{idx:02d}_{ex_slug}/: validate_command error ({type(e).__name__})"
                )
            finally:
                if tmp_dir and tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Language-specific checks ─────────────────────────────────────────
    for c_file in list(module_dir.glob("*.c")) + list(module_dir.glob("*.rs")):
        content = c_file.read_text(errors="replace")
        if "main(" not in content and "fn main" not in content:
            errors.append(f"{c_file.name}: missing main() function")

    return errors


# ── Phase 3b: LLM Quality Review ─────────────────────────────────────────────


async def _review_module(
    llm: LLMClient,
    module_index: int,
    module_dir: Path,
    module_spec: dict,
    model: str,
) -> ModuleReview:
    """Review a single module for pedagogical quality."""
    # Gather module content
    parts: list[str] = []
    readme_path = module_dir / "README.md"
    if readme_path.exists():
        parts.append(f"[README.md]\n{readme_path.read_text(errors='replace')}\n")

    for f in sorted(module_dir.iterdir()):
        if f.is_file() and f.name != "README.md":
            content = f.read_text(errors="replace")
            parts.append(f"[{f.name}]\n{content}\n")

    module_content = "\n---\n".join(parts)
    spec_json = json.dumps(module_spec, indent=2, ensure_ascii=False)

    messages = [{
        "role": "user",
        "content": (
            f"Review Module {module_index}: \"{module_spec.get('title', '')}\"\n\n"
            f"Module specification (what was requested):\n{spec_json}\n\n"
            f"Generated files:\n\n{module_content}"
        ),
    }]

    result = await llm.complete(
        messages=messages,
        model=model,
        system=REVIEW_SYSTEM_PROMPT,
        response_model=ModuleReview,
        max_tokens=4096,
    )

    review = result.structured
    assert isinstance(review, ModuleReview)
    # Ensure module_index matches
    review.module_index = module_index
    return review


async def _phase_review(
    llm: LLMClient,
    design: CurriculumDesign,
    analysis: Analysis,
    source_content: str,
    student_level: str,
    generate_model: str,
    review_model: str,
    course_dir: Path,
    module_outputs: dict[int, dict],
    max_revision_cycles: int = 1,
) -> None:
    """Phase 3: pre-flight validation + LLM quality review + targeted re-generation."""
    curriculum = design.curriculum
    _log_step("Phase 3: Reviewing generated modules...")
    _emit({"type": "phase", "phase": "review"})

    for cycle in range(max_revision_cycles + 1):
        if cycle > 0:
            _log(f"Revision cycle {cycle}/{max_revision_cycles}", _C.CYAN)

        # 3a: Pre-flight validation (with contract checks)
        preflight_failures: dict[int, list[str]] = {}
        for module in curriculum.modules:
            idx = module.module_index
            module_slug = f"module_{idx:02d}_{_slugify(module.title)}"
            module_dir = course_dir / module_slug
            if module_dir.exists():
                errors = _preflight_module(module_dir, module.model_dump())
                if errors:
                    preflight_failures[idx] = errors
                    _log(
                        f"Module {idx} pre-flight: {len(errors)} issues",
                        _C.YELLOW,
                    )

        # 3b: LLM quality review (for modules that pass pre-flight)
        reviews: list[ModuleReview] = []

        async def _review(mod: Any) -> None:
            idx = mod.module_index
            if idx in preflight_failures:
                reviews.append(ModuleReview(
                    module_index=idx,
                    verdict="revise",
                    issues=[],
                ))
                return

            module_slug = f"module_{idx:02d}_{_slugify(mod.title)}"
            module_dir_path = course_dir / module_slug
            if not module_dir_path.exists():
                return

            try:
                review = await _review_module(
                    llm=llm,
                    module_index=idx,
                    module_dir=module_dir_path,
                    module_spec=mod.model_dump(),
                    model=review_model,
                )
                reviews.append(review)
                verdict_color = _C.GREEN if review.verdict == "pass" else _C.YELLOW
                _log(
                    f"Module {idx} review: {review.verdict} "
                    f"({len(review.issues)} issues)",
                    verdict_color,
                )
            except Exception as e:
                print(f"  Module {idx} review error: {e}", file=sys.stderr)
                _log(f"Module {idx} review failed ({type(e).__name__})", _C.RED)
                reviews.append(ModuleReview(
                    module_index=idx,
                    verdict="pass",
                    issues=[],
                ))

        async with anyio.create_task_group() as tg:
            for mod in curriculum.modules:
                tg.start_soon(_review, mod)

        # Check overall result
        modules_to_revise = [r for r in reviews if r.verdict == "revise"]
        if not modules_to_revise:
            _log("All modules passed review", _C.GREEN)
            return

        if cycle >= max_revision_cycles:
            _log(
                f"{len(modules_to_revise)} modules still need revision "
                f"(max cycles reached)",
                _C.YELLOW,
            )
            return

        # Re-generate failed modules
        _log(
            f"Re-generating {len(modules_to_revise)} modules...",
            _C.CYAN,
        )

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

        async def _regenerate(mod: Any, review: ModuleReview) -> None:
            idx = mod.module_index
            feedback_parts: list[str] = []
            if idx in preflight_failures:
                feedback_parts.append(
                    "Pre-flight errors:\n" +
                    "\n".join(f"  - {e}" for e in preflight_failures[idx])
                )
            for issue in review.issues:
                feedback_parts.append(
                    f"- [{issue.criterion}] {issue.description}"
                    + (f" (file: {issue.file_path})" if issue.file_path else "")
                    + f"\n  Fix: {issue.suggested_fix}"
                )

            try:
                _, summary = await _generate_module_conversational(
                    llm=llm,
                    module_spec=mod.model_dump(),
                    course_context=course_context,
                    source_content=source_content,
                    student_level=student_level,
                    model=generate_model,
                    course_dir=course_dir,
                )
                module_outputs[idx] = summary
            except Exception as e:
                print(f"  Module {idx} re-generation error: {e}", file=sys.stderr)
                _log(f"Module {idx} re-generation failed ({type(e).__name__})", _C.RED)

        # Find module specs for modules that need revision
        module_map = {m.module_index: m for m in curriculum.modules}
        review_map = {r.module_index: r for r in modules_to_revise}

        async with anyio.create_task_group() as tg:
            for idx, review in review_map.items():
                if idx in module_map:
                    tg.start_soon(_regenerate, module_map[idx], review)


# ── Main pipeline ────────────────────────────────────────────────────────────


async def run_pipeline(
    url: str,
    user_level: str,
    refs: list[str] | None = None,
    output_dir: str = "./output",
    provider: str = "anthropic",
    api_key: str | None = None,
    design_model: str | None = None,
    generate_model: str | None = None,
    max_revision_cycles: int = 1,
    sources_dir: str | None = None,
    on_event: Any = None,
) -> dict:
    """Run the full course generation pipeline.

    Returns a dict with course_dir, total_cost_usd, and usage.
    """
    from .llm import PROVIDER_DEFAULTS

    token = _event_sink.set(on_event) if on_event is not None else None

    # Resolve models
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["anthropic"])
    design_model = design_model or defaults["design"]
    generate_model = generate_model or defaults["generate"]

    if provider == "mock":
        from .mock import MockLLMClient
        llm = MockLLMClient()
    else:
        llm = LLMClient(provider=provider, api_key=api_key)

    global _start_time
    _start_time = time.time()

    # ── Phase 0: Read preprocessed sources ────────────────────────────────
    _log_step("Reading preprocessed sources...")
    if sources_dir:
        source_content = await prepare_sources_with_summary(
            sources_dir=sources_dir,
            llm_client=llm,
            summary_model=generate_model,  # use cheap model for summarization
        )
    else:
        source_content = f"[No preprocessed sources — original URL: {url}]"

    _log(f"Source content: ~{len(source_content) // 4} tokens", _C.DIM)

    # ── Phase 1a: Analyze ─────────────────────────────────────────────────
    analysis = await _phase_analyze(
        llm=llm,
        source_content=source_content,
        url=url,
        model=design_model,
    )

    # Save analysis
    abs_output_dir = Path(output_dir).resolve()
    abs_output_dir.mkdir(parents=True, exist_ok=True)
    (abs_output_dir / "_analysis.json").write_text(
        analysis.model_dump_json(indent=2), encoding="utf-8"
    )

    # ── Phase 1b: Design Curriculum ───────────────────────────────────────
    design = await _phase_design(
        llm=llm,
        analysis=analysis,
        source_content=source_content,
        url=url,
        student_level=user_level,
        model=design_model,
    )

    curriculum = design.curriculum

    # Create course directory and write root files
    course_slug = _slugify(curriculum.course_title)
    course_dir = abs_output_dir / course_slug
    course_dir.mkdir(parents=True, exist_ok=True)

    (course_dir / "_curriculum.json").write_text(
        curriculum.model_dump_json(indent=2), encoding="utf-8"
    )
    (course_dir / "README.md").write_text(design.root_readme, encoding="utf-8")

    # Write requirements/setup file
    req_content = design.requirements.strip()
    if req_content:
        # Detect file type from content
        if req_content.startswith("[package]") or req_content.startswith("[workspace]"):
            filename = "Cargo.toml"
        elif req_content.startswith("{"):
            filename = "package.json"
        elif ":" in req_content.split("\n")[0] and not req_content.startswith("#"):
            filename = "Makefile"
        else:
            filename = "requirements.txt"
        (course_dir / filename).write_text(req_content + "\n", encoding="utf-8")

    _log(f"Course directory: {course_dir}", _C.GREEN)

    # Emit curriculum event for DAG visualization
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
        llm=llm,
        design=design,
        analysis=analysis,
        source_content=source_content,
        student_level=user_level,
        model=generate_model,
        course_dir=course_dir,
    )

    # ── Phase 3: Review (skip if no modules generated) ─────────────────
    if not module_outputs:
        _log("Skipping review — no modules were generated", _C.YELLOW)
    else:
        await _phase_review(
            llm=llm,
            design=design,
            analysis=analysis,
            source_content=source_content,
            student_level=user_level,
            generate_model=generate_model,
            review_model=design_model,
            course_dir=course_dir,
            module_outputs=module_outputs,
            max_revision_cycles=max_revision_cycles,
        )

    # ── Done ──────────────────────────────────────────────────────────────
    _emit({"type": "phase", "phase": "done"})
    total_elapsed = time.time() - _start_time
    mins, secs = divmod(int(total_elapsed), 60)

    # Count generated files
    generated_files = [
        f for f in course_dir.rglob("*")
        if f.is_file() and not f.name.startswith("_")
    ]
    dir_count = sum(1 for f in course_dir.rglob("*") if f.is_dir())

    # Get cumulative usage from the LLM client
    totals = llm.get_totals()

    print(file=sys.stderr)
    _log(
        f"{_C.BOLD}Done. {len(generated_files)} files in {dir_count} directories. "
        f"Time: {mins}m {secs}s",
        _C.GREEN,
    )
    _log(f"Course: {course_dir}", _C.GREEN)
    _log(
        f"Cost: ${totals['cost_usd']:.4f} | "
        f"Tokens: {totals['input_tokens']:,} in / {totals['output_tokens']:,} out | "
        f"API calls: {totals['api_calls']}",
        _C.DIM,
    )

    if token is not None:
        _event_sink.reset(token)

    return {
        "course_dir": str(course_dir),
        "total_cost_usd": totals["cost_usd"],
        "usage": totals,
    }
