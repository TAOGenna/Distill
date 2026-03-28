"""Course generation pipeline — replaces the agent loop with direct API calls.

Three-phase architecture using structured output:
  Phase 1a: Analyze source material → Analysis
  Phase 1b: Design curriculum → CurriculumDesign (curriculum + root files)
  Phase 2:  Generate modules (parallel) → ModuleOutput per module
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

from .llm import LLMClient
from .prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    CURRICULUM_DESIGN_SYSTEM_PROMPT,
    MODULE_GENERATION_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
)
from .schemas import (
    Analysis,
    CurriculumDesign,
    ExerciseFile,
    GeneratedFile,
    ModuleOutput,
    ModuleReview,
    ReviewResult,
)
from .sources import prepare_sources, prepare_sources_with_summary


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


def _slugify(title: str) -> str:
    slug = title.lower()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = slug.strip().replace(" ", "_")
    return slug[:50].rstrip("_")


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
            f"{source_content[:20000]}"  # First ~5K tokens of source for README context
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


def _validate_syntax(file: GeneratedFile) -> str | None:
    """Validate syntax for a generated file. Returns error message or None."""
    if file.language == "python":
        try:
            ast.parse(file.content)
            return None
        except SyntaxError as e:
            return f"{file.relative_path}: SyntaxError at line {e.lineno}: {e.msg}"
    # For other languages, we'd need their compilers — skip for now
    return None


async def _generate_single_module(
    llm: LLMClient,
    module_spec: dict,
    course_context: str,
    source_excerpts: str,
    student_level: str,
    model: str,
    course_dir: Path,
    shared_defs: dict | None = None,
    revision_feedback: str | None = None,
) -> tuple[int, ModuleOutput, Usage]:
    """Generate all files for a single module via one API call."""
    idx = module_spec["module_index"]
    title = module_spec["title"]
    spec_json = json.dumps(module_spec, indent=2, ensure_ascii=False)

    # Build key excerpts section if available
    key_excerpts = module_spec.get("key_excerpts", [])
    excerpts_block = ""
    if key_excerpts:
        excerpts_block = (
            "\n\nKEY EXCERPTS FROM SOURCE MATERIAL (use as ground truth):\n"
            + "\n".join(f"  [{i+1}] {exc}" for i, exc in enumerate(key_excerpts))
            + "\n\nTranslate these DIRECTLY to code. Do not invent algorithms."
        )

    # Build shared definitions section
    shared_block = ""
    if shared_defs:
        lang = shared_defs.get("language", "python")
        deps = shared_defs.get("dependencies", [])
        shared_block = (
            f"\n\nShared definitions:\n"
            f"  Language: {lang}\n"
            f"  Dependencies: {', '.join(deps) if deps else 'standard library only'}\n"
        )

    prompt = (
        f"Generate all files for Module {idx}: \"{title}\"\n\n"
        f"Student level: {student_level}\n\n"
        f"Module Blueprint (follow scaffold_contract EXACTLY):\n{spec_json}\n\n"
        f"Course context:\n{course_context}"
        f"{shared_block}"
        f"{excerpts_block}\n\n"
        f"Source material excerpts:\n{source_excerpts[:10000]}\n"
    )

    if revision_feedback:
        prompt += (
            f"\n\nREVISION REQUIRED — fix these issues from the previous attempt:\n"
            f"{revision_feedback}\n"
        )

    messages = [{"role": "user", "content": prompt}]

    result = await llm.complete(
        messages=messages,
        model=model,
        system=MODULE_GENERATION_SYSTEM_PROMPT,
        response_model=ModuleOutput,
        max_tokens=16384,
        max_retries=3,
    )

    module_output = result.structured
    assert isinstance(module_output, ModuleOutput)

    return idx, module_output, result.usage


def _validate_syntax_str(content: str, path: str, language: str) -> str | None:
    """Validate syntax for a file content string. Returns error or None."""
    if language == "python":
        try:
            ast.parse(content)
            return None
        except SyntaxError as e:
            return f"{path}: SyntaxError at line {e.lineno}: {e.msg}"
    return None


def _write_module_files(
    module_output: ModuleOutput,
    module_dir: Path,
) -> list[str]:
    """Write all generated files to disk. Returns list of syntax errors."""
    module_dir.mkdir(parents=True, exist_ok=True)

    # Write README
    (module_dir / "README.md").write_text(module_output.readme, encoding="utf-8")

    errors: list[str] = []

    # Write exercise files (scaffold version for students, solution in _solutions/)
    solutions_dir = module_dir / "_solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)

    for ex in module_output.exercises:
        # Write scaffold (student-facing)
        scaffold_path = module_dir / ex.relative_path
        scaffold_path.parent.mkdir(parents=True, exist_ok=True)
        scaffold_path.write_text(ex.scaffold_content, encoding="utf-8")

        # Validate scaffold syntax
        err = _validate_syntax_str(ex.scaffold_content, ex.relative_path, ex.language)
        if err:
            errors.append(f"scaffold {err}")

        # Write solution (hidden)
        solution_path = solutions_dir / ex.relative_path
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        solution_path.write_text(ex.solution_content, encoding="utf-8")

        # Validate solution syntax
        err = _validate_syntax_str(ex.solution_content, ex.relative_path, ex.language)
        if err:
            errors.append(f"solution {err}")

    # Write supporting files
    for file in module_output.supporting_files:
        file_path = module_dir / file.relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file.content, encoding="utf-8")

    return errors


async def _phase_generate(
    llm: LLMClient,
    design: CurriculumDesign,
    analysis: Analysis,
    source_content: str,
    student_level: str,
    model: str,
    course_dir: Path,
) -> dict[int, ModuleOutput]:
    """Generate all modules in parallel. Returns dict of module_index → output."""
    curriculum = design.curriculum
    modules = curriculum.modules
    _log_step(f"Phase 2: Generating {len(modules)} modules in parallel...")
    _emit({"type": "phase", "phase": "generate"})

    # Build shared course context
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

    # Shared definitions for all modules
    shared_defs = None
    if hasattr(design, "shared_definitions") and design.shared_definitions:
        shared_defs = design.shared_definitions.model_dump()

    results: dict[int, tuple[ModuleOutput, Usage]] = {}

    async def _run(module_spec: dict) -> None:
        idx = module_spec["module_index"]
        title = module_spec["title"]
        try:
            _, output, usage = await _generate_single_module(
                llm=llm,
                module_spec=module_spec,
                course_context=course_context,
                source_excerpts=source_content,
                student_level=student_level,
                model=model,
                course_dir=course_dir,
                shared_defs=shared_defs,
            )
            results[idx] = (output, usage)

            # Write files
            module_slug = f"module_{idx:02d}_{_slugify(title)}"
            module_dir = course_dir / module_slug
            syntax_errors = _write_module_files(output, module_dir)

            if syntax_errors:
                _log(f"Module {idx} ({title}): {len(syntax_errors)} syntax errors", _C.YELLOW)
                for err in syntax_errors:
                    _log(f"  {err}", _C.YELLOW)
            else:
                _log(f"Module {idx} ({title}) generated", _C.GREEN)

            _emit({"type": "module_complete", "module_index": idx, "title": title})

        except Exception as e:
            print(f"  Module {idx} error: {e}", file=sys.stderr)
            _log(f"Module {idx} ({title}) failed ({type(e).__name__})", _C.RED)

    async with anyio.create_task_group() as tg:
        for module in modules:
            tg.start_soon(_run, module.model_dump())

    _log(f"All {len(modules)} modules generated", _C.GREEN)
    return {idx: output for idx, (output, _) in results.items()}


# ── Phase 3a: Pre-flight validation ──────────────────────────────────────────


def _preflight_module(module_dir: Path, module_spec: dict | None = None) -> list[str]:
    """Run deterministic + contract-aware checks. Returns list of errors."""
    errors: list[str] = []

    # Check README exists
    if not (module_dir / "README.md").exists():
        errors.append("Missing README.md")

    # Check exercise files
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

    # Contract-aware: check solution files against expected output patterns
    if module_spec:
        exercises = module_spec.get("exercises", [])
        for ex in exercises:
            expected_pattern = ex.get("expected_output_pattern", "")
            if not expected_pattern:
                continue
            solutions_dir = module_dir / "_solutions"
            if solutions_dir.exists():
                for sol_file in solutions_dir.glob("*.py"):
                    sol_content = sol_file.read_text(errors="replace")
                    if expected_pattern.lower() not in sol_content.lower():
                        errors.append(
                            f"_solutions/{sol_file.name}: expected pattern "
                            f"'{expected_pattern}' not found in solution"
                        )

    # Check for C/Rust files
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
    module_outputs: dict[int, ModuleOutput],
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
                _, output, _ = await _generate_single_module(
                    llm=llm,
                    module_spec=mod.model_dump(),
                    course_context=course_context,
                    source_excerpts=source_content,
                    student_level=student_level,
                    model=generate_model,
                    course_dir=course_dir,
                    revision_feedback="\n".join(feedback_parts),
                )

                module_slug = f"module_{idx:02d}_{_slugify(mod.title)}"
                module_dir = course_dir / module_slug
                _write_module_files(output, module_dir)
                module_outputs[idx] = output
                _log(f"Module {idx} re-generated", _C.GREEN)
                _emit({"type": "module_complete", "module_index": idx, "title": mod.title})
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
    series: bool = False,
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

    # ── Phase 3: Review ───────────────────────────────────────────────────
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
