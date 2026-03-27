"""Scaffoldly agent — orchestrates course generation via Claude Agent SDK.

Three-phase architecture:
  Phase 1: Main agent (Opus) fetches, analyzes, designs, creates root files
  Phase 2: Orchestrator dispatches module generators (Sonnet) in parallel
  Phase 3: Main agent dispatches reviewer, fixes issues
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import anyio
from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)

from .system_prompt import MODULE_GENERATOR_SYSTEM_PROMPT, SYSTEM_PROMPT
from .tools import create_scaffoldly_server, get_state, reset_state


# ── ANSI colors ──────────────────────────────────────────────────────────────


class _C:
    """ANSI color codes. Disabled if stderr is not a terminal."""

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
    WHITE = "\033[37m" if _enabled else ""


# ── Logging ──────────────────────────────────────────────────────────────────

_start_time = 0.0


def _log(msg: str, color: str = "") -> None:
    elapsed = time.time() - _start_time if _start_time else 0
    mins, secs = divmod(int(elapsed), 60)
    ts = f"{_C.DIM}[{mins:02d}:{secs:02d}]{_C.RESET}"
    c = color or _C.RESET
    print(f"  {ts} {c}{msg}{_C.RESET}", file=sys.stderr, flush=True)


def _log_step(msg: str) -> None:
    """Log a step transition with a visible separator."""
    print(file=sys.stderr, flush=True)
    _log(f"{_C.BOLD}{msg}", _C.CYAN)


# ── Reviewer sub-agent ─────────────────────────────────────────────────────────

REVIEWER_AGENT = AgentDefinition(
    description="Reviews generated course files for pedagogical quality and correctness",
    prompt="""\
You are a strict reviewer of CS231n-style programming coursework.
You have access to Read and Bash tools. Your job is to review generated
course files and report issues.

EFFICIENCY: Read multiple files in parallel (issue multiple Read tool calls
in a single response). For syntax checking, run ONE comprehensive command:
  For Python: python3 -c "import ast, pathlib; errs = []; [errs.append(f'{f}: {e}') if not (lambda p: (ast.parse(p.read_text()), True))[-1] else None for f in pathlib.Path('COURSE_DIR').rglob('*.py')]; print('\\n'.join(errs) if errs else 'All files OK')"
  Or simply: find COURSE_DIR -name '*.py' -exec python3 -c "import ast; ast.parse(open('{}').read())" \\;

CHECK EACH OF THESE (report PASS or FAIL with specifics):

1. PROJECT STRUCTURE: Is the course well-organized into modules/directories?
   Are there clear READMEs explaining how to work through each module?
2. SCAFFOLDING: Do exercise files use clear TODO markers with line-count
   hints (~N lines) guiding the student on what to implement? For debug/explore
   exercises, is the provided code realistic and domain-appropriate?
3. DOCUMENTATION: Do scaffolded functions have thorough docstrings/comments
   explaining the algorithm step by step?
4. MILESTONES: Does every exercise end with a __main__ block (or main())
   that prints educational output when the student runs it? Does the output
   connect to the source material's insights (measurements, comparisons)?
   There should be NO separate test files or test frameworks.
5. PROGRESSIVE: Do later exercises build on earlier ones?
6. COMPILATION/SYNTAX: Use the ONE comprehensive command above. Report
   any files that fail.
7. REALISM: Is baked-in data realistic (not placeholder)?
8. INLINE QUESTIONS: Are there conceptual questions that force reflection?
9. TANGIBLE OUTCOME: Does each module produce a visible result?
10. ORGANIZATION: Would a student know where to start and what order to follow?

End with a VERDICT: PASS (ship it) or REVISE (list specific fixes needed).
""",
    tools=["Read", "Bash"],
    model="sonnet",
)


# ── Programmatic module generation ─────────────────────────────────────────────


def _slugify(title: str) -> str:
    slug = title.lower()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = slug.strip().replace(" ", "_")
    return slug[:40].rstrip("_")


async def _generate_module(
    module: dict,
    course_dir: str,
    course_context: str,
    student_level: str,
    generate_model: str,
) -> dict:
    """Generate a single module using standalone query(). Returns cost info."""
    idx = module["module_index"]
    title = module["title"]
    module_slug = f"module_{idx:02d}_{_slugify(title)}"
    module_dir = str(Path(course_dir) / module_slug)

    # Create module directory
    Path(module_dir).mkdir(parents=True, exist_ok=True)

    module_spec = json.dumps(module, indent=2, ensure_ascii=False)

    prompt = (
        f"Generate all source files for Module {idx}: \"{title}\"\n\n"
        f"Course directory: {course_dir}\n"
        f"Module directory: {module_dir}\n\n"
        f"Student level: {student_level}\n\n"
        f"Module specification:\n{module_spec}\n\n"
        f"{course_context}\n\n"
        f"Write all files into {module_dir}/. "
        f"After writing, validate syntax with Bash."
    )

    options = ClaudeAgentOptions(
        system_prompt=MODULE_GENERATOR_SYSTEM_PROMPT,
        model=generate_model,
        allowed_tools=["Bash", "Read", "Write"],
        permission_mode="bypassPermissions",
        max_turns=30,
    )

    cost = 0.0
    usage: dict[str, int] = {}
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, ResultMessage):
            cost = msg.total_cost_usd or 0.0
            if msg.usage:
                usage = {
                    k: v
                    for k, v in msg.usage.items()
                    if isinstance(v, (int, float))
                }

    _log(f"Module {idx} ({title}) generated", _C.GREEN)
    return {"module_index": idx, "title": title, "cost": cost, "usage": usage}


async def _dispatch_modules_parallel(
    curriculum: dict,
    analysis: dict,
    course_dir: str,
    student_level: str,
    generate_model: str,
) -> list[dict]:
    """Dispatch all module generators in parallel using standalone query()."""
    # Build shared course context from analysis
    concept_lines = "\n".join(
        f"  - {c['name']} ({c['priority']}): {c['description']}"
        for c in analysis.get("key_concepts", [])
    )
    course_context = (
        f"Course: {curriculum['course_title']}\n"
        f"Description: {curriculum['course_description']}\n"
        f"Content type: {analysis.get('content_type', 'unknown')}\n\n"
        f"Source material summary:\n{analysis.get('summary', '')}\n\n"
        f"Key concepts:\n{concept_lines}"
    )

    modules = curriculum["modules"]
    _log_step(f"  Dispatching {len(modules)} module generators in parallel...")

    results: list[dict] = []

    async with anyio.create_task_group() as tg:
        for module in modules:

            async def _run(m: dict = module) -> None:
                result = await _generate_module(
                    module=m,
                    course_dir=course_dir,
                    course_context=course_context,
                    student_level=student_level,
                    generate_model=generate_model,
                )
                results.append(result)

            tg.start_soon(_run)

    total_gen_cost = sum(r["cost"] for r in results)
    _log(
        f"All {len(modules)} modules generated (cost: ${total_gen_cost:.4f})",
        _C.GREEN,
    )
    return results


# ── Message processing ─────────────────────────────────────────────────────────


def _process_message(
    message: Any,
    step_transition: Any,
    accumulate_result: Any,
) -> None:
    """Process a single message from the agent stream."""
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                _log(block.text[:200])
            elif isinstance(block, ToolUseBlock):
                args_str = ", ".join(
                    f"{k}={str(v)[:60]}"
                    for k, v in (block.input or {}).items()
                )
                _log(f"{block.name}({args_str})", _C.BLUE)
                if block.name == "mcp__scaffoldly__submit_analysis":
                    step_transition("analyze")
                elif block.name == "mcp__scaffoldly__submit_curriculum":
                    step_transition("design")
            elif isinstance(block, ToolResultBlock):
                if block.is_error:
                    content_text = ""
                    if isinstance(block.content, list):
                        content_text = " ".join(
                            item.get("text", "")
                            if isinstance(item, dict)
                            else str(item)
                            for item in block.content
                        )
                    elif isinstance(block.content, str):
                        content_text = block.content
                    _log(f"ERROR: {content_text[:200]}", _C.RED)
    elif isinstance(message, TaskStartedMessage):
        task_desc = message.description or message.task_type or "unknown"
        _log(f"sub-agent started: {task_desc}", _C.MAGENTA)
        if "review" in task_desc.lower():
            step_transition("review")
    elif isinstance(message, TaskNotificationMessage):
        status = message.status if hasattr(message, "status") else "unknown"
        summary = (message.summary or "")[:150]
        color = _C.GREEN if status == "completed" else _C.MAGENTA
        _log(
            f"sub-agent {status}: {summary}" if summary else f"sub-agent {status}",
            color,
        )
    elif isinstance(message, ResultMessage):
        accumulate_result(message)


# ── Main agent ──────────────────────────────────────────────────────────────────


async def run_agent(
    url: str,
    user_level: str,
    refs: list[str] | None = None,
    series: bool = False,
    output_dir: str = "./output",
    model: str | None = None,
    generate_model: str = "sonnet",
    effort: str | None = "high",
    max_turns: int = 50,
    sources_dir: str | None = None,
) -> dict:
    """Run the Scaffoldly agent to generate coursework from a URL.

    Three-phase architecture:
      Phase 1: Main agent (model) fetches, analyzes, designs, creates root files
      Phase 2: Orchestrator dispatches module generators in parallel
      Phase 3: Main agent dispatches reviewer, fixes issues

    Returns a dict with course_dir, total_cost_usd, and usage.
    """
    reset_state(output_dir)

    server = create_scaffoldly_server()
    abs_output_dir = str(Path(output_dir).resolve())

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
        model=model,
        effort=effort,
        allowed_tools=[
            "Bash",
            "Read",
            "Write",
            "Edit",
            "mcp__scaffoldly__submit_analysis",
            "mcp__scaffoldly__submit_curriculum",
            "reviewer",
        ],
        mcp_servers={"scaffoldly": server},
        agents={"reviewer": REVIEWER_AGENT},
    )

    # Build the prompt — use preprocessed sources if available, else raw URLs
    manifest_path = Path(sources_dir) / "manifest.json" if sources_dir else None
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        phase1_prompt = (
            f"Generate a CS231n-style progressive course from preprocessed sources.\n"
            f"  Sources directory: {sources_dir}\n"
            f"  Original URL: {url}\n"
            f"  Student level: {user_level}\n"
            f"  Output directory: {abs_output_dir}\n\n"
            f"Source manifest:\n{json.dumps(manifest, indent=2)}\n\n"
            f"Follow your workflow: consume → analyze → design → create root files → stop.\n"
            f"Module generation will be handled automatically after you finish."
        )
    else:
        source_lines = [f"  Focus URL: {url}"]
        if refs:
            if series:
                source_lines.append(
                    "  Mode: SERIES (sources form an ordered progression)"
                )
                for i, ref in enumerate(refs, start=2):
                    source_lines.append(f"  Part {i}: {ref}")
            else:
                source_lines.append(
                    "  Mode: REFERENCE (focus + supplementary context)"
                )
                for ref in refs:
                    source_lines.append(f"  Reference: {ref}")
        sources_block = "\n".join(source_lines)

        phase1_prompt = (
            f"Generate a CS231n-style progressive course from these sources:\n"
            f"{sources_block}\n"
            f"  Student level: {user_level}\n"
            f"  Output directory: {abs_output_dir}\n\n"
            f"Follow your workflow: fetch → analyze → design → create root files → stop.\n"
            f"Module generation will be handled automatically after you finish."
        )

    global _start_time
    _start_time = time.time()
    step_start = _start_time
    current_step = "fetch"
    total_cost = 0.0
    total_usage: dict[str, int] = {}

    def _step_transition(new_step: str) -> None:
        nonlocal step_start, current_step
        elapsed = time.time() - step_start
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        _log_step(f"  {current_step} completed ({time_str})")
        current_step = new_step
        step_start = time.time()

    def _accumulate_result(msg: ResultMessage) -> None:
        nonlocal total_cost
        total_cost += msg.total_cost_usd or 0.0
        if msg.usage:
            for k, v in msg.usage.items():
                if isinstance(v, (int, float)):
                    total_usage[k] = total_usage.get(k, 0) + v

    _log_step("  Starting agent...")

    async with ClaudeSDKClient(options=options) as client:
        # ── Phase 1: Fetch → Analyze → Design → Root files ──────────────
        await client.query(phase1_prompt)
        async for message in client.receive_response():
            _process_message(message, _step_transition, _accumulate_result)

        _step_transition("generate")

        # ── Phase 2: Parallel module generation ─────────────────────────
        state = get_state()
        curriculum = state.get("curriculum")
        analysis = state.get("analysis")
        course_dir = state.get("course_dir")

        if not curriculum or not course_dir:
            _log("No curriculum found — skipping module generation", _C.RED)
        else:
            gen_results = await _dispatch_modules_parallel(
                curriculum=curriculum,
                analysis=analysis or {},
                course_dir=course_dir,
                student_level=user_level,
                generate_model=generate_model,
            )
            # Accumulate module generation costs (best-effort — never
            # let bookkeeping crash a successful generation run)
            try:
                for r in gen_results:
                    total_cost += r["cost"]
                    for k, v in r.get("usage", {}).items():
                        if isinstance(v, (int, float)):
                            total_usage[k] = total_usage.get(k, 0) + v
            except Exception as exc:  # noqa: BLE001
                _log(f"Warning: failed to accumulate usage stats: {exc}", _C.YELLOW)

        _step_transition("review")

        # ── Phase 3: Review → Fix ───────────────────────────────────────
        if course_dir and curriculum:
            n_modules = len(curriculum["modules"])
            phase3_prompt = (
                f"All {n_modules} modules have been generated in:\n"
                f"  {course_dir}\n\n"
                f"Proceed to step 5 (REVIEW). Dispatch the `reviewer` "
                f"sub-agent to check course quality. If it says REVISE, "
                f"fix the specific issues and re-review.\n\n"
                f"Then proceed to step 6 (FINISH) — summarize what was "
                f"generated."
            )
            await client.query(phase3_prompt)
            async for message in client.receive_response():
                _process_message(
                    message, _step_transition, _accumulate_result
                )

        _step_transition("done")
        total_elapsed = time.time() - _start_time
        mins, secs = divmod(int(total_elapsed), 60)
        print(file=sys.stderr)
        _log(f"{_C.BOLD}Agent finished. Total time: {mins}m {secs}s", _C.GREEN)

    # Post-run validation
    state = get_state()
    course_dir = state.get("course_dir") or abs_output_dir
    course_path = Path(course_dir)
    if course_path.exists():
        generated_files = [
            f
            for f in course_path.rglob("*")
            if f.is_file() and not f.name.startswith("_")
        ]
        all_files = list(course_path.rglob("*"))
        file_count = len(generated_files)
        dir_count = sum(1 for f in all_files if f.is_dir())
        _log(
            f"{course_dir}: {file_count} files in {dir_count} directories",
            _C.GREEN,
        )
        if file_count == 0:
            _log(
                "No course files were generated! Check errors above.", _C.RED
            )
    else:
        _log(f"Course directory does not exist: {course_dir}", _C.RED)

    return {
        "course_dir": course_dir,
        "total_cost_usd": total_cost,
        "usage": total_usage or None,
    }


def run_agent_sync(
    url: str,
    user_level: str,
    refs: list[str] | None = None,
    series: bool = False,
    output_dir: str = "./output",
    model: str | None = None,
    generate_model: str = "sonnet",
    effort: str | None = "high",
    max_turns: int = 50,
    sources_dir: str | None = None,
) -> dict:
    """Synchronous wrapper around run_agent for CLI use."""
    return anyio.run(
        lambda: run_agent(
            url,
            user_level,
            refs,
            series,
            output_dir,
            model,
            generate_model,
            effort,
            max_turns,
            sources_dir,
        )
    )
