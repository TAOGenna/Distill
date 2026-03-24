"""Scaffoldly agent — orchestrates course generation via Claude Agent SDK."""

from __future__ import annotations

import sys
import time

from pathlib import Path

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
)

from .system_prompt import SYSTEM_PROMPT
from .tools import create_scaffoldly_server, get_state, reset_state


# ── ANSI colors ──────────────────────────────────────────────────────────────

class _C:
    """ANSI color codes. Disabled if stderr is not a terminal."""
    _enabled = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    RESET   = "\033[0m"   if _enabled else ""
    BOLD    = "\033[1m"    if _enabled else ""
    DIM     = "\033[2m"    if _enabled else ""
    RED     = "\033[31m"   if _enabled else ""
    GREEN   = "\033[32m"   if _enabled else ""
    YELLOW  = "\033[33m"   if _enabled else ""
    BLUE    = "\033[34m"   if _enabled else ""
    MAGENTA = "\033[35m"   if _enabled else ""
    CYAN    = "\033[36m"   if _enabled else ""
    WHITE   = "\033[37m"   if _enabled else ""


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


# ── Sub-agent definitions ───────────────────────────────────────────────────────

REVIEWER_AGENT = AgentDefinition(
    description="Reviews generated course files for pedagogical quality and correctness",
    prompt="""\
You are a strict reviewer of CS231n-style programming coursework.
You have access to Read and Bash tools. Your job is to review generated
course files and report issues.

CHECK EACH OF THESE (report PASS or FAIL with specifics):

1. PROJECT STRUCTURE: Is the course well-organized into modules/directories?
   Are there clear READMEs explaining how to work through each module?
2. SCAFFOLDING: Do exercise files use clear TODO markers and comments
   guiding the student on what to implement?
3. DOCUMENTATION: Do scaffolded functions have thorough docstrings/comments
   explaining the algorithm step by step?
4. MILESTONES: Does every exercise end with a __main__ block (or main())
   that prints educational output when the student runs it? Does the output
   connect to the source material's insights (measurements, comparisons)?
   There should be NO separate test files or test frameworks.
5. PROGRESSIVE: Do later exercises build on earlier ones?
6. COMPILATION/SYNTAX: Use Bash to compile or syntax-check ALL source files.
   For Python: `python3 -c "import ast; ast.parse(open('file').read())"`
   For C/C++: `gcc -fsyntax-only file.c` or `g++ -fsyntax-only file.cpp`
   For Rust: `rustc --edition 2021 --crate-type lib file.rs`
7. REALISM: Is baked-in data realistic (not placeholder)?
8. INLINE QUESTIONS: Are there conceptual questions that force reflection?
9. TANGIBLE OUTCOME: Does each module produce a visible result?
10. ORGANIZATION: Would a student know where to start and what order to follow?

End with a VERDICT: PASS (ship it) or REVISE (list specific fixes needed).
""",
    tools=["Read", "Bash"],
    model="sonnet",
)

_MODULE_GENERATOR_PROMPT = """\
You are a module generator for CS231n-style coursework. You will be given
a module specification, course context, and student level.

Generate well-organized source files for the module:
- Exercise files with scaffolded code (TODO markers, docstrings, hints)
- Each exercise MUST end with a __main__ block (or main()) that runs the
  student's code and prints educational output (measurements, comparisons,
  behaviors that connect to the source material's insights)
- A module README explaining the exercises and how to work through them.
  Include 2-4 analytical questions at Level 3+ depth (analysis/synthesis,
  not recall). See the course context for the question rubric.
- Any supporting files (data, configs, Makefiles, etc.)
- Do NOT create test files or use test frameworks

Use Write to create each file. Use Bash to validate syntax/compilation.
Choose the right language and file structure for the domain.
"""


def _make_module_generator(model: str = "sonnet") -> AgentDefinition:
    """Create a module_generator agent with the specified model."""
    return AgentDefinition(
        description="Generates source files for a single module of the course",
        prompt=_MODULE_GENERATOR_PROMPT,
        tools=["Bash", "Read", "Write"],
        model=model,
    )


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
) -> dict:
    """Run the Scaffoldly agent to generate coursework from a URL.

    The main agent (model) handles analysis and curriculum design.
    Module generation is delegated to sub-agents (generate_model) to
    reduce cost — file generation is mechanical and doesn't need the
    most capable model.

    Returns a dict with course_dir, total_cost_usd, and usage.
    """
    reset_state(output_dir)

    server = create_scaffoldly_server()

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
            "module_generator",
            "reviewer",
        ],
        mcp_servers={"scaffoldly": server},
        agents={
            "reviewer": REVIEWER_AGENT,
            "module_generator": _make_module_generator(generate_model),
        },
    )

    # Build the prompt with optional multi-source context
    source_lines = [f"  Focus URL: {url}"]
    if refs:
        if series:
            source_lines.append(f"  Mode: SERIES (sources form an ordered progression)")
            for i, ref in enumerate(refs, start=2):
                source_lines.append(f"  Part {i}: {ref}")
        else:
            source_lines.append(f"  Mode: REFERENCE (focus + supplementary context)")
            for ref in refs:
                source_lines.append(f"  Reference: {ref}")
    sources_block = "\n".join(source_lines)

    prompt = (
        f"Generate a CS231n-style progressive course from these sources:\n"
        f"{sources_block}\n"
        f"  Student level: {user_level}\n"
        f"  Output directory: {output_dir}\n\n"
        f"Follow the workflow in your instructions exactly: "
        f"fetch → analyze → design → generate → review → fix → finish."
    )

    global _start_time
    _start_time = time.time()
    step_start = _start_time
    current_step = "fetch"

    def _step_transition(new_step: str) -> None:
        nonlocal step_start, current_step
        elapsed = time.time() - step_start
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        _log_step(f"  {current_step} completed ({time_str})")
        current_step = new_step

    _log_step("  Starting agent...")

    total_cost_usd = None
    usage = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        _log(block.text[:200])
                    elif isinstance(block, ToolUseBlock):
                        args_str = ", ".join(
                            f"{k}={str(v)[:60]}" for k, v in (block.input or {}).items()
                        )
                        _log(f"{block.name}({args_str})", _C.BLUE)
                        # Track workflow step transitions
                        if block.name == "mcp__scaffoldly__submit_analysis":
                            _step_transition("analyze")
                        elif block.name == "mcp__scaffoldly__submit_curriculum":
                            _step_transition("design")
                    elif isinstance(block, ToolResultBlock):
                        if block.is_error:
                            content_text = ""
                            if isinstance(block.content, list):
                                content_text = " ".join(
                                    item.get("text", "") if isinstance(item, dict) else str(item)
                                    for item in block.content
                                )
                            elif isinstance(block.content, str):
                                content_text = block.content
                            _log(f"ERROR: {content_text[:200]}", _C.RED)
            elif isinstance(message, TaskStartedMessage):
                task_desc = message.description or message.task_type or "unknown"
                _log(f"sub-agent started: {task_desc}", _C.MAGENTA)
                if current_step == "design" and "module" in task_desc.lower():
                    _step_transition("generate")
                elif "review" in task_desc.lower():
                    _step_transition("review")
            elif isinstance(message, TaskNotificationMessage):
                status = message.status if hasattr(message, "status") else "unknown"
                summary = (message.summary or "")[:150]
                color = _C.GREEN if status == "completed" else _C.MAGENTA
                _log(
                    f"sub-agent {status}: {summary}" if summary else f"sub-agent {status}",
                    color,
                )
            elif isinstance(message, ResultMessage):
                _step_transition("done")
                total_cost_usd = message.total_cost_usd
                usage = message.usage
                total_elapsed = time.time() - _start_time
                mins, secs = divmod(int(total_elapsed), 60)
                print(file=sys.stderr)
                _log(f"{_C.BOLD}Agent finished. Total time: {mins}m {secs}s", _C.GREEN)

    state = get_state()
    course_dir = state.get("course_dir") or output_dir

    # Post-run validation: count generated files
    course_path = Path(course_dir)
    if course_path.exists():
        generated_files = [f for f in course_path.rglob("*") if f.is_file() and not f.name.startswith("_")]
        all_files = list(course_path.rglob("*"))
        file_count = len(generated_files)
        dir_count = sum(1 for f in all_files if f.is_dir())
        _log(f"{course_dir}: {file_count} files in {dir_count} directories", _C.GREEN)
        if file_count == 0:
            _log("No course files were generated! Check errors above.", _C.RED)
    else:
        _log(f"Course directory does not exist: {course_dir}", _C.RED)

    return {
        "course_dir": course_dir,
        "total_cost_usd": total_cost_usd,
        "usage": usage,
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
) -> dict:
    """Synchronous wrapper around run_agent for CLI use."""
    return anyio.run(
        lambda: run_agent(
            url, user_level, refs, series, output_dir,
            model, generate_model, effort, max_turns,
        )
    )
