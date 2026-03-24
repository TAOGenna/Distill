"""Scaffoldly agent — orchestrates course generation via Claude Agent SDK."""

from __future__ import annotations

import sys

import anyio
from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from .system_prompt import SYSTEM_PROMPT
from .tools import create_scaffoldly_server, get_state, reset_state


def _log(msg: str) -> None:
    print(f"  → {msg}", file=sys.stderr, flush=True)


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

    _log("Starting Scaffoldly agent...")

    total_cost_usd = None
    usage = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        _log(block.text[:200])
            elif isinstance(message, ResultMessage):
                total_cost_usd = message.total_cost_usd
                usage = message.usage
                _log("Agent finished.")

    state = get_state()
    course_dir = state.get("course_dir", output_dir)
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
