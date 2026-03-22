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
4. TESTS: Is there a test file for each exercise that validates correctness?
   Do tests have helpful assertion messages?
5. PROGRESSIVE: Do later exercises build on earlier ones?
6. COMPILATION/SYNTAX: Use Bash to compile or syntax-check ALL source files.
   For Python: `python3 -c "import ast; ast.parse(open('file').read())"`
   For C/C++: `gcc -fsyntax-only file.c` or `g++ -fsyntax-only file.cpp`
   For Rust: `rustc --edition 2021 --crate-type lib file.rs`
7. REALISM: Is test data realistic (not placeholder)?
8. INLINE QUESTIONS: Are there conceptual questions that force reflection?
9. TANGIBLE OUTCOME: Does each module produce a visible result?
10. ORGANIZATION: Would a student know where to start and what order to follow?

End with a VERDICT: PASS (ship it) or REVISE (list specific fixes needed).
""",
    tools=["Read", "Bash"],
    model="sonnet",
)

MODULE_GENERATOR_AGENT = AgentDefinition(
    description="Generates source files for a single module of the course",
    prompt="""\
You are a module generator for CS231n-style coursework. You will be given
a module specification, course context, and student level.

Generate well-organized source files for the module:
- Exercise files with scaffolded code (TODO markers, docstrings, hints)
- Test files that validate correct implementations
- A module README explaining the exercises and how to work through them
- Any supporting files (data, configs, Makefiles, etc.)

Use Write to create each file. Use Bash to validate syntax/compilation.
Choose the right language and file structure for the domain.
""",
    tools=["Bash", "Read", "Write"],
    model="inherit",
)


# ── Main agent ──────────────────────────────────────────────────────────────────


async def run_agent(
    url: str,
    user_level: str,
    output_dir: str = "./output",
    model: str | None = None,
    effort: str | None = "high",
    max_turns: int = 50,
) -> str:
    """Run the Scaffoldly agent to generate coursework from a URL.

    Returns the path to the generated course directory.
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
            "module_generator": MODULE_GENERATOR_AGENT,
        },
    )

    prompt = (
        f"Generate a CS231n-style progressive course from this URL:\n"
        f"  URL: {url}\n"
        f"  Student level: {user_level}\n"
        f"  Output directory: {output_dir}\n\n"
        f"Follow the workflow in your instructions exactly: "
        f"fetch → analyze → design → generate → review → fix → finish."
    )

    _log("Starting Scaffoldly agent...")

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        _log(block.text[:200])
            elif isinstance(message, ResultMessage):
                _log("Agent finished.")

    state = get_state()
    course_dir = state.get("course_dir", output_dir)
    return course_dir


def run_agent_sync(
    url: str,
    user_level: str,
    output_dir: str = "./output",
    model: str | None = None,
    effort: str | None = "high",
    max_turns: int = 50,
) -> str:
    """Synchronous wrapper around run_agent for CLI use."""
    return anyio.run(
        lambda: run_agent(url, user_level, output_dir, model, effort, max_turns)
    )
