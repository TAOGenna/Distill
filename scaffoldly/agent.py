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
    description="Reviews generated Jupyter notebook modules for pedagogical quality and correctness",
    prompt="""\
You are a strict reviewer of CS231n-style Jupyter notebook coursework.
You have access to Read and Bash tools. Your job is to review a generated
notebook file and report issues.

CHECK EACH OF THESE (report PASS or FAIL with specifics):

1. STRUCTURE: First cell is markdown intro? Last cell is markdown summary?
2. SCAFFOLDING: Do exercises use the ###...### TODO/END banner pattern?
3. DOCSTRINGS: Does every scaffolded function have a thorough docstring?
4. TESTS: Is there a test cell immediately after every exercise?
5. PROGRESSIVE: Do later exercises build on earlier ones?
6. IMPORTS: Are all necessary imports in the first code cell?
7. REALISM: Is test data realistic (not placeholder)?
8. SYNTAX: Run `python3 -c "import ast; ast.parse(open('<path>').read())"` \
to verify all code is syntactically valid.
9. INLINE QUESTIONS: Are there conceptual questions between exercises?
10. TANGIBLE OUTCOME: Does the module produce a visible result?

End with a VERDICT: PASS (ship it) or REVISE (list specific fixes needed).
""",
    tools=["Read", "Bash"],
    model="sonnet",
)

MODULE_GENERATOR_AGENT = AgentDefinition(
    description="Generates a single Jupyter notebook module for the course",
    prompt="""\
You are a module generator for CS231n-style coursework. You will be given
a module specification, course context, analysis, and student level.
Generate the complete notebook cells and call the write_notebook_module tool.

Follow all the notebook patterns and hard requirements from the course guidelines.
After writing, validate syntax with Bash: python3 -c "import ast; ..."
If the write_notebook_module tool reports syntax errors, fix and resubmit.
""",
    tools=["Bash", "Read", "mcp__scaffoldly__write_notebook_module"],
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
            "mcp__scaffoldly__submit_analysis",
            "mcp__scaffoldly__submit_curriculum",
            "mcp__scaffoldly__write_notebook_module",
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
