"""Custom SDK MCP tools for the Scaffoldly agent.

These tools give Claude structured ways to submit pipeline outputs
(analysis, curriculum) while we handle validation and persistence.
The agent writes actual course files (source code, tests, READMEs)
directly using the built-in Write tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from .schemas import Analysis, Curriculum


# ── Shared state ────────────────────────────────────────────────────────────────
# The agent populates these progressively. The CLI reads them after the run.

_state: dict[str, Any] = {
    "analysis": None,
    "curriculum": None,
    "output_dir": "./output",
    "course_dir": None,
}


def get_state() -> dict[str, Any]:
    return _state


def reset_state(output_dir: str = "./output") -> None:
    _state["analysis"] = None
    _state["curriculum"] = None
    _state["output_dir"] = output_dir
    _state["course_dir"] = None


def _slugify(title: str) -> str:
    slug = title.lower()
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = slug.strip().replace(" ", "_")
    return slug[:50].rstrip("_")


# ── Tool: submit_analysis ───────────────────────────────────────────────────────


@tool(
    "submit_analysis",
    "Submit the structured analysis of the source material. "
    "Call this after you have fetched and studied the content.",
    {
        "title": str,
        "summary": str,
        "domain": str,
        "overall_difficulty": str,
        "key_concepts": list,
        "prerequisites": list,
        "code_patterns": list,
        "learning_goals": list,
    },
)
async def submit_analysis(args: dict) -> dict:
    try:
        analysis = Analysis(**args)
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Validation error: {e}. Please fix and resubmit."}]}

    _state["analysis"] = analysis.model_dump()

    out = Path(_state["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "_analysis.json").write_text(json.dumps(_state["analysis"], indent=2, ensure_ascii=False))

    return {
        "content": [{
            "type": "text",
            "text": (
                f"Analysis saved. Found {len(analysis.key_concepts)} concepts, "
                f"{len(analysis.prerequisites)} prerequisites, "
                f"primary language(s): {', '.join(p.language for p in analysis.code_patterns)}. "
                "Now design the curriculum with submit_curriculum."
            ),
        }]
    }


# ── Tool: submit_curriculum ─────────────────────────────────────────────────────


@tool(
    "submit_curriculum",
    "Submit the course curriculum design. "
    "Call this after submit_analysis, before generating modules.",
    {
        "course_title": str,
        "course_description": str,
        "modules": list,
    },
)
async def submit_curriculum(args: dict) -> dict:
    try:
        curriculum = Curriculum(**args)
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Validation error: {e}. Please fix and resubmit."}]}

    _state["curriculum"] = curriculum.model_dump()

    course_slug = _slugify(curriculum.course_title)
    course_dir = Path(_state["output_dir"]) / course_slug
    course_dir.mkdir(parents=True, exist_ok=True)
    _state["course_dir"] = str(course_dir)

    (course_dir / "_curriculum.json").write_text(
        json.dumps(_state["curriculum"], indent=2, ensure_ascii=False)
    )

    module_titles = "\n".join(
        f"  {m.module_index}. {m.title}" for m in curriculum.modules
    )
    return {
        "content": [{
            "type": "text",
            "text": (
                f"Curriculum saved to {course_dir}/_curriculum.json\n"
                f"Course directory: {course_dir}\n\n"
                f"{len(curriculum.modules)} modules:\n{module_titles}\n\n"
                "Now generate the course project. Use Write to create source "
                "files, tests, READMEs, and any other files directly in the "
                "course directory. Use Bash to compile/run tests and validate."
            ),
        }]
    }


# ── MCP Server ──────────────────────────────────────────────────────────────────


def create_scaffoldly_server():
    return create_sdk_mcp_server(
        name="scaffoldly",
        version="0.2.0",
        tools=[submit_analysis, submit_curriculum],
    )
