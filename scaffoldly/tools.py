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
    _state["output_dir"] = str(Path(output_dir).resolve())
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
    Analysis.model_json_schema(),
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

    # Summarize concept triage for the agent
    essential = [c.name for c in analysis.key_concepts if c.priority == "essential"]
    supporting = [c.name for c in analysis.key_concepts if c.priority == "supporting"]
    contextual = [c.name for c in analysis.key_concepts if c.priority == "contextual"]

    triage_summary = (
        f"\n\nConcept triage:"
        f"\n  Essential ({len(essential)}): {', '.join(essential) or 'none'}"
        f"\n  Supporting ({len(supporting)}): {', '.join(supporting) or 'none'}"
        f"\n  Contextual ({len(contextual)}): {', '.join(contextual) or 'none'}"
        f"\n\nReminder: essential concepts MUST have exercises. "
        f"Supporting concepts must appear in exercises or analytical questions. "
        f"Contextual concepts go in the 'What's Next' README section only."
    )

    return {
        "content": [{
            "type": "text",
            "text": (
                f"Analysis saved. Found {len(analysis.key_concepts)} concepts, "
                f"{len(analysis.prerequisites)} prerequisites, "
                f"primary language(s): {', '.join(p.language for p in analysis.code_patterns)}. "
                f"{triage_summary}"
                "\n\nNow design the curriculum with submit_curriculum."
            ),
        }]
    }


# ── Tool: submit_curriculum ─────────────────────────────────────────────────────


@tool(
    "submit_curriculum",
    "Submit the course curriculum design. "
    "Call this after submit_analysis, before generating modules.",
    Curriculum.model_json_schema(),
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

    module_lines = []
    for m in curriculum.modules:
        deps = f" (depends on: {m.depends_on})" if m.depends_on else " (no prerequisites)"
        module_lines.append(f"  {m.module_index}. {m.title}{deps}")
    module_listing = "\n".join(module_lines)

    # Lightweight coverage check against analysis triage
    coverage_note = ""
    if _state["analysis"]:
        analysis_concepts = _state["analysis"].get("key_concepts", [])
        essential_names = {
            c["name"] for c in analysis_concepts if c.get("priority") == "essential"
        }
        covered_names: set[str] = set()
        for m in curriculum.modules:
            covered_names.update(m.concepts_covered)
        uncovered = essential_names - covered_names
        if uncovered:
            coverage_note = (
                f"\n\n⚠ COVERAGE GAP: these essential concepts are not covered "
                f"by any module: {', '.join(sorted(uncovered))}. "
                f"Consider adding them to an existing module before generating."
            )
        else:
            coverage_note = (
                "\n\n✓ All essential concepts are covered by at least one module."
            )

    return {
        "content": [{
            "type": "text",
            "text": (
                f"Curriculum saved to {course_dir}/_curriculum.json\n"
                f"Course directory: {course_dir}\n\n"
                f"{len(curriculum.modules)} modules:\n{module_listing}\n"
                f"{coverage_note}\n\n"
                "NEXT: Re-read the source material's quantitative claims "
                "(numbers, benchmarks, measurements). Then:\n"
                "1. Create the course root README.md and requirements file.\n"
                "2. Create module directories with mkdir.\n"
                "3. Summarize the key quantitative claims that should appear "
                "as exercise milestone targets.\n"
                "4. STOP. Module generation will proceed automatically."
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
