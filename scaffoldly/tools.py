"""Custom SDK MCP tools for the Scaffoldly agent.

These tools give Claude structured ways to submit pipeline outputs
(analysis, curriculum, notebook modules) while we handle validation,
persistence, and notebook assembly.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from .notebook import _slugify, cells_to_notebook, create_course_readme_notebook, save_notebook
from .schemas import Analysis, Curriculum, ModuleNotebook


# ── Shared state ────────────────────────────────────────────────────────────────
# The agent populates these progressively. The CLI reads them after the run.

_state: dict[str, Any] = {
    "analysis": None,
    "curriculum": None,
    "modules": [],          # list of ModuleNotebook dicts
    "output_dir": "./output",
}


def get_state() -> dict[str, Any]:
    """Return the shared pipeline state (for the CLI to inspect after a run)."""
    return _state


def reset_state(output_dir: str = "./output") -> None:
    """Reset state for a fresh run."""
    _state["analysis"] = None
    _state["curriculum"] = None
    _state["modules"] = []
    _state["output_dir"] = output_dir


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
    """Validate and store the analysis."""
    try:
        analysis = Analysis(**args)
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Validation error: {e}. Please fix and resubmit."}]}

    _state["analysis"] = analysis.model_dump()

    # Persist to disk
    out = Path(_state["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "_analysis.json").write_text(json.dumps(_state["analysis"], indent=2, ensure_ascii=False))

    return {
        "content": [{
            "type": "text",
            "text": (
                f"Analysis saved. Found {len(analysis.key_concepts)} concepts, "
                f"{len(analysis.prerequisites)} prerequisites. "
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
    """Validate and store the curriculum, generate the overview notebook."""
    try:
        curriculum = Curriculum(**args)
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Validation error: {e}. Please fix and resubmit."}]}

    _state["curriculum"] = curriculum.model_dump()

    # Determine course directory
    course_slug = _slugify(0, curriculum.course_title).lstrip("00_")
    course_dir = Path(_state["output_dir"]) / course_slug
    course_dir.mkdir(parents=True, exist_ok=True)
    _state["course_dir"] = str(course_dir)

    # Persist curriculum JSON
    (course_dir / "_curriculum.json").write_text(
        json.dumps(_state["curriculum"], indent=2, ensure_ascii=False)
    )

    # Generate overview notebook
    overview_nb = create_course_readme_notebook(
        curriculum.course_title,
        curriculum.course_description,
        [m.model_dump() for m in curriculum.modules],
    )
    save_notebook(overview_nb, course_dir / "00_overview.ipynb")

    module_titles = "\n".join(
        f"  {m.module_index}. {m.title}" for m in curriculum.modules
    )
    return {
        "content": [{
            "type": "text",
            "text": (
                f"Curriculum saved with {len(curriculum.modules)} modules:\n"
                f"{module_titles}\n\n"
                f"Overview notebook written to {course_dir}/00_overview.ipynb\n"
                "Now generate each module with write_notebook_module."
            ),
        }]
    }


# ── Tool: write_notebook_module ─────────────────────────────────────────────────


@tool(
    "write_notebook_module",
    "Write a complete Jupyter notebook for one module of the course. "
    "Call this once per module, in order.",
    {
        "module_index": int,
        "title": str,
        "cells": list,
    },
)
async def write_notebook_module(args: dict) -> dict:
    """Validate cells, check Python syntax, assemble and save the notebook."""
    try:
        module = ModuleNotebook(**args)
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Validation error: {e}. Please fix and resubmit."}]}

    # Syntax-check all code cells
    syntax_errors = []
    for i, cell in enumerate(module.cells):
        if cell.cell_type == "code":
            try:
                ast.parse(cell.source)
            except SyntaxError as e:
                syntax_errors.append(f"Cell {i}: {e.msg} (line {e.lineno})")

    if syntax_errors:
        error_list = "\n".join(syntax_errors)
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"Syntax errors found in {len(syntax_errors)} code cell(s):\n"
                    f"{error_list}\n\n"
                    "Fix the code and call write_notebook_module again."
                ),
            }]
        }

    # Build and save the notebook
    course_dir = Path(_state.get("course_dir", _state["output_dir"]))
    slug = _slugify(module.module_index, module.title)
    nb_path = course_dir / f"{slug}.ipynb"

    cell_dicts = [c.model_dump() for c in module.cells]
    nb = cells_to_notebook(cell_dicts)
    save_notebook(nb, nb_path)

    _state["modules"].append(module.model_dump())

    return {
        "content": [{
            "type": "text",
            "text": (
                f"Module {module.module_index} '{module.title}' written to {nb_path} "
                f"({len(module.cells)} cells, all syntax valid)."
            ),
        }]
    }


# ── MCP Server ──────────────────────────────────────────────────────────────────


def create_scaffoldly_server():
    """Create the in-process MCP server with all Scaffoldly tools."""
    return create_sdk_mcp_server(
        name="scaffoldly",
        version="0.2.0",
        tools=[submit_analysis, submit_curriculum, write_notebook_module],
    )
