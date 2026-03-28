"""Validation helpers for the Scaffoldly pipeline.

Formerly contained MCP tool definitions for the Claude Agent SDK.
Now provides pure Python validation functions used by pipeline.py.
"""

from __future__ import annotations

from .schemas import Analysis, Curriculum


def validate_analysis(data: dict) -> Analysis:
    """Validate raw dict against the Analysis schema."""
    return Analysis(**data)


def validate_curriculum(data: dict) -> Curriculum:
    """Validate raw dict against the Curriculum schema."""
    return Curriculum(**data)


def check_coverage(analysis: Analysis, curriculum: Curriculum) -> list[str]:
    """Check that essential concepts have exercises.

    Returns a list of warning messages (empty if all good).
    """
    warnings: list[str] = []

    essential_names = {
        c.name for c in analysis.key_concepts if c.priority == "essential"
    }
    supporting_names = {
        c.name for c in analysis.key_concepts if c.priority == "supporting"
    }

    covered_names: set[str] = set()
    for m in curriculum.modules:
        covered_names.update(m.concepts_covered)

    uncovered_essential = essential_names - covered_names
    if uncovered_essential:
        warnings.append(
            f"Essential concepts not covered by any module: "
            f"{', '.join(sorted(uncovered_essential))}"
        )

    uncovered_supporting = supporting_names - covered_names
    if uncovered_supporting:
        warnings.append(
            f"Supporting concepts not covered by any module: "
            f"{', '.join(sorted(uncovered_supporting))}"
        )

    return warnings
