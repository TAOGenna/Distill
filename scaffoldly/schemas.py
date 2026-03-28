"""Pydantic models for structured LLM output throughout the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Stage 1: Analysis ──────────────────────────────────────────────────────────


class Concept(BaseModel):
    name: str
    description: str
    importance: Literal["core", "supporting", "tangential"]
    difficulty: Literal["beginner", "intermediate", "advanced", "expert"]
    source: str = Field(
        default="focus",
        description="Which source this concept came from: 'focus' for the "
        "primary URL, or the URL/title of a reference source. Concepts from "
        "refs are typically 'supporting' or 'contextual' priority.",
    )
    priority: Literal["essential", "supporting", "contextual"] = Field(
        description="Triage classification. 'essential': the system doesn't make "
        "sense without it — must have exercises. 'supporting': deepens understanding "
        "— should appear in at least one exercise or analytical question. "
        "'contextual': operational/tangential — belongs in What's Next section."
    )
    priority_rationale: str = Field(
        description="Why this concept was classified at this priority level. "
        "This reasoning helps the curriculum designer make better scoping decisions."
    )


class Prerequisite(BaseModel):
    name: str
    why_needed: str
    difficulty: Literal["beginner", "intermediate", "advanced", "expert"]


class CodePattern(BaseModel):
    description: str
    language: str
    concepts_demonstrated: list[str]


class Analysis(BaseModel):
    title: str
    summary: str
    domain: str
    content_type: Literal[
        "systems_engineering",
        "ml_research",
        "tutorial",
        "library_walkthrough",
    ] = Field(
        description="The type of source material. This determines the pedagogy "
        "strategy: milestone style, scaffolding approach, and how math is presented."
    )
    overall_difficulty: Literal["beginner", "intermediate", "advanced", "expert"]
    key_concepts: list[Concept]
    prerequisites: list[Prerequisite]
    code_patterns: list[CodePattern]
    learning_goals: list[str]


# ── Stage 2: Curriculum Design ─────────────────────────────────────────────────


_SCAFFOLDING_MAP: dict[str, str] = {
    # canonical
    "heavy": "heavy", "medium": "medium", "light": "light", "none": "none",
    # common synonyms models use
    "full": "heavy", "high": "heavy",
    "partial": "medium", "moderate": "medium", "guided": "medium",
    "minimal": "light", "low": "light", "lite": "light",
    "zero": "none", "no": "none",
}


class Exercise(BaseModel):
    title: str
    type: Literal[
        "implement", "fill_blank", "debug", "analyze", "extend",
        "contrastive", "comparative", "explore",
    ]
    description: str
    scaffolding_level: Literal["heavy", "medium", "light", "none"]
    what_is_provided: str
    what_student_writes: str
    milestone: str = Field(
        description="What the student sees when they run the exercise. "
        "Describe the output — printed measurements, saved plots, or "
        "visualizations that reproduce a key insight from the source material."
    )

    @field_validator("scaffolding_level", mode="before")
    @classmethod
    def normalize_scaffolding(cls, v: str) -> str:
        if isinstance(v, str):
            mapped = _SCAFFOLDING_MAP.get(v.lower().strip())
            if mapped:
                return mapped
        return v  # let Pydantic raise the literal error if still invalid


class InlineQuestion(BaseModel):
    question: str
    context: str


class Module(BaseModel):
    module_index: int
    title: str
    description: str
    learning_objectives: list[str]
    concepts_covered: list[str]
    depends_on: list[int] = Field(
        default_factory=list,
        description="Module indices this module requires as prerequisites. "
        "Empty list means this module can be started independently. "
        "Used to generate the learning path in the course README.",
    )
    exercises: list[Exercise]
    inline_questions: list[InlineQuestion]
    visible_outcome: str


class Curriculum(BaseModel):
    course_title: str
    course_description: str
    modules: list[Module]


# ── Stage 2b: Curriculum Design (with root files) ────────────────────────────


class CurriculumDesign(BaseModel):
    """Phase 1b output — curriculum plus root project files."""

    curriculum: Curriculum
    root_readme: str = Field(
        description="Full content of the course root README.md. Includes setup "
        "instructions, module order/learning path, and What's Next section."
    )
    requirements: str = Field(
        description="Content of requirements.txt (Python) or equivalent setup "
        "file (Cargo.toml, Makefile, package.json) for the course."
    )


# ── Stage 3: Module Generation ────────────────────────────────────────────────


class GeneratedFile(BaseModel):
    """A single file produced by the module generator."""

    relative_path: str = Field(
        description="Path relative to the module directory, e.g. 'ex01_basic.py' "
        "or 'data/sample.csv'. Do not include the module directory name."
    )
    content: str = Field(
        description="Full file content."
    )
    language: str = Field(
        description="Programming language or file type for syntax validation "
        "routing, e.g. 'python', 'c', 'rust', 'markdown', 'json', 'csv'."
    )


class ModuleOutput(BaseModel):
    """Phase 2 output — all files for a single module."""

    readme: str = Field(
        description="Full content of the module README.md. Includes learning "
        "objectives, exercise walkthrough, and 2-4 analytical questions at "
        "Level 3+ depth."
    )
    files: list[GeneratedFile] = Field(
        description="All exercise and supporting files for this module."
    )


# ── Stage 4: Review ───────────────────────────────────────────────────────────


class ReviewIssue(BaseModel):
    """A single issue found during quality review."""

    criterion: str = Field(
        description="Which rubric criterion this issue falls under, e.g. "
        "'progressive_difficulty', 'realism', 'question_depth', 'scaffolding'."
    )
    description: str = Field(
        description="What is wrong — specific and actionable."
    )
    file_path: str | None = Field(
        default=None,
        description="Which file has the issue (relative to module dir), if specific."
    )
    suggested_fix: str = Field(
        description="Concrete revision instruction the module generator can act on."
    )


class ModuleReview(BaseModel):
    """Review result for a single module."""

    module_index: int
    verdict: Literal["pass", "revise"] = Field(
        description="'pass' if the module meets quality standards, "
        "'revise' if it needs changes."
    )
    issues: list[ReviewIssue] = Field(
        default_factory=list,
        description="List of issues found. Empty if verdict is 'pass'."
    )


class ReviewResult(BaseModel):
    """Phase 3b output — review of all modules."""

    modules: list[ModuleReview]
    overall_verdict: Literal["pass", "revise"] = Field(
        description="'pass' if all modules pass, 'revise' if any need changes."
    )


