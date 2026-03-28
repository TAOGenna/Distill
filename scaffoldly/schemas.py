"""Pydantic models for structured LLM output throughout the pipeline.

Blueprint architecture: Phase 1 produces rich contracts (scaffold_contract,
key_excerpts, validation_criteria) that constrain Phase 2 generation.
"""

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


# ── Stage 2: Blueprint (Curriculum Design) ───────────────────────────────────


_SCAFFOLDING_MAP: dict[str, str] = {
    "heavy": "heavy", "medium": "medium", "light": "light", "none": "none",
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
    what_is_provided: str = Field(
        description="What working code the student receives (~65% of file). "
        "Be specific: 'class with __init__, import block, __main__ test harness'."
    )
    what_student_writes: str = Field(
        description="What the student implements (~35%). Each TODO block with "
        "line counts: 'backward() ~8-12 lines, compute_loss() ~5-8 lines'."
    )
    milestone: str = Field(
        description="What the student sees when they run the exercise. "
        "Be specific: 'prints gradient table, all errors < 1e-5'."
    )
    key_insight: str = Field(
        default="",
        description="The single most important thing this exercise teaches. "
        "Goes in docstring and hints."
    )
    common_mistakes: str = Field(
        default="",
        description="Common student mistakes, semicolon-separated. "
        "Become warnings in scaffold comments."
    )
    expected_output_pattern: str = Field(
        default="",
        description="String that should appear in stdout when correct. "
        "E.g., 'relative error', 'pages/sec'. Used for validation."
    )

    @field_validator("scaffolding_level", mode="before")
    @classmethod
    def normalize_scaffolding(cls, v: str) -> str:
        if isinstance(v, str):
            mapped = _SCAFFOLDING_MAP.get(v.lower().strip())
            if mapped:
                return mapped
        return v


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
        description="Module indices this module requires as prerequisites.",
    )
    exercises: list[Exercise]
    inline_questions: list[InlineQuestion]
    visible_outcome: str
    key_excerpts: list[str] = Field(
        default_factory=list,
        description="Verbatim excerpts (200-500 chars each) from the source material "
        "containing the algorithms, formulas, pseudocode, or techniques this "
        "module's exercises must implement. These are injected into the module "
        "generator prompt to ground it in the source material's actual content. "
        "Extract the EXACT text — do not paraphrase."
    )


class Curriculum(BaseModel):
    course_title: str
    course_description: str
    modules: list[Module]


# ── Stage 2b: Curriculum Design (with root files) ────────────────────────────


def _unescape_content(v: str) -> str:
    """Fix double-escaped newlines that some models produce in JSON strings."""
    if isinstance(v, str) and "\\n" in v:
        v = v.replace("\\\\n", "\n").replace("\\n", "\n")
        v = v.replace("\\\\t", "\t").replace("\\t", "\t")
    return v


class SharedDefinitions(BaseModel):
    """Shared conventions across all modules."""

    language: str = Field(
        description="Primary programming language for the course: 'python', 'c', 'rust', etc."
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Real packages/libraries used across modules. "
        "E.g., ['numpy', 'torch', 'matplotlib'] or ['pthread', 'cilk']."
    )
    naming_convention: str = Field(
        default="snake_case",
        description="Variable/function naming convention: 'snake_case' or 'camelCase'."
    )


class CurriculumDesign(BaseModel):
    """Phase 1b output — Blueprint with rich contracts for module generation."""

    curriculum: Curriculum
    shared_definitions: SharedDefinitions = Field(
        description="Shared conventions (language, dependencies, naming) for all modules."
    )
    root_readme: str = Field(
        description="Full content of the course root README.md. Includes setup "
        "instructions, module order/learning path, and What's Next section."
    )
    requirements: str = Field(
        description="Content of requirements.txt (Python) or equivalent setup "
        "file (Cargo.toml, Makefile, package.json) for the course."
    )

    @field_validator("root_readme", "requirements", mode="before")
    @classmethod
    def fix_content_escaping(cls, v: str) -> str:
        return _unescape_content(v)


# ── Stage 3: Module Generation ────────────────────────────────────────────────


_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python", ".pyw": "python",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".js": "javascript", ".ts": "typescript",
    ".java": "java",
    ".md": "markdown", ".txt": "text",
    ".json": "json", ".csv": "csv", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".sh": "shell", ".bash": "shell",
    ".html": "html", ".css": "css",
}


class ExerciseFile(BaseModel):
    """A single exercise with both scaffold (student) and solution versions."""

    relative_path: str = Field(
        description="Path relative to the module directory, e.g. 'ex01_basic_dp.py'."
    )
    scaffold_content: str = Field(
        description="The STUDENT version of the file. Contains ~65% provided code "
        "(imports, class structures, __main__ block, docstrings, data fixtures) "
        "and ~35% TODO blocks where the student writes code. TODO blocks must "
        "include line count hints (e.g., '# YOUR CODE HERE - 8-12 lines'). "
        "Must parse/compile without errors. NotImplementedError in TODO zones."
    )
    solution_content: str = Field(
        description="The COMPLETE working version with all TODOs filled in. "
        "Must produce the output described in milestone when executed. "
        "Same structure as scaffold — identical imports, classes, __main__ block — "
        "but with solution code replacing TODO markers."
    )
    language: str = Field(
        default="",
        description="Programming language. Auto-detected from extension if omitted."
    )

    @field_validator("language", mode="before")
    @classmethod
    def infer_language(cls, v: str, info) -> str:
        if v:
            return v
        path = info.data.get("relative_path", "")
        if path:
            import os
            ext = os.path.splitext(path)[1].lower()
            return _EXT_TO_LANGUAGE.get(ext, "text")
        return "text"

    @field_validator("scaffold_content", "solution_content", mode="before")
    @classmethod
    def fix_file_escaping(cls, v: str) -> str:
        return _unescape_content(v)


class GeneratedFile(BaseModel):
    """A supporting (non-exercise) file produced by the module generator."""

    relative_path: str = Field(
        description="Path relative to the module directory, e.g. 'data/sample.csv' "
        "or 'Makefile'. Do not include the module directory name."
    )
    content: str = Field(
        description="Full file content."
    )
    language: str = Field(
        default="",
        description="File type. Auto-detected from extension if omitted."
    )

    @field_validator("language", mode="before")
    @classmethod
    def infer_language(cls, v: str, info) -> str:
        if v:
            return v
        path = info.data.get("relative_path", "")
        if path:
            import os
            ext = os.path.splitext(path)[1].lower()
            return _EXT_TO_LANGUAGE.get(ext, "text")
        return "text"

    @field_validator("content", mode="before")
    @classmethod
    def fix_file_escaping(cls, v: str) -> str:
        return _unescape_content(v)


class ModuleOutput(BaseModel):
    """Phase 2 output — all files for a single module."""

    readme: str = Field(
        description="Full content of the module README.md. Includes learning "
        "objectives, exercise walkthrough, 2-4 analytical questions at Level 3+ "
        "depth, and hints addressing common_mistakes from the Blueprint."
    )
    exercises: list[ExerciseFile] = Field(
        description="Exercise files with both scaffold and solution versions. "
        "Generate scaffold FIRST (think about pedagogy), then fill in the solution."
    )
    supporting_files: list[GeneratedFile] = Field(
        default_factory=list,
        description="Any additional files: data fixtures, configs, Makefiles, "
        "helper modules, header files. Optional."
    )

    @field_validator("readme", mode="before")
    @classmethod
    def fix_readme_escaping(cls, v: str) -> str:
        return _unescape_content(v)


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
