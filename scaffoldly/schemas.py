"""Pydantic models for structured LLM output throughout the pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Stage 1: Analysis ──────────────────────────────────────────────────────────


class Concept(BaseModel):
    name: str
    description: str
    importance: Literal["core", "supporting", "tangential"]
    difficulty: Literal["beginner", "intermediate", "advanced", "expert"]


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


class Exercise(BaseModel):
    title: str
    type: Literal["implement", "fill_blank", "debug", "analyze", "extend"]
    description: str
    scaffolding_level: Literal["heavy", "medium", "light", "none"]
    what_is_provided: str
    what_student_writes: str
    milestone: str = Field(
        description="What the student sees when they run the exercise. "
        "Describe the output — printed measurements, saved plots, or "
        "visualizations that reproduce a key insight from the source material."
    )


class InlineQuestion(BaseModel):
    question: str
    context: str


class Module(BaseModel):
    module_index: int
    title: str
    description: str
    learning_objectives: list[str]
    concepts_covered: list[str]
    exercises: list[Exercise]
    inline_questions: list[InlineQuestion]
    visible_outcome: str


class Curriculum(BaseModel):
    course_title: str
    course_description: str
    modules: list[Module]


