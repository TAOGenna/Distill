"""Prompt templates for the multi-stage coursework generation pipeline."""

ANALYSIS_PROMPT = """You are an expert technical educator analyzing source material for coursework creation.

Analyze the following technical content thoroughly. Your analysis will be used to design a progressive programming course.

Return your analysis as a JSON object with this exact structure:

{{
  "title": "Concise title for this material",
  "summary": "2-3 sentence summary of what this covers",
  "domain": "Primary technical domain (e.g., systems programming, machine learning, web development, graphics)",
  "overall_difficulty": "beginner | intermediate | advanced | expert",
  "key_concepts": [
    {{
      "name": "concept name",
      "description": "brief description",
      "importance": "core | supporting | tangential",
      "difficulty": "beginner | intermediate | advanced | expert"
    }}
  ],
  "prerequisites": [
    {{
      "name": "prerequisite topic",
      "why_needed": "why this is needed to understand the material",
      "difficulty": "beginner | intermediate | advanced | expert"
    }}
  ],
  "code_patterns": [
    {{
      "description": "what this code pattern does",
      "language": "programming language",
      "concepts_demonstrated": ["concept1", "concept2"]
    }}
  ],
  "learning_goals": [
    "Goal 1: what someone should understand after mastering this"
  ]
}}

SOURCE MATERIAL:
{content}"""


CURRICULUM_PROMPT = """You are designing a progressive programming course based on technical content analysis.
Your courses follow Stanford CS231n methodology — the gold standard in scaffolded technical education.

STUDENT PROFILE:
{user_level}

CONTENT ANALYSIS:
{analysis}

Design a course with 3-6 modules that progressively builds the student's understanding from their current level to mastery of the source material.

CRITICAL DESIGN PRINCIPLES (from Stanford CS231n):
1. Start with foundations the student needs but may lack
2. Each module: 3-5 focused exercises, building from easy to hard
3. Early modules: HEAVY scaffolding (student fills in 3-10 lines within a provided function)
4. Later modules: LIGHTER scaffolding (student implements entire functions or small systems)
5. EVERY exercise must have an immediate automated test that validates correctness
6. Include "Inline Questions" — conceptual questions between exercises that force reflection
7. Each module should produce a visible, satisfying result (a plot, a benchmark, a working demo)
8. Difficulty should increase WITHIN each module AND across modules
9. Later modules should reuse code/concepts from earlier modules

Return a JSON object with this exact structure:

{{
  "course_title": "Course title",
  "course_description": "1-2 sentence description",
  "modules": [
    {{
      "module_index": 0,
      "title": "Module title",
      "description": "What this module covers and why it matters",
      "learning_objectives": ["Objective 1", "Objective 2"],
      "concepts_covered": ["concept1", "concept2"],
      "exercises": [
        {{
          "title": "Exercise title",
          "type": "implement | fill_blank | debug | analyze | extend",
          "description": "What the student does",
          "scaffolding_level": "heavy | medium | light | none",
          "what_is_provided": "What code/structure is given",
          "what_student_writes": "What the student must implement",
          "test_strategy": "How to validate correctness"
        }}
      ],
      "inline_questions": [
        {{
          "question": "Conceptual question",
          "context": "What was just implemented that motivates this question"
        }}
      ],
      "visible_outcome": "What satisfying result the student sees at the end"
    }}
  ]
}}"""


MODULE_PROMPT = """You are generating a Jupyter notebook assignment module in the style of Stanford CS231n.
This is the most important step — the quality of these exercises determines whether someone actually learns.

COURSE CONTEXT:
{curriculum}

MODULE TO GENERATE:
{module_spec}

SOURCE MATERIAL ANALYSIS:
{analysis}

STUDENT LEVEL:
{user_level}

Generate a complete Jupyter notebook as a JSON array of cells. Follow these patterns EXACTLY:

## PATTERN 1: Scaffolded function (HEAVY scaffolding)
```python
def function_name(arg1, arg2):
    \"\"\"
    [Thorough docstring explaining the algorithm step by step]

    The approach:
    1. First, we do X because...
    2. Then, we compute Y using...
    3. Finally, we return Z

    Args:
        arg1: Description including type and shape if applicable
        arg2: Description

    Returns:
        result: Description including expected type/shape
    \"\"\"
    result = None
    ###########################################################################
    # TODO: [Clear, specific description of what to implement]               #
    #                                                                         #
    # Hint: [Concrete hint about the approach or key function to use]         #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return result
```

## PATTERN 2: Test cell (IMMEDIATELY after each exercise)
```python
# ===== Test your implementation =====
result = function_name(test_input)
expected = known_correct_output

# Specific assertion with helpful error message
assert some_condition, f"Expected {{expected}}, got {{result}}"
print("✓ Test passed!")

# Optional: additional edge case tests
result2 = function_name(edge_case_input)
assert another_condition, "Edge case failed: [description]"
print("✓ Edge case test passed!")
```

## PATTERN 3: Inline conceptual question
```markdown
---
**Inline Question:** [Thought-provoking question about what was just implemented]

*Your answer:* [leave blank for student to fill in]

---
```

## PATTERN 4: Progressive reveal (later exercises build on earlier ones)
```python
# Now let's use our function_from_exercise_1 in a more complex context
def advanced_function(x):
    # This uses function_from_exercise_1 internally
    intermediate = function_from_exercise_1(x)
    ###########################################################################
    # TODO: [Build on the previous exercise]                                  #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
```

## OUTPUT FORMAT

Return a JSON object:
{{
  "cells": [
    {{"cell_type": "markdown", "source": "# Module Title\\n\\nIntroduction..."}},
    {{"cell_type": "code", "source": "import numpy as np\\n# Setup code..."}},
    {{"cell_type": "markdown", "source": "## Exercise 1: Title\\n\\nExplanation..."}},
    {{"cell_type": "code", "source": "def func():\\n    # scaffolded code..."}},
    {{"cell_type": "code", "source": "# Test\\nassert ..."}},
    {{"cell_type": "markdown", "source": "**Inline Question:** ..."}},
    ...more cells...
    {{"cell_type": "markdown", "source": "## Summary\\n\\nIn this module you learned..."}}
  ]
}}

## HARD REQUIREMENTS
1. ALL code cells must contain syntactically valid Python (this will be checked)
2. Tests must ACTUALLY PASS when the TODO sections are correctly implemented
3. Every scaffolded function MUST have a comprehensive docstring
4. Use the EXACT TODO/END banner comment pattern shown above (with ###... borders)
5. Each exercise MUST build on previous ones where possible
6. Include ALL necessary imports in the first code cell
7. DO NOT use placeholder data — create realistic test data for the domain
8. The first cell MUST be a markdown introduction explaining what this module covers
9. The last cell MUST be a markdown summary of what was learned
10. Make exercises SPECIFIC to the source material — not generic programming exercises
11. When the student completes all exercises, they should have built something tangible related to the source material
12. Include at least one visualization or print output that shows the result working"""


BLOG_INDEX_PROMPT = """You are analyzing a blog's index/home page to find individual blog post links.

Given this HTML content from a blog, extract all links to individual blog posts.
Return a JSON object:

{{
  "blog_name": "Name of the blog or author",
  "posts": [
    {{
      "title": "Post title",
      "url": "Full URL to the post",
      "description": "Brief description if available"
    }}
  ]
}}

Only include actual blog post links, not navigation, social media, or other links.

HTML CONTENT:
{content}"""
