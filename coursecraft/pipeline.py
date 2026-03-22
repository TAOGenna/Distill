"""Multi-stage LLM pipeline for coursework generation."""

import json
import re
import sys
from pathlib import Path

import anthropic

from .ingest import ingest
from .notebook import (
    _slugify,
    cells_to_notebook,
    create_course_readme_notebook,
    save_notebook,
)
from .prompts import ANALYSIS_PROMPT, CURRICULUM_PROMPT, MODULE_PROMPT


def _log(msg: str) -> None:
    """Print a status message."""
    print(f"  → {msg}", file=sys.stderr, flush=True)


def _extract_json(text: str) -> dict | list:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Try to find JSON in code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        text = fence_match.group(1)

    # Try direct parse
    text = text.strip()
    return json.loads(text)


class CoursePipeline:
    """Orchestrates the multi-stage coursework generation pipeline."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _call_llm(self, system: str, prompt: str, max_tokens: int = 16000) -> str:
        """Make a single LLM call and return the text response."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system=system,
        )
        return message.content[0].text

    def analyze(self, content: dict) -> dict:
        """Stage 1: Analyze source material and extract concepts."""
        _log("Analyzing source material...")

        # Truncate very long content to fit context
        material = content["content"]
        if len(material) > 80_000:
            material = material[:80_000] + "\n\n[Content truncated for analysis]"

        prompt = ANALYSIS_PROMPT.format(content=material)
        response = self._call_llm(
            system="You are an expert technical educator. Always respond with valid JSON.",
            prompt=prompt,
        )

        try:
            analysis = _extract_json(response)
        except json.JSONDecodeError:
            # Retry with explicit instruction
            response = self._call_llm(
                system="You are an expert technical educator. You MUST respond with ONLY valid JSON, no other text.",
                prompt=prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no explanation, just the JSON object.",
            )
            analysis = _extract_json(response)

        _log(
            f"Found {len(analysis.get('key_concepts', []))} concepts, "
            f"{len(analysis.get('prerequisites', []))} prerequisites"
        )
        return analysis

    def design_curriculum(self, analysis: dict, user_level: str) -> dict:
        """Stage 2: Design the course curriculum based on analysis + user level."""
        _log("Designing curriculum...")

        prompt = CURRICULUM_PROMPT.format(
            user_level=user_level,
            analysis=json.dumps(analysis, indent=2),
        )
        response = self._call_llm(
            system="You are a world-class course designer following Stanford CS231n methodology. Always respond with valid JSON.",
            prompt=prompt,
            max_tokens=16000,
        )

        curriculum = _extract_json(response)
        n_modules = len(curriculum.get("modules", []))
        _log(f"Designed {n_modules} modules")
        return curriculum

    def generate_module(
        self,
        module_spec: dict,
        curriculum: dict,
        analysis: dict,
        user_level: str,
    ) -> list[dict]:
        """Stage 3: Generate a single module's notebook cells."""
        module_idx = module_spec.get("module_index", 0)
        module_title = module_spec.get("title", f"Module {module_idx}")
        _log(f"Generating module {module_idx}: {module_title}...")

        # Build a compact curriculum context (just titles + current module detail)
        curriculum_context = {
            "course_title": curriculum.get("course_title", ""),
            "modules_overview": [
                {"index": m["module_index"], "title": m["title"]}
                for m in curriculum.get("modules", [])
            ],
        }

        prompt = MODULE_PROMPT.format(
            curriculum=json.dumps(curriculum_context, indent=2),
            module_spec=json.dumps(module_spec, indent=2),
            analysis=json.dumps(analysis, indent=2),
            user_level=user_level,
        )

        response = self._call_llm(
            system=(
                "You are generating a Jupyter notebook assignment. "
                "You MUST return valid JSON with a 'cells' array. "
                "Each cell has 'cell_type' (markdown or code) and 'source' (string). "
                "All Python code must be syntactically valid."
            ),
            prompt=prompt,
            max_tokens=16000,
        )

        result = _extract_json(response)
        cells = result if isinstance(result, list) else result.get("cells", [])
        _log(f"  Generated {len(cells)} cells")
        return cells

    def run(
        self,
        url: str,
        user_level: str,
        output_dir: str = "./output",
    ) -> str:
        """Run the full pipeline: ingest → analyze → design → generate.

        Returns the path to the generated course directory.
        """
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  CourseCraft — Generating coursework", file=sys.stderr)
        print(f"  Source: {url}", file=sys.stderr)
        print(f"  Level: {user_level}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

        # Stage 0: Ingest content
        _log("Fetching content...")
        content = ingest(url)
        _log(f"Fetched: {content['title']}")

        # Stage 1: Analyze
        analysis = self.analyze(content)

        # Stage 2: Design curriculum
        curriculum = self.design_curriculum(analysis, user_level)

        # Stage 3: Generate each module
        course_title = curriculum.get("course_title", content["title"])
        course_slug = _slugify(0, course_title).lstrip("00_")
        course_dir = Path(output_dir) / course_slug
        course_dir.mkdir(parents=True, exist_ok=True)

        modules = curriculum.get("modules", [])

        # Save analysis and curriculum for reference
        (course_dir / "_analysis.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False)
        )
        (course_dir / "_curriculum.json").write_text(
            json.dumps(curriculum, indent=2, ensure_ascii=False)
        )

        # Generate overview notebook
        overview_nb = create_course_readme_notebook(
            course_title,
            curriculum.get("course_description", ""),
            modules,
        )
        save_notebook(overview_nb, course_dir / "00_overview.ipynb")

        # Generate each module notebook
        for module_spec in modules:
            cells = self.generate_module(
                module_spec, curriculum, analysis, user_level
            )
            nb = cells_to_notebook(cells)
            slug = _slugify(module_spec["module_index"], module_spec["title"])
            save_notebook(nb, course_dir / f"{slug}.ipynb")

        _log(f"\nCourse generated at: {course_dir}")
        return str(course_dir)
