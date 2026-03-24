"""CLI entry point for Scaffoldly."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="scaffoldly",
        description=(
            "Generate progressive, CS231n-style coursework from technical "
            "blogs, repos, or papers using an AI agent powered by Claude."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── generate ────────────────────────────────────────────────────────────
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a full course from a URL",
    )
    gen_parser.add_argument(
        "url",
        help="Primary URL — the focus source that drives the curriculum",
    )
    gen_parser.add_argument(
        "--ref",
        action="append",
        default=[],
        dest="refs",
        metavar="URL",
        help=(
            "Additional reference URL (repeatable). "
            "Skimmed for supplementary concepts, not deeply analyzed. "
            "Use with --series if the sources form an ordered progression."
        ),
    )
    gen_parser.add_argument(
        "--series",
        action="store_true",
        default=False,
        help=(
            "Treat the primary URL and all --ref URLs as an ordered series "
            "(e.g. Part 1 → Part 2 → Part 3). The curriculum spans the full "
            "arc. Without this flag, refs are treated as supplementary context."
        ),
    )
    gen_parser.add_argument(
        "--level",
        required=True,
        help=(
            'Description of the student\'s current level, e.g. '
            '"junior Python developer, new to systems programming"'
        ),
    )
    gen_parser.add_argument(
        "--output",
        default="./output",
        help="Output directory for generated course (default: ./output)",
    )
    gen_parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Model for analysis and curriculum design (default: claude-opus-4-6)",
    )
    gen_parser.add_argument(
        "--generate-model",
        default="sonnet",
        dest="generate_model",
        help=(
            "Model for file generation sub-agents (default: sonnet). "
            "File generation is mostly mechanical — a cheaper model keeps "
            "costs down without sacrificing learning quality."
        ),
    )
    gen_parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "max"],
        default="high",
        help="Agent effort level (default: high)",
    )
    gen_parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum agent turns before stopping (default: 30)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        _cmd_generate(args)


def _cmd_generate(args):
    """Generate a full course from a URL using the agent."""
    from .agent import run_agent_sync

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Scaffoldly — Generating coursework", file=sys.stderr)
    print(f"  Source: {args.url}", file=sys.stderr)
    if args.refs:
        mode = "series" if args.series else "reference"
        print(f"  Refs ({mode}): {len(args.refs)} additional source(s)", file=sys.stderr)
    print(f"  Level:  {args.level}", file=sys.stderr)
    print(f"  Models: {args.model} (design) → {args.generate_model} (generate)", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    result = run_agent_sync(
        url=args.url,
        refs=args.refs,
        series=args.series,
        user_level=args.level,
        output_dir=args.output,
        model=args.model,
        generate_model=args.generate_model,
        effort=args.effort,
        max_turns=args.max_turns,
    )

    course_dir = result["course_dir"]
    total_cost_usd = result["total_cost_usd"]
    usage = result["usage"]

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Course generated: {course_dir}", file=sys.stderr)
    if total_cost_usd is not None:
        print(f"  Total cost:       ${total_cost_usd:.4f}", file=sys.stderr)
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        total_input = input_tokens + cache_creation + cache_read
        print(f"  Input tokens:     {total_input:,}", file=sys.stderr)
        if cache_read:
            print(f"    (cache read:    {cache_read:,})", file=sys.stderr)
        if cache_creation:
            print(f"    (cache write:   {cache_creation:,})", file=sys.stderr)
        if input_tokens:
            print(f"    (uncached:      {input_tokens:,})", file=sys.stderr)
        print(f"  Output tokens:    {output_tokens:,}", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    print(f"\nCourse generated: {course_dir}")
    print(f"  Open the course directory to start learning.\n")
