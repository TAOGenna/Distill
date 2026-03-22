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
        help="URL of a blog post, GitHub repo, or technical article",
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
        help="Claude model to use (default: claude-opus-4-6)",
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
    print(f"  Level:  {args.level}", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    course_dir = run_agent_sync(
        url=args.url,
        user_level=args.level,
        output_dir=args.output,
        model=args.model,
        effort=args.effort,
        max_turns=args.max_turns,
    )

    print(f"\nCourse generated: {course_dir}")
    print(f"  Open the notebooks in Jupyter to start learning.\n")
