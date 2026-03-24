"""CLI entry point for Scaffoldly."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ── ANSI colors ──────────────────────────────────────────────────────────────

class _C:
    _enabled = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    RESET   = "\033[0m"   if _enabled else ""
    BOLD    = "\033[1m"    if _enabled else ""
    DIM     = "\033[2m"    if _enabled else ""
    RED     = "\033[31m"   if _enabled else ""
    GREEN   = "\033[32m"   if _enabled else ""
    YELLOW  = "\033[33m"   if _enabled else ""
    CYAN    = "\033[36m"   if _enabled else ""
    WHITE   = "\033[37m"   if _enabled else ""


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

    _err = lambda s="": print(s, file=sys.stderr)

    _err()
    _err(f"  {_C.CYAN}{_C.BOLD}Scaffoldly{_C.RESET}")
    _err(f"  {_C.DIM}{'─' * 50}{_C.RESET}")
    _err(f"  {_C.DIM}Source:{_C.RESET}  {args.url}")
    if args.refs:
        mode = "series" if args.series else "reference"
        _err(f"  {_C.DIM}Refs:{_C.RESET}    {len(args.refs)} additional source(s) ({mode})")
    _err(f"  {_C.DIM}Level:{_C.RESET}   {args.level}")
    _err(f"  {_C.DIM}Models:{_C.RESET}  {args.model} (design) → {args.generate_model} (generate)")
    _err(f"  {_C.DIM}{'─' * 50}{_C.RESET}")
    _err()

    wall_start = time.time()
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

    # Count generated files
    course_path = Path(course_dir) if course_dir else None
    file_count = 0
    if course_path and course_path.exists():
        generated_files = [f for f in course_path.rglob("*") if f.is_file() and not f.name.startswith("_")]
        file_count = len(generated_files)

    wall_elapsed = time.time() - wall_start
    wall_mins, wall_secs = divmod(int(wall_elapsed), 60)

    _err()
    _err(f"  {_C.DIM}{'─' * 50}{_C.RESET}")
    _err()

    if file_count == 0:
        _err(f"  {_C.RED}{_C.BOLD}FAILED — No course files were generated{_C.RESET}")
        _err()
        _err(f"  {_C.DIM}Check the logs above for {_C.RED}ERROR{_C.DIM} messages.{_C.RESET}")
        _err(f"  {_C.DIM}Common causes:{_C.RESET}")
        _err(f"    {_C.DIM}- MCP tool validation errors (submit_analysis / submit_curriculum){_C.RESET}")
        _err(f"    {_C.DIM}- Sub-agent dispatch failures{_C.RESET}")
        _err()
        _print_stats(
            _err, course_dir, file_count, wall_mins, wall_secs,
            total_cost_usd, usage,
        )
        sys.exit(1)

    _err(f"  {_C.GREEN}{_C.BOLD}Course generated{_C.RESET}")
    _err(f"  {course_dir} ({file_count} files)")
    _err()
    _print_stats(
        _err, course_dir, file_count, wall_mins, wall_secs,
        total_cost_usd, usage,
    )

    print(f"\n{course_dir}")


def _print_stats(_err, course_dir, file_count, wall_mins, wall_secs,
                 total_cost_usd, usage):
    """Print the stats block (time, cost, tokens)."""
    _err(f"  {_C.DIM}Time:{_C.RESET}    {wall_mins}m {wall_secs}s")
    if total_cost_usd is not None:
        _err(f"  {_C.DIM}Cost:{_C.RESET}    ${total_cost_usd:.4f}")
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        total_input = input_tokens + cache_creation + cache_read
        _err(f"  {_C.DIM}Tokens:{_C.RESET}  {total_input:,} in / {output_tokens:,} out")
        parts = []
        if cache_read:
            parts.append(f"cache read: {cache_read:,}")
        if cache_creation:
            parts.append(f"cache write: {cache_creation:,}")
        if input_tokens:
            parts.append(f"uncached: {input_tokens:,}")
        if parts:
            _err(f"  {_C.DIM}         ({', '.join(parts)}){_C.RESET}")
    _err()
