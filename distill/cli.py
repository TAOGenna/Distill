"""CLI entry point for Distill — launches the web UI server."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="distill",
        description=(
            "Generate progressive, CS231n-style coursework from technical "
            "blogs, repos, or papers. Opens a web UI for course generation."
        ),
    )

    # Top-level flags (backward-compat: `distill --port 9000`)
    parser.add_argument(
        "--port",
        type=int,
        default=8420,
        help="Port for the web UI (default: 8420)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        default=False,
        help="Don't open the browser automatically",
    )

    sub = parser.add_subparsers(dest="command")

    # publish subcommand
    pub = sub.add_parser(
        "publish",
        help="Build a static site from courses and push to GitHub Pages",
    )
    pub.add_argument(
        "--to",
        type=str,
        metavar="PATH",
        help="Destination directory (e.g. /path/to/your-site/courses)",
    )
    pub.add_argument(
        "--output-dir",
        type=str,
        help="Course output directory (default: from config)",
    )
    pub.add_argument(
        "courses",
        nargs="*",
        help="Specific course names to publish (default: all)",
    )

    args = parser.parse_args()

    if args.command == "publish":
        from .publish import publish

        publish(
            output_dir=args.output_dir,
            to=args.to,
            courses=args.courses or None,
        )
    else:
        from .server import serve

        serve(port=args.port, open_browser=not args.no_open)
