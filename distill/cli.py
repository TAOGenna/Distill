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

    args = parser.parse_args()

    from .server import serve

    serve(port=args.port, open_browser=not args.no_open)
