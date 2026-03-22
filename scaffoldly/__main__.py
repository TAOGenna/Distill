"""CLI entry point: python -m scaffoldly"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="scaffoldly",
        description=(
            "Generate progressive, CS231n-style coursework from technical "
            "blogs, repos, or papers. Transforms expert-level content into "
            "scaffolded exercises with tests, inline questions, and "
            "progressive difficulty."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- generate command ---
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
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use (default: claude-sonnet-4-20250514)",
    )

    # --- analyze command (lightweight, just shows analysis) ---
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a URL and show extracted concepts (no course generation)",
    )
    analyze_parser.add_argument("url", help="URL to analyze")
    analyze_parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use",
    )

    # --- list-posts command (for blog indexes) ---
    list_parser = subparsers.add_parser(
        "list-posts",
        help="List blog posts from a blog index page",
    )
    list_parser.add_argument("url", help="Blog index URL")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "list-posts":
        _cmd_list_posts(args)
    elif args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "generate":
        _cmd_generate(args)


def _cmd_list_posts(args):
    """List blog posts from an index page."""
    from .ingest import ingest_blog_index

    posts = ingest_blog_index(args.url)
    if not posts:
        print("No blog posts found at that URL.")
        return

    print(f"\nFound {len(posts)} posts:\n")
    for i, post in enumerate(posts, 1):
        print(f"  {i}. {post['title']}")
        print(f"     {post['url']}\n")


def _cmd_analyze(args):
    """Analyze a URL and print the extracted concepts."""
    import json

    from .ingest import ingest
    from .pipeline import CoursePipeline

    content = ingest(args.url)
    print(f"\nFetched: {content['title']}\n", file=sys.stderr)

    pipeline = CoursePipeline(model=args.model)
    analysis = pipeline.analyze(content)

    print(json.dumps(analysis, indent=2, ensure_ascii=False))


def _cmd_generate(args):
    """Generate a full course from a URL."""
    from .pipeline import CoursePipeline

    pipeline = CoursePipeline(model=args.model)
    course_dir = pipeline.run(
        url=args.url,
        user_level=args.level,
        output_dir=args.output,
    )
    print(f"\n✓ Course generated: {course_dir}")
    print(f"  Open the notebooks in Jupyter to start learning.\n")


if __name__ == "__main__":
    main()
