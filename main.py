"""Unified CLI entrypoint for PageIndex legal document RAG system."""

import argparse
import sys

from ingest import main as ingest_main
from search import main as search_main


def main():
    parser = argparse.ArgumentParser(
        prog="pageindex",
        description="PageIndex: Tree-based RAG for Indian Court Judgments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ingest subcommand ---
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDFs into the index")
    ingest_parser.add_argument("--pdf-dir", default=None, help="Directory containing PDFs")
    ingest_parser.add_argument("--limit", type=int, default=0, help="Max PDFs to process (0=all)")
    ingest_parser.add_argument("--resume", action="store_true", help="Skip already-ingested cases")
    ingest_parser.add_argument("--db-path", default=None, help="Override DB path")
    ingest_parser.add_argument("--workers", type=int, default=4, help="Max concurrent Ollama calls per PDF (default: 4)")

    # --- search subcommand ---
    search_parser = subparsers.add_parser("search", help="Search ingested judgments")
    search_parser.add_argument("--query", required=True, help="Legal question to search for")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of candidate cases")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")
    search_parser.add_argument("--db-path", default=None, help="Override DB path")

    args = parser.parse_args()

    # Re-inject args into sys.argv so the submodule's main() parses them correctly
    if args.command == "ingest":
        sys.argv = ["ingest.py"]
        if args.pdf_dir:
            sys.argv += ["--pdf-dir", args.pdf_dir]
        if args.limit:
            sys.argv += ["--limit", str(args.limit)]
        if args.resume:
            sys.argv.append("--resume")
        if args.db_path:
            sys.argv += ["--db-path", args.db_path]
        if args.workers != 4:
            sys.argv += ["--workers", str(args.workers)]
        ingest_main()

    elif args.command == "search":
        sys.argv = ["search.py", "--query", args.query]
        if args.top_k != 5:
            sys.argv += ["--top-k", str(args.top_k)]
        if args.json:
            sys.argv.append("--json")
        if args.db_path:
            sys.argv += ["--db-path", args.db_path]
        search_main()


if __name__ == "__main__":
    main()
