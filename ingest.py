"""PDF ingestion pipeline: extract text, build tree index via Ollama, store in SQLite."""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pymupdf
from dotenv import load_dotenv
from tqdm import tqdm

from db import get_case_id, init_db, store_case, store_pages, store_tree, case_exists
from llm_client import call_ollama, LLMError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PDF_DIR = os.getenv("PDF_DIR", "./data/dataset_pdfs")
BATCH_SIZE = 10
MAX_BATCH_CHARS = 96_000  # ~24K tokens for qwen2.5:7b (32K context)

OLLAMA_PROMPT_TEMPLATE = """You are a legal document analyst specializing in Indian Supreme Court judgments.

Analyze the following pages from an Indian court judgment and produce a structured summary node.

Context: Indian legal judgments typically contain:
- Ratio decidendi (binding legal principle)
- Obiter dicta (observations not binding)
- References to IPC sections, Constitutional Articles, CrPC provisions
- Writ petitions, Special Leave Petitions (SLPs), impugned orders
- Arguments from appellant/respondent counsel
- Discussion of precedent cases

Pages {start_page} to {end_page}:
---
{page_text}
---

Return a single JSON object with exactly these fields:
{{
    "title": "Brief descriptive title for this section",
    "summary": "2-4 sentence summary capturing key legal points, holdings, or arguments",
    "key_topics": ["list", "of", "key", "legal", "topics", "mentioned"],
    "section_type": "one of: facts|arguments|analysis|judgment|procedural|other",
    "start_page": {start_page},
    "end_page": {end_page}
}}

Return ONLY the JSON object, no other text."""


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from PDF, returning [(page_number, text)] for non-empty pages."""
    pages = []
    doc = pymupdf.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if text and text.strip():
                pages.append((i, text.strip()))
    finally:
        doc.close()
    return pages


def build_batch_prompt(batch: list[tuple[int, str]]) -> str:
    """Build Ollama prompt for a batch of pages."""
    start_page = batch[0][0]
    end_page = batch[-1][0]

    combined_text = "\n\n".join(
        f"[Page {pn}]\n{text}" for pn, text in batch
    )

    # Truncate if too long
    if len(combined_text) > MAX_BATCH_CHARS:
        combined_text = combined_text[:MAX_BATCH_CHARS] + "\n... [truncated]"

    return OLLAMA_PROMPT_TEMPLATE.format(
        start_page=start_page,
        end_page=end_page,
        page_text=combined_text,
    )


def validate_node(node: dict, batch: list[tuple[int, str]]) -> dict:
    """Validate and fix node fields."""
    start_page = batch[0][0]
    end_page = batch[-1][0]

    valid_types = {"facts", "arguments", "analysis", "judgment", "procedural", "other"}

    node.setdefault("title", f"Pages {start_page}-{end_page}")
    node.setdefault("summary", "")
    node.setdefault("key_topics", [])
    node.setdefault("section_type", "other")
    node["start_page"] = start_page
    node["end_page"] = end_page

    if node["section_type"] not in valid_types:
        node["section_type"] = "other"
    if not isinstance(node["key_topics"], list):
        node["key_topics"] = []

    return node


def _process_batch(batch: list[tuple[int, str]], filename: str) -> dict:
    """Process a single batch through Ollama. Thread-safe."""
    prompt = build_batch_prompt(batch)
    try:
        node = call_ollama(prompt, expect_json=True)
        node = validate_node(node, batch)
    except LLMError as e:
        start_page, end_page = batch[0][0], batch[-1][0]
        log.warning("Ollama failed for pages %d-%d of %s: %s. Using fallback.",
                    start_page, end_page, filename, e)
        node = {
            "title": f"Pages {start_page}-{end_page}",
            "summary": " ".join(text[:200] for _, text in batch)[:500],
            "key_topics": [],
            "section_type": "other",
            "start_page": start_page,
            "end_page": end_page,
        }
    return node


def process_pdf(pdf_path: str, conn, workers: int = 4) -> bool:
    """Process a single PDF: extract, index, store. Returns True on success."""
    filename = os.path.basename(pdf_path)
    case_id = get_case_id(filename)

    log.info("Processing: %s (id: %s)", filename, case_id)

    # Extract pages
    try:
        pages = extract_pages(pdf_path)
    except Exception as e:
        log.error("Failed to extract PDF %s: %s", filename, e)
        return False

    if not pages:
        log.warning("No text extracted from %s, skipping", filename)
        return False

    page_count = pages[-1][0]  # highest page number
    batches = [pages[i:i + BATCH_SIZE] for i in range(0, len(pages), BATCH_SIZE)]

    # Build tree nodes via Ollama (parallel)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_batch, batch, filename) for batch in batches]
        nodes = [f.result() for f in futures]  # preserves submission order

    # Build tree
    tree = {
        "case_id": case_id,
        "total_pages": page_count,
        "nodes": nodes,
    }

    # Store in DB (single transaction)
    try:
        store_case(conn, case_id, filename, page_count)
        store_pages(conn, case_id, pages)
        store_tree(conn, case_id, json.dumps(tree))
        conn.commit()
        log.info("Stored %s: %d pages, %d nodes", filename, len(pages), len(nodes))
        return True
    except Exception as e:
        conn.rollback()
        log.error("DB error for %s: %s", filename, e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Ingest legal PDFs into PageIndex")
    parser.add_argument("--pdf-dir", default=PDF_DIR, help="Directory containing PDFs")
    parser.add_argument("--limit", type=int, default=0, help="Max PDFs to process (0=all)")
    parser.add_argument("--resume", action="store_true", help="Skip already-ingested cases")
    parser.add_argument("--db-path", default=None, help="Override DB path")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent Ollama calls per PDF (default: 4)")
    args = parser.parse_args()

    if not os.path.isdir(args.pdf_dir):
        log.error("PDF directory not found: %s", args.pdf_dir)
        sys.exit(1)

    conn = init_db(args.db_path)

    # Collect PDF files
    pdf_files = sorted(
        f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf")
    )
    log.info("Found %d PDFs in %s", len(pdf_files), args.pdf_dir)

    # Filter for resume
    if args.resume:
        before = len(pdf_files)
        pdf_files = [
            f for f in pdf_files if not case_exists(conn, get_case_id(f))
        ]
        log.info("Resume mode: skipping %d already-ingested, %d remaining",
                 before - len(pdf_files), len(pdf_files))

    # Apply limit
    if args.limit > 0:
        pdf_files = pdf_files[: args.limit]

    if not pdf_files:
        log.info("No PDFs to process.")
        conn.close()
        return

    success = 0
    failed = 0

    for filename in tqdm(pdf_files, desc="Ingesting PDFs"):
        pdf_path = os.path.join(args.pdf_dir, filename)
        if process_pdf(pdf_path, conn, workers=args.workers):
            success += 1
        else:
            failed += 1

    log.info("Done: %d succeeded, %d failed out of %d", success, failed, success + failed)
    conn.close()


if __name__ == "__main__":
    main()
