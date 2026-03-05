"""Search pipeline: FTS5 discovery -> Gemini tree nav -> page retrieval -> answer generation."""

import argparse
import json
import logging
import os
import re
import sys

import pymupdf
from dotenv import load_dotenv

from db import init_db, search_cases, get_tree, get_page_range
from llm_client import call_gemini, call_ollama, LLMError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TREE_NAV_SYSTEM_PROMPT = """You are navigating an Indian legal judgment to find the most relevant sections for a query.

You will receive a tree structure describing sections of a court judgment. Each node has:
- title: section heading
- summary: brief description of content
- key_topics: important legal topics covered
- section_type: facts, arguments, analysis, judgment, procedural, or other
- start_page / end_page: page range

Your task: identify which nodes are most relevant to answering the user's query.

Consider:
- For questions about legal principles: look for "analysis" and "judgment" sections
- For factual background: look for "facts" sections
- For procedural history: look for "procedural" sections
- For counsels' positions: look for "arguments" sections

Return JSON only:
{"relevant_nodes": [0, 2], "reasoning": "brief explanation", "confidence": 0.0}

Where relevant_nodes contains the 0-based indices of the most relevant nodes (pick 1-3 nodes).
confidence is between 0.0 and 1.0."""

ANSWER_SYSTEM_PROMPT = """You are a legal research assistant specializing in Indian Supreme Court jurisprudence.

Answer the user's legal question based ONLY on the provided judgment excerpts. Be precise and cite specific pages.

Rules:
- Cite every factual claim with the format: (Case Name, p. X)
- If multiple pages support a point, cite all: (Case Name, pp. X-Y)
- Distinguish between ratio decidendi (binding) and obiter dicta (persuasive)
- Note any dissenting opinions separately
- If the provided text does not contain enough information to answer, say so clearly
- Do not invent or assume facts not present in the excerpts"""


def parse_gemini_json(text: str) -> dict:
    """Parse JSON from Gemini response, handling markdown code blocks."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip().rstrip("`")
    return json.loads(cleaned)


def step1_discover_cases(conn, query: str, top_k: int = 5) -> list[dict]:
    """Step 1: FTS5 search to find candidate cases."""
    candidates = search_cases(conn, query, limit=top_k)
    log.info("Found %d candidate cases", len(candidates))
    return candidates


def step2_navigate_trees(conn, candidates: list[dict], query: str) -> list[dict]:
    """Step 2: For each candidate, use Gemini to identify relevant tree nodes.

    Returns list of {case_id, filename, nodes: [node_dicts], confidence}.
    """
    results = []

    for candidate in candidates:
        case_id = candidate["case_id"]
        filename = candidate["filename"]
        tree = get_tree(conn, case_id)

        if tree is None:
            log.warning("No tree found for %s, skipping", filename)
            continue

        nodes = tree.get("nodes", [])
        if not nodes:
            log.warning("Empty tree for %s, skipping", filename)
            continue

        # Build tree summary for Gemini
        tree_desc = f"Case: {filename}\nTotal pages: {tree.get('total_pages', '?')}\n\nSections:\n"
        for i, node in enumerate(nodes):
            tree_desc += (
                f"\n[{i}] {node.get('title', 'Untitled')}\n"
                f"    Type: {node.get('section_type', '?')}\n"
                f"    Pages: {node.get('start_page', '?')}-{node.get('end_page', '?')}\n"
                f"    Summary: {node.get('summary', '')}\n"
                f"    Topics: {', '.join(node.get('key_topics', []))}\n"
            )

        prompt = f"Query: {query}\n\n{tree_desc}"

        try:
            response = call_gemini(prompt, system_prompt=TREE_NAV_SYSTEM_PROMPT)
            nav_result = parse_gemini_json(response)
            relevant_indices = nav_result.get("relevant_nodes", [])
            confidence = nav_result.get("confidence", 0.0)

            # Validate indices
            selected_nodes = []
            for idx in relevant_indices:
                if isinstance(idx, int) and 0 <= idx < len(nodes):
                    selected_nodes.append(nodes[idx])

            if selected_nodes:
                results.append({
                    "case_id": case_id,
                    "filename": filename,
                    "nodes": selected_nodes,
                    "confidence": confidence,
                    "reasoning": nav_result.get("reasoning", ""),
                })

        except (LLMError, json.JSONDecodeError) as e:
            log.warning("Tree navigation failed for %s: %s", filename, e)
            continue

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def step3_retrieve_pages(conn, nav_results: list[dict]) -> list[dict]:
    """Step 3: Fetch actual page text for selected nodes.

    Returns list of {case_id, filename, pages: [{page_number, page_text}]}.
    """
    results = []
    for item in nav_results:
        all_pages = []
        seen_pages = set()
        for node in item["nodes"]:
            start = node.get("start_page", 1)
            end = node.get("end_page", start)
            pages = get_page_range(conn, item["case_id"], start, end)
            for p in pages:
                if p["page_number"] not in seen_pages:
                    all_pages.append(p)
                    seen_pages.add(p["page_number"])

        all_pages.sort(key=lambda x: x["page_number"])
        results.append({
            "case_id": item["case_id"],
            "filename": item["filename"],
            "pages": all_pages,
            "confidence": item["confidence"],
        })
    return results


def step4_generate_answer(query: str, retrieved: list[dict]) -> str:
    """Step 4: Send retrieved pages to Gemini for answer generation."""
    if not retrieved:
        return "No relevant cases found for your query."

    # Build context from all retrieved pages
    context_parts = []
    for item in retrieved:
        case_name = os.path.splitext(item["filename"])[0]
        context_parts.append(f"\n--- {case_name} ---")
        for page in item["pages"]:
            context_parts.append(f"\n[Page {page['page_number']}]\n{page['page_text']}")

    context = "\n".join(context_parts)

    # Trim context if too large (Gemini has large context but be reasonable)
    if len(context) > 500_000:
        context = context[:500_000] + "\n... [truncated]"

    prompt = f"Question: {query}\n\nJudgment Excerpts:\n{context}"

    try:
        answer = call_gemini(prompt, system_prompt=ANSWER_SYSTEM_PROMPT)
        return answer
    except LLMError as e:
        return f"Error generating answer: {e}"


def format_output(query: str, answer: str, retrieved: list[dict], as_json: bool = False) -> str:
    """Format the final output."""
    if as_json:
        return json.dumps({
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "case": os.path.splitext(item["filename"])[0],
                    "pages": [p["page_number"] for p in item["pages"]],
                    "confidence": item["confidence"],
                }
                for item in retrieved
            ],
        }, indent=2)

    # Markdown format
    lines = [
        "## Answer\n",
        answer,
        "\n## Sources\n",
    ]
    for item in retrieved:
        case_name = os.path.splitext(item["filename"])[0]
        page_nums = [str(p["page_number"]) for p in item["pages"]]
        lines.append(f"- **{case_name}** — pages {', '.join(page_nums)}")

    lines.append("\n## Closest Matching Cases (by similarity)\n")
    for rank, item in enumerate(retrieved, 1):
        case_name = os.path.splitext(item["filename"])[0]
        lines.append(f"{rank}. **{item['filename']}** (confidence: {item['confidence']:.2f})")

    return "\n".join(lines)


PDF_QUERY_PROMPT = """Read these pages from an Indian court judgment. List the key legal topics, acts, and parties as a comma-separated list of keywords only. No JSON. No explanation. Example: Prevention of Corruption Act, disqualification, IPC Section 109, legislative assembly, conviction

Pages:
{text}"""


def query_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF and generate a keyword search query using Ollama."""
    doc = pymupdf.open(pdf_path)
    try:
        pages_text = []
        for i, page in enumerate(doc):
            if i >= 5:
                break
            text = page.get_text().strip()
            if text:
                pages_text.append(text)
    finally:
        doc.close()

    if not pages_text:
        raise ValueError(f"No text extracted from {pdf_path}")

    combined = "\n\n".join(pages_text)[:12000]
    prompt = PDF_QUERY_PROMPT.format(text=combined)
    raw = call_ollama(prompt, expect_json=False).strip()

    # If model still returned JSON, extract key_topics values as fallback
    if raw.startswith("{"):
        try:
            import re as _re
            topics = _re.findall(r'"([^"]{4,})"', raw)
            raw = ", ".join(topics[:10]) if topics else raw[:200]
        except Exception:
            pass

    # Strip quotes and truncate
    query = raw.strip('"\'').split("\n")[0][:300]
    log.info("Generated query from PDF: %s", query)
    return query


def run_search(query: str, conn, top_k: int = 5, as_json: bool = False) -> str:
    """Execute the full 4-step search pipeline."""
    log.info("Query: %s", query)

    # Step 1: Discover cases
    candidates = step1_discover_cases(conn, query, top_k=top_k)
    if not candidates:
        return "No matching cases found in the database."

    # Step 2: Navigate trees
    nav_results = step2_navigate_trees(conn, candidates, query)
    if not nav_results:
        return "Found matching cases but could not identify relevant sections."

    # Step 3: Retrieve pages
    retrieved = step3_retrieve_pages(conn, nav_results)

    # Step 4: Generate answer
    answer = step4_generate_answer(query, retrieved)

    return format_output(query, answer, retrieved, as_json=as_json)


def main():
    parser = argparse.ArgumentParser(description="Search ingested legal judgments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Legal question to search for")
    group.add_argument("--pdf", help="Path to a PDF — finds the most similar ingested cases")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate cases")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--db-path", default=None, help="Override DB path")
    args = parser.parse_args()

    conn = init_db(args.db_path)

    if args.pdf:
        if not os.path.isfile(args.pdf):
            log.error("PDF not found: %s", args.pdf)
            sys.exit(1)
        log.info("Analysing PDF: %s", args.pdf)
        query = query_from_pdf(args.pdf)
        print(f"\nGenerated query: {query}\n")
    else:
        query = args.query

    output = run_search(query, conn, top_k=args.top_k, as_json=args.json)
    print(output)
    conn.close()


if __name__ == "__main__":
    main()
