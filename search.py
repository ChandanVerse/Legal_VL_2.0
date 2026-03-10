"""Search pipeline: FTS5 discovery -> LLM tree nav -> page retrieval -> answer generation."""

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from db import init_db, search_cases, get_tree, get_page_range
from llm_client import call_gemini_text, call_ollama, LLMError
from pdf_utils import extract_pdf_text

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


def _make_source_dict(item: dict) -> dict:
    """Build the standard source dict from a retrieved item."""
    return {
        "case": os.path.splitext(item["filename"])[0],
        "pages": [p["page_number"] for p in item["pages"]],
        "confidence": round(item["confidence"], 2),
    }


def parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip().rstrip("`")
    return json.loads(cleaned)


def _navigate_single(candidate: dict, query: str) -> dict | None:
    """Navigate one case's tree and return a result dict, or None on failure.
    Opens its own DB connection so it is safe to run in a thread pool.
    """
    case_id = candidate["case_id"]
    filename = candidate["filename"]
    conn = init_db()
    tree = get_tree(conn, case_id)

    if tree is None:
        log.warning("No tree found for %s, skipping", filename)
        return None

    nodes = tree.get("nodes", [])
    if not nodes:
        log.warning("Empty tree for %s, skipping", filename)
        return None

    tree_desc = f"Case: {filename}\nTotal pages: {tree.get('total_pages', '?')}\n\nSections:\n"
    for i, node in enumerate(nodes):
        tree_desc += (
            f"\n[{i}] {node.get('title', 'Untitled')}\n"
            f"    Type: {node.get('section_type', '?')}\n"
            f"    Pages: {node.get('start_page', '?')}-{node.get('end_page', '?')}\n"
            f"    Summary: {node.get('summary', '')}\n"
            f"    Topics: {', '.join(node.get('key_topics', []))}\n"
        )

    try:
        response = call_gemini_text(f"Query: {query}\n\n{tree_desc}", system_prompt=TREE_NAV_SYSTEM_PROMPT)
        nav_result = parse_llm_json(response)
        relevant_indices = nav_result.get("relevant_nodes", [])
        selected_nodes = [
            nodes[idx] for idx in relevant_indices
            if isinstance(idx, int) and 0 <= idx < len(nodes)
        ]
        if selected_nodes:
            return {
                "case_id": case_id,
                "filename": filename,
                "nodes": selected_nodes,
                "confidence": nav_result.get("confidence", 0.0),
                "reasoning": nav_result.get("reasoning", ""),
            }
    except (LLMError, json.JSONDecodeError) as e:
        log.warning("Tree navigation failed for %s: %s", filename, e)

    return None


def step2_navigate_trees(conn, candidates: list[dict], query: str) -> list[dict]:
    """Step 2: For each candidate, use LLM to identify relevant tree nodes (parallel).

    Returns list of {case_id, filename, nodes: [node_dicts], confidence}.
    """
    results = []
    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        futures = {executor.submit(_navigate_single, c, query): c for c in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

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


def build_context_text(retrieved: list[dict], max_context_chars: int = 30_000) -> str:
    """Build a context string from retrieved pages, capped at max_context_chars."""
    context_parts = []
    char_count = 0
    for item in retrieved:
        case_name = os.path.splitext(item["filename"])[0]
        header = f"\n--- {case_name} ---"
        context_parts.append(header)
        char_count += len(header)
        for page in item["pages"]:
            chunk = f"\n[Page {page['page_number']}]\n{page['page_text']}"
            if char_count + len(chunk) > max_context_chars:
                context_parts.append("\n... [truncated]")
                break
            context_parts.append(chunk)
            char_count += len(chunk)
    return "\n".join(context_parts)


def get_retrieved_context(
    query: str, conn, top_k: int = 5, max_context_chars: int = 30_000
) -> tuple[str, list[dict]]:
    """Steps 1-3: FTS5 discovery -> tree navigation -> page retrieval.

    Returns:
        (context_text, retrieved_list) where retrieved_list contains source info.
        Both are empty if no results are found.
    """
    candidates = search_cases(conn, query, limit=top_k)
    log.info("Found %d candidate cases", len(candidates))
    if not candidates:
        return "", []

    nav_results = step2_navigate_trees(conn, candidates, query)
    if not nav_results:
        return "", []

    retrieved = step3_retrieve_pages(conn, nav_results)
    context_text = build_context_text(retrieved, max_context_chars)
    return context_text, retrieved


def step4_generate_answer(query: str, context_text: str) -> str:
    """Step 4: Send context text to LLM for answer generation."""
    if not context_text:
        return "No relevant cases found for your query."

    prompt = f"Question: {query}\n\nJudgment Excerpts:\n{context_text}"

    try:
        answer = call_gemini_text(prompt, system_prompt=ANSWER_SYSTEM_PROMPT)
        return answer
    except LLMError as e:
        return f"Error generating answer: {e}"


def format_output(query: str, answer: str, retrieved: list[dict], as_json: bool = False) -> str:
    """Format the final output."""
    if as_json:
        return json.dumps({
            "query": query,
            "answer": answer,
            "sources": [_make_source_dict(item) for item in retrieved],
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
    combined = extract_pdf_text(pdf_path)[:12000]
    prompt = PDF_QUERY_PROMPT.format(text=combined)
    raw = call_ollama(prompt, expect_json=False).strip()

    # If model still returned JSON, extract key_topics values as fallback
    if raw.startswith("{"):
        try:
            topics = re.findall(r'"([^"]{4,})"', raw)
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

    context_text, retrieved = get_retrieved_context(query, conn, top_k)
    if not retrieved:
        return "No matching cases found in the database."

    answer = step4_generate_answer(query, context_text)
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
