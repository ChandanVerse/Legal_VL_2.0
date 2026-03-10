"""Gemini tool-calling orchestration for PDF queries."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types

from db import init_db
from llm_client import call_gemini, _response_text, LLMError
from pdf_utils import extract_pdf_text
from search import get_retrieved_context, _make_source_dict

log = logging.getLogger(__name__)

# ─── Tool definition ──────────────────────────────────────────────────────────

SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_corpus",
            description=(
                "Search the ingested Indian Supreme Court judgment corpus. "
                "Returns relevant case excerpts matching the query."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query for finding relevant cases.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of candidate cases to retrieve (1-5).",
                    },
                },
                "required": ["query"],
            },
        )
    ]
)

TOOL_SYSTEM_PROMPT = """You are a legal research assistant specializing in Indian Supreme Court jurisprudence.

You have been given the text of an uploaded PDF (a court judgment or legal document). Your job is to:
1. Read and understand the PDF content.
2. Use the search_corpus tool to find related cases in the corpus. You may call it multiple times with different queries to cover different legal issues.
3. After retrieving sufficient context, synthesize a comprehensive answer with citations.

Strategy for tool calls:
- Extract key legal issues, acts, and principles from the PDF.
- Search for each major issue separately for better results.
- Use specific legal terms, section numbers, and case names in your queries.

Citation rules:
- Cite every factual claim: (Case Name, p. X)
- Distinguish ratio decidendi (binding) from obiter dicta (persuasive)
- Note dissenting opinions separately
- If retrieved text is insufficient, say so clearly
- Do not invent facts not present in the retrieved excerpts"""

MAX_CONTEXT_CHARS = 28_000


def extract_pdf_text(pdf_path: str, max_pages: int = 5) -> str:
    """Extract text from the first N pages of a PDF using pymupdf."""
    doc = pymupdf.open(pdf_path)
    try:
        pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text = page.get_text().strip()
            if text:
                pages.append(f"[Page {i + 1}]\n{text}")
    finally:
        doc.close()

    if not pages:
        raise ValueError(f"No text extracted from {pdf_path}")

    return "\n\n".join(pages)


def _dispatch_tool_call(fc) -> tuple[str, str, list[dict]]:
    """Execute a single search_corpus tool call. Opens its own DB connection (thread-safe).
    Returns (call_name, result_text, sources).
    """
    if fc.name != "search_corpus":
        return fc.name, f'{{"error": "Unknown tool: {fc.name}"}}', []

    args = fc.args or {}
    query = args.get("query", "")
    top_k = args.get("top_k", 3)
    if isinstance(top_k, str):
        top_k = int(top_k) if top_k.isdigit() else 3
    top_k = max(1, min(top_k, 5))
    log.info("search_corpus(query=%r, top_k=%d)", query, top_k)

    conn = init_db()
    context_text, retrieved = get_retrieved_context(query, conn, top_k=top_k)
    conn.close()
    sources = [_make_source_dict(item) for item in retrieved]
    return fc.name, context_text or "No matching cases found for this query.", sources


def run_tool_calling_loop(
    pdf_text: str,
    user_prompt: str,
    max_rounds: int = 3,
) -> dict:
    """Orchestrate Gemini tool-calling loop for PDF queries.

    Returns: {"reply": str, "sources": list[dict]}
    """
    if len(pdf_text) > 15_000:
        log.warning("PDF text truncated from %d to 15,000 chars for prompt", len(pdf_text))

    user_content = f"## Uploaded PDF Content\n\n{pdf_text[:15_000]}"
    if user_prompt:
        user_content += f"\n\n## User Request\n\n{user_prompt}"
    else:
        user_content += "\n\n## User Request\n\nFind similar cases and analyze the key legal issues in this document."

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_content)]),
    ]

    all_sources: list[dict] = []
    seen_cases: set[str] = set()
    total_context_chars = 0
    budget_lock = threading.Lock()

    for round_num in range(max_rounds):
        log.info("Tool calling round %d/%d", round_num + 1, max_rounds)

        try:
            response = call_gemini(
                contents=contents,
                system_instruction=TOOL_SYSTEM_PROMPT,
                tools=[SEARCH_TOOL],
            )
        except LLMError as e:
            log.error("Gemini call failed: %s", e)
            raise

        function_calls = response.function_calls
        if not function_calls:
            return {"reply": _response_text(response), "sources": all_sources}

        # Append the model turn (with function calls) to the conversation
        contents.append(response.candidates[0].content)

        # Dispatch all tool calls in parallel
        func_response_parts: list[types.Part] = [None] * len(function_calls)

        with ThreadPoolExecutor(max_workers=len(function_calls)) as executor:
            futures = {
                executor.submit(_dispatch_tool_call, fc): i
                for i, fc in enumerate(function_calls)
            }
            for future in as_completed(futures):
                idx = futures[future]
                fc_name, context_text, sources = future.result()

                # Deduplicate sources
                for source in sources:
                    if source["case"] not in seen_cases:
                        seen_cases.add(source["case"])
                        all_sources.append(source)

                # Apply context budget (thread-safe)
                with budget_lock:
                    remaining = MAX_CONTEXT_CHARS - total_context_chars
                    if remaining <= 0:
                        tool_result = "Context budget exhausted. Please synthesize your answer from the results already provided."
                    elif context_text and context_text != "No matching cases found for this query.":
                        tool_result = context_text[:remaining]
                        total_context_chars += len(tool_result)
                    else:
                        tool_result = context_text

                func_response_parts[idx] = types.Part.from_function_response(
                    name="search_corpus",
                    response={"result": tool_result},
                )

        contents.append(types.Content(role="user", parts=func_response_parts))

    # Max rounds exhausted — get final answer without tools
    log.info("Max rounds reached, requesting final answer")
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text="Please synthesize your final answer now based on all the search results above.")],
        )
    )

    try:
        reply = _response_text(call_gemini(contents=contents, system_instruction=TOOL_SYSTEM_PROMPT))
    except LLMError as e:
        reply = f"Error generating final answer: {e}"

    return {"reply": reply, "sources": all_sources}
