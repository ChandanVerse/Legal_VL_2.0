# PageIndex — Legal Judgment RAG System

A retrieval-augmented generation (RAG) system for querying Indian Supreme Court judgments. No vector embeddings — uses **FTS5 full-text search** for discovery and **LLM-generated tree indexes** for precise section navigation.

## How It Works

```
PDF → Text Extraction → Ollama (local) → Tree Index → SQLite

Query → FTS5 Search → Gemini Tree Nav → Page Retrieval → Gemini Answer

PDF Upload → Text Extraction → Gemini Tool Calling → search_corpus() → Gemini Answer
```

**Ingestion** (offline, run once):
1. Extract page text from PDFs via PyMuPDF
2. Batch pages (10 at a time) and send to a local Ollama model (legal-mistral)
3. Ollama produces a structured JSON node per batch: title, summary, key topics, section type, page range
4. All pages + tree index stored in SQLite with FTS5 (porter stemmer)

**Search** (4-step pipeline):
1. **FTS5 discovery** — BM25 ranked search finds candidate cases
2. **Tree navigation** — Gemini reads each case's section tree and selects the most relevant nodes for the query
3. **Page retrieval** — Fetch raw page text for selected sections
4. **Answer generation** — Gemini produces a cited answer from the retrieved pages

**PDF Query** (tool-calling flow):
1. User uploads a PDF + optional prompt via the frontend
2. First 5 pages extracted with PyMuPDF (no Ollama needed)
3. PDF text + prompt sent to Gemini with a `search_corpus` tool definition
4. Gemini decides what to search for, calls `search_corpus` one or more times (max 3 rounds)
5. Tool results (retrieved case excerpts) fed back into the conversation
6. Gemini synthesizes a final answer with citations from all retrieved context

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.11+ |
| [Ollama](https://ollama.com) | Running locally with `legal-mistral` pulled |
| Gemini API key | Free tier works; set `GEMINI_API_KEY` in `.env` |

## Setup

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd Legal_VL_2.0
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -e .

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**.env file:**
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash     # optional, this is the default
OLLAMA_MODEL=legal-mistral            # optional, this is the default
OLLAMA_BASE_URL=http://localhost:11434  # optional
DB_PATH=./legal.db                # optional
```

## Usage

### Ingest PDFs

```bash
# Ingest a single test PDF
.venv/Scripts/python.exe ingest.py --pdf-dir ./data/test_case --limit 1

# Ingest all PDFs in the dataset
.venv/Scripts/python.exe ingest.py --pdf-dir ./data/dataset_pdfs

# Resume an interrupted ingestion (skips already-processed cases)
.venv/Scripts/python.exe ingest.py --pdf-dir ./data/dataset_pdfs --resume

# Control parallelism (Ollama calls per PDF)
.venv/Scripts/python.exe ingest.py --pdf-dir ./data/dataset_pdfs --workers 8
```

> **Note:** Ollama must be running (`ollama serve`) before ingestion.

### Search

```bash
# Natural language query
.venv/Scripts/python.exe search.py --query "right to life under Article 21"

# Return more candidate cases
.venv/Scripts/python.exe search.py --query "preventive detention" --top-k 10

# JSON output (for programmatic use)
.venv/Scripts/python.exe search.py --query "bail conditions" --json

# Find cases similar to a PDF (auto-generates query from the document)
.venv/Scripts/python.exe search.py --pdf ./data/test_case/my_case.pdf
```

## Project Structure

```
Legal_VL_2.0/
├── db.py              # SQLite schema, FTS5 search, CRUD helpers
├── llm_client.py      # Ollama and Gemini API wrappers with retry logic
├── ingest.py          # PDF → tree index pipeline (CLI)
├── search.py          # 4-step search pipeline (CLI)
├── tool_calling.py    # Gemini tool-calling orchestration for PDF queries
├── api.py             # FastAPI backend (chat, PDF query, ingest endpoints)
├── pyproject.toml
├── .env               # API keys (not committed)
├── data/
│   ├── dataset_pdfs/  # 200 Supreme Court judgments
│   └── test_case/     # Single PDF for quick testing
└── frontend/          # React + Vite UI
```

## Database Schema

```sql
cases           -- case_id (SHA-256 of filename), filename, page_count
case_pages      -- case_id, page_number, page_text
case_trees      -- case_id, tree (JSON: nodes with titles/summaries/page ranges)
case_pages_fts  -- FTS5 virtual table over page_text (porter tokenizer)
```

## Configuration Reference

| Env Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | _(required)_ | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for search + tool calling |
| `OLLAMA_MODEL` | `legal-mistral` | Ollama model for indexing |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DB_PATH` | `./legal.db` | SQLite database path |

## Dependencies

- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF text extraction
- [google-genai](https://googleapis.github.io/python-genai/) — Gemini API (new SDK)
- [requests](https://requests.readthedocs.io) — Ollama HTTP calls
- [python-dotenv](https://pypi.org/project/python-dotenv/) — `.env` loading
- [tqdm](https://tqdm.github.io) — Progress bars
