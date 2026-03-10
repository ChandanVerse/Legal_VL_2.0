"""FastAPI backend for the Legal RAG chatbot."""

import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from db import init_db
from ingest import process_pdf as _process_pdf
from llm_client import call_gemini_chat, LLMError
from search import ANSWER_SYSTEM_PROMPT, get_retrieved_context, _make_source_dict
from pdf_utils import extract_pdf_text
from tool_calling import run_tool_calling_loop

app = FastAPI(title="Legal RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DB ──────────────────────────────────────────────────────────────────────

_thread_local = threading.local()

def get_conn():
    """Return a per-thread SQLite connection (safe for FastAPI's thread pool)."""
    if not hasattr(_thread_local, "conn"):
        _thread_local.conn = init_db()
    return _thread_local.conn

# ─── Session store ────────────────────────────────────────────────────────────

@dataclass
class Session:
    messages: list[dict] = field(default_factory=list)
    context: str = ""
    retrieved: list[dict] = field(default_factory=list)
    context_locked: bool = False
    sources_sent: bool = False
    last_active: float = field(default_factory=time.monotonic)

sessions: dict[str, Session] = {}
_SESSION_TTL = 3600  # 1 hour

def _prune_sessions() -> None:
    now = time.monotonic()
    stale = [sid for sid, s in sessions.items() if now - s.last_active > _SESSION_TTL]
    for sid in stale:
        del sessions[sid]

# ─── Ingest job store ─────────────────────────────────────────────────────────

ingest_jobs: dict[str, dict] = {}
_INGEST_JOB_TTL = 300  # prune finished jobs after 5 minutes

def _prune_ingest_jobs() -> None:
    now = time.monotonic()
    stale = [jid for jid, j in ingest_jobs.items()
             if j.get("finished_at") and now - j["finished_at"] > _INGEST_JOB_TTL]
    for jid in stale:
        del ingest_jobs[jid]

def _run_ingest(job_id: str, tmp_path: str) -> None:
    ingest_jobs[job_id]["status"] = "processing"
    try:
        conn = init_db()
        ok = _process_pdf(tmp_path, conn)
        conn.close()
        ingest_jobs[job_id]["status"] = "done" if ok else "failed"
        if not ok:
            ingest_jobs[job_id]["error"] = "Ingest failed — is Ollama running?"
    except Exception as e:
        ingest_jobs[job_id]["status"] = "failed"
        ingest_jobs[job_id]["error"] = str(e)
    finally:
        ingest_jobs[job_id]["finished_at"] = time.monotonic()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

SYSTEM_PROMPT_TEMPLATE = (
    ANSWER_SYSTEM_PROMPT
    + "\n\nThe following judgment excerpts are your ONLY source of information "
      "for this conversation:\n\n{context}"
)

# ─── Models ───────────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    content: str

class MessageResponse(BaseModel):
    reply: str
    sources: list[dict]

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/chat/new")
def new_chat():
    _prune_sessions()
    session_id = str(uuid.uuid4())
    sessions[session_id] = Session()
    return {"session_id": session_id}


@app.post("/chat/{session_id}/message", response_model=MessageResponse)
def send_message(session_id: str, req: MessageRequest):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    query = req.content.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message content is empty")

    # First message: run RAG and lock context for the session
    if not session.context_locked:
        context_text, retrieved = get_retrieved_context(query, get_conn())
        session.context = context_text
        session.retrieved = retrieved
        session.context_locked = True

    system_content = (
        SYSTEM_PROMPT_TEMPLATE.format(context=session.context)
        if session.context
        else ANSWER_SYSTEM_PROMPT
    )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(session.messages)
    messages.append({"role": "user", "content": query})

    try:
        reply = call_gemini_chat(messages)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))

    session.messages.append({"role": "user", "content": query})
    session.messages.append({"role": "assistant", "content": reply})
    session.last_active = time.monotonic()

    sources = []
    if not session.sources_sent and session.retrieved:
        sources = [_make_source_dict(item) for item in session.retrieved]
        session.sources_sent = True

    return MessageResponse(reply=reply, sources=sources)


@app.delete("/chat/{session_id}", status_code=204)
def delete_chat(session_id: str):
    sessions.pop(session_id, None)


# ─── PDF query route ─────────────────────────────────────────────────────────

@app.post("/chat/{session_id}/pdf-query", response_model=MessageResponse)
async def pdf_query(session_id: str, file: UploadFile = File(...), prompt: str = Form("")):
    """Upload a PDF and search the corpus for similar cases."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save to temp file
    content = await file.read()
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="lex_query_")
    os.write(fd, content)
    os.close(fd)

    try:
        pdf_text = extract_pdf_text(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    try:
        result = run_tool_calling_loop(pdf_text, prompt)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))

    user_content = prompt if prompt else f"Find similar cases for: {file.filename}"
    session.messages.append({"role": "user", "content": user_content})
    session.messages.append({"role": "assistant", "content": result["reply"]})
    session.context_locked = True

    return MessageResponse(reply=result["reply"], sources=result["sources"])


# ─── Ingest routes ────────────────────────────────────────────────────────────

@app.post("/ingest/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    job_id = str(uuid.uuid4())
    content = await file.read()

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix=f"lex_{job_id}_")
    os.write(fd, content)
    os.close(fd)

    ingest_jobs[job_id] = {"status": "queued", "filename": file.filename, "error": None}
    background_tasks.add_task(_run_ingest, job_id, tmp_path)

    return {"job_id": job_id, "filename": file.filename}


@app.get("/ingest/status/{job_id}")
def get_ingest_status(job_id: str):
    _prune_ingest_jobs()
    job = ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ─── File serving ─────────────────────────────────────────────────────────────

PDF_DIR = Path(__file__).parent / "data" / "dataset_pdfs"

@app.get("/files/{filename}")
def download_pdf(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files")
    path = PDF_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})
