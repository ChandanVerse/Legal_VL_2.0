"""Database layer for PageIndex legal document RAG system."""

import hashlib
import json
import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "./legal.db")


def get_case_id(filename: str) -> str:
    """Deterministic case ID from filename stem (SHA-256, 16 hex chars)."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return hashlib.sha256(stem.encode("utf-8")).hexdigest()[:16]


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Initialize database with schema and return connection."""
    db_path = db_path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id   TEXT PRIMARY KEY,
            filename  TEXT NOT NULL,
            page_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS case_pages (
            case_id     TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            page_text   TEXT NOT NULL,
            PRIMARY KEY (case_id, page_number),
            FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS case_trees (
            case_id    TEXT PRIMARY KEY,
            tree       TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS case_pages_fts USING fts5(
            page_text,
            content='case_pages',
            content_rowid='rowid',
            tokenize='porter'
        );

        CREATE TRIGGER IF NOT EXISTS case_pages_ai AFTER INSERT ON case_pages BEGIN
            INSERT INTO case_pages_fts(rowid, page_text)
            VALUES (new.rowid, new.page_text);
        END;

        CREATE TRIGGER IF NOT EXISTS case_pages_ad AFTER DELETE ON case_pages BEGIN
            INSERT INTO case_pages_fts(case_pages_fts, rowid, page_text)
            VALUES ('delete', old.rowid, old.page_text);
        END;
    """)

    return conn


def store_case(conn: sqlite3.Connection, case_id: str, filename: str, page_count: int):
    """Insert a case record."""
    conn.execute(
        "INSERT OR REPLACE INTO cases (case_id, filename, page_count) VALUES (?, ?, ?)",
        (case_id, filename, page_count),
    )


def store_pages(conn: sqlite3.Connection, case_id: str, pages: list[tuple[int, str]]):
    """Bulk insert pages for a case. pages = [(page_number, page_text), ...]."""
    conn.executemany(
        "INSERT OR REPLACE INTO case_pages (case_id, page_number, page_text) VALUES (?, ?, ?)",
        [(case_id, pn, txt) for pn, txt in pages],
    )


def store_tree(conn: sqlite3.Connection, case_id: str, tree_json: str):
    """Insert tree JSON for a case."""
    conn.execute(
        "INSERT OR REPLACE INTO case_trees (case_id, tree) VALUES (?, ?)",
        (case_id, tree_json),
    )


def get_page_range(
    conn: sqlite3.Connection, case_id: str, start: int, end: int
) -> list[dict]:
    """Fetch pages in [start, end] range for a case."""
    rows = conn.execute(
        "SELECT page_number, page_text FROM case_pages "
        "WHERE case_id = ? AND page_number >= ? AND page_number <= ? "
        "ORDER BY page_number",
        (case_id, start, end),
    ).fetchall()
    return [{"page_number": r["page_number"], "page_text": r["page_text"]} for r in rows]


def get_tree(conn: sqlite3.Connection, case_id: str) -> dict | None:
    """Load and parse tree JSON for a case."""
    row = conn.execute(
        "SELECT tree FROM case_trees WHERE case_id = ?", (case_id,)
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["tree"])


def case_exists(conn: sqlite3.Connection, case_id: str) -> bool:
    """Check if a case is already ingested."""
    row = conn.execute(
        "SELECT 1 FROM cases WHERE case_id = ?", (case_id,)
    ).fetchone()
    return row is not None


def search_cases(
    conn: sqlite3.Connection, query: str, limit: int = 5
) -> list[dict]:
    """FTS5 BM25 search over page text, results grouped by case.

    Returns list of dicts: {case_id, filename, page_number, snippet, rank}.
    """
    # Sanitize: strip FTS5 special chars, keep only alphanumeric words
    import re as _re
    clean = _re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    tokens = [w for w in clean.split() if len(w) >= 3]
    if not tokens:
        tokens = query.split()[:5]
    fts_query = " OR ".join(tokens[:30])  # cap to avoid huge queries

    rows = conn.execute(
        """
        SELECT
            cp.case_id,
            c.filename,
            cp.page_number,
            snippet(case_pages_fts, 0, '>>>', '<<<', '...', 48) AS snippet,
            rank
        FROM case_pages_fts
        JOIN case_pages cp ON case_pages_fts.rowid = cp.rowid
        JOIN cases c ON cp.case_id = c.case_id
        WHERE case_pages_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (fts_query, limit * 3),  # fetch extra to allow grouping
    ).fetchall()

    # Group by case, keep best (lowest rank) per case
    seen: dict[str, dict] = {}
    for r in rows:
        cid = r["case_id"]
        if cid not in seen:
            seen[cid] = {
                "case_id": cid,
                "filename": r["filename"],
                "page_number": r["page_number"],
                "snippet": r["snippet"],
                "rank": r["rank"],
            }
    results = sorted(seen.values(), key=lambda x: x["rank"])
    return results[:limit]


if __name__ == "__main__":
    conn = init_db()
    print(f"Database initialized at {DB_PATH}")
    conn.close()
