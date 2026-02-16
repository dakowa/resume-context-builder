"""
Enhanced Database Module for Resume Context Builder

Features:
- Multiple context databases support
- Information decay (auto-remove old data)
- Time-based filtering
- Fixed deletion functionality
"""

import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Any

from sqlalchemy import create_engine, text, Engine
import uuid

# State directory for all context databases
STATE_DIR = Path.home() / ".context-packager-state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Default context database
DEFAULT_CONTEXT_DB = "default"


def get_database_url(context_db: str = DEFAULT_CONTEXT_DB) -> str:
    """Get database URL for a specific context database.
    
    Args:
        context_db: Name of the context database (default: "default")
    """
    # Prefer explicit env vars; fallback to stateful SQLite file
    url = (
        os.getenv("CONTEXT_DB_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("PGSERVER_URL")
        or os.getenv("PGSERVER")
    )
    if url:
        return url
    
    # Use separate database file for each context
    db_file = STATE_DIR / f"context_{context_db}.db"
    return f"sqlite:///{db_file.as_posix()}"


def get_engine(context_db: str = DEFAULT_CONTEXT_DB, echo: bool = False) -> Engine:
    """Get SQLAlchemy engine for a specific context database."""
    engine = create_engine(
        get_database_url(context_db), 
        echo=echo, 
        future=True, 
        pool_pre_ping=True
    )
    init_schema(engine)
    return engine


def list_context_databases() -> List[str]:
    """List all available context databases."""
    databases = []
    for db_file in STATE_DIR.glob("context_*.db"):
        # Extract context name from filename (context_NAME.db)
        name = db_file.stem.replace("context_", "")
        if name:
            databases.append(name)
    return sorted(databases)


def delete_context_database(context_db: str) -> bool:
    """Delete a context database entirely.
    
    Returns True if deleted, False if not found or is default.
    """
    if context_db == DEFAULT_CONTEXT_DB:
        return False  # Protect default database
    
    db_file = STATE_DIR / f"context_{context_db}.db"
    if db_file.exists():
        try:
            db_file.unlink()
            return True
        except Exception:
            pass
    return False


def init_schema(engine: Engine) -> None:
    """Initialize database schema with enhanced tables."""
    with engine.begin() as conn:
        # Chunks table with context support and timestamps
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT,
                    chunk_name TEXT,
                    hash TEXT UNIQUE,
                    content TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    source_date TEXT  -- Original document date if available
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at)
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_source_date ON chunks(source_date)
                """
            )
        )
        
        # Meta nodes table
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS meta_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT,
                    name TEXT,
                    normalized_name TEXT UNIQUE
                )
                """
            )
        )
        
        # Chunk-meta edges
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chunk_meta_edges (
                    chunk_id INTEGER,
                    meta_node_id INTEGER,
                    weight REAL,
                    PRIMARY KEY (chunk_id, meta_node_id)
                )
                """
            )
        )
        
        # File index with enhanced tracking
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS file_index (
                    path TEXT PRIMARY KEY,
                    sha256 TEXT,
                    params_sig TEXT,
                    updated_at TEXT,
                    file_date TEXT  -- Original file modification date
                )
                """
            )
        )
        
        # Jobs metadata
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    input_dir TEXT,
                    md_out_dir TEXT,
                    sla_minutes INTEGER,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
        )
        
        # Backfill migration: ensure sla_minutes column exists
        try:
            if engine.dialect.name == "sqlite":
                rows = conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
                cols = [r[1] for r in rows]
                if "sla_minutes" not in cols:
                    conn.execute(text("ALTER TABLE jobs ADD COLUMN sla_minutes INTEGER"))
            else:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS sla_minutes INTEGER"))
        except Exception:
            pass
        
        # Job runs with enhanced tracking
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS job_runs (
                    id TEXT PRIMARY KEY,
                    job_id TEXT,
                    input_dir TEXT,
                    md_out_dir TEXT,
                    status TEXT,
                    progress INTEGER,
                    processed_files INTEGER,
                    total_files INTEGER,
                    chunks_upserted INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    last_message TEXT,
                    log TEXT,
                    cancel_requested INTEGER DEFAULT 0,
                    error TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_job_runs_started_at ON job_runs(started_at)
                """
            )
        )
        
        # Feedback tracking
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    chunk_id INTEGER,
                    score REAL,
                    created_at TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query)
                """
            )
        )
        
        # Search history
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    params_json TEXT,
                    result_count INTEGER,
                    created_at TEXT
                )
                """
            )
        )
        
        # Decay tracking table
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS decay_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT,
                    details TEXT,
                    executed_at TEXT
                )
                """
            )
        )
        
        # Prepare pgvector schema if available
        try:
            if engine.dialect.name == "postgresql":
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS chunk_vectors (
                            chunk_id BIGINT PRIMARY KEY,
                            svd vector,
                            updated_at TIMESTAMP
                        )
                        """
                    )
                )
        except Exception:
            pass


# ============================================================================
# Context Database Management
# ============================================================================

def ensure_job(
    job_id: str, 
    name: str | None, 
    input_dir: str, 
    md_out_dir: str | None, 
    sla_minutes: int | None = None,
    context_db: str = DEFAULT_CONTEXT_DB
) -> None:
    """Upsert a job metadata row for visibility in UI."""
    engine = get_engine(context_db)
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO jobs(id, name, input_dir, md_out_dir, sla_minutes, created_at, updated_at)
                VALUES(:id, :name, :in_dir, :out_dir, :sla, :now, :now)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    input_dir=excluded.input_dir,
                    md_out_dir=excluded.md_out_dir,
                    sla_minutes=excluded.sla_minutes,
                    updated_at=excluded.updated_at
                """
            ),
            {
                "id": job_id, 
                "name": name, 
                "in_dir": input_dir, 
                "out_dir": md_out_dir, 
                "sla": (int(sla_minutes) if sla_minutes is not None else None), 
                "now": now
            },
        )


def start_job_run(
    job_id: str | None, 
    input_dir: str, 
    md_out_dir: str | None,
    context_db: str = DEFAULT_CONTEXT_DB
) -> str:
    """Create a job_run row and return its id."""
    engine = get_engine(context_db)
    run_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO job_runs(id, job_id, input_dir, md_out_dir, status, progress, processed_files, total_files, chunks_upserted, started_at, finished_at, last_message, log, cancel_requested, error)
                VALUES(:id, :job_id, :in_dir, :out_dir, 'running', 0, 0, 0, 0, :now, NULL, '', '', 0, NULL)
                """
            ),
            {"id": run_id, "job_id": job_id, "in_dir": input_dir, "out_dir": md_out_dir, "now": now},
        )
    return run_id


def update_job_run(
    run_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    processed_files: int | None = None,
    total_files: int | None = None,
    chunks_upserted: int | None = None,
    last_message: str | None = None,
    append_log: str | None = None,
    error: str | None = None,
    context_db: str = DEFAULT_CONTEXT_DB
) -> None:
    """Partial update of a job_run row."""
    engine = get_engine(context_db)
    sets: list[str] = []
    params: dict = {"id": run_id}
    if status is not None:
        sets.append("status=:status")
        params["status"] = status
    if progress is not None:
        sets.append("progress=:progress")
        params["progress"] = int(max(0, min(100, progress)))
    if processed_files is not None:
        sets.append("processed_files=:pf")
        params["pf"] = int(processed_files)
    if total_files is not None:
        sets.append("total_files=:tf")
        params["tf"] = int(total_files)
    if chunks_upserted is not None:
        sets.append("chunks_upserted=:cu")
        params["cu"] = int(chunks_upserted)
    if last_message is not None:
        sets.append("last_message=:lm")
        params["lm"] = str(last_message)
    if error is not None:
        sets.append("error=:err")
        params["err"] = str(error)
    stmt = "UPDATE job_runs SET " + ", ".join(sets) + " WHERE id=:id"
    if sets:
        with engine.begin() as conn:
            conn.execute(text(stmt), params)
    if append_log:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE job_runs
                    SET log=COALESCE(log,'') || :chunk
                    WHERE id=:id
                    """
                ),
                {"id": run_id, "chunk": (append_log or "")},
            )


def finish_job_run(
    run_id: str, 
    *, 
    status: str, 
    error: str | None = None,
    context_db: str = DEFAULT_CONTEXT_DB
) -> None:
    """Mark a run finished and set status and optional error."""
    engine = get_engine(context_db)
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE job_runs
                SET status=:status, finished_at=:now, error=:error
                WHERE id=:id
                """
            ),
            {"id": run_id, "status": status, "now": now, "error": error},
        )


def request_cancel_run(run_id: str, context_db: str = DEFAULT_CONTEXT_DB) -> None:
    """Signal a running job to cancel."""
    engine = get_engine(context_db)
    with engine.begin() as conn:
        conn.execute(text("UPDATE job_runs SET cancel_requested=1 WHERE id=:id"), {"id": run_id})


def is_cancel_requested(run_id: str, context_db: str = DEFAULT_CONTEXT_DB) -> bool:
    engine = get_engine(context_db)
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT cancel_requested FROM job_runs WHERE id=:id"), 
            {"id": run_id}
        ).fetchone()
        try:
            return bool(row[0]) if row else False
        except Exception:
            return False


def fetch_recent_runs(
    limit: int = 50, 
    job_id: str | None = None,
    context_db: str = DEFAULT_CONTEXT_DB
) -> List[Tuple]:
    """Return recent job runs, newest first."""
    engine = get_engine(context_db)
    params: dict = {"limit": int(limit)}
    query = """SELECT id, job_id, input_dir, md_out_dir, status, progress, processed_files, total_files, chunks_upserted, started_at, finished_at, last_message, substr(log, -5000), cancel_requested, error FROM job_runs"""
    if job_id:
        query += " WHERE job_id=:job_id"
        params["job_id"] = job_id
    query += " ORDER BY started_at DESC LIMIT :limit"
    with engine.begin() as conn:
        rows = conn.execute(text(query), params)
        return [tuple(r) for r in rows]


def fetch_jobs(context_db: str = DEFAULT_CONTEXT_DB) -> List[Tuple]:
    """Return jobs list."""
    engine = get_engine(context_db)
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, name, input_dir, md_out_dir, created_at, updated_at, sla_minutes FROM jobs ORDER BY updated_at DESC")
        )
        return [tuple(r) for r in rows]


def fetch_last_success(job_id: str, context_db: str = DEFAULT_CONTEXT_DB) -> str | None:
    """Return ISO timestamp of last successful run for a job."""
    engine = get_engine(context_db)
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT started_at FROM job_runs WHERE job_id=:jid AND status='success' ORDER BY started_at DESC LIMIT 1"),
            {"jid": job_id}
        ).fetchone()
        return str(row[0]) if row else None


# ============================================================================
# Chunk Operations
# ============================================================================

def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def upsert_chunks(
    engine: Engine, 
    records: List[Tuple[str, str, str]],
    source_date: str | None = None
) -> None:
    """Upsert chunk records with optional source date.
    
    records items: (path, chunk_name, content)
    """
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        for path, chunk_name, content in records:
            h = compute_hash(content)
            conn.execute(
                text(
                    """
                    INSERT INTO chunks(path, chunk_name, hash, content, created_at, updated_at, source_date)
                    VALUES(:path, :chunk_name, :hash, :content, :now, :now, :source_date)
                    ON CONFLICT(hash) DO UPDATE SET
                        path=excluded.path,
                        chunk_name=excluded.chunk_name,
                        content=excluded.content,
                        updated_at=excluded.updated_at,
                        source_date=COALESCE(excluded.source_date, chunks.source_date)
                    """
                ),
                {
                    "path": path, 
                    "chunk_name": chunk_name, 
                    "hash": h, 
                    "content": content, 
                    "now": now,
                    "source_date": source_date
                },
            )


def get_file_index(
    engine: Engine, 
    path: str
) -> Tuple[str, str] | None:
    """Return (sha256, params_sig) for a path if recorded."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT sha256, params_sig FROM file_index WHERE path=:p"), 
            {"p": path}
        ).fetchone()
        return (row[0], row[1]) if row else None


def upsert_file_index(
    engine: Engine, 
    path: str, 
    sha256: str, 
    params_sig: str | None,
    file_date: str | None = None
) -> None:
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO file_index(path, sha256, params_sig, updated_at, file_date)
                VALUES(:path, :sha, :psig, :now, :file_date)
                ON CONFLICT(path) DO UPDATE SET
                    sha256=excluded.sha256,
                    params_sig=excluded.params_sig,
                    updated_at=excluded.updated_at,
                    file_date=COALESCE(excluded.file_date, file_index.file_date)
                """
            ),
            {"path": path, "sha": sha256, "psig": params_sig, "now": now, "file_date": file_date},
        )


def delete_chunks_by_path(
    engine: Engine, 
    path: str
) -> int:
    """Delete all chunks that belong to a specific file path.
    
    Returns number of rows deleted.
    """
    with engine.begin() as conn:
        # First get the count
        count_result = conn.execute(
            text("SELECT COUNT(*) FROM chunks WHERE path=:p"), 
            {"p": path}
        ).fetchone()
        count = count_result[0] if count_result else 0
        
        # Then delete
        conn.execute(text("DELETE FROM chunks WHERE path=:p"), {"p": path})
        return count


def delete_file_index(
    engine: Engine, 
    path: str
) -> None:
    """Remove file from file_index."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM file_index WHERE path=:p"), {"p": path})


def fetch_chunk_ids_by_path(
    engine: Engine, 
    path: str
) -> List[int]:
    """Return chunk IDs for a given file path."""
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id FROM chunks WHERE path=:p"), 
            {"p": path}
        )
        return [int(r[0]) for r in rows]


def fetch_all_chunks(
    engine: Engine,
    since: str | None = None,
    until: str | None = None
) -> List[Tuple[int, str, str, str]]:
    """Return list of (id, path, chunk_name, content) with optional time filtering.
    
    Args:
        since: ISO datetime string - only return chunks created after this date
        until: ISO datetime string - only return chunks created before this date
    """
    query = "SELECT id, path, chunk_name, content FROM chunks WHERE 1=1"
    params: Dict[str, Any] = {}
    
    if since:
        query += " AND created_at >= :since"
        params["since"] = since
    if until:
        query += " AND created_at <= :until"
        params["until"] = until
    
    with engine.begin() as conn:
        rows = conn.execute(text(query), params)
        return [(r[0], r[1], r[2], r[3]) for r in rows]


def count_chunks(
    engine: Engine, 
    like: str | None = None,
    since: str | None = None,
    until: str | None = None
) -> int:
    """Return total number of chunks with optional filters."""
    query = "SELECT COUNT(1) FROM chunks WHERE 1=1"
    params: Dict[str, Any] = {}
    
    if like:
        query += " AND (path LIKE :like OR chunk_name LIKE :like OR content LIKE :like)"
        params["like"] = like
    if since:
        query += " AND created_at >= :since"
        params["since"] = since
    if until:
        query += " AND created_at <= :until"
        params["until"] = until
    
    with engine.begin() as conn:
        row = conn.execute(text(query), params).fetchone()
        return int(row[0]) if row else 0


def fetch_chunks(
    engine: Engine, 
    limit: int = 50, 
    offset: int = 0, 
    like: str | None = None,
    since: str | None = None,
    until: str | None = None
) -> List[Tuple[int, str, str, str]]:
    """Paginated fetch with optional time filtering."""
    base = "SELECT id, path, chunk_name, content FROM chunks WHERE 1=1"
    params: Dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
    
    if like:
        base += " AND (path LIKE :like OR chunk_name LIKE :like OR content LIKE :like)"
        params["like"] = like
    if since:
        base += " AND created_at >= :since"
        params["since"] = since
    if until:
        base += " AND created_at <= :until"
        params["until"] = until
    
    base += " ORDER BY id DESC LIMIT :limit OFFSET :offset"
    
    with engine.begin() as conn:
        rows = conn.execute(text(base), params)
        return [(r[0], r[1], r[2], r[3]) for r in rows]


def delete_chunks_by_ids(
    engine: Engine, 
    ids: List[int]
) -> int:
    """Delete chunks by id list. Returns number of rows deleted."""
    if not ids:
        return 0
    safe_ids = []
    for i in ids:
        try:
            safe_ids.append(int(i))
        except Exception:
            continue
    if not safe_ids:
        return 0
    placeholders = ", ".join(str(i) for i in sorted(set(safe_ids)))
    with engine.begin() as conn:
        res = conn.execute(text(f"DELETE FROM chunks WHERE id IN ({placeholders})"))
        try:
            return int(res.rowcount)
        except Exception:
            return 0


def fetch_chunk_by_id(
    engine: Engine, 
    chunk_id: int
) -> Tuple[int, str, str, str] | None:
    """Fetch a single chunk by id."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, path, chunk_name, content FROM chunks WHERE id=:id"), 
            {"id": int(chunk_id)}
        ).fetchone()
        return (row[0], row[1], row[2], row[3]) if row else None


# ============================================================================
# Information Decay System
# ============================================================================

def apply_information_decay(
    engine: Engine,
    max_age_days: int = 365,
    dry_run: bool = False
) -> Dict[str, int]:
    """Automatically remove data older than specified age.
    
    Args:
        engine: Database engine
        max_age_days: Maximum age in days (default: 365 = 1 year)
        dry_run: If True, only count what would be deleted without actually deleting
    
    Returns:
        Dict with counts of deleted items by category
    """
    cutoff_date = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
    results = {"chunks": 0, "file_index": 0, "job_runs": 0, "feedback": 0}
    
    with engine.begin() as conn:
        # Count chunks to delete
        count_result = conn.execute(
            text("SELECT COUNT(*) FROM chunks WHERE created_at < :cutoff"),
            {"cutoff": cutoff_date}
        ).fetchone()
        results["chunks"] = count_result[0] if count_result else 0
        
        # Count file_index entries to delete
        count_result = conn.execute(
            text("SELECT COUNT(*) FROM file_index WHERE updated_at < :cutoff"),
            {"cutoff": cutoff_date}
        ).fetchone()
        results["file_index"] = count_result[0] if count_result else 0
        
        # Count old job_runs
        count_result = conn.execute(
            text("SELECT COUNT(*) FROM job_runs WHERE started_at < :cutoff"),
            {"cutoff": cutoff_date}
        ).fetchone()
        results["job_runs"] = count_result[0] if count_result else 0
        
        # Count old feedback
        count_result = conn.execute(
            text("SELECT COUNT(*) FROM feedback WHERE created_at < :cutoff"),
            {"cutoff": cutoff_date}
        ).fetchone()
        results["feedback"] = count_result[0] if count_result else 0
        
        if not dry_run:
            # Actually delete
            conn.execute(text("DELETE FROM chunks WHERE created_at < :cutoff"), {"cutoff": cutoff_date})
            conn.execute(text("DELETE FROM file_index WHERE updated_at < :cutoff"), {"cutoff": cutoff_date})
            conn.execute(text("DELETE FROM job_runs WHERE started_at < :cutoff"), {"cutoff": cutoff_date})
            conn.execute(text("DELETE FROM feedback WHERE created_at < :cutoff"), {"cutoff": cutoff_date})
            
            # Log the decay action
            conn.execute(
                text("INSERT INTO decay_log(action, details, executed_at) VALUES(:action, :details, :now)"),
                {
                    "action": "decay_cleanup",
                    "details": f"Deleted chunks:{results['chunks']}, files:{results['file_index']}, jobs:{results['job_runs']}, feedback:{results['feedback']}",
                    "now": datetime.utcnow().isoformat()
                }
            )
    
    return results


def get_decay_log(
    engine: Engine,
    limit: int = 10
) -> List[Tuple[str, str, str]]:
    """Get recent decay operations."""
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT action, details, executed_at FROM decay_log ORDER BY executed_at DESC LIMIT :limit"),
            {"limit": limit}
        )
        return [(r[0], r[1], r[2]) for r in rows]


# ============================================================================
# Time Range Utilities
# ============================================================================

def get_time_range(
    range_str: str
) -> Tuple[str | None, str | None]:
    """Convert a time range string to ISO datetime strings.
    
    Args:
        range_str: One of:
            - "1d" or "1day" - last 1 day
            - "1w" or "1week" - last 1 week
            - "1m" or "1month" - last 1 month
            - "3m" or "3months" - last 3 months
            - "6m" or "6months" - last 6 months
            - "1y" or "1year" - last 1 year
            - "all" - no time limit
    
    Returns:
        Tuple of (since, until) - until is None for all ranges (means up to now)
    """
    now = datetime.utcnow()
    until = None  # Up to now
    
    range_str = range_str.lower().strip()
    
    if range_str in ("all", "everything", "*"):
        return (None, None)
    
    # Parse number and unit
    import re
    match = re.match(r'(\d+)\s*([a-z]+)', range_str)
    if not match:
        return (None, None)
    
    num = int(match.group(1))
    unit = match.group(2)
    
    if unit in ('d', 'day', 'days'):
        since = (now - timedelta(days=num)).isoformat()
    elif unit in ('w', 'week', 'weeks'):
        since = (now - timedelta(weeks=num)).isoformat()
    elif unit in ('m', 'month', 'months'):
        since = (now - timedelta(days=num * 30)).isoformat()  # Approximate
    elif unit in ('y', 'year', 'years'):
        since = (now - timedelta(days=num * 365)).isoformat()
    else:
        return (None, None)
    
    return (since, until)


# ============================================================================
# Meta Node Operations
# ============================================================================

def upsert_meta_node(
    engine: Engine, 
    node_type: str, 
    name: str
) -> int:
    """Upsert a meta node and return its ID."""
    norm = f"{node_type}::{name.strip().lower()}"
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM meta_nodes WHERE normalized_name=:n"), 
            {"n": norm}
        ).fetchone()
        if row:
            return int(row[0])
        
        res = conn.execute(
            text("INSERT INTO meta_nodes(type, name, normalized_name) VALUES(:t, :n, :nn)"),
            {"t": node_type, "n": name.strip(), "nn": norm}
        )
        if res.lastrowid:
            return int(res.lastrowid)
        row = conn.execute(
            text("SELECT id FROM meta_nodes WHERE normalized_name=:n"), 
            {"n": norm}
        ).fetchone()
        return int(row[0])


def link_chunk_to_meta(
    engine: Engine, 
    chunk_id: int, 
    meta_id: int, 
    weight: float = 1.0
) -> None:
    """Link a chunk to a meta node."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO chunk_meta_edges(chunk_id, meta_node_id, weight)
                VALUES(:cid, :mid, :w)
                ON CONFLICT(chunk_id, meta_node_id) DO UPDATE SET weight=excluded.weight
                """
            ),
            {"cid": chunk_id, "mid": meta_id, "w": weight}
        )


def fetch_meta_nodes_for_chunks(
    engine: Engine, 
    chunk_ids: List[int]
) -> Dict[int, List[Tuple[str, str]]]:
    """Return {chunk_id: [(type, name), ...]}."""
    if not chunk_ids:
        return {}
    ids_str = ",".join(str(i) for i in chunk_ids)
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT e.chunk_id, m.type, m.name
                FROM chunk_meta_edges e
                JOIN meta_nodes m ON e.meta_node_id = m.id
                WHERE e.chunk_id IN ({ids_str})
                """
            )
        )
        res = {}
        for cid, t, n in rows:
            res.setdefault(cid, []).append((t, n))
        return res


# ============================================================================
# pgvector Operations (Postgres only)
# ============================================================================

def persist_chunk_vectors(
    engine: Engine, 
    rows: List[Tuple[int, List[float]]]
) -> int:
    """Persist SVD vectors to Postgres pgvector table."""
    if not rows:
        return 0
    if engine.dialect.name != "postgresql":
        return 0
    
    with engine.begin() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS chunk_vectors (
                        chunk_id BIGINT PRIMARY KEY,
                        svd vector,
                        updated_at TIMESTAMP
                    )
                    """
                )
            )
        except Exception:
            pass
    
    from math import ceil
    BATCH = 500
    updated = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i : i + BATCH]
        payload = []
        ts = datetime.utcnow()
        for cid, vec in batch:
            literal = "[" + ",".join(f"{float(x):.6f}" for x in (vec or [])) + "]"
            payload.append({"id": int(cid), "svd": literal, "ts": ts})
        with engine.begin() as conn:
            try:
                conn.execute(
                    text(
                        """
                        INSERT INTO chunk_vectors(chunk_id, svd, updated_at)
                        VALUES(:id, :svd::vector, :ts)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            svd = excluded.svd,
                            updated_at = excluded.updated_at
                        """
                    ),
                    payload,
                )
                updated += len(batch)
            except Exception:
                pass
    return updated


def fetch_chunk_vectors(
    engine: Engine, 
    ids: List[int]
) -> Dict[int, List[float]]:
    """Fetch SVD vectors from pgvector table."""
    result: Dict[int, List[float]] = {}
    if not ids:
        return result
    if engine.dialect.name != "postgresql":
        return result
    
    safe_ids = []
    for x in ids:
        try:
            safe_ids.append(int(x))
        except Exception:
            continue
    if not safe_ids:
        return result
    
    placeholders = ", ".join(str(i) for i in sorted(set(safe_ids)))
    with engine.begin() as conn:
        try:
            rows = conn.execute(
                text(f"SELECT chunk_id, svd::text FROM chunk_vectors WHERE chunk_id IN ({placeholders})")
            )
            for cid, vec_txt in rows:
                try:
                    s = str(vec_txt or "").strip()
                    if s.startswith("[") and s.endswith("]"):
                        parts = s[1:-1].split(",")
                        vec = [float(p) for p in parts if p.strip()]
                        result[int(cid)] = vec
                except Exception:
                    continue
        except Exception:
            return result
    return result


# ============================================================================
# Feedback Operations
# ============================================================================

def record_feedback(
    query: str, 
    chunk_id: int, 
    score: float,
    context_db: str = DEFAULT_CONTEXT_DB
) -> None:
    """Record user feedback."""
    engine = get_engine(context_db)
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO feedback(query, chunk_id, score, created_at) VALUES(:q, :cid, :s, :now)"),
            {"q": query.strip(), "cid": chunk_id, "s": score, "now": now}
        )


def fetch_feedback_for_query(
    query: str,
    context_db: str = DEFAULT_CONTEXT_DB
) -> List[Tuple[int, float]]:
    """Return list of (chunk_id, score) for a query."""
    engine = get_engine(context_db)
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT chunk_id, score FROM feedback WHERE query=:q"),
            {"q": query.strip()}
        ).fetchall()
        return [(r[0], r[1]) for r in rows]


# ============================================================================
# Search History
# ============================================================================

def record_search_history(
    query: str, 
    params: dict, 
    count: int,
    context_db: str = DEFAULT_CONTEXT_DB
) -> int:
    """Log a search event."""
    import json
    engine = get_engine(context_db)
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        res = conn.execute(
            text("INSERT INTO search_history(query, params_json, result_count, created_at) VALUES(:q, :p, :c, :now)"),
            {"q": query, "p": json.dumps(params), "c": count, "now": now}
        )
        if res.lastrowid:
            return int(res.lastrowid)
        return 0


def fetch_search_history(
    limit: int = 100,
    context_db: str = DEFAULT_CONTEXT_DB
) -> List[Dict]:
    """Return recent searches."""
    import json
    engine = get_engine(context_db)
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, query, params_json, result_count, created_at FROM search_history ORDER BY id DESC LIMIT :limit"),
            {"limit": limit}
        )
        out = []
        for r in rows:
            try:
                params = json.loads(r[2])
            except Exception:
                params = {}
            out.append({
                "id": r[0], "query": r[1], "params": params, "count": r[3], "created_at": r[4]
            })
        return out


# ============================================================================
# Factory Reset
# ============================================================================

def factory_reset_db(context_db: str = DEFAULT_CONTEXT_DB) -> None:
    """Dangerous: wipe all tables and reinitialize schema."""
    url = get_database_url(context_db)
    try:
        if url.startswith("sqlite///") or url.startswith("sqlite:///"):
            p = (STATE_DIR / f'context_{context_db}.db')
            if p.exists():
                p.unlink()
            return
        
        engine = create_engine(url, future=True, pool_pre_ping=True)
        with engine.begin() as conn:
            for tbl in ("apscheduler_jobs", "chunks", "file_index", "jobs", "job_runs", "decay_log"):
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
                except Exception:
                    pass
        init_schema(engine)
    except Exception:
        return
