"""Database initialization and connection management.

Handles schema creation, PRAGMA configuration, and provides connection
factories for the writer thread and read-only workers.

Usage:
    from db import init_db, get_writer_conn, get_reader_conn

    # First time: create DB and schema
    init_db("/mnt/ssd2/NLP_Project/nlp.db")

    # Writer (single thread only):
    conn = get_writer_conn("/mnt/ssd2/NLP_Project/nlp.db")

    # Readers (any number of processes/threads):
    conn = get_reader_conn("/mnt/ssd2/NLP_Project/nlp.db")
"""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

# PRAGMA profiles
_PRAGMAS_COMMON = [
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
    ("temp_store", "MEMORY"),
    ("busy_timeout", "60000"),
]

_PRAGMAS_WRITER_INGEST = [
    # 8 GB page cache: -(8 * 1024 * 1024) = -8388608 KiB
    ("cache_size", "-8388608"),
    # Disable mmap during writes to avoid TLB shootdowns on 128-thread CPU
    ("mmap_size", "0"),
    # Let WAL grow to ~400 MB before auto-checkpoint (50000 * 8KB pages)
    ("wal_autocheckpoint", "50000"),
]

_PRAGMAS_READER = [
    # 4 GB page cache for readers
    ("cache_size", "-4194304"),
    # 32 GB mmap window for read-heavy access
    ("mmap_size", "34359738368"),
]

# Schema DDL

_SCHEMA_DDL = """
-- =================================================================
-- STAGE 1: File Discovery
-- =================================================================
CREATE TABLE IF NOT EXISTS documents (
    doc_id              INTEGER PRIMARY KEY,
    fullpath            TEXT    NOT NULL UNIQUE,
    parent_dir          TEXT    NOT NULL,
    filename            TEXT    NOT NULL,
    extension           TEXT    NOT NULL,
    filesize_bytes      INTEGER NOT NULL,
    file_hash_sha256    TEXT,
    conversion_status   TEXT    NOT NULL DEFAULT 'pending'
                        CHECK(conversion_status IN (
                            'pending', 'queued', 'in_progress', 'success',
                            'failed', 'skipped', 'needs_ocr'
                        )),
    extraction_method   TEXT,
    extraction_time_ms  INTEGER,
    error_message       TEXT,
    source_location     TEXT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')),
    updated_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now'))
);

-- =================================================================
-- Schema metadata (for migrations)
-- =================================================================
CREATE TABLE IF NOT EXISTS schema_info (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
"""

# Post-discovery indices -- created after bulk insert for speed
_INDICES_STAGE1 = """
CREATE INDEX IF NOT EXISTS idx_doc_extension   ON documents(extension);
CREATE INDEX IF NOT EXISTS idx_doc_conversion  ON documents(conversion_status);
CREATE INDEX IF NOT EXISTS idx_doc_hash        ON documents(file_hash_sha256);
CREATE INDEX IF NOT EXISTS idx_doc_parent      ON documents(parent_dir);
CREATE INDEX IF NOT EXISTS idx_doc_source      ON documents(source_location);
CREATE INDEX IF NOT EXISTS idx_doc_filesize    ON documents(filesize_bytes);
"""


def _apply_pragmas(conn: sqlite3.Connection, pragmas: list[tuple[str, str]]):
    """Apply a list of (pragma_name, value) pairs."""
    for name, value in pragmas:
        conn.execute(f"PRAGMA {name} = {value}")


def init_db(db_path: str) -> sqlite3.Connection:
    """Create the database file and apply schema.

    MUST be called exactly once before any other DB operations.
    Sets page_size (immutable after first table creation), applies
    schema DDL, and records the schema version.

    Returns a writer connection ready for ingestion.
    """
    p = Path(db_path)
    is_new = not p.exists() or p.stat().st_size == 0

    conn = sqlite3.connect(db_path, isolation_level=None)

    if is_new:
        # page_size must be set before any tables are created
        conn.execute("PRAGMA page_size = 8192")

    _apply_pragmas(conn, _PRAGMAS_COMMON)
    _apply_pragmas(conn, _PRAGMAS_WRITER_INGEST)

    # Create tables
    conn.executescript(_SCHEMA_DDL)

    # Record schema version
    conn.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )

    return conn


def get_writer_conn(db_path: str, phase: str = "ingest") -> sqlite3.Connection:
    """Open a writer connection with appropriate PRAGMAs.

    Args:
        db_path: Path to the database file.
        phase: 'ingest' for bulk writes, 'steady' for post-ingestion.
    """
    conn = sqlite3.connect(db_path, isolation_level=None)
    _apply_pragmas(conn, _PRAGMAS_COMMON)

    if phase == "ingest":
        _apply_pragmas(conn, _PRAGMAS_WRITER_INGEST)
    else:
        # Steady-state: enable mmap, lower WAL threshold
        conn.execute("PRAGMA cache_size = -8388608")
        conn.execute("PRAGMA mmap_size = 34359738368")
        conn.execute("PRAGMA wal_autocheckpoint = 1000")

    return conn


def get_reader_conn(db_path: str) -> sqlite3.Connection:
    """Open a read-only connection. Safe to call from any thread/process."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, isolation_level=None)
    _apply_pragmas(conn, _PRAGMAS_COMMON)
    _apply_pragmas(conn, _PRAGMAS_READER)
    return conn


def create_stage1_indices(conn: sqlite3.Connection):
    """Create indices on the documents table after bulk insertion."""
    conn.executescript(_INDICES_STAGE1)
    conn.execute("ANALYZE")


def finalize_ingestion(conn: sqlite3.Connection):
    """Run after all major writes complete. Checkpoints WAL and re-enables mmap."""
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA mmap_size = 34359738368")
    conn.execute("PRAGMA wal_autocheckpoint = 1000")
