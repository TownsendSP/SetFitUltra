"""EPUB to Markdown extraction pipeline

Implements Stage 2a: Parallel EPUB extraction with SSD staging.

Usage:
    python src/extraction_epub.py --db nlp.db --staging-dir staging/ --workers 96
"""

import os
import time
import shutil
import sqlite3
import argparse
import subprocess
import threading
import logging
import zipfile
import re
from pathlib import Path
from multiprocessing import Pool, Queue, Process

try:
    from db import get_writer_conn, get_reader_conn
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_writer_conn, get_reader_conn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Sentinel for signaling the end of the queue
SENTINEL = None


def extract_epub_worker(item):
    doc_id, staged_path, original_path = item
    start_time = time.monotonic()

    try:
        text_blocks = []
        # Read directly into memory.
        with zipfile.ZipFile(staged_path, 'r') as z:
            for filename in z.namelist():
                # Only parse content files
                if filename.endswith(('.html', '.htm', '.xhtml', '.xml')):
                    try:
                        content = z.read(filename).decode('utf-8', errors='ignore')
                        text = re.sub(r'<[^>]+>', ' ', content)
                        text = re.sub(r'\s+', ' ', text).strip()
                        if text:
                            text_blocks.append(text)
                    except Exception:
                        pass

        text = "\n\n".join(text_blocks)

        byte_count = len(text.encode('utf-8'))
        char_count = len(text)
        word_count = len(text.split())
        line_count = text.count('\n') + 1

        quality_score = 1.0 if char_count > 1000 else 0.5
        if char_count < 100:
            quality_score = 0.1

        return {
            'doc_id': doc_id,
            'text': text,
            'byte_count': byte_count,
            'char_count': char_count,
            'word_count': word_count,
            'line_count': line_count,
            'quality_score': quality_score,
            'status': 'success',
            'method': 'zip_regex',
            'time_ms': int((time.monotonic() - start_time) * 1000),
            'error': None
        }
    except zipfile.BadZipFile:
        return {'doc_id': doc_id, 'status': 'failed', 'method': 'zip_regex',
                'time_ms': int((time.monotonic() - start_time) * 1000), 'error': 'BadZipFile'}
    except Exception as e:
        return {'doc_id': doc_id, 'status': 'failed', 'method': 'zip_regex',
                'time_ms': int((time.monotonic() - start_time) * 1000), 'error': str(e)}


def db_writer_worker(queue, db_path, batch_size=500):
    conn = get_writer_conn(db_path, phase='ingest')

    # Ensure the target table exists (redundant but safe)
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS markdown_content
                 (
                     doc_id        INTEGER PRIMARY KEY REFERENCES documents (doc_id) ON DELETE CASCADE,
                     markdown_text TEXT    NOT NULL,
                     byte_count    INTEGER NOT NULL,
                     char_count    INTEGER NOT NULL,
                     word_count    INTEGER NOT NULL,
                     line_count    INTEGER NOT NULL,
                     quality_score REAL,
                     quality_notes TEXT
                 )
                 """)

    batch = []
    last_flush = time.monotonic()
    total_success = 0
    total_failed = 0

    logger.info("DB Writer process started")

    while True:
        try:
            item = queue.get(timeout=1.0)
        except Exception:  # Empty queue
            if batch and (time.monotonic() - last_flush) > 10.0:
                _commit_batch(conn, batch)
                batch.clear()
                last_flush = time.monotonic()
            continue

        if item is SENTINEL:
            break

        batch.append(item)
        if item['status'] == 'success':
            total_success += 1
        else:
            total_failed += 1

        if len(batch) >= batch_size:
            _commit_batch(conn, batch)
            batch.clear()
            last_flush = time.monotonic()

    if batch:
        _commit_batch(conn, batch)

    logger.info(f"DB Writer finished. Commited: {total_success} success, {total_failed} failed")
    conn.close()


def _commit_batch(conn, batch):
    conn.execute("BEGIN")
    for item in batch:
        doc_id = item['doc_id']
        if item['status'] == 'success':
            conn.execute(
                """INSERT OR REPLACE INTO markdown_content
                   (doc_id, markdown_text, byte_count, char_count, word_count, line_count, quality_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, item['text'], item['byte_count'], item['char_count'],
                 item['word_count'], item['line_count'], item['quality_score'])
            )
            conn.execute(
                """UPDATE documents
                   SET conversion_status  = 'success',
                       extraction_method  = ?,
                       extraction_time_ms = ?,
                       error_message      = NULL,
                       updated_at         = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                   WHERE doc_id = ?""",
                (item['method'], item['time_ms'], doc_id)
            )
        else:
            conn.execute(
                """UPDATE documents
                   SET conversion_status  = 'failed',
                       extraction_method  = ?,
                       extraction_time_ms = ?,
                       error_message      = ?,
                       updated_at         = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                   WHERE doc_id = ?""",
                (item['method'], item['time_ms'], item['error'], doc_id)
            )
    conn.execute("COMMIT")


def main():
    parser = argparse.ArgumentParser(description="Parallel EPUB Extraction Pipeline")
    parser.add_argument("--db", default="nlp.db", help="Path to SQLite database")
    parser.add_argument("--staging-dir", default="staging/", help="Path to staging directory on SSD")
    parser.add_argument("--batch-size", type=int, default=5000, help="Number of files to stage per loop")
    parser.add_argument("--workers", type=int, default=96, help="Parallel worker count (Threadripper optimize)")
    args = parser.parse_args()

    staging_path = Path(args.staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)

    reader_conn = get_reader_conn(args.db)
    pending = reader_conn.execute(
        "SELECT doc_id, fullpath FROM documents WHERE extension = 'epub' AND conversion_status = 'pending'"
    ).fetchall()
    reader_conn.close()

    total = len(pending)
    if total == 0:
        logger.info("No pending EPUBs found. Exiting.")
        return

    logger.info(f"Starting extraction for {total} EPUBs using {args.workers} workers.")

    result_queue = Queue(maxsize=args.batch_size)
    writer_proc = Process(target=db_writer_worker, args=(result_queue, args.db))
    writer_proc.start()

    try:
        for i in range(0, total, args.batch_size):
            batch_docs = pending[i: i + args.batch_size]
            current_staged = []

            logger.info(f"Staging batch {i // args.batch_size + 1} ({len(batch_docs)} files)...")
            stage_start = time.monotonic()
            for doc_id, fullpath in batch_docs:
                staged_file = staging_path / f"{doc_id}.epub"
                try:
                    shutil.copy2(fullpath, staged_file)
                    current_staged.append((doc_id, str(staged_file), fullpath))
                except Exception as e:
                    logger.error(f"Failed to stage {doc_id}: {e}")
                    result_queue.put({
                        'doc_id': doc_id, 'status': 'failed', 'method': 'staging',
                        'time_ms': 0, 'error': f"Staging failed: {e}"
                    })

            stage_dur = time.monotonic() - stage_start
            logger.info(f"Staging complete in {stage_dur:.1f}s ({len(current_staged)} files).")

            logger.info(f"Extracting batch {i // args.batch_size + 1}...")
            extract_start = time.monotonic()
            with Pool(processes=args.workers) as pool:
                for result in pool.imap_unordered(extract_epub_worker, current_staged):
                    result_queue.put(result)

            extract_dur = time.monotonic() - extract_start
            logger.info(f"Extraction complete in {extract_dur:.1f}s.")

            for _, staged_file, _ in current_staged:
                try:
                    os.remove(staged_file)
                except OSError:
                    pass

            logger.info(
                f"Progress: {min(i + args.batch_size, total)}/{total} ({(i + len(batch_docs)) / total * 100:.1f}%)")

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Shutting down...")
    finally:
        result_queue.put(SENTINEL)
        writer_proc.join()

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
