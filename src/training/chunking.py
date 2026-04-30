import os
import time
import hashlib
import sqlite3
import argparse
import logging
from multiprocessing import Pool, Queue, Process
from transformers import AutoTokenizer
import transformers

transformers.logging.set_verbosity_error()

try:
    from db import get_writer_conn, get_reader_conn
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_writer_conn, get_reader_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SENTINEL = None


def chunk_text(text, tokenizer, chunk_size=500, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i: i + chunk_size]
        if len(chunk_tokens) < 10:
            continue

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append({
            'text': chunk_text,
            'token_count': len(chunk_tokens),
            'char_offset': 0,
            'index': len(chunks)
        })

        if i + chunk_size >= len(tokens):
            break

    return chunks


import re


def clean_markdown_global(text):
    if not text: return ""
    text = re.sub(r'\{#.*?\}', '', text)  # Remove Pandoc IDs
    text = re.sub(r'\[.*?\]\{.*?\}', '', text)
    text = re.sub(r'(?i)^#\s+(Table of Contents|Index|Bibliography|References).*?(\n#|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_quality_chunk(text):
    if len(re.findall(r'\[\d+\]|\(\w+,\s\d{4}\)', text)) > 5:
        return False
    alphanumeric_ratio = len(re.findall(r'\w', text)) / (len(text) + 1)
    if alphanumeric_ratio < 0.4:
        return False
    if len(text.split()) < 20:
        return False
    return True


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def chunking_worker(item):
    doc_id, text, model_path = item

    try:
        text = clean_markdown_global(text)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        raw_chunks = chunk_text(text, tokenizer)
        results = []
        for c in raw_chunks:
            if not is_quality_chunk(c['text']):
                continue

            md5 = hashlib.md5(c['text'].encode('utf-8')).hexdigest()
            results.append({
                'doc_id': doc_id,
                'chunk_index': len(results),
                'chunk_text': c['text'],
                'token_count': c['token_count'],
                'byte_count': len(c['text'].encode('utf-8')),
                'char_offset': 0,
                'text_hash_md5': md5
            })

        return {'doc_id': doc_id, 'chunks': results, 'status': 'success'}
    except Exception as e:
        return {'doc_id': doc_id, 'status': 'failed', 'error': str(e)}


def db_writer_worker(queue, db_path, batch_size=1000):
    conn = get_writer_conn(db_path, phase='ingest')

    batch = []
    last_flush = time.monotonic()
    total_chunks = 0

    logger.info("Chunk DB Writer started")

    while True:
        try:
            item = queue.get(timeout=1.0)
        except Exception:
            if batch and (time.monotonic() - last_flush) > 5.0:
                _commit_chunks(conn, batch)
                batch.clear()
                last_flush = time.monotonic()
            continue

        if item is SENTINEL:
            break

        if item['status'] == 'success':
            batch.extend(item['chunks'])
            total_chunks += len(item['chunks'])

        if len(batch) >= batch_size:
            _commit_chunks(conn, batch)
            batch.clear()
            last_flush = time.monotonic()

    if batch:
        _commit_chunks(conn, batch)

    logger.info(f"Chunk DB Writer finished. Total chunks: {total_chunks}")
    conn.close()


def _commit_chunks(conn, chunks):
    conn.execute("BEGIN")
    for c in chunks:
        conn.execute(
            """INSERT OR REPLACE INTO chunks 
               (doc_id, chunk_index, chunk_text, token_count, byte_count, char_offset, text_hash_md5)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (c['doc_id'], c['chunk_index'], c['chunk_text'], c['token_count'],
             c['byte_count'], c['char_offset'], c['text_hash_md5'])
        )
    conn.execute("COMMIT")


def main():
    parser = argparse.ArgumentParser(description="Parallel Chunking Pipeline")
    parser.add_argument("--db", default="nlp.db", help="Path to SQLite database")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Tokenizer model name")
    parser.add_argument("--workers", type=int, default=64, help="Parallel worker count")
    args = parser.parse_args()

    # Get documents that haven't been chunked yet
    reader_conn = get_reader_conn(args.db)
    pending = reader_conn.execute(
        """SELECT m.doc_id, m.markdown_text
           FROM markdown_content m
                    LEFT JOIN chunks c ON m.doc_id = c.doc_id
           WHERE c.doc_id IS NULL"""
    ).fetchall()
    reader_conn.close()

    if not pending:
        logger.info("No documents found for chunking.")
        return

    logger.info(f"Chunking {len(pending)} documents using {args.workers} workers...")

    result_queue = Queue(maxsize=1000)
    writer_proc = Process(target=db_writer_worker, args=(result_queue, args.db))
    writer_proc.start()

    try:
        AutoTokenizer.from_pretrained(args.model, local_files_only=True)

        items = [(row[0], row[1], args.model) for row in pending]

        from tqdm import tqdm
        with Pool(processes=args.workers) as pool:
            for result in tqdm(pool.imap_unordered(chunking_worker, items), total=len(items),
                               desc="Chunking documents"):
                result_queue.put(result)

    finally:
        result_queue.put(SENTINEL)
        writer_proc.join()

    logger.info("Chunking complete.")


if __name__ == "__main__":
    main()
