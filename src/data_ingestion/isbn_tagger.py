import os
import sys
import json
import logging
import re
from collections import defaultdict
import time
import duckdb

try:
    from db import get_reader_conn, get_writer_conn
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_reader_conn, get_writer_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def normalize_tag(tag_name):
    tag = tag_name.lower().strip()
    tag = re.sub(r'[^a-z0-9\s]', '', tag)
    tag = re.sub(r'\s+', '_', tag)
    return tag

def normalize_isbn(isbn):
    if not isbn:
        return ""
    isbn_str = str(isbn).strip("'\"")
    return re.sub(r'[^0-9X]', '', isbn_str.upper())

def process_local_dump(db_path):
    duckdb_path = "ol_metadata.duckdb"
    if not os.path.exists(duckdb_path):
        logger.error(f"DuckDB index not found: {duckdb_path}. Please run build_ol_index.py first.")
        return

    reader = get_reader_conn(db_path)
    # Get all docs with an ISBN that don't have ISBN tags yet
    query = """
        SELECT doc_id, isbn 
        FROM documents 
        WHERE isbn IS NOT NULL 
          AND doc_id NOT IN (SELECT DISTINCT doc_id FROM document_tags WHERE source IN ('isbn', 'isbn_none'))
    """
    pending = reader.execute(query).fetchall()
    reader.close()
    
    if not pending:
        logger.info("No pending ISBNs to look up.")
        return

    isbn_to_docs = defaultdict(list)
    for doc_id, isbn in pending:
        norm = normalize_isbn(isbn)
        if norm:
            isbn_to_docs[norm].append(doc_id)
            
    unique_isbns = list(isbn_to_docs.keys())
    logger.info(f"Looking up metadata for {len(unique_isbns)} unique ISBNs (covering {len(pending)} docs) using DuckDB index...")

    doc_to_subjects = defaultdict(set)
    found_isbns = set()

    start_time = time.monotonic()
    try:
        conn = duckdb.connect(duckdb_path, read_only=True)
        conn.execute("INSTALL json; LOAD json;")
        
        batch_size = 500
        for i in range(0, len(unique_isbns), batch_size):
            batch = unique_isbns[i:i+batch_size]
            placeholders = ', '.join(['?'] * len(batch))
            query = f"""
                SELECT iw.isbn, ws.subjects 
                FROM isbn_work iw
                JOIN work_subjects ws ON iw.work_key = ws.work_key
                WHERE iw.isbn IN ({placeholders})
            """
            results = conn.execute(query, batch).fetchall()
            
            for matched_isbn, subjects_json in results:
                found_isbns.add(matched_isbn)
                try:
                    subjects = json.loads(subjects_json)
                    for subj in subjects:
                        subj_name = subj if isinstance(subj, str) else subj.get('name', '')
                        if subj_name:
                            norm_name = normalize_tag(subj_name)
                            if norm_name:
                                for doc_id in isbn_to_docs[matched_isbn]:
                                    doc_to_subjects[doc_id].add(norm_name)
                except Exception as e:
                    logger.warning(f"Error parsing subjects for ISBN {matched_isbn}: {e}")
        
        conn.close()
    except Exception as e:
        logger.error(f"DuckDB lookup failed: {e}")
        return

    pass1_time = time.monotonic() - start_time
    logger.info(f"DuckDB lookup complete in {pass1_time:.2f}s. Found subjects for {len(found_isbns)} / {len(unique_isbns)} ISBNs.")

    logger.info("Writing DuckDB tags to database...")
    writer = get_writer_conn(db_path, phase='steady')
    writer.execute("BEGIN")
    
    docs_with_subjects = set(doc_to_subjects.keys())
    all_pending_docs = set([doc_id for doc_id, _ in pending])
    docs_without_subjects = list(all_pending_docs - docs_with_subjects)
    
    tags_inserted = 0
    tag_id_cache = {}
    
    for doc_id, subjects in doc_to_subjects.items():
        for norm_name in subjects:
            if norm_name not in tag_id_cache:
                writer.execute("INSERT OR IGNORE INTO tags (tag_name) VALUES (?)", (norm_name,))
                tag_id_res = writer.execute("SELECT tag_id FROM tags WHERE tag_name = ?", (norm_name,)).fetchone()
                if tag_id_res:
                    tag_id_cache[norm_name] = tag_id_res[0]
            
            if norm_name in tag_id_cache:
                tag_id = tag_id_cache[norm_name]
                writer.execute(
                    "INSERT OR IGNORE INTO document_tags (doc_id, tag_id, source, confidence) VALUES (?, ?, ?, ?)",
                    (doc_id, tag_id, 'isbn', 1.0)
                )
                tags_inserted += 1
    
    writer.execute("COMMIT")

    unmatched_marked = 0
    for doc_id in docs_without_subjects:
        writer.execute(
            "INSERT OR IGNORE INTO document_tags (doc_id, tag_id, source, confidence) VALUES (?, ?, ?, ?)",
            (doc_id, 0, 'isbn_none', 0.0)
        )
        unmatched_marked += 1

    writer.execute("COMMIT")

    writer.close()
    
    logger.info(f"Finished! Inserted {tags_inserted} tag links from DuckDB index.")
    if unmatched_marked > 0:
        logger.info(f"Marked {unmatched_marked} documents as isbn_none (no local subjects found).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="High-speed local OpenLibrary ISBN tagger (DuckDB only)")
    parser.add_argument("--db", default="nlp.db", help="Path to SQLite database")
    args = parser.parse_args()
    
    process_local_dump(args.db)