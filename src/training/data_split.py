import argparse
import random
import sqlite3
import sys
import time
import os
from collections import defaultdict

try:
    from db import get_writer_conn, get_reader_conn
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_writer_conn, get_reader_conn


def create_chunk_split(db_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, clear=False):
    t0 = time.monotonic()
    conn = get_writer_conn(db_path)

    if clear:
        print("Clearing existing chunk split data...")
        conn.execute("DELETE FROM chunk_sample_tags")
        conn.execute("DELETE FROM chunk_train_samples")
        conn.commit()

    print("Collecting tagged chunks...")
    query = """
            SELECT c.chunk_id, dt.tag_id
            FROM chunks c
                     JOIN cleaned_document_tags dt ON c.doc_id = dt.doc_id \
            """
    rows = conn.execute(query).fetchall()

    chunk_to_tags = defaultdict(list)
    for cid, tid in rows:
        chunk_to_tags[cid].append(tid)

    all_chunk_ids = list(chunk_to_tags.keys())
    print(f"Total tagged chunks found: {len(all_chunk_ids):,}")

    print(f"Performing stratified split (seed={seed})...")
    random.seed(seed)

    cat_groups = defaultdict(list)
    for cid in all_chunk_ids:
        primary_tag = chunk_to_tags[cid][0]
        cat_groups[primary_tag].append(cid)

    train_ids, val_ids, test_ids = [], [], []
    for cat, cids in cat_groups.items():
        random.shuffle(cids)
        n = len(cids)
        tr = int(train_ratio * n)
        vl = int((train_ratio + val_ratio) * n)
        train_ids.extend(cids[:tr])
        val_ids.extend(cids[tr:vl])
        test_ids.extend(cids[vl:])

    print("Inserting split assignments...")
    conn.execute("BEGIN")
    conn.executemany("INSERT OR IGNORE INTO chunk_train_samples (chunk_id, split) VALUES (?, 'train')",
                     [(x,) for x in train_ids])
    conn.executemany("INSERT OR IGNORE INTO chunk_train_samples (chunk_id, split) VALUES (?, 'val')",
                     [(x,) for x in val_ids])
    conn.executemany("INSERT OR IGNORE INTO chunk_train_samples (chunk_id, split) VALUES (?, 'test')",
                     [(x,) for x in test_ids])
    conn.execute("COMMIT")

    print("Populating chunk_sample_tags...")
    conn.execute("BEGIN")
    conn.execute("""
                 INSERT OR IGNORE INTO chunk_sample_tags (sample_id, tag_id, confidence)
                 SELECT s.sample_id, dt.tag_id, dt.confidence
                 FROM chunk_train_samples s
                          JOIN chunks c ON s.chunk_id = c.chunk_id
                          JOIN cleaned_document_tags dt ON c.doc_id = dt.doc_id
                 """)
    conn.execute("COMMIT")

    total = conn.execute("SELECT COUNT(*) FROM chunk_train_samples").fetchone()[0]
    conn.close()
    print(f"Split creation complete in {time.monotonic() - t0:.1f}s. Total samples: {total:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="nlp.db")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    create_chunk_split(args.db, clear=args.clear)
