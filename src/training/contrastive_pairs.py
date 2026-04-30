import sqlite3
import random
import logging
from typing import Dict, List, Tuple
import os
import sys
import time

try:
    from db import get_writer_conn, get_reader_conn
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_writer_conn, get_reader_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_samples_by_tag(db_path: str) -> Dict[int, List[int]]:
    t0 = time.time()
    conn = get_reader_conn(db_path)
    query = """
            SELECT c.chunk_id, dt.tag_id
            FROM chunk_train_samples s
                     JOIN chunks c ON s.chunk_id = c.chunk_id
                     JOIN cleaned_document_tags dt ON c.doc_id = dt.doc_id
            WHERE s.split = 'train' \
            """
    rows = conn.execute(query).fetchall()
    conn.close()

    tag_samples = {}
    for chunk_id, tag_id in rows:
        if tag_id not in tag_samples:
            tag_samples[tag_id] = []
        tag_samples[tag_id].append(chunk_id)

    logger.info(f"Loaded {len(rows):,} tagged TRAIN IDs in {time.time() - t0:.1f}s.")
    return tag_samples


from tqdm import tqdm


def generate_contrastive_pairs(
        db_path: str,
        num_positive_pairs: int = 50000,
        num_negative_pairs: int = 50000,
        seed: int = 42
):
    t0 = time.time()
    random.seed(seed)
    tag_samples = get_samples_by_tag(db_path)

    if not tag_samples:
        logger.error("No high-quality tagged samples found in TRAIN split.")
        return

    valid_tags = [t for t, s in tag_samples.items() if len(s) >= 2]
    all_tags = list(tag_samples.keys())

    # 1. Generate Positive Pairs
    logger.info(f"Generating {num_positive_pairs:,} positive pairs...")
    pos_pairs = []
    random_tags = random.choices(valid_tags, k=num_positive_pairs)
    for tag in tqdm(random_tags, desc="Positive Pairs"):
        sA, sB = random.sample(tag_samples[tag], 2)
        pos_pairs.append((sA, sB, 1.0))

    # 2. Generate Negative Pairs
    logger.info(f"Generating {num_negative_pairs:,} negative pairs...")
    neg_pairs = []
    for _ in tqdm(range(num_negative_pairs), desc="Negative Pairs"):
        tA, tB = random.sample(all_tags, 2)
        sA = random.choice(tag_samples[tA])
        sB = random.choice(tag_samples[tB])
        neg_pairs.append((sA, sB, 0.0))

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)

    logger.info(f"Writing {len(all_pairs):,} pairs to DB...")
    writer = get_writer_conn(db_path, phase='steady')
    writer.execute("BEGIN")
    writer.execute("DELETE FROM contrastive_pairs")
    writer.commit()

    batch_size = 50000
    for i in tqdm(range(0, len(all_pairs), batch_size), desc="DB Insert"):
        batch = all_pairs[i:i + batch_size]
        writer.executemany(
            "INSERT OR IGNORE INTO contrastive_pairs (sample_a_id, sample_b_id, similarity) VALUES (?, ?, ?)",
            batch
        )
        writer.commit()

    final_count = writer.execute("SELECT COUNT(*) FROM contrastive_pairs").fetchone()[0]
    writer.close()
    logger.info(f"Wrote {final_count:,} unique pairs in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="nlp.db")
    parser.add_argument("--pos", type=int, default=100000)
    parser.add_argument("--neg", type=int, default=100000)
    args = parser.parse_args()

    generate_contrastive_pairs(args.db, args.pos, args.neg)
