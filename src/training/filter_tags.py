"""Tag Vocabulary Optimization and Deduplication.

Writes cleaned results to NEW tables (cleaned_tags, cleaned_document_tags)
instead of modifying originals. Point your training pipeline at these.

Steps:
1. Regex junk removal.
2. Synonym merging (Prefix/Suffix stripping + Semantic Clustering).
3. Frequency pruning (< N occurrences).
4. Cross-source dedup (keep highest confidence per doc+tag).
"""

import os
import re
import json
import sqlite3
import argparse
import logging
import time
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

try:
    from db import get_writer_conn, get_reader_conn
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_writer_conn, get_reader_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

JUNK_PATTERNS = [
    r'^[0-9]',                          # Numeric leading (ISBNs)
    r'^[a-z]{2,4}[0-9]{4,}$',           # BISAC codes (fic009000)
    r'^nyt',                             # NYT list refs
    r'produktform',                      # Product format
    r'^large_type_books$',
    r'^large_print',
    r'^hardcover$|^paperback$|^audiobook$|^ebook$',
    r'^lexile',                          # Reading levels
    r'^none$',                           # none tag
    r'fictitious_character',             # Character names
]

STRIP_PATTERNS = [
    (r'^fiction_', ''),
    (r'_fiction$', ''),
    (r'_general$', ''),
]


# Step 1: Load & Regex Filter

def load_tags(db_path):
    conn = get_reader_conn(db_path)
    query = """
        SELECT t.tag_id, t.tag_name, COUNT(dt.doc_id) as doc_count
        FROM tags t
        LEFT JOIN document_tags dt ON t.tag_id = dt.tag_id
        GROUP BY t.tag_id
    """
    rows = conn.execute(query).fetchall()
    conn.close()
    return {r[1]: {'id': r[0], 'count': r[2]} for r in rows}


def apply_regex_filters(tags_dict):
    """Step 1: Identify junk tags by regex patterns and length."""
    survivors = {}
    junk_ids = set()
    for name, data in tags_dict.items():
        is_junk = False
        if len(name) > 60:
            is_junk = True
        else:
            for pattern in JUNK_PATTERNS:
                if re.search(pattern, name, re.IGNORECASE):
                    is_junk = True
                    break
        if not is_junk:
            survivors[name] = data
        else:
            junk_ids.add(data['id'])

    logger.info(f"Step 1: {len(junk_ids):,} junk tags removed. {len(survivors):,} surviving.")
    return survivors, junk_ids



# Step 2a: Prefix / Suffix Merging


def apply_prefix_suffix_merges(tags_dict):
    """Strip mechanical prefixes/suffixes to merge trivial variants."""
    merged = {}  # canonical_name -> {ids: [], count: N, best_id: ID, max_count: N}

    for name, data in tags_dict.items():
        new_name = name
        for pattern, replacement in STRIP_PATTERNS:
            new_name = re.sub(pattern, replacement, new_name)
        if not new_name or len(new_name) < 2:
            new_name = name

        if new_name not in merged:
            merged[new_name] = {
                'ids': [], 'count': 0,
                'best_id': data['id'], 'max_count': data['count'],
            }
        merged[new_name]['ids'].append(data['id'])
        merged[new_name]['count'] += data['count']
        if data['count'] > merged[new_name]['max_count']:
            merged[new_name]['max_count'] = data['count']
            merged[new_name]['best_id'] = data['id']

    logger.info(f"Step 2a: {len(tags_dict):,} -> {len(merged):,} tags via prefix/suffix stripping.")
    return merged



# Step 2b: Semantic Merging


def semantic_merging(merged_2a, model_path, threshold=0.85):
    """Merge near-duplicates using MiniLM cosine similarity."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading encoder from {model_path} on {device}...")
    model = SentenceTransformer(model_path).to(device)

    names = list(merged_2a.keys())
    counts = [merged_2a[n]['count'] for n in names]

    logger.info(f"Encoding {len(names):,} tags...")
    readable = [n.replace('_', ' ') for n in names]
    embeddings = model.encode(readable, convert_to_tensor=True,
                              show_progress_bar=True, batch_size=512)

    logger.info("Computing pairwise cosine similarity...")
    cos_sim = util.cos_sim(embeddings, embeddings)

    merge_map = {}   # loser_name -> winner_name
    visited = set()
    sorted_indices = np.argsort(counts)[::-1]
    proposed_groups = []

    for i in tqdm(range(len(names)), desc="Clustering"):
        idx = int(sorted_indices[i])
        name = names[idx]
        if name in visited:
            continue
        similar_indices = torch.where(cos_sim[idx] > threshold)[0].tolist()
        group = []
        for s_idx in similar_indices:
            s_name = names[s_idx]
            if s_name != name and s_name not in visited:
                merge_map[s_name] = name
                visited.add(s_name)
                group.append(s_name)
        if group:
            proposed_groups.append({'winner': name, 'merged': group})
        visited.add(name)

    logger.info(f"Step 2b: {len(merge_map):,} semantic merges across {len(proposed_groups)} groups.")
    return merge_map, proposed_groups



# Build final mapping & write cleaned tables
def build_id_remap(junk_ids, merged_2a, semantic_map):
    """Build old_tag_id -> new_tag_id mapping. Returns (remap_dict, winner_names).
    
    remap_dict: {old_id: new_id} for ALL surviving original tag IDs (including identity maps).
    winner_names: {new_id: canonical_name}
    """
    remap = {}        # old_tag_id -> canonical_tag_id
    winner_names = {} # canonical_tag_id -> display name

    for name_2a, data in merged_2a.items():
        # Resolve semantic chain
        final_name = semantic_map.get(name_2a, name_2a)
        if final_name not in merged_2a:
            final_name = name_2a
        final_id = merged_2a[final_name]['best_id']
        winner_names[final_id] = final_name

        for orig_id in data['ids']:
            if orig_id not in junk_ids:
                remap[orig_id] = final_id

    logger.info(f"ID remap built: {len(remap):,} old IDs -> {len(winner_names):,} canonical IDs.")
    return remap, winner_names


def write_cleaned_tables(db_path, remap, winner_names, min_count):
    """Write cleaned_tags and cleaned_document_tags to the database."""
    writer = get_writer_conn(db_path, phase='steady')

    logger.info("Creating cleaned tables...")
    writer.execute("DROP TABLE IF EXISTS cleaned_document_tags")
    writer.execute("DROP TABLE IF EXISTS cleaned_tags")
    writer.execute("""
        CREATE TABLE cleaned_tags (
            tag_id      INTEGER PRIMARY KEY,
            tag_name    TEXT    NOT NULL UNIQUE COLLATE NOCASE
        )
    """)
    writer.execute("""
        CREATE TABLE cleaned_document_tags (
            doc_id      INTEGER NOT NULL,
            tag_id      INTEGER NOT NULL,
            confidence  REAL,
            PRIMARY KEY (doc_id, tag_id)
        )
    """)
    writer.commit()

    logger.info("Loading remap into temp table...")
    writer.execute("CREATE TEMP TABLE id_remap (old_id INTEGER PRIMARY KEY, new_id INTEGER)")
    writer.executemany("INSERT INTO id_remap VALUES (?, ?)",
                       [(old, new) for old, new in remap.items()])
    writer.commit()

    logger.info("Populating cleaned_document_tags (remap + dedup)...")
    t0 = time.monotonic()
    writer.execute("BEGIN")
    writer.execute("""
        INSERT INTO cleaned_document_tags (doc_id, tag_id, confidence)
        SELECT doc_id, new_tag_id, MAX(confidence)
        FROM (
            SELECT dt.doc_id, r.new_id AS new_tag_id, dt.confidence
            FROM document_tags dt
            JOIN id_remap r ON dt.tag_id = r.old_id
            WHERE dt.source NOT IN ('isbn_none', 'predicted')
        )
        GROUP BY doc_id, new_tag_id
    """)
    writer.execute("COMMIT")
    total_assignments = writer.execute("SELECT COUNT(*) FROM cleaned_document_tags").fetchone()[0]
    logger.info(f"  {total_assignments:,} assignments written ({time.monotonic() - t0:.1f}s)")

    logger.info(f"Step 3: Pruning tags with < {min_count} occurrences...")
    t0 = time.monotonic()
    writer.execute("BEGIN")
    writer.execute(f"""
        DELETE FROM cleaned_document_tags
        WHERE tag_id IN (
            SELECT tag_id FROM cleaned_document_tags
            GROUP BY tag_id HAVING COUNT(*) < ?
        )
    """, (min_count,))
    writer.execute("COMMIT")
    after_prune = writer.execute("SELECT COUNT(*) FROM cleaned_document_tags").fetchone()[0]
    logger.info(f"  {total_assignments - after_prune:,} assignments pruned ({time.monotonic() - t0:.1f}s)")

    logger.info("Populating cleaned_tags vocabulary...")
    surviving_ids = set(
        r[0] for r in writer.execute("SELECT DISTINCT tag_id FROM cleaned_document_tags").fetchall()
    )
    tag_rows = [(tid, winner_names[tid]) for tid in surviving_ids if tid in winner_names]
    writer.execute("BEGIN")
    writer.executemany("INSERT INTO cleaned_tags (tag_id, tag_name) VALUES (?, ?)", tag_rows)
    writer.execute("COMMIT")

    writer.execute("CREATE INDEX IF NOT EXISTS idx_cdt_doc ON cleaned_document_tags(doc_id)")
    writer.execute("CREATE INDEX IF NOT EXISTS idx_cdt_tag ON cleaned_document_tags(tag_id)")

    final_tags = writer.execute("SELECT COUNT(*) FROM cleaned_tags").fetchone()[0]
    final_assigns = writer.execute("SELECT COUNT(*) FROM cleaned_document_tags").fetchone()[0]
    final_docs = writer.execute("SELECT COUNT(DISTINCT doc_id) FROM cleaned_document_tags").fetchone()[0]

    writer.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE — cleaned tables written:")
    logger.info(f"  cleaned_tags:           {final_tags:,} tags")
    logger.info(f"  cleaned_document_tags:  {final_assigns:,} assignments")
    logger.info(f"  Documents covered:      {final_docs:,}")
    logger.info(f"{'='*60}")



def main():
    parser = argparse.ArgumentParser(description="Clean tags -> write to cleaned_* tables")
    parser.add_argument("--db", default="nlp.db")
    parser.add_argument("--encoder", default="models/final_deep/stage1_encoder")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for semantic merge")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Minimum document occurrences to keep a tag")
    parser.add_argument("--commit", action="store_true",
                        help="Write cleaned tables to DB (default: dry run)")
    parser.add_argument("--skip-semantic", action="store_true",
                        help="Skip semantic merge (useful for testing)")
    parser.add_argument("--merges-file", default=None,
                        help="Load semantic merges from a previous run's JSON instead of recomputing")
    args = parser.parse_args()

    # 1. Load and Regex
    logger.info("Loading tags from database...")
    raw_tags = load_tags(args.db)
    logger.info(f"Loaded {len(raw_tags):,} tags.")
    after_step1, junk_ids = apply_regex_filters(raw_tags)

    merged_2a = apply_prefix_suffix_merges(after_step1)

    if args.merges_file and os.path.exists(args.merges_file):
        logger.info(f"Loading semantic merges from {args.merges_file}...")
        with open(args.merges_file) as f:
            proposed_json = json.load(f)
        semantic_map = {}
        for group in proposed_json:
            for loser in group['merged']:
                semantic_map[loser] = group['winner']
        logger.info(f"Loaded {len(semantic_map):,} semantic merges from file.")
    elif args.skip_semantic:
        semantic_map = {}
        proposed_json = []
        logger.info("Skipping semantic merge.")
    else:
        semantic_map, proposed_json = semantic_merging(
            merged_2a, args.encoder, args.threshold
        )
        with open("proposed_tag_merges.json", "w") as f:
            json.dump(proposed_json, f, indent=2)
        logger.info("Proposed merges saved to proposed_tag_merges.json")

    remap, winner_names = build_id_remap(junk_ids, merged_2a, semantic_map)

    canonical_ids = set(remap.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY:")
    logger.info(f"  Original tags:          {len(raw_tags):,}")
    logger.info(f"  After junk removal:     {len(after_step1):,}")
    logger.info(f"  After prefix/suffix:    {len(merged_2a):,}")
    logger.info(f"  After semantic merge:   {len(canonical_ids):,}")
    logger.info(f"  Frequency threshold:    {args.min_count} (applied on write)")
    logger.info(f"{'='*60}\n")

    if not args.commit:
        logger.info("DRY RUN. Review proposed_tag_merges.json, then re-run with --commit.")
        return

    write_cleaned_tables(args.db, remap, winner_names, args.min_count)


if __name__ == "__main__":
    main()
