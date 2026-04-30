import os
import zipfile
import logging
import sqlite3
import argparse
import time
from lxml import etree
from pathlib import Path
from multiprocessing import Pool, cpu_count

try:
    from db import get_writer_conn, get_reader_conn
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_writer_conn, get_reader_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

NS = {
    'n': 'urn:oasis:names:tc:opendocument:xmlns:container',
    'opf': 'http://www.idpf.org/2007/opf',
    'dc': 'http://purl.org/dc/elements/1.1/'
}


def extract_metadata_from_epub(epub_path):
    """Extracts Dublin Core metadata from an EPUB file."""
    try:
        with zipfile.ZipFile(epub_path, 'r') as z:
            try:
                container_data = z.read('META-INF/container.xml')
                container_root = etree.fromstring(container_data)
                opf_path = container_root.xpath('//n:rootfile/@full-path', namespaces=NS)[0]
            except (KeyError, IndexError, etree.XMLSyntaxError):
                return None

            try:
                opf_data = z.read(opf_path)
                opf_root = etree.fromstring(opf_data)
            except (KeyError, etree.XMLSyntaxError):
                return None

            metadata = {
                'subjects': opf_root.xpath('//dc:subject/text()', namespaces=NS),
                'description': opf_root.xpath('//dc:description/text()', namespaces=NS),
                'creator': opf_root.xpath('//dc:creator/text()', namespaces=NS),
                'title': opf_root.xpath('//dc:title/text()', namespaces=NS),
                'language': opf_root.xpath('//dc:language/text()', namespaces=NS),
                'identifier': opf_root.xpath('//dc:identifier/text()', namespaces=NS)
            }
            return metadata
    except Exception as e:
        return None


def normalize_tag(tag_name):
    import re
    """Basic normalization: lowercase, underscores, strip symbols."""
    tag = tag_name.lower().strip()
    tag = re.sub(r'[^a-z0-9\s]', '', tag)
    tag = re.sub(r'\s+', '_', tag)
    return tag


def process_item(item):
    doc_id, fullpath = item
    metadata = extract_metadata_from_epub(fullpath)
    if metadata:
        return doc_id, metadata
    return None


def main():
    parser = argparse.ArgumentParser(description="Direct EPUB Metadata Tag Extraction")
    parser.add_argument("--db", default="nlp.db", help="Path to SQLite database")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Parallel workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents to process")
    args = parser.parse_args()

    reader = get_reader_conn(args.db)
    query = """
            SELECT doc_id, fullpath
            FROM documents
            WHERE extension = 'epub'
              AND doc_id NOT IN (SELECT DISTINCT doc_id FROM document_tags WHERE source = 'epub_metadata') \
            """
    if args.limit > 0:
        query += f" LIMIT {args.limit}"

    pending = reader.execute(query).fetchall()
    reader.close()

    if not pending:
        logger.info("No pending EPUBs for metadata extraction.")
        return

    logger.info(f"Extracting metadata from {len(pending)} EPUBs using {args.workers} workers...")

    start_time = time.time()
    from tqdm import tqdm

    writer = get_writer_conn(args.db, phase='steady')
    tag_id_cache = {}
    tags_linked = 0
    batch_results = []

    def flush_batch(results):
        nonlocal tags_linked
        writer.execute("BEGIN")
        for doc_id, meta in results:
            subjects = meta.get('subjects', [])
            for raw_subj in subjects:
                sub_parts = [s.strip() for s in raw_subj.split(',')]
                for part in sub_parts:
                    norm_name = normalize_tag(part)
                    if not norm_name or len(norm_name) < 2: continue

                    if norm_name not in tag_id_cache:
                        writer.execute("INSERT OR IGNORE INTO tags (tag_name) VALUES (?)", (norm_name,))
                        tid_res = writer.execute("SELECT tag_id FROM tags WHERE tag_name = ?", (norm_name,)).fetchone()
                        if tid_res:
                            tag_id_cache[norm_name] = tid_res[0]

                    if norm_name in tag_id_cache:
                        tag_id = tag_id_cache[norm_name]
                        writer.execute(
                            "INSERT OR IGNORE INTO document_tags (doc_id, tag_id, source, confidence) VALUES (?, ?, ?, ?)",
                            (doc_id, tag_id, 'epub_metadata', 1.0)
                        )
                        tags_linked += 1
        writer.execute("COMMIT")

    with Pool(processes=args.workers) as pool:
        for res in tqdm(pool.imap_unordered(process_item, pending), total=len(pending), desc="Extracting Metadata"):
            if res:
                batch_results.append(res)

            if len(batch_results) >= 1000:
                flush_batch(batch_results)
                batch_results = []

    if batch_results:
        flush_batch(batch_results)

    writer.close()
    logger.info(f"Finished! Linked {tags_linked} tags from EPUB metadata in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    main()
