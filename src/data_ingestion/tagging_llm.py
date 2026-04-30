"""LLM-based tagging using GLM 4.7
"""

import os
import json
import sqlite3
import argparse
import logging
import concurrent.futures
from pathlib import Path
from openai import OpenAI

try:
    from db import get_writer_conn, get_reader_conn
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_writer_conn, get_reader_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_tags(db_path):
    conn = get_reader_conn(db_path)
    tags = conn.execute("SELECT tag_id, tag_name FROM tags").fetchall()
    conn.close()
    return {t[1].lower(): t[0] for t in tags}


def tag_text_llm(client, model, text, tag_vocab, context_desc="document"):
    tag_list = ", ".join(tag_vocab.keys())

    prompt = f"""You are a specialized scientific document tagger. 
Analyze the following {context_desc} and select all applicable tags from the provided vocabulary.

VOCABULARY:
{tag_list}

{context_desc.upper()} CONTENT:
{text[:4000]}  # Limit context for speed

RESPONSE FORMAT:
Return ONLY a JSON array of tag names.
Example: ["astrophysics", "machine_learning"]
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if "glm" not in model.lower() else None,
            # GLM might not support json_object mode
            temperature=0.1
        )
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        tags = json.loads(content)
        if isinstance(tags, dict):
            tags = tags.get("tags", tags.get("categories", []))

        valid_tags = [t.lower().strip() for t in tags if t.lower().strip() in tag_vocab]
        return valid_tags
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return []


def process_document(doc_id, text, client, model, tag_vocab, db_path):
    """Tags a document and its representative chunks."""
    logger.info(f"Tagging doc_id: {doc_id}")

    doc_tags = tag_text_llm(client, model, text, tag_vocab, "document")

    if doc_tags:
        conn = get_writer_conn(db_path, phase='steady')
        conn.execute("BEGIN")
        for tag_name in doc_tags:
            tag_id = tag_vocab[tag_name]
            conn.execute(
                "INSERT OR IGNORE INTO document_tags (doc_id, tag_id, source, confidence) VALUES (?, ?, ?, ?)",
                (doc_id, tag_id, 'llm', 1.0)
            )
        conn.execute("COMMIT")
        conn.close()

    reader_conn = get_reader_conn(db_path)
    chunks = reader_conn.execute(
        "SELECT chunk_id, chunk_text FROM chunks WHERE doc_id = ? LIMIT 3",
        (doc_id,)
    ).fetchall()
    reader_conn.close()

    for chunk_id, chunk_text in chunks:
        c_tags = tag_text_llm(client, model, chunk_text, tag_vocab, "passage")
        if c_tags:
            conn = get_writer_conn(db_path, phase='steady')
            conn.execute("BEGIN")
            for tag_name in c_tags:
                tag_id = tag_vocab[tag_name]
                conn.execute(
                    "INSERT OR IGNORE INTO chunk_tags (chunk_id, tag_id, source, confidence) VALUES (?, ?, ?, ?)",
                    (chunk_id, tag_id, 'llm', 1.0)
                )
            conn.execute("COMMIT")
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="LLM Tagging Pipeline")
    parser.add_argument("--db", default="nlp.db", help="Path to SQLite database")
    parser.add_argument("--endpoint", default="http://localhost:6767/v1", help="OpenAI-compatible endpoint")
    parser.add_argument("--model", default="glm47", help="Model name")
    parser.add_argument("--limit", type=int, default=100, help="Number of documents to tag in this run")
    parser.add_argument("--workers", type=int, default=10, help="Parallel API requests")
    args = parser.parse_args()

    tag_vocab = get_tags(args.db)
    logger.info(f"Loaded {len(tag_vocab)} tags from vocabulary.")

    client = OpenAI(base_url=args.endpoint, api_key="sk-no-key-required", timeout=600)

    reader_conn = get_reader_conn(args.db)
    pending = reader_conn.execute(
        """SELECT m.doc_id, m.markdown_text
           FROM markdown_content m
                    LEFT JOIN document_tags dt ON m.doc_id = dt.doc_id AND dt.source = 'llm'
           WHERE dt.doc_id IS NULL
           LIMIT ?""",
        (args.limit,)
    ).fetchall()
    reader_conn.close()

    if not pending:
        logger.info("No documents pending LLM tagging.")
        return

    logger.info(f"Tagging {len(pending)} documents with {args.workers} concurrent workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_document, row[0], row[1], client, args.model, tag_vocab, args.db)
            for row in pending
        ]
        concurrent.futures.wait(futures)

    logger.info("Tagging run complete.")


if __name__ == "__main__":
    main()
