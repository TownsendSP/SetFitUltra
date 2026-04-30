import re
import sqlite3
import logging
import os
import sys
from tqdm import tqdm

try:
    from db import get_reader_conn, get_writer_conn
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.db import get_reader_conn, get_writer_conn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Regex for ISBN-10 and ISBN-13
ISBN_RE = re.compile(
    r'(?:ISBN(?:-1[03])?:?\s+)?(?=[-0-9 ]{10,17})(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]',
    re.IGNORECASE)


def clean_isbn(isbn_str):
    cleaned = re.sub(r'[^0-9X]', '', isbn_str.upper())
    if len(cleaned) == 10 or len(cleaned) == 13:
        return cleaned
    return None


def extract_from_db(db_path):
    reader = get_reader_conn(db_path)
    writer = get_writer_conn(db_path, phase='steady')

    count_query = """
                  SELECT count(doc_id)
                  FROM markdown_content
                  WHERE doc_id IN (SELECT doc_id FROM documents WHERE isbn IS NULL) \
                  """
    total_docs = reader.execute(count_query).fetchone()[0]

    if total_docs == 0:
        logger.info("No documents need ISBN scanning.")
        reader.close()
        writer.close()
        return

    logger.info(f"Scanning {total_docs} documents for ISBNs in batches...")

    cursor = reader.cursor()
    cursor.execute("""
                   SELECT doc_id, SUBSTR(markdown_text, 1, 50000)
                   FROM markdown_content
                   WHERE doc_id IN (SELECT doc_id FROM documents WHERE isbn IS NULL)
                   """)

    found_count = 0
    batch_size = 1000

    writer.execute("BEGIN")

    with tqdm(total=total_docs, desc="Extracting ISBNs") as pbar:
        while True:
            docs = cursor.fetchmany(batch_size)
            if not docs:
                break

            for doc_id, text in docs:
                if not text:
                    pbar.update(1)
                    continue

                matches = ISBN_RE.findall(text)
                for m in matches:
                    cleaned = clean_isbn(m)
                    if cleaned:
                        writer.execute("UPDATE documents SET isbn = ? WHERE doc_id = ?", (cleaned, doc_id))
                        found_count += 1
                        break
                pbar.update(1)

            writer.execute("COMMIT")
            writer.execute("BEGIN")

    writer.execute("COMMIT")
    writer.close()
    reader.close()
    logger.info(f"Updated {found_count} documents with extracted ISBNs.")


if __name__ == "__main__":
    extract_from_db("nlp.db")
