"""Stage 3d: arXiv Dataset Ingestion.

Reads the Kaggle arXiv metadata JSON and ingests abstracts + categories
into the database for training data augmentation.

Usage:
    python -m src.arxiv_ingest --arxiv-file /tank/nlpdataset/arxivdata/arxiv-metadata-oai-snapshot.json \
                                --db /mnt/ssd2/NLP_Project/nlp.db \
                                --sample 10000
"""

import argparse
import json
import sys
import time
from pathlib import Path

from db import get_writer_conn, init_db


# arXiv category to tag mapping
ARXIV_CATEGORY_TAGS = {
    "cs.AI": ["artificial_intelligence", "machine_learning"],
    "cs.AR": ["computer_architecture", "hardware"],
    "cs.CC": ["computational_complexity", "theory"],
    "cs.CE": ["computational_engineering"],
    "cs.CG": ["computer_graphics", "geometry"],
    "cs.CL": ["natural_language_processing", "computational_linguistics"],
    "cs.CR": ["cryptography", "security", "cybersecurity"],
    "cs.CV": ["computer_vision", "image_processing"],
    "cs.CY": ["computers_and_society"],
    "cs.DB": ["databases", "data_management"],
    "cs.DC": ["distributed_computing", "parallel_computing"],
    "cs.DL": ["digital_libraries"],
    "cs.DM": ["discrete_mathematics"],
    "cs.DS": ["data_structures", "algorithms"],
    "cs.ET": ["emerging_technologies"],
    "cs.FL": ["formal_languages", "automata_theory"],
    "cs.GL": ["general_literature"],
    "cs.GR": ["graphics"],
    "cs.GT": ["game_theory", "economics"],
    "cs.HC": ["human_computer_interaction"],
    "cs.IR": ["information_retrieval", "search"],
    "cs.IT": ["information_theory"],
    "cs.LG": ["machine_learning", "deep_learning"],
    "cs.LO": ["logic", "formal_methods"],
    "cs.MA": ["multiagent_systems"],
    "cs.MM": ["multimedia"],
    "cs.MS": ["mathematical_software"],
    "cs.NA": ["numerical_analysis"],
    "cs.NE": ["neural_networks", "evolutionary_computation"],
    "cs.NI": ["networking", "internet_architecture"],
    "cs.OH": ["other_computer_science"],
    "cs.OS": ["operating_systems"],
    "cs.PF": ["performance"],
    "cs.PL": ["programming_languages", "compilers"],
    "cs.RO": ["robotics"],
    "cs.SC": ["symbolic_computation"],
    "cs.SD": ["sound", "audio"],
    "cs.SE": ["software_engineering"],
    "cs.SI": ["social_networks"],
    "cs.SY": ["systems_and_control"],
    "physics": ["physics"],
    "astro-ph": ["astrophysics", "astronomy"],
    "cond-mat": ["condensed_matter", "materials_science"],
    "gr-qc": ["general_relativity", "quantum_cosmology"],
    "hep-ex": ["high_energy_physics", "experimental"],
    "hep-lat": ["lattice_field_theory"],
    "hep-ph": ["high_energy_physics", "phenomenology"],
    "hep-th": ["high_energy_physics", "theory"],
    "math-ph": ["mathematical_physics"],
    "nlin": ["nonlinear_dynamics"],
    "nucl-ex": ["nuclear_experiment"],
    "nucl-th": ["nuclear_theory"],
    "quant-ph": ["quantum_physics"],
    "math": ["mathematics"],
    "math.AG": ["algebraic_geometry"],
    "math.AT": ["algebraic_topology"],
    "math.AP": ["analysis", "pdes"],
    "math.CT": ["category_theory"],
    "math.CA": ["classical_analysis"],
    "math.CO": ["combinatorics"],
    "math.AC": ["commutative_algebra"],
    "math.CV": ["complex_variables"],
    "math.DG": ["differential_geometry"],
    "math.DS": ["dynamical_systems"],
    "math.FA": ["functional_analysis"],
    "math.GM": ["general_mathematics"],
    "math.GN": ["general_topology"],
    "math.GT": ["geometric_topology"],
    "math.GR": ["group_theory"],
    "math.HO": ["history_of_mathematics"],
    "math.IT": ["information_theory"],
    "math.KT": ["k_theory"],
    "math.LO": ["logic"],
    "math.MP": ["mathematical_physics"],
    "math.MG": ["metric_geometry"],
    "math.NT": ["number_theory"],
    "math.NA": ["numerical_analysis"],
    "math.OA": ["operator_algebras"],
    "math.OC": ["optimization_and_control"],
    "math.PR": ["probability"],
    "math.QA": ["quantum_algebra"],
    "math.RT": ["representation_theory"],
    "math.RA": ["rings_and_algebras"],
    "math.SP": ["spectral_theory"],
    "math.ST": ["statistics"],
    "math.SG": ["symplectic_geometry"],
    "q-bio": ["quantitative_biology"],
    "q-bio.BM": ["biomolecules"],
    "q-bio.CB": ["cell_biology"],
    "q-bio.GN": ["genomics"],
    "q-bio.MN": ["molecular_networks"],
    "q-bio.NC": ["neurons_and_cognition"],
    "q-bio.OT": ["other_quantitative_biology"],
    "q-bio.PE": ["populations_and_evolution"],
    "q-bio.QM": ["quantitative_methods"],
    "q-bio.SC": ["subcellular_processes"],
    "q-bio.TO": ["tissues_and_organs"],
    "q-fin": ["quantitative_finance"],
    "q-fin.CP": ["computational_finance"],
    "q-fin.EC": ["economics"],
    "q-fin.GN": ["general_finance"],
    "q-fin.MF": ["mathematical_finance"],
    "q-fin.PM": ["portfolio_management"],
    "q-fin.PR": ["pricing_of_securities"],
    "q-fin.RM": ["risk_management"],
    "q-fin.ST": ["statistical_finance"],
    "q-fin.TR": ["trading_and_market_microstructure"],
    "stat": ["statistics"],
    "stat.AP": ["applications"],
    "stat.CO": ["computation"],
    "stat.ML": ["machine_learning"],
    "stat.ME": ["methodology"],
    "stat.OT": ["other_statistics"],
    "stat.TH": ["theory"],
    "eess": ["electrical_engineering"],
    "eess.AS": ["audio_and_speech_processing"],
    "eess.IV": ["image_and_video_processing"],
    "eess.SP": ["signal_processing"],
    "eess.SY": ["systems_and_control"],
    "econ": ["economics"],
    "econ.EM": ["econometrics"],
    "econ.GN": ["general_economics"],
    "econ.TH": ["theoretical_economics"],
}


def parse_arxiv_record(line: str) -> dict | None:
    try:
        record = json.loads(line)

        arxiv_id = record.get("id")
        title = record.get("title", "").replace("\n", " ").strip()
        abstract = record.get("abstract", "").replace("\n", " ").strip()
        categories = record.get("categories", "").strip()
        authors = record.get("authors", "").strip()
        update_date = record.get("update_date")

        if not abstract or not categories:
            return None

        category_list = categories.split()
        tags = []
        for cat in category_list:
            if cat in ARXIV_CATEGORY_TAGS:
                tags.extend(ARXIV_CATEGORY_TAGS[cat])

        tags = list(set(tags))

        if not tags:
            return None

        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "categories": categories,
            "authors": authors,
            "update_date": update_date,
            "tags": tags,
        }
    except Exception as e:
        return None


def ingest_arxiv(
    arxiv_file: str,
    db_path: str,
    sample_size: int | None = None,
    batch_size: int = 5000,
) -> dict:
    t0 = time.monotonic()

    if not Path(db_path).exists():
        conn = init_db(db_path)
    else:
        conn = get_writer_conn(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS arxiv_papers (
            arxiv_id        TEXT    PRIMARY KEY,
            title           TEXT    NOT NULL,
            abstract        TEXT    NOT NULL,
            categories      TEXT    NOT NULL,
            authors         TEXT,
            update_date     TEXT,
            created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now'))
        )
    """)

    print(f"Reading arXiv metadata from {arxiv_file}...")

    total_read = 0
    total_parsed = 0
    total_inserted = 0
    total_skipped = 0
    batch = []

    with open(arxiv_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            total_read += 1

            if sample_size and total_read > sample_size:
                break

            # Parse record
            record = parse_arxiv_record(line)
            if record is None:
                continue

            total_parsed += 1
            batch.append(record)

            # Batch commit
            if len(batch) >= batch_size:
                conn.execute("BEGIN")
                for r in batch:
                    try:
                        conn.execute(
                            """INSERT OR IGNORE INTO arxiv_papers
                               (arxiv_id, title, abstract, categories, authors, update_date)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                r["arxiv_id"],
                                r["title"],
                                r["abstract"],
                                r["categories"],
                                r["authors"],
                                r["update_date"],
                            ),
                        )
                        if conn.execute("SELECT changes()").fetchone()[0] > 0:
                            total_inserted += 1
                        else:
                            total_skipped += 1
                    except Exception as e:
                        print(
                            f"  ERROR inserting {r['arxiv_id']}: {e}", file=sys.stderr
                        )
                        total_skipped += 1
                conn.execute("COMMIT")

                # Progress
                elapsed = time.monotonic() - t0
                rate = total_read / elapsed if elapsed > 0 else 0
                print(
                    f"  {total_read:>8,} read, {total_parsed:>8,} parsed, "
                    f"{total_inserted:>8,} inserted  ({rate:,.0f} rec/sec)"
                )

                batch.clear()

    # Final batch
    if batch:
        conn.execute("BEGIN")
        for r in batch:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO arxiv_papers
                       (arxiv_id, title, abstract, categories, authors, update_date)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        r["arxiv_id"],
                        r["title"],
                        r["abstract"],
                        r["categories"],
                        r["authors"],
                        r["update_date"],
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    total_inserted += 1
                else:
                    total_skipped += 1
            except Exception as e:
                print(f"  ERROR inserting {r['arxiv_id']}: {e}", file=sys.stderr)
                total_skipped += 1
        conn.execute("COMMIT")

    # Create index
    print("Creating index on categories...")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_arxiv_categories ON arxiv_papers(categories)"
    )
    conn.execute("ANALYZE")

    # Verify
    total_in_db = conn.execute("SELECT COUNT(*) FROM arxiv_papers").fetchone()[0]
    conn.close()

    total_time = time.monotonic() - t0
    print(f"\narXiv ingestion complete:")
    print(f"  Read: {total_read:,} records")
    print(f"  Parsed: {total_parsed:,} records")
    print(f"  Inserted: {total_inserted:,} records")
    print(f"  Skipped: {total_skipped:,} records")
    print(f"  Total in DB: {total_in_db:,} records")
    print(f"  Time: {total_time:.1f}s")

    return {
        "total_read": total_read,
        "total_parsed": total_parsed,
        "total_inserted": total_inserted,
        "total_skipped": total_skipped,
        "total_in_db": total_in_db,
        "time_seconds": total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3d: Ingest arXiv metadata for training augmentation."
    )
    parser.add_argument(
        "--arxiv-file", required=True, help="Path to arxiv-metadata-oai-snapshot.json"
    )
    parser.add_argument(
        "--db", default="/mnt/ssd2/NLP_Project/nlp.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--sample", type=int, help="Only ingest first N records (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Rows per DB commit (default: 5000)",
    )

    args = parser.parse_args()

    result = ingest_arxiv(
        arxiv_file=args.arxiv_file,
        db_path=args.db,
        sample_size=args.sample,
        batch_size=args.batch_size,
    )

    sys.exit(0 if result["total_inserted"] > 0 else 1)


if __name__ == "__main__":
    main()
