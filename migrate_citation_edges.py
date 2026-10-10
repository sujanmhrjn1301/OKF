"""
migrate_citation_edges.py -- Migrate Cross-Reference Graph Edges to Supabase

This script:
  1. Creates the table knowledge.citation_edges and indexes if not exists.
  2. Ingests all edges from data/okf/constitution/article_graph.json into Supabase.
  3. Verifies the inserted count directly from Supabase.
  4. Makes retrieval completely independent of local disk files.

Usage:
  python migrate_citation_edges.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

import auth_service

load_dotenv(Path(__file__).with_name(".env"))

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "data" / "okf" / "constitution" / "article_graph.json"


def setup_citation_edges_table(conn):
    """Create knowledge.citation_edges table and indexes if not exist."""
    sql = """
    CREATE TABLE IF NOT EXISTS knowledge.citation_edges (
        id BIGSERIAL PRIMARY KEY,
        source_document_id TEXT NOT NULL REFERENCES knowledge.documents(id) ON DELETE CASCADE,
        source_unit_number INTEGER NOT NULL,
        target_document_id TEXT NOT NULL REFERENCES knowledge.documents(id) ON DELETE CASCADE,
        target_unit_number INTEGER NOT NULL,
        relation_type TEXT NOT NULL DEFAULT 'refers_to',
        metadata JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(source_document_id, source_unit_number, target_document_id, target_unit_number)
    );

    CREATE INDEX IF NOT EXISTS idx_citation_source 
    ON knowledge.citation_edges(source_document_id, source_unit_number);

    CREATE INDEX IF NOT EXISTS idx_citation_target 
    ON knowledge.citation_edges(target_document_id, target_unit_number);
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("[MIGRATION] Table knowledge.citation_edges verified/created.")


def upload_graph_edges(conn, graph: dict[str, list[int]], source_doc: str = "constitution", target_doc: str = "constitution"):
    """Insert graph edges into knowledge.citation_edges."""
    rows = []
    for src_str, targets in graph.items():
        try:
            src_num = int(src_str)
        except ValueError:
            continue
        for tgt_num in targets:
            rows.append((source_doc, src_num, target_doc, int(tgt_num), "refers_to"))

    if not rows:
        print("[MIGRATION] No edges to upload.")
        return 0

    sql = """
    INSERT INTO knowledge.citation_edges (
        source_document_id, source_unit_number,
        target_document_id, target_unit_number,
        relation_type
    ) VALUES %s
    ON CONFLICT (source_document_id, source_unit_number, target_document_id, target_unit_number)
    DO NOTHING;
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=100)
    conn.commit()
    print(f"[MIGRATION] Processed {len(rows)} edges for {source_doc}.")
    return len(rows)


def verify_cloud_edges(conn):
    """Query Supabase to verify edges stored in the cloud."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM knowledge.citation_edges;")
        total_count = cur.fetchone()[0]

        cur.execute("""
            SELECT source_document_id, source_unit_number, target_document_id, target_unit_number
            FROM knowledge.citation_edges
            ORDER BY source_unit_number
            LIMIT 5;
        """)
        samples = cur.fetchall()

    print(f"[VERIFY] Total edges in Supabase: {total_count}")
    print("[VERIFY] Sample edges in cloud:")
    for src_doc, src_num, tgt_doc, tgt_num in samples:
        print(f"   {src_doc} #{src_num} -> {tgt_doc} #{tgt_num}")
    return total_count


def main():
    print("=" * 60)
    print("  Migrating Article Graph to Supabase Cloud")
    print("=" * 60)

    if not GRAPH_PATH.exists():
        print(f"[ERROR] Graph file not found at {GRAPH_PATH}")
        sys.exit(1)

    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    print(f"[LOAD] Loaded {len(graph)} source articles from {GRAPH_PATH.name}")

    conn = auth_service.get_db_connection()
    try:
        setup_citation_edges_table(conn)
        upload_graph_edges(conn, graph, source_doc="constitution", target_doc="constitution")
        verify_cloud_edges(conn)
        print("\n[SUCCESS] All legal relationships are now safely stored in Supabase!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
