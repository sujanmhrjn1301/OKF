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


def upload_graph_edges(conn, graph: dict, source_doc: str = "constitution", target_doc: str = "constitution"):
    """Insert provision-to-provision graph edges into knowledge.citation_edges.

    Accepts both formats:
      - legacy: {"100": [76]}
      - nested: {"100": {"articles": [76], "clauses": [...], "schedules": [...]}}
    Only provision-level links (articles / sections / constitution_articles) become DB edges;
    finer-grained links (clauses, sub-sections, schedules, parts, chapters) stay in the JSON
    and are attached as edge metadata for context.
    """
    rows = []
    for src_str, targets in graph.items():
        try:
            src_num = int(src_str)
        except ValueError:
            continue

        if isinstance(targets, list):  # legacy format
            for tgt_num in targets:
                rows.append((source_doc, src_num, target_doc, int(tgt_num), "refers_to", "{}"))
            continue

        DOC_TITLES = {
            "constitution": "The Constitution of Nepal, 2015",
            "criminal_procedure": "The National Criminal Procedure (Code) Act, 2017",
        }
        source_title = DOC_TITLES.get(source_doc, source_doc)
        detail = {k: v for k, v in targets.items() if k not in ("articles", "sections", "constitution_articles")}

        same_doc_key = "articles" if source_doc == "constitution" else "sections"
        for tgt_num in targets.get(same_doc_key, []):
            rel = "refers_to_article" if same_doc_key == "articles" else "refers_to_section"
            pinpoints = []
            if source_doc == "constitution":
                pinpoints = [c for c in targets.get("article_clauses", []) if c.startswith(f"{tgt_num}(")]
            else:
                pinpoints = [s for s in targets.get("section_subsections", []) if s.startswith(f"{tgt_num}(")]

            unit_label = "Article" if source_doc == "constitution" else "Section"
            pinpoint_str = f" ({pinpoints[0]})" if pinpoints else ""
            edge_meta = {
                "scope": "INTRA_DOCUMENT",
                "source_document_title": source_title,
                "target_document_title": source_title,
                "relationship_label": f"{unit_label} {src_num} -> {unit_label} {tgt_num}{pinpoint_str} (Within Same Document)",
                "relationship_description": f"{unit_label} {src_num} refers internally within {source_title} to {unit_label} {tgt_num}{pinpoint_str}",
                "pinpoint": pinpoints,
            }
            for k, v in detail.items():
                if k not in edge_meta:
                    edge_meta[k] = v

            rows.append((source_doc, src_num, target_doc, int(tgt_num), rel, json.dumps(edge_meta)))

        for tgt_num in targets.get("constitution_articles", []):
            target_title = DOC_TITLES.get("constitution", "The Constitution of Nepal, 2015")
            const_pinpoints = [c for c in targets.get("constitution_article_clauses", []) if c.startswith(f"{tgt_num}(")]
            pinpoint_str = f" ({const_pinpoints[0]})" if const_pinpoints else ""
            edge_meta = {
                "scope": "CROSS_DOCUMENT",
                "source_document_title": source_title,
                "target_document_title": target_title,
                "relationship_label": f"Section {src_num} ({source_title}) -> Article {tgt_num}{pinpoint_str} ({target_title}) [Cross-Act Citation]",
                "relationship_description": f"Section {src_num} of {source_title} refers across legislation to Article {tgt_num}{pinpoint_str} of {target_title}",
                "pinpoint": const_pinpoints,
            }
            for k, v in detail.items():
                if k not in edge_meta:
                    edge_meta[k] = v

            rows.append((source_doc, src_num, "constitution", int(tgt_num), "cites_constitutional_article", json.dumps(edge_meta)))

    if not rows:
        print("[MIGRATION] No edges to upload.")
        return 0

    sql = """
    INSERT INTO knowledge.citation_edges (
        source_document_id, source_unit_number,
        target_document_id, target_unit_number,
        relation_type, metadata
    ) VALUES %s
    ON CONFLICT (source_document_id, source_unit_number, target_document_id, target_unit_number)
    DO UPDATE SET relation_type = EXCLUDED.relation_type, metadata = EXCLUDED.metadata;
    """

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur, sql, rows, template="(%s, %s, %s, %s, %s, %s::jsonb)", page_size=100
        )
    conn.commit()
    print(f"[MIGRATION] Processed {len(rows)} edges for {source_doc}.")
    return len(rows)


def verify_cloud_edges(conn):
    """Query Supabase to verify edges stored in the cloud."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM knowledge.citation_edges;")
        total_count = cur.fetchone()[0]

        cur.execute("""
            SELECT source_document_id, target_document_id, COUNT(*)
            FROM knowledge.citation_edges
            GROUP BY source_document_id, target_document_id
            ORDER BY source_document_id;
        """)
        breakdown = cur.fetchall()

        cur.execute("""
            SELECT source_document_id, source_unit_number, target_document_id, target_unit_number, relation_type, metadata
            FROM knowledge.citation_edges
            WHERE metadata::text != '{}'
            ORDER BY source_document_id, source_unit_number
            LIMIT 4;
        """)
        samples = cur.fetchall()

    print(f"\n[VERIFY] Total edges in Supabase: {total_count}")
    print("[VERIFY] Breakdown by Act:")
    for src_doc, tgt_doc, cnt in breakdown:
        print(f"   * {src_doc} -> {tgt_doc}: {cnt} edges")
    print("\n[VERIFY] Sample edges with rich metadata stored in cloud:")
    for src_doc, src_num, tgt_doc, tgt_num, rel, meta in samples:
        print(f"   * {src_doc} #{src_num} -> {tgt_doc} #{tgt_num} ({rel}) | meta: {meta}")
    return total_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate citation graphs to Supabase Cloud")
    parser.add_argument("--act", type=str, default=None, help="Name of act to migrate (e.g. criminal_procedure)")
    parser.add_argument("--all", action="store_true", help="Migrate Constitution and all available Acts")
    parser.add_argument("--reset", action="store_true", help="Truncate table before migrating fresh edges")
    args = parser.parse_args()

    print("=" * 60)
    print("  Migrating Legal Citation Graph to Supabase Cloud")
    print("=" * 60)

    conn = auth_service.get_db_connection()
    try:
        setup_citation_edges_table(conn)

        if args.reset:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE knowledge.citation_edges;")
            conn.commit()
            print("[MIGRATION] Table knowledge.citation_edges truncated for clean reload.")

        # 1. Migrate Constitution if requested or default
        if not args.act or args.all:
            const_path = BASE_DIR / "data" / "okf" / "constitution" / "article_graph.json"
            if const_path.exists():
                with open(const_path, "r", encoding="utf-8") as f:
                    graph = json.load(f)
                print(f"[LOAD] Loaded {len(graph)} constitution source articles from {const_path.name}")
                upload_graph_edges(conn, graph, source_doc="constitution", target_doc="constitution")

        # 2. Migrate Act graph (e.g. criminal_procedure)
        act_names = []
        if args.act:
            act_names = [args.act]
        elif args.all:
            # Check for any act_graph.json under data/okf/
            for d in (BASE_DIR / "data" / "okf").iterdir():
                if d.is_dir() and d.name != "constitution" and (d / "act_graph.json").exists():
                    act_names.append(d.name)

        for act_name in act_names:
            act_path = BASE_DIR / "data" / "okf" / act_name / "act_graph.json"
            if act_path.exists():
                with open(act_path, "r", encoding="utf-8") as f:
                    graph = json.load(f)
                print(f"[LOAD] Loaded {len(graph)} source sections from {act_name}/{act_path.name}")
                upload_graph_edges(conn, graph, source_doc=act_name, target_doc=act_name)
            else:
                print(f"[WARNING] act_graph.json not found for {act_name} at {act_path}")

        verify_cloud_edges(conn)
        print("\n[SUCCESS] Legal relationships are safely stored in Supabase!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
