"""
verify_cloud_graph_sync.py -- Automated Cloud Graph Verification & Audit Tool

Performs an automated 1:1 audit between local graph files
(article_graph.json and act_graph.json) and Supabase's knowledge.citation_edges.

Usage:
    python verify_cloud_graph_sync.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import auth_service

CONST_GRAPH_PATH = BASE_DIR / "data" / "okf" / "constitution" / "article_graph.json"
ACT_GRAPH_PATH = BASE_DIR / "data" / "okf" / "criminal_procedure" / "act_graph.json"


def audit_cloud_sync():
    print("=" * 70)
    print("  LEGAL KNOWLEDGE GRAPH -- SUPABASE AUTOMATED AUDIT")
    print("=" * 70)

    # 1. Load Local Graphs
    with open(CONST_GRAPH_PATH, "r", encoding="utf-8") as f:
        const_data = json.load(f)

    with open(ACT_GRAPH_PATH, "r", encoding="utf-8") as f:
        act_data = json.load(f)

    # 2. Query Supabase
    conn = auth_service.get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_document_id, source_unit_number,
                       target_document_id, target_unit_number,
                       relation_type, metadata
                FROM knowledge.citation_edges;
            """)
            cloud_rows = cur.fetchall()
    finally:
        conn.close()

    # Index cloud edges: (src_doc, src_unit) -> set of (tgt_doc, tgt_unit, rel)
    cloud_edges: set[tuple[str, int, str, int, str]] = set()
    cloud_provisions: set[tuple[str, int]] = set()

    for src_doc, src_unit, tgt_doc, tgt_unit, rel, meta in cloud_rows:
        cloud_edges.add((src_doc, int(src_unit), tgt_doc, int(tgt_unit), rel))
        cloud_provisions.add((src_doc, int(src_unit)))

    # ----------------------------------------------------
    # AUDIT PART 1: Constitution
    # ----------------------------------------------------
    print("\n[1] AUDIT: Constitution of Nepal (article_graph.json)")
    expected_const_edges = 0
    missing_const = []

    for art_str, data in const_data.items():
        art_num = int(art_str)
        for tgt_art in data.get("articles", []):
            expected_const_edges += 1
            key = ("constitution", art_num, "constitution", int(tgt_art), "refers_to_article")
            if key not in cloud_edges:
                missing_const.append(f"Article {art_num} -> Article {tgt_art}")

    actual_const_edges = len([e for e in cloud_edges if e[0] == "constitution" and e[2] == "constitution"])
    print(f"    * Expected Article-to-Article Edges: {expected_const_edges}")
    print(f"    * Actual Edges in Supabase:          {actual_const_edges}")
    if missing_const:
        print(f"    [FAIL] Missing {len(missing_const)} edges in Supabase: {missing_const[:5]}")
    else:
        print("    [PASS] 100% of Constitutional Article links are synced in Supabase!")

    # ----------------------------------------------------
    # AUDIT PART 2: Criminal Procedure Code
    # ----------------------------------------------------
    print("\n[2] AUDIT: Criminal Procedure Code (act_graph.json)")
    expected_act_intra = 0
    expected_act_cross = 0
    missing_act = []

    for sec_str, data in act_data.items():
        sec_num = int(sec_str)
        for tgt_sec in data.get("sections", []):
            expected_act_intra += 1
            key = ("criminal_procedure", sec_num, "criminal_procedure", int(tgt_sec), "refers_to_section")
            if key not in cloud_edges:
                missing_act.append(f"Section {sec_num} -> Section {tgt_sec}")

        for tgt_art in data.get("constitution_articles", []):
            expected_act_cross += 1
            key = ("criminal_procedure", sec_num, "constitution", int(tgt_art), "cites_constitutional_article")
            if key not in cloud_edges:
                missing_act.append(f"Section {sec_num} -> Constitution Article {tgt_art}")

    actual_act_intra = len([e for e in cloud_edges if e[0] == "criminal_procedure" and e[2] == "criminal_procedure"])
    actual_act_cross = len([e for e in cloud_edges if e[0] == "criminal_procedure" and e[2] == "constitution"])

    print(f"    * Expected Section-to-Section Edges: {expected_act_intra}")
    print(f"    * Actual Intra-Act Edges in Supabase: {actual_act_intra}")
    print(f"    * Expected Cross-Act Edges (Act->Const): {expected_act_cross}")
    print(f"    * Actual Cross-Act Edges in Supabase:    {actual_act_cross}")

    if missing_act:
        print(f"    [FAIL] Missing {len(missing_act)} edges in Supabase: {missing_act[:5]}")
    else:
        print("    [PASS] 100% of Statutory Section & Cross-Act links are synced in Supabase!")

    # ----------------------------------------------------
    # AUDIT PART 3: Grand Total
    # ----------------------------------------------------
    total_expected = expected_const_edges + expected_act_intra + expected_act_cross
    total_actual = len(cloud_edges)
    print("\n" + "=" * 70)
    print(f"  GRAND TOTAL: {total_actual} edges verified in Supabase (Expected: {total_expected})")
    if total_actual == total_expected and not missing_const and not missing_act:
        print("  VERIFICATION STATUS: 100% MATCH -- NO MANUAL COUNTING NEEDED!")
    else:
        print("  VERIFICATION STATUS: DISCREPANCY DETECTED")
    print("=" * 70)


if __name__ == "__main__":
    audit_cloud_sync()
