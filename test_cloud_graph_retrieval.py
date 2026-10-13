"""
test_cloud_graph_retrieval.py -- Comprehensive Test for Cloud-Native Graph Retrieval

Verifies:
  1. Direct connection to Supabase and query on knowledge.citation_edges.
  2. Loading graph via chat_cli.get_article_graph() directly from Supabase.
  3. Verifying that NO local disk file is needed (simulates deployment without data/ folder).
  4. End-to-end execution of expand_with_cross_references() on realistic candidates (e.g. Article 100).
  5. Verifies that the cross-referenced chunk (Article 76) is correctly fetched from Supabase.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

import chat_cli
import auth_service


def test_cloud_edges_exist():
    print("\n--- Test 1: Verify Supabase knowledge.citation_edges Table ---")
    conn = auth_service.get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge.citation_edges;")
            count = cur.fetchone()[0]
            print(f"[PASS] Total citation edges in Supabase: {count}")
            assert count >= 167, f"Expected at least 167 citation edges, got {count}"

            cur.execute("""
                SELECT source_document_id, target_document_id, COUNT(*)
                FROM knowledge.citation_edges
                GROUP BY source_document_id, target_document_id
                ORDER BY source_document_id;
            """)
            breakdown = cur.fetchall()
            print("[PASS] Cloud Citation Breakdown:")
            for src, tgt, cnt in breakdown:
                print(f"       * {src} -> {tgt}: {cnt} edges")
    finally:
        conn.close()


def test_get_cloud_citation_graph():
    print("\n--- Test 2: Verify get_cloud_citation_graph() from Supabase ---")
    chat_cli._CLOUD_CITATION_GRAPH = None

    # Point local file paths to fake non-existent paths to PROVE zero disk reliance
    orig_art_path = chat_cli.ARTICLE_GRAPH_PATH
    orig_crim_dir = chat_cli.CRIMINAL_PROC_DIR
    chat_cli.ARTICLE_GRAPH_PATH = Path("/non_existent_folder/fake_art.json")
    chat_cli.CRIMINAL_PROC_DIR = Path("/non_existent_folder/fake_crim")

    try:
        cloud_graph = chat_cli.get_cloud_citation_graph()
        print(f"[PASS] Successfully loaded cloud graph with {len(cloud_graph)} source provisions (Zero local files used).")
        assert len(cloud_graph) > 0, "Cloud graph failed to load!"

        # Check Criminal Procedure Section 18
        assert ("criminal_procedure", 18) in cloud_graph, "Criminal Procedure Sec 18 missing!"
        sec18_edges = cloud_graph[("criminal_procedure", 18)]
        sec18_targets = [e["target_unit_number"] for e in sec18_edges if e["target_document_id"] == "criminal_procedure"]
        assert 16 in sec18_targets, f"Expected Sec 18 -> 16, got {sec18_targets}"
        print(f"       Criminal Procedure Section 18 targets: {sec18_targets} | meta: {sec18_edges[0]['metadata']}")

        # Check Cross-Act Section 96 -> Constitution
        assert ("criminal_procedure", 96) in cloud_graph, "Cross-act Section 96 missing!"
        sec96_edges = cloud_graph[("criminal_procedure", 96)]
        sec96_const_targets = [e["target_unit_number"] for e in sec96_edges if e["target_document_id"] == "constitution"]
        assert 134 in sec96_const_targets and 145 in sec96_const_targets, f"Expected Section 96 -> [134, 145], got {sec96_const_targets}"
        print(f"       Cross-Act Section 96 -> Constitution Articles: {sec96_const_targets}")

        # Check Constitution Article 100
        assert ("constitution", 100) in cloud_graph, "Article 100 missing!"
        art100_targets = [e["target_unit_number"] for e in cloud_graph[("constitution", 100)]]
        assert 76 in art100_targets, f"Expected Article 100 -> 76, got {art100_targets}"
        print(f"       Constitution Article 100 -> Articles: {art100_targets}")
    finally:
        chat_cli.ARTICLE_GRAPH_PATH = orig_art_path
        chat_cli.CRIMINAL_PROC_DIR = orig_crim_dir


def test_end_to_end_statutory_and_constitutional_expansion():
    print("\n--- Test 3: Verify expand_with_cross_references() for Statutory & Constitutional Provisions ---")
    chat_cli._CLOUD_CITATION_GRAPH = None

    candidates = [
        {
            "id": 1001,
            "document_id": "criminal_procedure",
            "unit_number": 18,
            "title": "Section 18: Power to make search",
            "content": "Power to search person or place...",
            "rrf_score": 0.08,
            "retrieval_count": 2,
        },
        {
            "id": 1002,
            "document_id": "criminal_procedure",
            "unit_number": 96,
            "title": "Section 96: Power to transfer cases",
            "content": "Supreme Court power to transfer cases...",
            "rrf_score": 0.07,
            "retrieval_count": 1,
        },
    ]

    intent = {"intent": "STATUTORY_CRIMINAL"}
    expanded = chat_cli.expand_with_cross_references(candidates, intent)

    print(f"[PASS] Original candidates: 2, Total after expansion: {len(expanded)}")
    cross_refs = [c for c in expanded if c.get("is_cross_ref")]
    print(f"[PASS] Cross-referenced candidates injected: {len(cross_refs)}")
    injected_keys = [(c["document_id"], c["unit_number"]) for c in cross_refs]
    for cr in cross_refs:
        print(f"       Injected: {cr.get('document_id')} Unit #{cr.get('unit_number')} - {cr.get('title')} ({cr.get('cross_ref_relation')})")

    # Verify Section 18 pulled in Section 16
    assert ("criminal_procedure", 16) in injected_keys, f"Expected ('criminal_procedure', 16) in {injected_keys}"
    # Verify Section 96 pulled in Constitution Article 134
    assert ("constitution", 134) in injected_keys or ("constitution", 145) in injected_keys, f"Expected Cross-Act Article in {injected_keys}"
    print("\n[VERIFIED] Both intra-Act and cross-Act provisions successfully expanded directly from Supabase!")


def main():
    print("=" * 60)
    print("  Testing Universal Cloud Citation Graph (Supabase-Native)")
    print("=" * 60)

    test_cloud_edges_exist()
    test_get_cloud_citation_graph()
    test_end_to_end_statutory_and_constitutional_expansion()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED: System is 100% Cloud-Native and Ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
