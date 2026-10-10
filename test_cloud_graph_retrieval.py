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

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
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
            assert count > 0, "No citation edges found in Supabase!"
    finally:
        conn.close()


def test_get_article_graph_from_cloud():
    print("\n--- Test 2: Verify get_article_graph() loads from Supabase ---")
    # Invalidate cache
    chat_cli._ARTICLE_GRAPH = None

    # Point local file path to a fake non-existent path to PROVE no disk reliance
    original_path = chat_cli.ARTICLE_GRAPH_PATH
    chat_cli.ARTICLE_GRAPH_PATH = Path("/non_existent_folder/fake_graph.json")

    try:
        graph = chat_cli.get_article_graph()
        print(f"[PASS] Successfully loaded {len(graph)} source articles from Supabase (Zero local files used).")
        assert len(graph) > 0, "Graph failed to load from Supabase!"
        assert "100" in graph, "Article 100 missing from cloud graph!"
        assert 76 in graph["100"], f"Expected Article 100 -> 76, got {graph['100']}"
        print(f"       Article 100 -> {graph['100']}")
        print(f"       Article 273 has {len(graph.get('273', []))} referenced fundamental rights.")
    finally:
        chat_cli.ARTICLE_GRAPH_PATH = original_path


def test_end_to_end_cross_ref_expansion():
    print("\n--- Test 3: Verify expand_with_cross_references() via Cloud ---")
    # Simulate candidates retrieved by vector search (e.g. Article 100 - Vote of Confidence)
    candidates = [
        {
            "id": 100,
            "document_id": "constitution",
            "unit_number": 100,
            "title": "Article 100 -- Provisions relating to Vote of Confidence and Motion of No Confidence",
            "content": "Sample content of Article 100...",
            "rrf_score": 0.05,
            "retrieval_count": 2,
        }
    ]

    intent = {"intent": "CONSTITUTIONAL_DIRECT"}

    # Run expansion
    expanded = chat_cli.expand_with_cross_references(candidates, intent)

    print(f"[PASS] Original candidates: 1, Expanded total candidates: {len(expanded)}")
    cross_refs = [c for c in expanded if c.get("is_cross_ref")]
    print(f"[PASS] Cross-referenced candidates injected: {len(cross_refs)}")
    for cr in cross_refs:
        print(f"       Injected: {cr.get('document_id')} Unit #{cr.get('unit_number')} - {cr.get('title')}")

    # Check if Article 76 was pulled in from Supabase
    target_units = [cr.get("unit_number") for cr in cross_refs]
    assert 76 in target_units, f"Expected Article 76 to be injected via cross-ref, got {target_units}"
    print("\n[VERIFIED] Article 76 was successfully pulled from Supabase via cloud graph expansion!")


def main():
    print("=" * 60)
    print("  Testing Cloud-Native Legal Knowledge Graph")
    print("=" * 60)

    test_cloud_edges_exist()
    test_get_article_graph_from_cloud()
    test_end_to_end_cross_ref_expansion()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED: System is 100% Cloud-Native and Ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
