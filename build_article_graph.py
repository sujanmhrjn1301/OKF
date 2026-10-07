"""
build_article_graph.py -- Build Article Cross-Reference Graph

Parses all OKF article markdown files and extracts inter-article references
(e.g., "Article 133", "pursuant to Article 144") to create a static adjacency
map stored as JSON.

Usage:
    python build_article_graph.py

Output:
    okf_output/article_graph.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OKF_DIR = Path(__file__).with_name("okf_output")
ARTICLES_DIR = OKF_DIR / "articles"
OUTPUT_PATH = OKF_DIR / "article_graph.json"

# Matches "Article 25", "Articles 133 or 144", "Article 133, 144 or 145"
ARTICLE_REF_RE = re.compile(
    r"(?:Article|Articles|Art\.)\s*(\d{1,3}(?:\s*(?:,|or|and|/)\s*\d{1,3})*)",
    re.IGNORECASE,
)
# Extract individual numbers from a matched group like "133, 144 or 145"
NUM_RE = re.compile(r"\d{1,3}")

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def extract_article_number(filepath: Path) -> int | None:
    """Extract the article number from the filename (e.g., article-025.md -> 25)."""
    m = re.search(r"article-(\d+)", filepath.stem)
    return int(m.group(1)) if m else None


def extract_body(text: str) -> str:
    """Strip YAML frontmatter and return only the body text."""
    m = FRONTMATTER_RE.match(text)
    return text[m.end():] if m else text


def find_references(body: str, self_article: int) -> list[int]:
    """Find all article numbers referenced in the body text, excluding self-references."""
    refs = set()
    for match in ARTICLE_REF_RE.finditer(body):
        nums = NUM_RE.findall(match.group(1))
        for n in nums:
            art_num = int(n)
            if art_num != self_article and 1 <= art_num <= 400:
                refs.add(art_num)
    return sorted(refs)


def build_graph() -> dict[str, list[int]]:
    """Build the full cross-reference adjacency map."""
    graph: dict[str, list[int]] = {}

    article_files = sorted(ARTICLES_DIR.glob("article-*.md"))
    print(f"[GRAPH] Scanning {len(article_files)} article files...")

    total_refs = 0
    for filepath in article_files:
        art_num = extract_article_number(filepath)
        if art_num is None:
            continue

        try:
            text = filepath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  WARNING: Could not read {filepath.name}: {e}")
            continue

        body = extract_body(text)
        refs = find_references(body, art_num)

        if refs:
            graph[str(art_num)] = refs
            total_refs += len(refs)

    print(f"[GRAPH] Found {total_refs} cross-references across {len(graph)} articles")
    return graph


def main():
    print("=" * 60)
    print("  Building Article Cross-Reference Graph")
    print("=" * 60)

    graph = build_graph()

    # Save to JSON
    OUTPUT_PATH.write_text(
        json.dumps(graph, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\n[GRAPH] Saved to {OUTPUT_PATH}")

    # Show a sample
    print("\n--- Sample Cross-References ---")
    sample_keys = list(graph.keys())[:10]
    for key in sample_keys:
        refs = graph[key]
        print(f"  Article {key} -> {refs}")

    print(f"\n[DONE] Total articles with outgoing refs: {len(graph)}")


if __name__ == "__main__":
    main()
