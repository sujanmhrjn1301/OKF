"""
build_article_graph.py -- Build Article Cross-Reference Graph (Constitution of Nepal)

Parses all OKF article markdown files and extracts multi-dimensional legal links:
  1. articles         -- other Articles cited (e.g. "Article 133", "Articles 133 or 144")
  2. article_clauses  -- pinpoint citations (e.g. "clause (1) of Article 134" -> "134(1)")
  3. clauses          -- internal clauses of the same Article (e.g. "pursuant to clause (4)")
  4. sub_clauses      -- internal sub-clauses (e.g. "sub-clause (a)")
  5. schedules        -- Constitutional Schedules (e.g. "Schedule-5")
  6. parts            -- Constitutional Parts (e.g. "Part 3")

Usage:
    python ingestion/build_article_graph.py

Output:
    data/okf/constitution/article_graph.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE_PROJECT_DIR = Path(__file__).resolve().parent.parent
OKF_DIR = BASE_PROJECT_DIR / "data" / "okf" / "constitution"
ARTICLES_DIR = OKF_DIR / "articles"
OUTPUT_PATH = OKF_DIR / "article_graph.json"

MAX_ARTICLE = 308
MAX_SCHEDULE = 9
MAX_PART = 35

# Matches "Article 25", "Articles 133 or 144", "Article 133, 144 or 145"
ARTICLE_REF_RE = re.compile(
    r"(?:Article|Articles|Art\.)\s*(\d{1,3}(?:\s*(?:,|or|and|/)\s*\d{1,3})*)",
    re.IGNORECASE,
)
# "clause (1) of Article 134", "clauses (1) and (2) of Article 134"
ARTICLE_CLAUSE_RE = re.compile(
    r"clauses?\s*((?:\(\d{1,2}\)\s*(?:,|or|and)?\s*)+)of\s+Article\s+(\d{1,3})",
    re.IGNORECASE,
)
CLAUSE_RE = re.compile(
    r"(?<!sub-)(?<!sub\s)clauses?\s*((?:\(\d{1,2}\)\s*(?:,|or|and)?\s*)+)",
    re.IGNORECASE,
)
SUB_CLAUSE_RE = re.compile(r"sub-?\s*clauses?\s*\(([a-z])\)", re.IGNORECASE)
SCHEDULE_RE = re.compile(
    r"(?:Schedule|Schedules)\s*-?\s*(\d{1,2}(?:\s*(?:,|or|and|/)\s*\d{1,2})*)",
    re.IGNORECASE,
)
PART_RE = re.compile(r"\bParts?\s*-?\s*(\d{1,2})\b", re.IGNORECASE)
NUM_RE = re.compile(r"\d{1,3}")

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def extract_article_number(filepath: Path) -> int | None:
    """Extract the article number from the filename (e.g., article-025.md -> 25)."""
    m = re.search(r"article-(\d+)", filepath.stem)
    return int(m.group(1)) if m else None


def extract_body(text: str) -> str:
    """Strip YAML frontmatter and structural markdown (titles, headers, clause index)."""
    m = FRONTMATTER_RE.match(text)
    body = text[m.end():] if m else text
    body = re.sub(r"^\s*#\s*Article\s*\d+.*?\n", "", body, flags=re.IGNORECASE)
    body = re.sub(r">\s*\*\*Part\s*\d+:.*?\n", "", body, flags=re.IGNORECASE)   # "> **Part 3:** ..." header
    body = re.sub(r">\s*\*\*Page:.*?\n", "", body, flags=re.IGNORECASE)
    body = re.sub(r"##\s*Clause Index.*", "", body, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"###\s*Clause\s*\(\d+\)", "", body, flags=re.IGNORECASE)
    body = re.sub(r"####\s*Proviso.*?\n", "", body, flags=re.IGNORECASE)
    body = re.sub(r"####\s*Sub-clause.*?\n", "", body, flags=re.IGNORECASE)
    body = re.sub(r"^\s*-\s*\*\*\s*(?:Sub-)?Clause\s*\([^)]*\)\s*\*\*\s*:?", "", body, flags=re.MULTILINE | re.IGNORECASE)
    # De-hyphenate PDF line breaks ("Schedule-\n5", "sub-\nclause")
    body = re.sub(r"Schedule-\s*\n\s*(\d+)", r"Schedule-\1", body, flags=re.IGNORECASE)
    body = re.sub(r"sub-\s*\n\s*clause", "sub-clause", body, flags=re.IGNORECASE)
    return body


def find_references(body: str, self_article: int) -> dict[str, list]:
    """Extract all link types from an Article body, excluding self-references."""
    # 1. Pinpoint "clause (x) of Article N" (captured first, then removed so the
    #    clause numbers aren't mistaken for this Article's own clauses)
    article_clauses: set[str] = set()
    pinpoint_articles: set[int] = set()
    for m in ARTICLE_CLAUSE_RE.finditer(body):
        art = int(m.group(2))
        if art == self_article or not (1 <= art <= MAX_ARTICLE):
            continue
        pinpoint_articles.add(art)
        for c in re.findall(r"\d{1,2}", m.group(1)):
            article_clauses.add(f"{art}({int(c)})")
    internal_body = ARTICLE_CLAUSE_RE.sub(" ", body)

    # 2. Articles
    articles = set(pinpoint_articles)
    for m in ARTICLE_REF_RE.finditer(body):
        for n in NUM_RE.findall(m.group(1)):
            art = int(n)
            if art != self_article and 1 <= art <= MAX_ARTICLE:
                articles.add(art)

    # 3. Internal clauses / sub-clauses
    clauses = {int(n) for grp in CLAUSE_RE.findall(internal_body) for n in re.findall(r"\d{1,2}", grp)}
    sub_clauses = {s.lower() for s in SUB_CLAUSE_RE.findall(internal_body)}

    # 4. Schedules
    schedules = {
        int(n) for m in SCHEDULE_RE.finditer(body) for n in NUM_RE.findall(m.group(1))
        if 1 <= int(n) <= MAX_SCHEDULE
    }

    # 5. Parts
    parts = {int(n) for n in PART_RE.findall(body) if 1 <= int(n) <= MAX_PART}

    result: dict[str, list] = {}
    if articles:
        result["articles"] = sorted(articles)
    if article_clauses:
        result["article_clauses"] = sorted(article_clauses, key=lambda s: (int(s.split("(")[0]), s))
    if clauses:
        result["clauses"] = sorted(clauses)
    if sub_clauses:
        result["sub_clauses"] = sorted(sub_clauses)
    if schedules:
        result["schedules"] = sorted(schedules)
    if parts:
        result["parts"] = sorted(parts)
    return result


def build_graph() -> dict[str, dict[str, list]]:
    """Build the full multi-dimensional cross-reference map."""
    graph: dict[str, dict[str, list]] = {}

    article_files = sorted(ARTICLES_DIR.glob("article-*.md"))
    print(f"[GRAPH] Scanning {len(article_files)} article files...")

    totals = {k: 0 for k in ("articles", "article_clauses", "clauses", "sub_clauses", "schedules", "parts")}
    for filepath in article_files:
        art_num = extract_article_number(filepath)
        if art_num is None:
            continue

        try:
            text = filepath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  WARNING: Could not read {filepath.name}: {e}")
            continue

        refs = find_references(extract_body(text), art_num)
        if refs:
            graph[str(art_num)] = refs
            for k, v in refs.items():
                totals[k] += len(v)

    print(f"[GRAPH] Articles with references: {len(graph)} / {len(article_files)}")
    for k, v in totals.items():
        print(f"   * {k:<16} {v}")
    return graph


def main():
    print("=" * 60)
    print("  Building Article Cross-Reference Graph (Constitution)")
    print("=" * 60)

    graph = build_graph()

    OUTPUT_PATH.write_text(
        json.dumps(graph, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\n[GRAPH] Saved to {OUTPUT_PATH}")

    print("\n--- Key Verification Samples ---")
    for key in ["18", "46", "57", "100", "109", "273"]:
        if key in graph:
            print(f"  Article {key}: {json.dumps(graph[key])}")

    print(f"\n[DONE] Total articles with outgoing refs: {len(graph)}")


if __name__ == "__main__":
    main()
