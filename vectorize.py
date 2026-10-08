"""
vectorize.py -- Load OKF legal bundles into Supabase (Two-Tier pgvector store)

Architecture:
  - knowledge.documents       : Document-level registry (Constitution, Criminal Procedure Code, future Acts)
  - knowledge.document_chunks : Chunk-level vectors (1536-dim via OpenRouter openai/text-embedding-3-small)
  - Fast, lightweight, zero-PyTorch dependency for instant deployment on Render.

Usage:
  python vectorize.py --source constitution              # ingest constitution
  python vectorize.py --source criminal_procedure        # ingest criminal procedure code
  python vectorize.py --source all                       # ingest all sources
  python vectorize.py --query "right to education"       # cross-statute search
  python vectorize.py --query "investigation" --source criminal_procedure  # filtered search
  python vectorize.py --setup-only                       # only run DB schema setup
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
import requests
import yaml
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

# -- Configuration -----------------------------------------------------------
load_dotenv(Path(__file__).with_name(".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL_NAME = "openai/text-embedding-3-small"
EMBEDDING_DIM = 1536

DB_HOST = os.getenv("SUPABASE_DB_HOST", "")
DB_PORT = os.getenv("SUPABASE_DB_PORT", "5432")
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")

BASE_DIR = Path(__file__).resolve().parent
DOCS_TABLE = "knowledge.documents"
CHUNKS_TABLE = "knowledge.document_chunks"
BATCH_SIZE = 40

# -- Source Registry ----------------------------------------------------------
# Registry of legal materials. To add a new law in the future, simply add an entry here!
SOURCE_CONFIGS: dict[str, dict[str, Any]] = {
    "constitution": {
        "id": "constitution",
        "title": "Constitution of Nepal, 2015",
        "short_title": "Constitution",
        "doc_category": "constitution",
        "status": "active",
        "year_bs": 2072,
        "year_ad": 2015,
        "act_number": "Constitution 2072",
        "jurisdiction": "Nepal",
        "language": "en",
        "tags": ["constitutional_law", "fundamental_rights", "state_structure", "nepal"],
        "dir": BASE_DIR / "data" / "okf" / "constitution",
    },
    "criminal_procedure": {
        "id": "criminal_procedure",
        "title": "The National Criminal Procedure (Code) Act, 2017",
        "short_title": "National Criminal Procedure Code",
        "doc_category": "code",
        "status": "active",
        "year_bs": 2074,
        "year_ad": 2017,
        "act_number": "Act No. 36 of 2074",
        "jurisdiction": "Nepal",
        "language": "en",
        "tags": ["criminal_procedure", "investigation", "bail", "trial", "nepal_law"],
        "dir": BASE_DIR / "data" / "okf" / "criminal_procedure",
    },
}


# -- Data Model --------------------------------------------------------------
@dataclass
class OKFDocument:
    file_path: str
    doc_type: str
    title: str
    description: str
    content: str
    metadata: dict
    chunks: list[str]


# -- Parsing & Chunking -------------------------------------------------------
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_okf_file(filepath: Path, okf_dir: Path | None = None) -> OKFDocument | None:
    base = okf_dir or filepath.parent
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  WARNING: Could not read {filepath}: {e}")
        return None

    m = FRONTMATTER_RE.match(text)
    if not m:
        return None

    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}

    body = text[m.end():].strip()

    return OKFDocument(
        file_path=str(filepath.relative_to(base)),
        doc_type=meta.get("type", "unknown"),
        title=meta.get("title", filepath.stem),
        description=meta.get("description", ""),
        content=body,
        metadata=meta,
        chunks=[],
    )


def chunk_document(doc: OKFDocument, max_chunk_size: int = 800) -> list[str]:
    """Split a document into logical chunks for embedding."""
    chunks: list[str] = []
    content = doc.content
    prefix = f"{doc.title}\n{doc.description}\n".strip()

    sections = re.split(r"(?=^#{2,4}\s)", content, flags=re.MULTILINE)

    current_chunk = prefix
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(current_chunk) + len(section) > max_chunk_size and current_chunk != prefix:
            chunks.append(current_chunk.strip())
            current_chunk = f"{prefix}\n\n{section}" if prefix else section
        else:
            current_chunk += "\n\n" + section

    if current_chunk.strip() and current_chunk.strip() != prefix:
        chunks.append(current_chunk.strip())

    if not chunks:
        chunks = [f"{prefix}\n\n{content[:max_chunk_size]}".strip()]

    return chunks


# -- Database Helpers ---------------------------------------------------------
def get_db_connection():
    """Connect to Supabase Postgres directly with SSL."""
    return psycopg2.connect(
        host=DB_HOST,
        port=int(DB_PORT),
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
        connect_timeout=15,
    )


def setup_database() -> None:
    """Ensure two-tier schema and search functions exist."""
    print("[DB] Verifying/creating schema in Supabase...")
    from setup_industry_standard_db import SETUP_SQL

    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(SETUP_SQL)
    cur.close()
    conn.close()
    print("[DB] Schema and HNSW index ready.\n")


# -- Embeddings via OpenRouter (Lightweight, No PyTorch) ----------------------
def embed_chunks(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """Generate 1536-dim embeddings using OpenRouter API."""
    if not OPENROUTER_API_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY missing in .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[list[float]] = []
    total = len(texts)
    print(f"[EMBED] Embedding {total} chunks via OpenRouter ({MODEL_NAME})...")

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "model": MODEL_NAME,
            "input": batch,
        }

        # Retry logic with backoff
        for attempt in range(3):
            try:
                resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    # Sort by index to preserve exact order
                    data_sorted = sorted(data, key=lambda x: x["index"])
                    all_embeddings.extend([item["embedding"] for item in data_sorted])
                    done = min(i + batch_size, total)
                    print(f"  [{done}/{total}] chunks embedded...")
                    break
                elif resp.status_code == 429:
                    print(f"  Rate limited, waiting 3s (attempt {attempt+1})...")
                    time.sleep(3)
                else:
                    raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Request error ({e}), retrying...")
                time.sleep(2)

    return all_embeddings


# -- Ingestion Pipeline -------------------------------------------------------
def extract_hierarchy_info(doc: OKFDocument, meta: dict, source_id: str) -> tuple[str, int | None, int | None, str]:
    """Extract (structural_ref, unit_number, parent_number, parent_title) cleanly."""
    doc_type = doc.doc_type

    if source_id == "constitution":
        art_num = meta.get("article_number")
        part_num = meta.get("part_number")
        part_title = meta.get("part_title", "")
        if art_num:
            structural_ref = f"Article {art_num}"
            unit_number = int(art_num)
        elif "Schedule" in doc.title:
            structural_ref = doc.title
            unit_number = None
        else:
            structural_ref = doc.title
            unit_number = None
        parent_number = int(part_num) if part_num is not None else None
        return structural_ref, unit_number, parent_number, part_title

    else:  # Acts / Codes (Criminal Procedure Code)
        sec_num = meta.get("section") or meta.get("section_number")
        chap_num = meta.get("chapter_number")
        chap_title = meta.get("chapter_title", "")
        if sec_num:
            structural_ref = f"Section {sec_num}"
            unit_number = int(sec_num)
        elif "Chapter" in doc.title:
            structural_ref = doc.title
            unit_number = None
        elif "Schedule" in doc.title:
            structural_ref = doc.title
            unit_number = None
        else:
            structural_ref = doc.title
            unit_number = None
        parent_number = int(chap_num) if chap_num is not None else None
        return structural_ref, unit_number, parent_number, chap_title


def ingest(source: str) -> None:
    """Full pipeline: parse OKF -> embed via OpenRouter -> insert into 2-tier Supabase schema."""
    if source not in SOURCE_CONFIGS:
        sys.exit(f"ERROR: Unknown source '{source}'. Available: {list(SOURCE_CONFIGS.keys())}")

    config = SOURCE_CONFIGS[source]
    okf_dir = config["dir"]
    doc_id = config["id"]
    title = config["title"]

    print(f"\n" + "=" * 65)
    print(f"[INGEST] Source: {doc_id} -> {title}")
    print(f"[INGEST] Directory: {okf_dir}")
    print("=" * 65)

    files = sorted(okf_dir.rglob("*.md"))
    print(f"[INGEST] Found {len(files)} OKF Markdown files")

    docs: list[OKFDocument] = []
    for f in files:
        doc = parse_okf_file(f, okf_dir)
        if doc:
            doc.chunks = chunk_document(doc)
            docs.append(doc)

    print(f"[INGEST] Parsed {len(docs)} documents")
    total_chunks = sum(len(d.chunks) for d in docs)
    print(f"[INGEST] Total chunks to embed: {total_chunks}")

    # Prepare chunks and metadata
    all_chunks_info = []
    chunk_texts = []
    for doc in docs:
        meta = doc.metadata
        ref, unit_no, parent_no, parent_title = extract_hierarchy_info(doc, meta, doc_id)
        for chunk_idx, chunk_text in enumerate(doc.chunks):
            chunk_texts.append(chunk_text)
            all_chunks_info.append({
                "doc": doc,
                "chunk_idx": chunk_idx,
                "chunk_text": chunk_text,
                "meta": meta,
                "ref": ref,
                "unit_no": unit_no,
                "parent_no": parent_no,
                "parent_title": parent_title,
            })

    # Generate embeddings via OpenRouter
    embeddings = embed_chunks(chunk_texts)

    # Database insertion
    print("\n[DB] Connecting to Supabase...")
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    # 1. Clean previous data for this document (Cascades automatically to chunks!)
    print(f"[DB] Cleaning previous data for '{doc_id}'...")
    cur.execute(f"DELETE FROM {DOCS_TABLE} WHERE id = %s;", (doc_id,))

    # 2. Register Document in knowledge.documents
    print(f"[DB] Registering document in {DOCS_TABLE}...")
    insert_doc_sql = f"""
        INSERT INTO {DOCS_TABLE} (
            id, title, short_title, doc_category, status,
            year_bs, year_ad, act_number, jurisdiction, language,
            total_units, total_chunks, tags, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    cur.execute(insert_doc_sql, (
        config["id"],
        config["title"],
        config["short_title"],
        config["doc_category"],
        config["status"],
        config["year_bs"],
        config["year_ad"],
        config["act_number"],
        config["jurisdiction"],
        config["language"],
        len(docs),
        total_chunks,
        config["tags"],
        json.dumps(config, default=str),
    ))

    # 3. Insert Chunks in batches
    print(f"[DB] Inserting {len(all_chunks_info)} chunks into {CHUNKS_TABLE}...")
    insert_chunk_sql = f"""
        INSERT INTO {CHUNKS_TABLE} (
            document_id, file_path, chunk_index, doc_type,
            structural_ref, unit_number, parent_number, parent_title,
            title, description, content, page,
            clause_count, sub_clause_count, proviso_count, tags,
            metadata, embedding
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s
        );
    """

    rows = []
    for item, emb in zip(all_chunks_info, embeddings):
        d = item["doc"]
        m = item["meta"]
        rows.append((
            doc_id,
            d.file_path,
            item["chunk_idx"],
            d.doc_type,
            item["ref"],
            item["unit_no"],
            item["parent_no"],
            item["parent_title"],
            d.title,
            d.description or "",
            item["chunk_text"],
            m.get("page"),
            m.get("clause_count", 0),
            m.get("sub_clause_count", 0) or m.get("sub_sections_count", 0),
            m.get("proviso_count", 0),
            m.get("tags", []),
            json.dumps(m, default=str),
            str(emb),
        ))

    psycopg2.extras.execute_batch(cur, insert_chunk_sql, rows, page_size=100)
    print(f"[INGEST] Successfully ingested '{doc_id}' ({len(rows)} chunks stored).")

    cur.close()
    conn.close()


# -- Search -------------------------------------------------------------------
def search(
    query: str,
    top_k: int = 5,
    source: str | None = None,
    category: str | None = None,
    unit_number: int | None = None,
) -> None:
    """Semantic search over the Two-Tier OKF vector store."""
    print(f'\nSearching: "{query}"')
    if source:
        print(f"  Filtering by document: {source}")
    if category:
        print(f"  Filtering by category: {category}")
    if unit_number:
        print(f"  Filtering by unit/section #: {unit_number}")
    print("-" * 65)

    # Embed query
    query_emb = embed_chunks([query])[0]

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Use unified RPC function
    cur.execute(
        """
        SELECT * FROM knowledge.match_documents(
            query_embedding := %s::vector,
            match_threshold := 0.20,
            match_count := %s,
            filter_document_id := %s,
            filter_doc_category := %s,
            filter_unit_number := %s
        );
        """,
        (str(query_emb), top_k, source, category, unit_number),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        print("No matching results found.")
        return

    for i, r in enumerate(rows, 1):
        sim = r["similarity"]
        doc_title = r["document_title"]
        s_ref = r["structural_ref"] or r["title"]
        content_preview = r["content"][:220].replace("\n", " ")

        print(f"\n  [{i}] {s_ref} | {doc_title}")
        print(f"      Similarity: {sim:.4f} | Category: {r['doc_category']} | Type: {r['doc_type']} | Page: {r['page']}")
        if r["parent_title"]:
            print(f"      Chapter/Part: {r['parent_title']} (No. {r['parent_number']})")
        print(f"      Snippet: {content_preview}...")

    print()


# -- CLI ----------------------------------------------------------------------
def main() -> None:
    available_sources = list(SOURCE_CONFIGS.keys())

    parser = argparse.ArgumentParser(description="OKF -> Supabase Two-Tier Vector Store")
    parser.add_argument(
        "--source", "-s", type=str, default=None,
        help=f"Source to ingest. Options: {available_sources + ['all']}.",
    )
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--unit", "-u", type=int, help="Filter by section/article number")
    parser.add_argument("--category", "-c", type=str, help="Filter by category (constitution, code, act)")
    parser.add_argument("--setup-only", action="store_true", help="Only setup database schema")
    args = parser.parse_args()

    print("=" * 65)
    print("  OKF Knowledge Store (Two-Tier pgvector + OpenRouter)")
    print("=" * 65)

    if args.setup_only:
        setup_database()
        return

    if args.query:
        search(args.query, top_k=args.top_k, source=args.source, category=args.category, unit_number=args.unit)
        return

    if not args.source:
        sys.exit("ERROR: --source required for ingest. Use: --source constitution, --source criminal_procedure, or --source all")

    # Ingest
    if args.source == "all":
        for src_key in available_sources:
            ingest(src_key)
    else:
        ingest(args.source)

    # Quick test search
    print("\n" + "=" * 65)
    print("  Cross-Statute Test Search: 'arrest without warrant'")
    print("=" * 65)
    search("arrest without warrant", top_k=3)


if __name__ == "__main__":
    main()
