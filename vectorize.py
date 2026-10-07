"""
vectorize.py -- Load OKF bundle into Supabase (pgvector) for semantic retrieval

Reads  :  data/okf/<source>/  (OKF Markdown files with YAML frontmatter)
Writes :  Supabase table `knowledge.okf_documents` with vector embeddings

Usage:
  python vectorize.py --source constitution              # ingest constitution
  python vectorize.py --source penal_code                # ingest penal code
  python vectorize.py --source all                       # ingest all sources
  python vectorize.py --query "right to education"       # search all
  python vectorize.py --query "definitions" --source penal_code  # filtered search
  python vectorize.py --setup-only                       # just create table
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("ERROR: python-dotenv required.  pip install python-dotenv")

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit("ERROR: psycopg2 required.  pip install psycopg2-binary")

try:
    from supabase import create_client, Client
except ImportError:
    sys.exit("ERROR: supabase required.  pip install supabase")

# -- Configuration -----------------------------------------------------------
load_dotenv(Path(__file__).with_name(".env"))

hf_token = os.getenv("HF_TOKEN", "")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
DB_HOST = os.getenv("SUPABASE_DB_HOST", "")
DB_PORT = os.getenv("SUPABASE_DB_PORT", "5432")
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TABLE_NAME = "knowledge.okf_documents"
BATCH_SIZE = 50

# -- Source Registry ----------------------------------------------------------
# Maps source keys to their OKF directory and human-readable title.
# Add new entries here when you ingest a new Act/law.
SOURCE_CONFIGS = {
    "constitution": {
        "dir": BASE_DIR / "data" / "okf" / "constitution",
        "title": "Constitution of Nepal, 2015",
    },
    "penal_code": {
        "dir": BASE_DIR / "data" / "okf" / "penal_code",
        "title": "The National Criminal Procedure (Code) Act, 2017",
    },
}

def get_okf_dir(source: str) -> Path:
    """Resolve the OKF directory for a given source key."""
    if source not in SOURCE_CONFIGS:
        sys.exit(f"ERROR: Unknown source '{source}'. Available: {list(SOURCE_CONFIGS.keys())}")
    d = SOURCE_CONFIGS[source]["dir"]
    if not d.exists():
        sys.exit(f"ERROR: OKF directory not found: {d}")
    return d


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


# -- Parsing ------------------------------------------------------------------
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
    """Split a document into chunks for embedding.

    Strategy: split by clause/section headings, include title for context.
    """
    chunks: list[str] = []
    content = doc.content
    prefix = f"{doc.title}\n{doc.description}\n"

    sections = re.split(r"(?=^#{2,4}\s)", content, flags=re.MULTILINE)

    current_chunk = prefix
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(current_chunk) + len(section) > max_chunk_size and current_chunk != prefix:
            chunks.append(current_chunk.strip())
            current_chunk = prefix + section
        else:
            current_chunk += "\n\n" + section

    if current_chunk.strip() and current_chunk.strip() != prefix.strip():
        chunks.append(current_chunk.strip())

    if not chunks:
        chunks = [prefix + "\n\n" + content[:max_chunk_size]]

    return chunks


# -- Database Setup (direct psycopg2) ----------------------------------------
def get_db_connection():
    """Connect directly to the Supabase Postgres database."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=int(DB_PORT),
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
        connect_timeout=10,
    )
    conn.autocommit = True
    return conn


def setup_database() -> None:
    """Create the table, indexes, and search function via direct SQL."""
    print("[DB] Connecting to Supabase Postgres directly...")
    conn = get_db_connection()
    cur = conn.cursor()
    print("[DB] Connected. Setting up schema...")

    statements = [
        # Enable pgvector
        "CREATE EXTENSION IF NOT EXISTS vector;",

        "CREATE SCHEMA IF NOT EXISTS knowledge;",
        # Create table
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id BIGSERIAL PRIMARY KEY,
            file_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            doc_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            content TEXT NOT NULL,
            part_number INTEGER,
            part_title TEXT,
            article_number INTEGER,
            page INTEGER,
            clause_count INTEGER DEFAULT 0,
            sub_clause_count INTEGER DEFAULT 0,
            proviso_count INTEGER DEFAULT 0,
            tags TEXT[],
            metadata JSONB,
            embedding VECTOR({EMBEDDING_DIM}),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,

        # Indexes for metadata filtering
        f"CREATE INDEX IF NOT EXISTS idx_okf_doc_type ON {TABLE_NAME}(doc_type);",
        f"CREATE INDEX IF NOT EXISTS idx_okf_part ON {TABLE_NAME}(part_number);",
        f"CREATE INDEX IF NOT EXISTS idx_okf_article ON {TABLE_NAME}(article_number);",
        f"CREATE INDEX IF NOT EXISTS idx_okf_page ON {TABLE_NAME}(page);",

        # Search function: hybrid (cosine similarity + metadata filters)
        f"""
        CREATE OR REPLACE FUNCTION knowledge.match_okf_documents(
            query_embedding VECTOR({EMBEDDING_DIM}),
            match_threshold FLOAT DEFAULT 0.3,
            match_count INT DEFAULT 10,
            filter_part_number INT DEFAULT NULL,
            filter_doc_type TEXT DEFAULT NULL
        )
        RETURNS TABLE (
            id BIGINT,
            file_path TEXT,
            chunk_index INTEGER,
            doc_type TEXT,
            title TEXT,
            content TEXT,
            part_number INTEGER,
            article_number INTEGER,
            page INTEGER,
            clause_count INTEGER,
            sub_clause_count INTEGER,
            proviso_count INTEGER,
            tags TEXT[],
            similarity FLOAT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                d.id,
                d.file_path,
                d.chunk_index,
                d.doc_type,
                d.title,
                d.content,
                d.part_number,
                d.article_number,
                d.page,
                d.clause_count,
                d.sub_clause_count,
                d.proviso_count,
                d.tags,
                (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity
            FROM {TABLE_NAME} d
            WHERE
                (1 - (d.embedding <=> query_embedding)) > match_threshold
                AND (filter_part_number IS NULL OR d.part_number = filter_part_number)
                AND (filter_doc_type IS NULL OR d.doc_type = filter_doc_type)
            ORDER BY d.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        """,
    ]

    for i, stmt in enumerate(statements, 1):
        try:
            cur.execute(stmt)
            print(f"  [{i}/{len(statements)}] OK")
        except Exception as e:
            print(f"  [{i}/{len(statements)}] WARNING: {e}")

    cur.close()
    conn.close()
    print("[DB] Schema setup complete.\n")


# -- Embedding ----------------------------------------------------------------
def load_model():
    print(f"[MODEL] Loading {MODEL_NAME}...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("ERROR: sentence-transformers required.  pip install sentence-transformers")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[MODEL] Loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(model, chunks: list[str]) -> list[list[float]]:
    embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


# -- Ingest Pipeline ----------------------------------------------------------
def collect_okf_files(okf_dir: Path) -> list[Path]:
    return sorted(okf_dir.rglob("*.md"))


def ingest(model, source: str) -> None:
    """Full pipeline: parse -> chunk -> embed -> insert via psycopg2.
    
    Only deletes and re-inserts data for the specified source,
    leaving other sources untouched.
    """
    source_config = SOURCE_CONFIGS[source]
    okf_dir = source_config["dir"]
    source_title = source_config["title"]

    print(f"\n[INGEST] Source: {source} ({source_title})")
    print(f"[INGEST] OKF Dir: {okf_dir}")

    files = collect_okf_files(okf_dir)
    print(f"[INGEST] Found {len(files)} OKF files")

    # Parse
    docs: list[OKFDocument] = []
    for f in files:
        doc = parse_okf_file(f, okf_dir)
        if doc:
            doc.chunks = chunk_document(doc)
            docs.append(doc)
    print(f"[INGEST] Parsed {len(docs)} documents")

    total_chunks = sum(len(d.chunks) for d in docs)
    print(f"[INGEST] Total chunks to embed: {total_chunks}")

    # Collect all chunks across all documents
    all_chunks_info = []
    chunk_texts = []
    for doc in docs:
        meta = doc.metadata
        for chunk_idx, chunk_text in enumerate(doc.chunks):
            chunk_texts.append(chunk_text)
            all_chunks_info.append((doc, chunk_idx, chunk_text, meta))

    print(f"[INGEST] Encoding {len(chunk_texts)} chunks with {MODEL_NAME}...")
    embeddings = model.encode(chunk_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # Connect
    print("[INGEST] Connecting to database...")
    conn = get_db_connection()
    cur = conn.cursor()

    # Clear ONLY this source's data (preserve other sources)
    print(f"[INGEST] Clearing existing '{source}' data only...")
    cur.execute(f"DELETE FROM {TABLE_NAME} WHERE source_document = %s;", (source,))

    # Prepare batch rows
    print("[INGEST] Inserting into database in batches...")
    insert_sql = f"""
        INSERT INTO {TABLE_NAME}
            (source_document, source_title, file_path, chunk_index,
             doc_type, title, description, content,
             part_number, part_title, article_number,
             section_number, chapter_number, page,
             clause_count, sub_clause_count, proviso_count, tags,
             metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    rows = []
    for (doc, chunk_idx, chunk_text, meta), emb in zip(all_chunks_info, embeddings):
        rows.append((
            source,
            source_title,
            doc.file_path,
            chunk_idx,
            doc.doc_type,
            doc.title,
            doc.description or "",
            chunk_text,
            meta.get("part_number"),
            meta.get("part_title", ""),
            meta.get("article_number"),
            meta.get("section") or meta.get("section_number"),
            meta.get("chapter_number"),
            meta.get("page"),
            meta.get("clause_count", 0),
            meta.get("sub_clause_count", 0) or meta.get("sub_sections_count", 0),
            meta.get("proviso_count", 0),
            meta.get("tags", []),
            json.dumps(meta, default=str),
            str(emb.tolist() if hasattr(emb, "tolist") else emb),
        ))

    psycopg2.extras.execute_batch(cur, insert_sql, rows, page_size=100)
    rows_inserted = len(rows)

    # Create vector index (HNSW for best similarity search performance)
    print("[INGEST] Creating vector index...")
    try:
        cur.execute("DROP INDEX IF EXISTS idx_okf_documents_embedding;")
        cur.execute(
            f"""
            CREATE INDEX idx_okf_documents_embedding ON {TABLE_NAME}
            USING hnsw (embedding vector_cosine_ops);
            """
        )
        print("  HNSW vector index created.")
    except Exception as e:
        print(f"  Note on vector index: {e}")

    cur.close()
    conn.close()

    print(f"\n[INGEST] Done! Inserted {rows_inserted} chunks for '{source}' into '{TABLE_NAME}'")


# -- Search -------------------------------------------------------------------
def search(
    model,
    query: str,
    top_k: int = 5,
    part_number: int | None = None,
    source: str | None = None,
) -> None:
    """Semantic search over the OKF vector store."""
    print(f'\nSearching: "{query}"')
    if source:
        print(f"  Filtering by source: {source}")
    if part_number:
        print(f"  Filtering by Part {part_number}")
    print("-" * 60)

    # Embed query
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    # Search via direct SQL (more reliable than RPC for complex types)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    sql = f"""
        SELECT
            id, source_document, source_title, file_path, chunk_index,
            doc_type, title, content,
            part_number, article_number, section_number, chapter_number,
            page, clause_count, sub_clause_count, proviso_count, tags,
            (1 - (embedding <=> %s::vector)) AS similarity
        FROM {TABLE_NAME}
        WHERE (1 - (embedding <=> %s::vector)) > 0.3
    """
    params: list = [str(query_embedding), str(query_embedding)]

    if source is not None:
        sql += " AND source_document = %s"
        params.append(source)

    if part_number is not None:
        sql += " AND part_number = %s"
        params.append(part_number)

    sql += f" ORDER BY embedding <=> %s::vector LIMIT %s"
    params.extend([str(query_embedding), top_k])

    cur.execute(sql, params)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        print("No results found.")
        return

    for i, row in enumerate(rows, 1):
        sim = row["similarity"]
        title = row["title"]
        page = row["page"]
        doc_type = row["doc_type"]
        src = row["source_document"]
        art_num = row["article_number"]
        sec_num = row["section_number"]
        clause_count = row["clause_count"]
        content_preview = row["content"][:250].replace("\n", " ")

        print(f"\n  [{i}] {title}")
        print(f"      Source: {src} | Similarity: {sim:.4f} | Type: {doc_type} | Page: {page}")
        if art_num:
            print(f"      Article: {art_num} | Clauses: {clause_count}")
        if sec_num:
            print(f"      Section: {sec_num} | Chapter: {row['chapter_number']}")
        print(f"      Preview: {content_preview}...")

    print()


# -- Main ---------------------------------------------------------------------
def main() -> None:
    import argparse

    available_sources = list(SOURCE_CONFIGS.keys())

    parser = argparse.ArgumentParser(description="OKF -> Supabase Vector Store (Multi-Source)")
    parser.add_argument(
        "--source", "-s", type=str, default=None,
        help=f"Source to ingest or search. Options: {available_sources + ['all']}. "
             f"Required for ingest, optional for search (NULL = search all)."
    )
    parser.add_argument("--query", "-q", type=str, help="Search query (skip ingest)")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--part", "-p", type=int, help="Filter by part number")
    parser.add_argument("--setup-only", action="store_true", help="Only create DB schema")
    args = parser.parse_args()

    print("=" * 60)
    print("  OKF -> Supabase pgvector (Multi-Source)")
    print("=" * 60)
    print(f"  Available sources: {available_sources}")

    if args.query:
        model = load_model()
        search(model, args.query, top_k=args.top_k, part_number=args.part, source=args.source)
    elif args.setup_only:
        setup_database()
    else:
        # Ingest mode: --source is required
        if not args.source:
            sys.exit("ERROR: --source is required for ingest. Use: --source constitution, --source penal_code, or --source all")

        setup_database()
        model = load_model()

        if args.source == "all":
            for src_key in available_sources:
                ingest(model, src_key)
        else:
            if args.source not in SOURCE_CONFIGS:
                sys.exit(f"ERROR: Unknown source '{args.source}'. Available: {available_sources}")
            ingest(model, args.source)

        # Demo search across all sources
        print("\n" + "=" * 60)
        print("  Demo Searches (all sources)")
        print("=" * 60)
        search(model, "definitions", top_k=3)


if __name__ == "__main__":
    main()
