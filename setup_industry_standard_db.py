"""
setup_industry_standard_db.py -- Setup Two-Tier Document Registry & Vector Chunks in Supabase

Architecture:
  - knowledge.documents       : Document-level registry (Constitution, Criminal Procedure Code, future Acts)
  - knowledge.document_chunks : Vector chunk table with 1536-dim embeddings (OpenAI text-embedding-3-small via OpenRouter)
  - HNSW index for high-speed cosine vector search
  - knowledge.match_documents : Unified multi-document retrieval RPC
  - knowledge.match_okf_documents : Backwards-compatible RPC
"""

import os
import sys
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv(Path(__file__).with_name(".env"))

DB_HOST = os.getenv("SUPABASE_DB_HOST", "")
DB_PORT = int(os.getenv("SUPABASE_DB_PORT", "5432"))
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")

SETUP_SQL = """
-- 1. Enable pgvector and ensure schema
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS knowledge;

-- 2. Drop old tables cleanly
DROP TABLE IF EXISTS knowledge.okf_documents CASCADE;
DROP TABLE IF EXISTS knowledge.document_chunks CASCADE;
DROP TABLE IF EXISTS knowledge.documents CASCADE;

-- 3. Document Registry Table (Tier 1)
CREATE TABLE knowledge.documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    short_title TEXT,
    doc_category TEXT NOT NULL DEFAULT 'act',
    status TEXT NOT NULL DEFAULT 'active',
    year_bs INTEGER,
    year_ad INTEGER,
    act_number TEXT,
    jurisdiction TEXT NOT NULL DEFAULT 'Nepal',
    language TEXT NOT NULL DEFAULT 'en',
    total_units INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Document Chunks Table (Tier 2)
CREATE TABLE knowledge.document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES knowledge.documents(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    doc_type TEXT NOT NULL,
    structural_ref TEXT,
    unit_number INTEGER,
    parent_number INTEGER,
    parent_title TEXT,
    title TEXT NOT NULL,
    description TEXT,
    content TEXT NOT NULL,
    page INTEGER,
    clause_count INTEGER DEFAULT 0,
    sub_clause_count INTEGER DEFAULT 0,
    proviso_count INTEGER DEFAULT 0,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Standard Indexes for Filtering
CREATE INDEX idx_chunks_document_id ON knowledge.document_chunks(document_id);
CREATE INDEX idx_chunks_structural_ref ON knowledge.document_chunks(structural_ref);
CREATE INDEX idx_chunks_unit_number ON knowledge.document_chunks(unit_number);
CREATE INDEX idx_chunks_parent_number ON knowledge.document_chunks(parent_number);
CREATE INDEX idx_chunks_doc_type ON knowledge.document_chunks(doc_type);
CREATE INDEX idx_chunks_doc_and_type ON knowledge.document_chunks(document_id, doc_type);

-- 6. HNSW Vector Index for Fast Cosine Distance Search
CREATE INDEX idx_chunks_embedding_hnsw ON knowledge.document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 7. Unified Document Search Function (RPC)
CREATE OR REPLACE FUNCTION knowledge.match_documents(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.25,
    match_count INT DEFAULT 10,
    filter_document_id TEXT DEFAULT NULL,
    filter_doc_category TEXT DEFAULT NULL,
    filter_doc_type TEXT DEFAULT NULL,
    filter_unit_number INT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    document_id TEXT,
    document_title TEXT,
    doc_category TEXT,
    structural_ref TEXT,
    unit_number INTEGER,
    parent_number INTEGER,
    parent_title TEXT,
    doc_type TEXT,
    title TEXT,
    content TEXT,
    file_path TEXT,
    page INTEGER,
    tags TEXT[],
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        d.title AS document_title,
        d.doc_category,
        c.structural_ref,
        c.unit_number,
        c.parent_number,
        c.parent_title,
        c.doc_type,
        c.title,
        c.content,
        c.file_path,
        c.page,
        c.tags,
        c.metadata,
        (1 - (c.embedding <=> query_embedding))::FLOAT AS similarity
    FROM knowledge.document_chunks c
    JOIN knowledge.documents d ON c.document_id = d.id
    WHERE
        (1 - (c.embedding <=> query_embedding)) > match_threshold
        AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
        AND (filter_doc_category IS NULL OR d.doc_category = filter_doc_category)
        AND (filter_doc_type IS NULL OR c.doc_type = filter_doc_type)
        AND (filter_unit_number IS NULL OR c.unit_number = filter_unit_number)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 8. Backwards-Compatible RPC (match_okf_documents)
CREATE OR REPLACE FUNCTION knowledge.match_okf_documents(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.25,
    match_count INT DEFAULT 10,
    filter_part_number INT DEFAULT NULL,
    filter_doc_type TEXT DEFAULT NULL,
    filter_source TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    source_document TEXT,
    source_title TEXT,
    file_path TEXT,
    chunk_index INTEGER,
    doc_type TEXT,
    title TEXT,
    content TEXT,
    part_number INTEGER,
    article_number INTEGER,
    section_number INTEGER,
    chapter_number INTEGER,
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
        c.id,
        c.document_id AS source_document,
        d.title AS source_title,
        c.file_path,
        c.chunk_index,
        c.doc_type,
        c.title,
        c.content,
        c.parent_number AS part_number,
        CASE WHEN d.doc_category = 'constitution' THEN c.unit_number ELSE NULL END AS article_number,
        CASE WHEN d.doc_category != 'constitution' THEN c.unit_number ELSE NULL END AS section_number,
        c.parent_number AS chapter_number,
        c.page,
        c.clause_count,
        c.sub_clause_count,
        c.proviso_count,
        c.tags,
        (1 - (c.embedding <=> query_embedding))::FLOAT AS similarity
    FROM knowledge.document_chunks c
    JOIN knowledge.documents d ON c.document_id = d.id
    WHERE
        (1 - (c.embedding <=> query_embedding)) > match_threshold
        AND (filter_source IS NULL OR c.document_id = filter_source)
        AND (filter_doc_type IS NULL OR c.doc_type = filter_doc_type)
        AND (filter_part_number IS NULL OR c.parent_number = filter_part_number)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""


def main():
    print("=" * 70)
    print("SETTING UP INDUSTRY-STANDARD TWO-TIER VECTOR SCHEMA IN SUPABASE")
    print("=" * 70)
    print(f"Host: {DB_HOST}")
    print(f"Database: {DB_NAME}")
    print(f"User: {DB_USER}")

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
        connect_timeout=15,
    )
    conn.autocommit = True
    cur = conn.cursor()

    print("\nExecuting schema migration SQL...")
    cur.execute(SETUP_SQL)
    print("Schema applied successfully!")

    # Verify tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'knowledge'
        ORDER BY table_name;
    """)
    tables = [row[0] for row in cur.fetchall()]
    print(f"\nTables in schema 'knowledge': {tables}")

    cur.close()
    conn.close()
    print("\nDone! Ready for vector ingestion.")


if __name__ == "__main__":
    main()
