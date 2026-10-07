"""
setup_hybrid_search.py -- Enable Full-Text Search (BM25) and Reciprocal Rank Fusion (RRF)
in Supabase PostgreSQL for hybrid retrieval.
"""

import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).with_name(".env"))

DB_HOST = os.getenv("SUPABASE_DB_HOST", "")
DB_PORT = int(os.getenv("SUPABASE_DB_PORT", "5432"))
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
        connect_timeout=10,
    )


HYBRID_SETUP_SQL = """
-- 1. Add Full-Text Search Vector Column
ALTER TABLE knowledge.okf_documents 
ADD COLUMN IF NOT EXISTS fts_tokens TSVECTOR;

-- 2. Populate Full-Text Search Index from Title, Description, and Content
UPDATE knowledge.okf_documents
SET fts_tokens = to_tsvector(
    'english',
    COALESCE(title, '') || ' ' || 
    COALESCE(description, '') || ' ' || 
    COALESCE(part_title, '') || ' ' || 
    COALESCE(array_to_string(tags, ' '), '') || ' ' || 
    COALESCE(content, '')
);

-- 3. Create GIN Index for High-Speed Full-Text Search
CREATE INDEX IF NOT EXISTS idx_okf_fts ON knowledge.okf_documents USING GIN(fts_tokens);

-- 4. Create Trigger to keep FTS tokens automatically updated
CREATE OR REPLACE FUNCTION knowledge.update_okf_fts()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fts_tokens := to_tsvector(
        'english',
        COALESCE(NEW.title, '') || ' ' || 
        COALESCE(NEW.description, '') || ' ' || 
        COALESCE(NEW.part_title, '') || ' ' || 
        COALESCE(array_to_string(NEW.tags, ' '), '') || ' ' || 
        COALESCE(NEW.content, '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_okf_fts ON knowledge.okf_documents;
CREATE TRIGGER trg_update_okf_fts
BEFORE INSERT OR UPDATE ON knowledge.okf_documents
FOR EACH ROW
EXECUTE FUNCTION knowledge.update_okf_fts();

-- 5. Stored Procedure: Reciprocal Rank Fusion (RRF) Hybrid Search
CREATE OR REPLACE FUNCTION knowledge.hybrid_search_okf(
    query_text TEXT,
    query_embedding VECTOR(384),
    match_count INT DEFAULT 15,
    rrf_k INT DEFAULT 60
)
RETURNS TABLE (
    id BIGINT,
    file_path TEXT,
    chunk_index INTEGER,
    doc_type TEXT,
    title TEXT,
    content TEXT,
    part_number INTEGER,
    part_title TEXT,
    article_number INTEGER,
    page INTEGER,
    clause_count INTEGER,
    sub_clause_count INTEGER,
    proviso_count INTEGER,
    tags TEXT[],
    dense_rank INT,
    sparse_rank INT,
    rrf_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH dense_search AS (
        SELECT
            d.id,
            ROW_NUMBER() OVER (ORDER BY d.embedding <=> query_embedding)::INT AS rank
        FROM knowledge.okf_documents d
        WHERE (1 - (d.embedding <=> query_embedding)) > 0.28
        ORDER BY d.embedding <=> query_embedding
        LIMIT match_count * 2
    ),
    sparse_search AS (
        SELECT
            d.id,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(d.fts_tokens, plainto_tsquery('english', query_text)) DESC)::INT AS rank
        FROM knowledge.okf_documents d
        WHERE d.fts_tokens @@ plainto_tsquery('english', query_text)
        ORDER BY ts_rank_cd(d.fts_tokens, plainto_tsquery('english', query_text)) DESC
        LIMIT match_count * 2
    ),
    combined AS (
        SELECT
            COALESCE(d.id, s.id) AS id,
            COALESCE(d.rank, 999)::INT AS d_rank,
            COALESCE(s.rank, 999)::INT AS s_rank,
            (COALESCE(1.0 / (rrf_k + d.rank), 0.0) + COALESCE(1.0 / (rrf_k + s.rank), 0.0))::FLOAT AS score
        FROM dense_search d
        FULL OUTER JOIN sparse_search s ON d.id = s.id
    )
    SELECT
        doc.id,
        doc.file_path,
        doc.chunk_index,
        doc.doc_type,
        doc.title,
        doc.content,
        doc.part_number,
        doc.part_title,
        doc.article_number,
        doc.page,
        doc.clause_count,
        doc.sub_clause_count,
        doc.proviso_count,
        doc.tags,
        c.d_rank,
        c.s_rank,
        c.score
    FROM combined c
    JOIN knowledge.okf_documents doc ON c.id = doc.id
    ORDER BY c.score DESC
    LIMIT match_count;
END;
$$;
"""


def setup_hybrid():
    print("=" * 60)
    print("  Setting up PostgreSQL Hybrid Search (Dense + BM25 FTS + RRF)")
    print("=" * 60)
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    try:
        cur.execute(HYBRID_SETUP_SQL)
        print("[SUCCESS] Full-Text Search (GIN Index) & RRF Stored Procedure Active!")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    setup_hybrid()
