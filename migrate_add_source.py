"""
migrate_add_source.py -- Add multi-document source columns to knowledge.okf_documents

This migration:
  1. Adds new columns: source_document, source_title, section_number, chapter_number
  2. Backfills existing 839 Constitution rows with source_document='constitution'
  3. Creates new indexes for efficient filtered search
  4. Updates the search function to support source filtering

Safe to run multiple times (uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).with_name(".env"))

DB_HOST = os.getenv("SUPABASE_DB_HOST", "")
DB_PORT = int(os.getenv("SUPABASE_DB_PORT", "5432"))
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")


MIGRATION_SQL = """
-- ============================================================================
-- MIGRATION: Add multi-document source differentiation
-- ============================================================================

-- 1. Add new columns (safe: IF NOT EXISTS via DO block)
DO $$
BEGIN
    -- source_document: identifies which Act/law ('constitution', 'penal_code', etc.)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'knowledge' AND table_name = 'okf_documents' AND column_name = 'source_document'
    ) THEN
        ALTER TABLE knowledge.okf_documents ADD COLUMN source_document TEXT NOT NULL DEFAULT 'constitution';
    END IF;

    -- source_title: human-readable full title
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'knowledge' AND table_name = 'okf_documents' AND column_name = 'source_title'
    ) THEN
        ALTER TABLE knowledge.okf_documents ADD COLUMN source_title TEXT NOT NULL DEFAULT 'Constitution of Nepal, 2015';
    END IF;

    -- section_number: for statutory codes (Penal Code sections)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'knowledge' AND table_name = 'okf_documents' AND column_name = 'section_number'
    ) THEN
        ALTER TABLE knowledge.okf_documents ADD COLUMN section_number INTEGER;
    END IF;

    -- chapter_number: chapter within the Act
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'knowledge' AND table_name = 'okf_documents' AND column_name = 'chapter_number'
    ) THEN
        ALTER TABLE knowledge.okf_documents ADD COLUMN chapter_number INTEGER;
    END IF;
END $$;

-- 2. Backfill existing Constitution data
UPDATE knowledge.okf_documents
SET source_document = 'constitution',
    source_title = 'Constitution of Nepal, 2015'
WHERE source_document IS NULL OR source_document = 'constitution';

-- 3. Create new indexes
CREATE INDEX IF NOT EXISTS idx_okf_source ON knowledge.okf_documents(source_document);
CREATE INDEX IF NOT EXISTS idx_okf_section ON knowledge.okf_documents(section_number);
CREATE INDEX IF NOT EXISTS idx_okf_chapter ON knowledge.okf_documents(chapter_number);

-- Composite index for common query pattern: source + similarity search
CREATE INDEX IF NOT EXISTS idx_okf_source_doctype ON knowledge.okf_documents(source_document, doc_type);

-- 4. Update search function with source filtering
CREATE OR REPLACE FUNCTION knowledge.match_okf_documents(
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.3,
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
        d.id,
        d.source_document,
        d.source_title,
        d.file_path,
        d.chunk_index,
        d.doc_type,
        d.title,
        d.content,
        d.part_number,
        d.article_number,
        d.section_number,
        d.chapter_number,
        d.page,
        d.clause_count,
        d.sub_clause_count,
        d.proviso_count,
        d.tags,
        (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity
    FROM knowledge.okf_documents d
    WHERE
        (1 - (d.embedding <=> query_embedding)) > match_threshold
        AND (filter_part_number IS NULL OR d.part_number = filter_part_number)
        AND (filter_doc_type IS NULL OR d.doc_type = filter_doc_type)
        AND (filter_source IS NULL OR d.source_document = filter_source)
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""


def run_migration():
    print("=" * 60)
    print("  Migration: Add Multi-Document Source Columns")
    print("=" * 60)
    print(f"Connecting to {DB_HOST}:{DB_PORT}...")

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
        connect_timeout=10,
    )
    conn.autocommit = True
    cur = conn.cursor()

    try:
        # Check current state
        cur.execute("SELECT COUNT(*) FROM knowledge.okf_documents;")
        existing_count = cur.fetchone()[0]
        print(f"\n  Existing records: {existing_count}")

        # Run migration
        print("\n  Running migration SQL...")
        cur.execute(MIGRATION_SQL)

        # Verify
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'knowledge' AND table_name = 'okf_documents'
            ORDER BY ordinal_position;
        """)
        columns = [row[0] for row in cur.fetchall()]
        print(f"\n  Table columns after migration:")
        for col in columns:
            marker = " ← NEW" if col in ('source_document', 'source_title', 'section_number', 'chapter_number') else ""
            print(f"    - {col}{marker}")

        # Check backfill
        cur.execute("""
            SELECT source_document, COUNT(*) 
            FROM knowledge.okf_documents 
            GROUP BY source_document;
        """)
        sources = cur.fetchall()
        print(f"\n  Records by source after backfill:")
        for src, cnt in sources:
            print(f"    • {src}: {cnt} records")

        print(f"\n[SUCCESS] Migration complete! {existing_count} existing records preserved.")

    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run_migration()
