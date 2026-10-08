"""
db_schema.py -- Setup Supabase Database Schemas and Tables for:
1. Global Knowledge Base (OKF Constitution with Vector Embeddings)
2. App Users & Authentication (Signup, Login, Profiles)
3. User Vault (Private Uploaded PDFs, Document Chunks, isolated Vector Search)
4. Chat History (Per-user conversations and message history with sources)
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
    """Connect to Supabase PostgreSQL using pooler or direct host."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
        connect_timeout=10,
    )


SQL_SCHEMA_SETUP = """
-- 1. Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 2. Create Isolated Schemas
CREATE SCHEMA IF NOT EXISTS knowledge;
CREATE SCHEMA IF NOT EXISTS app_users;
CREATE SCHEMA IF NOT EXISTS user_vault;

-- ============================================================================
-- GLOBAL KNOWLEDGE REPOSITORY (Multi-Document Legal Knowledge Base)
-- ============================================================================
-- ============================================================================
-- GLOBAL KNOWLEDGE REPOSITORY (Two-Tier Multi-Document Legal Knowledge Base)
-- ============================================================================
CREATE TABLE IF NOT EXISTS knowledge.documents (
    id TEXT PRIMARY KEY,                       -- e.g. 'constitution', 'criminal_procedure'
    title TEXT NOT NULL,                       -- Official statutory title
    short_title TEXT,                          -- Human-readable short title
    doc_category TEXT NOT NULL DEFAULT 'act',  -- 'constitution', 'code', 'act', 'regulation', 'precedent'
    status TEXT NOT NULL DEFAULT 'active',     -- 'active', 'amended', 'repealed'
    year_bs INTEGER,                           -- Calendar year in BS (e.g. 2074)
    year_ad INTEGER,                           -- Calendar year in AD (e.g. 2017)
    act_number TEXT,                           -- e.g. 'Act No. 36 of 2074'
    jurisdiction TEXT NOT NULL DEFAULT 'Nepal',
    language TEXT NOT NULL DEFAULT 'en',       -- 'en', 'ne'
    total_units INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge.document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES knowledge.documents(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    doc_type TEXT NOT NULL,
    structural_ref TEXT,                       -- 'Article 18', 'Section 58', 'Schedule 1'
    unit_number INTEGER,                       -- Numeric article or section
    parent_number INTEGER,                     -- Part or Chapter number
    parent_title TEXT,                         -- Part or Chapter title
    title TEXT NOT NULL,
    description TEXT,
    content TEXT NOT NULL,
    page INTEGER,
    clause_count INTEGER DEFAULT 0,
    sub_clause_count INTEGER DEFAULT 0,
    proviso_count INTEGER DEFAULT 0,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(1536),                    -- OpenAI text-embedding-3-small (1536 dims)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON knowledge.document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_structural_ref ON knowledge.document_chunks(structural_ref);
CREATE INDEX IF NOT EXISTS idx_chunks_unit_num ON knowledge.document_chunks(unit_number);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_num ON knowledge.document_chunks(parent_number);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON knowledge.document_chunks(doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_and_type ON knowledge.document_chunks(document_id, doc_type);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON knowledge.document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Unified Multi-Document Retrieval RPC
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


-- ============================================================================
-- APP USERS & AUTHENTICATION
-- ============================================================================
CREATE TABLE IF NOT EXISTS app_users.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name VARCHAR(100) NOT NULL,
    middle_name VARCHAR(100),
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    phone_number VARCHAR(50),
    address TEXT,
    profession VARCHAR(100),
    organization VARCHAR(150),
    bio TEXT,
    avatar_url TEXT,
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON app_users.users(email);

CREATE TABLE IF NOT EXISTS app_users.email_otps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    otp_code VARCHAR(10) NOT NULL,
    purpose VARCHAR(50) DEFAULT 'signup',
    expires_at TIMESTAMPTZ NOT NULL,
    is_used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_email_otps_lookup ON app_users.email_otps(email, otp_code, is_used);
CREATE INDEX IF NOT EXISTS idx_email_otps_email_time ON app_users.email_otps(email, created_at DESC);



-- ============================================================================
-- USER VAULT: ISOLATED UPLOADED PDFS & CHUNKS (For "Chat with PDF only" feature)
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_vault.user_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES app_users.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL DEFAULT 0,
    storage_path TEXT,
    page_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'ready',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_docs_user_id ON user_vault.user_documents(user_id);

CREATE TABLE IF NOT EXISTS user_vault.user_document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES user_vault.user_documents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES app_users.users(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_chunks_user_id ON user_vault.user_document_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_user_chunks_doc_id ON user_vault.user_document_chunks(document_id);

-- Isolated search function for a specific user and optionally a specific PDF
CREATE OR REPLACE FUNCTION user_vault.match_user_document_chunks(
    p_user_id UUID,
    p_query_embedding VECTOR(384),
    p_document_id UUID DEFAULT NULL,
    p_match_threshold FLOAT DEFAULT 0.3,
    p_match_count INT DEFAULT 10
)
RETURNS TABLE (
    id BIGINT,
    document_id UUID,
    chunk_index INTEGER,
    page_number INTEGER,
    content TEXT,
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
        c.chunk_index,
        c.page_number,
        c.content,
        c.metadata,
        (1 - (c.embedding <=> p_query_embedding))::FLOAT AS similarity
    FROM user_vault.user_document_chunks c
    WHERE
        c.user_id = p_user_id
        AND (p_document_id IS NULL OR c.document_id = p_document_id)
        AND (1 - (c.embedding <=> p_query_embedding)) > p_match_threshold
    ORDER BY c.embedding <=> p_query_embedding
    LIMIT p_match_count;
END;
$$;


-- ============================================================================
-- USER CHAT HISTORY & CONVERSATION SESSIONS
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_vault.chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES app_users.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Conversation',
    chat_mode VARCHAR(50) DEFAULT 'constitution', -- 'constitution', 'criminal_procedure', 'all_laws', 'pdf_chat'
    document_id UUID REFERENCES user_vault.user_documents(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON user_vault.chat_sessions(user_id);

CREATE TABLE IF NOT EXISTS user_vault.chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES user_vault.chat_sessions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES app_users.users(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    sources JSONB DEFAULT '[]'::jsonb, -- cited clauses/articles or PDF page excerpts
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_msgs_session ON user_vault.chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_msgs_user ON user_vault.chat_messages(user_id);
"""


def setup_all():
    print("=" * 60)
    print("  Initializing Supabase Database Schemas & Tables")
    print("=" * 60)
    print(f"Connecting to {DB_HOST}:{DB_PORT} as {DB_USER}...")
    
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    try:
        cur.execute(SQL_SCHEMA_SETUP)
        print("\n[SUCCESS] Successfully initialized database schemas:")
        print("  1. Schema 'knowledge'   -> okf_documents (Multi-Document Legal Knowledge Base)")
        print("  2. Schema 'app_users'   -> users (Signup, Login, Profiles)")
        print("  3. Schema 'user_vault'  -> user_documents, user_document_chunks (Isolated User PDFs)")
        print("  4. Schema 'user_vault'  -> chat_sessions, chat_messages (User Chat History)")
    except Exception as e:
        print(f"[ERROR] Failed to run schema setup: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    setup_all()
