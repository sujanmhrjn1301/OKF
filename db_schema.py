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
-- GLOBAL KNOWLEDGE REPOSITORY (Constitution & OKF Documents)
-- ============================================================================
CREATE TABLE IF NOT EXISTS knowledge.okf_documents (
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
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_okf_doc_type ON knowledge.okf_documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_okf_part ON knowledge.okf_documents(part_number);
CREATE INDEX IF NOT EXISTS idx_okf_article ON knowledge.okf_documents(article_number);
CREATE INDEX IF NOT EXISTS idx_okf_page ON knowledge.okf_documents(page);

-- Hybrid search for global OKF documents
CREATE OR REPLACE FUNCTION knowledge.match_okf_documents(
    query_embedding VECTOR(384),
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
    FROM knowledge.okf_documents d
    WHERE
        (1 - (d.embedding <=> query_embedding)) > match_threshold
        AND (filter_part_number IS NULL OR d.part_number = filter_part_number)
        AND (filter_doc_type IS NULL OR d.doc_type = filter_doc_type)
    ORDER BY d.embedding <=> query_embedding
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
    chat_mode VARCHAR(50) DEFAULT 'constitution', -- 'constitution' OR 'pdf_chat'
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
        print("  1. Schema 'knowledge'   -> okf_documents (Global Constitution Store)")
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
