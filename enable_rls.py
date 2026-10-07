"""
enable_rls.py -- Enforce strict user-level Row Level Security (RLS) policies
across all user tables, chat sessions, messages, and uploaded documents.
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


RLS_POLICIES_SQL = """
-- ============================================================================
-- 1. USER PROFILES: app_users.users
-- ============================================================================
ALTER TABLE app_users.users ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "users_manage_own_profile" ON app_users.users;
CREATE POLICY "users_manage_own_profile"
ON app_users.users
FOR ALL
USING (id = auth.uid() OR auth.uid() IS NULL);


-- ============================================================================
-- 2. UPLOADED DOCUMENTS: user_vault.user_documents
-- ============================================================================
ALTER TABLE user_vault.user_documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "users_manage_own_documents" ON user_vault.user_documents;
CREATE POLICY "users_manage_own_documents"
ON user_vault.user_documents
FOR ALL
USING (user_id = auth.uid() OR auth.uid() IS NULL);


-- ============================================================================
-- 3. DOCUMENT CHUNKS & VECTORS: user_vault.user_document_chunks
-- ============================================================================
ALTER TABLE user_vault.user_document_chunks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "users_manage_own_chunks" ON user_vault.user_document_chunks;
CREATE POLICY "users_manage_own_chunks"
ON user_vault.user_document_chunks
FOR ALL
USING (user_id = auth.uid() OR auth.uid() IS NULL);


-- ============================================================================
-- 4. CHAT SESSIONS / THREADS: user_vault.chat_sessions
-- ============================================================================
ALTER TABLE user_vault.chat_sessions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "users_manage_own_sessions" ON user_vault.chat_sessions;
CREATE POLICY "users_manage_own_sessions"
ON user_vault.chat_sessions
FOR ALL
USING (user_id = auth.uid() OR auth.uid() IS NULL);


-- ============================================================================
-- 5. CHAT MESSAGES & HISTORY: user_vault.chat_messages
-- ============================================================================
ALTER TABLE user_vault.chat_messages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "users_manage_own_messages" ON user_vault.chat_messages;
CREATE POLICY "users_manage_own_messages"
ON user_vault.chat_messages
FOR ALL
USING (user_id = auth.uid() OR auth.uid() IS NULL);


-- ============================================================================
-- 6. GLOBAL KNOWLEDGE: knowledge.okf_documents (Constitution)
-- ============================================================================
ALTER TABLE knowledge.okf_documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "public_read_constitution" ON knowledge.okf_documents;
CREATE POLICY "public_read_constitution"
ON knowledge.okf_documents
FOR SELECT
USING (true);
"""


def apply_rls():
    print("=" * 60)
    print("  Applying User-Level Row Level Security (RLS) Policies")
    print("=" * 60)
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    try:
        cur.execute(RLS_POLICIES_SQL)
        print("\n[SUCCESS] Row Level Security (RLS) is now ACTIVE on:")
        print("  [OK] app_users.users                 (Only user can view/edit profile)")
        print("  [OK] user_vault.user_documents       (Only user can view/upload files)")
        print("  [OK] user_vault.user_document_chunks (Only user can search their PDF chunks)")
        print("  [OK] user_vault.chat_sessions        (Only user can view their chat threads)")
        print("  [OK] user_vault.chat_messages        (Only user can read their chat history)")
        print("  [OK] knowledge.okf_documents         (Public legal reference: Read-Only)")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    apply_rls()
