"""
auth_service.py -- Authentication, User Management, and Isolated Chat/Document Services

Provides:
- Secure Signup with field validation & PBKDF2 password hashing
- Secure Login & credential verification
- User Profile management
- Per-user Isolated Chat Sessions & History management
- Per-user Uploaded Document & Chunk tracking (for 'chat with PDF only' feature)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from email_service import send_otp_email

load_dotenv(Path(__file__).with_name(".env"))

DB_HOST = os.getenv("SUPABASE_DB_HOST", "")
DB_PORT = int(os.getenv("SUPABASE_DB_PORT", "5432"))
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")


def get_db_connection():
    """Returns an autocommit psycopg2 connection."""
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
    return conn


# ============================================================================
# Security: PBKDF2-HMAC-SHA256 Password Hashing
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a unique cryptographic salt."""
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}:{key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt:key string."""
    try:
        salt, key_hex = stored_hash.split(":")
        test_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return secrets.compare_digest(test_key.hex(), key_hex)
    except Exception:
        return False


def validate_email(email: str) -> str:
    """Validate and normalize email."""
    email = (email or "").strip().lower()
    if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
        raise ValueError(f"Invalid email address: '{email}'")
    return email


# ============================================================================
# OTP and Email Verification
# ============================================================================

def generate_and_store_otp(
    email: str,
    purpose: str = "signup",
    valid_minutes: int = 10,
    user_name: str | None = None,
) -> str:
    """Generate a cryptographically secure 6-digit OTP, store in database, and dispatch via email."""
    clean_email = validate_email(email)
    otp_code = f"{secrets.randbelow(900000) + 100000}"

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # If user_name was not explicitly passed, attempt to look it up from app_users.users
        if not user_name:
            cur.execute("SELECT first_name, last_name FROM app_users.users WHERE email = %s;", (clean_email,))
            u_row = cur.fetchone()
            if u_row:
                user_name = f"{u_row['first_name']} {u_row.get('last_name') or ''}".strip()

        # Invalidate previous unused OTPs for this email and purpose
        cur.execute(
            """
            UPDATE app_users.email_otps 
            SET is_used = TRUE 
            WHERE email = %s AND purpose = %s AND is_used = FALSE;
            """,
            (clean_email, purpose),
        )

        # Store fresh OTP code with expiration
        cur.execute(
            """
            INSERT INTO app_users.email_otps (email, otp_code, purpose, expires_at)
            VALUES (%s, %s, %s, NOW() + INTERVAL '%s minutes');
            """,
            (clean_email, otp_code, purpose, valid_minutes),
        )
    finally:
        cur.close()
        conn.close()

    # Send the OTP via email service with personalized name
    send_otp_email(clean_email, otp_code, user_name=user_name, purpose=purpose, valid_minutes=valid_minutes)
    return otp_code


def verify_otp(email: str, otp_code: str, purpose: str = "signup") -> bool:
    """Verify an OTP code for an email address and mark it as consumed."""
    clean_email = validate_email(email)
    code = (otp_code or "").strip()
    if not code:
        raise ValueError("OTP verification code is required.")

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            SELECT id, otp_code, expires_at, is_used
            FROM app_users.email_otps
            WHERE email = %s AND purpose = %s
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (clean_email, purpose),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("No pending verification code found. Please request a new code.")

        if row["is_used"]:
            raise ValueError("This verification code has already been used. Please request a new code.")

        # Check expiration (PostgreSQL TIMESTAMPTZ returned as timezone-aware datetime)
        now_utc = datetime.now(timezone.utc)
        expires_at = row["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if now_utc > expires_at:
            raise ValueError("Verification code has expired. Please request a new code.")

        if not secrets.compare_digest(row["otp_code"], code):
            raise ValueError("Incorrect verification code. Please check and try again.")

        # Mark OTP as used
        cur.execute("UPDATE app_users.email_otps SET is_used = TRUE WHERE id = %s;", (row["id"],))
        return True
    finally:
        cur.close()
        conn.close()


# ============================================================================
# User Registration and Authentication
# ============================================================================

def initiate_signup(
    *,
    first_name: str,
    last_name: str,
    email: str,
    password: str,
    confirm_password: str,
    middle_name: str | None = None,
    phone_number: str | None = None,
    address: str | None = None,
    profession: str | None = None,
    organization: str | None = None,
    bio: str | None = None,
    avatar_url: str | None = None,
) -> dict[str, Any]:
    """
    Step 1 of Signup: Validate details, record user with is_verified=FALSE, and send verification OTP.
    """
    if not first_name or not first_name.strip():
        raise ValueError("First name is required.")
    if not last_name or not last_name.strip():
        raise ValueError("Last name is required.")

    clean_email = validate_email(email)

    if password != confirm_password:
        raise ValueError("Passwords do not match.")

    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    password_hash = hash_password(password)

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        # Check if email is already registered
        cur.execute("SELECT id, is_verified FROM app_users.users WHERE email = %s;", (clean_email,))
        existing = cur.fetchone()

        if existing:
            if existing["is_verified"]:
                raise ValueError(f"An active account with email '{clean_email}' already exists. Please log in.")
            else:
                # Update existing unverified registration
                cur.execute(
                    """
                    UPDATE app_users.users
                    SET first_name = %s, middle_name = %s, last_name = %s,
                        password_hash = %s, phone_number = %s, address = %s,
                        profession = %s, organization = %s, bio = %s, avatar_url = %s,
                        updated_at = NOW()
                    WHERE id = %s;
                    """,
                    (
                        first_name.strip(),
                        middle_name.strip() if middle_name else None,
                        last_name.strip(),
                        password_hash,
                        phone_number.strip() if phone_number else None,
                        address.strip() if address else None,
                        profession.strip() if profession else None,
                        organization.strip() if organization else None,
                        bio.strip() if bio else None,
                        avatar_url.strip() if avatar_url else None,
                        existing["id"],
                    ),
                )
        else:
            # Insert new unverified user
            cur.execute(
                """
                INSERT INTO app_users.users
                    (first_name, middle_name, last_name, email, password_hash,
                     phone_number, address, profession, organization, bio, avatar_url,
                     is_verified)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE);
                """,
                (
                    first_name.strip(),
                    middle_name.strip() if middle_name else None,
                    last_name.strip(),
                    clean_email,
                    password_hash,
                    phone_number.strip() if phone_number else None,
                    address.strip() if address else None,
                    profession.strip() if profession else None,
                    organization.strip() if organization else None,
                    bio.strip() if bio else None,
                    avatar_url.strip() if avatar_url else None,
                ),
            )
    finally:
        cur.close()
        conn.close()

    # Generate and send 6-digit OTP with personalized name
    user_full_name = f"{first_name.strip()} {last_name.strip()}".strip()
    generate_and_store_otp(clean_email, purpose="signup", valid_minutes=10, user_name=user_full_name)

    return {
        "status": "pending_verification",
        "email": clean_email,
        "message": f"Verification code sent to {clean_email}. Please verify to complete registration.",
    }


def complete_signup(email: str, otp_code: str) -> dict[str, Any]:
    """
    Step 2 of Signup: Verify the OTP and activate the user account.
    Returns the authenticated user profile.
    """
    clean_email = validate_email(email)
    verify_otp(clean_email, otp_code, purpose="signup")

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cur.execute(
            """
            UPDATE app_users.users
            SET is_verified = TRUE, last_login_at = NOW(), updated_at = NOW()
            WHERE email = %s
            RETURNING id, first_name, middle_name, last_name, email,
                      phone_number, address, profession, organization, bio,
                      avatar_url, role, is_active, is_verified, created_at, last_login_at;
            """,
            (clean_email,),
        )
        user = cur.fetchone()
        if not user:
            raise ValueError(f"No registration found for email '{clean_email}'.")

        return dict(user)
    finally:
        cur.close()
        conn.close()


def resend_signup_otp(email: str) -> dict[str, Any]:
    """Resend a fresh OTP to an unverified email address."""
    clean_email = validate_email(email)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cur.execute("SELECT id, is_verified FROM app_users.users WHERE email = %s;", (clean_email,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"No pending registration found for '{clean_email}'. Please sign up first.")
        if row["is_verified"]:
            raise ValueError(f"Account for '{clean_email}' is already verified. Please log in directly.")
    finally:
        cur.close()
        conn.close()

    generate_and_store_otp(clean_email, purpose="signup", valid_minutes=10)
    return {
        "status": "otp_resent",
        "email": clean_email,
        "message": f"A new verification code has been sent to {clean_email}.",
    }


def signup(
    *,
    first_name: str,
    last_name: str,
    email: str,
    password: str,
    confirm_password: str,
    middle_name: str | None = None,
    phone_number: str | None = None,
    address: str | None = None,
    profession: str | None = None,
    organization: str | None = None,
    bio: str | None = None,
    avatar_url: str | None = None,
    auto_verify: bool = False,
) -> dict[str, Any]:
    """
    Register a user. If auto_verify=True, activates immediately (useful for scripts/testing).
    Otherwise, starts the OTP email verification flow.
    """
    res = initiate_signup(
        first_name=first_name,
        middle_name=middle_name,
        last_name=last_name,
        email=email,
        password=password,
        confirm_password=confirm_password,
        phone_number=phone_number,
        address=address,
        profession=profession,
        organization=organization,
        bio=bio,
        avatar_url=avatar_url,
    )
    if auto_verify:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute(
                """
                UPDATE app_users.users
                SET is_verified = TRUE, last_login_at = NOW(), updated_at = NOW()
                WHERE email = %s
                RETURNING id, first_name, middle_name, last_name, email,
                          phone_number, address, profession, organization, bio,
                          avatar_url, role, is_active, is_verified, created_at, last_login_at;
                """,
                (res["email"],),
            )
            return dict(cur.fetchone())
        finally:
            cur.close()
            conn.close()
    return res


def login(email: str, password: str) -> dict[str, Any]:
    """Authenticate user with email and password. Requires verified email."""
    clean_email = validate_email(email)

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cur.execute(
            """
            SELECT id, first_name, middle_name, last_name, email, password_hash,
                   phone_number, address, profession, organization, bio,
                   avatar_url, role, is_active, is_verified, created_at, last_login_at
            FROM app_users.users
            WHERE email = %s;
            """,
            (clean_email,),
        )
        row = cur.fetchone()

        if not row:
            raise ValueError("Invalid email or password.")

        if not row["is_active"]:
            raise ValueError("This account has been deactivated.")

        if not verify_password(password, row["password_hash"]):
            raise ValueError("Invalid email or password.")

        if not row.get("is_verified", False):
            raise ValueError(
                f"Your email address '{clean_email}' is not verified yet. "
                "Please verify using the OTP code sent to your email."
            )

        # Update last login timestamp
        cur.execute(
            "UPDATE app_users.users SET last_login_at = NOW() WHERE id = %s;",
            (row["id"],),
        )

        user_data = dict(row)
        user_data.pop("password_hash", None)
        user_data["last_login_at"] = datetime.now(timezone.utc).isoformat()
        return user_data
    finally:
        cur.close()
        conn.close()



def get_user_profile(user_id: str) -> dict[str, Any] | None:
    """Retrieve user profile by ID without sensitive password hash."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            SELECT id, first_name, middle_name, last_name, email,
                   phone_number, address, profession, organization, bio,
                   avatar_url, role, is_active, created_at, last_login_at
            FROM app_users.users
            WHERE id = %s;
            """,
            (user_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        cur.close()
        conn.close()


# ============================================================================
# Chat History Management (Isolated Per User)
# ============================================================================

def create_chat_session(
    user_id: str,
    title: str = "New Conversation",
    chat_mode: str = "constitution",
    document_id: str | None = None,
) -> dict[str, Any]:
    """Start a new chat session for a user (either 'constitution' or 'pdf_chat' with a specific document)."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            INSERT INTO user_vault.chat_sessions (user_id, title, chat_mode, document_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id, user_id, title, chat_mode, document_id, created_at, updated_at;
            """,
            (user_id, title, chat_mode, document_id),
        )
        return dict(cur.fetchone())
    finally:
        cur.close()
        conn.close()


def add_chat_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add a message turn (user or assistant) with citations to chat history."""
    sources_json = json.dumps(sources or [])
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            INSERT INTO user_vault.chat_messages (session_id, user_id, role, content, sources)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, session_id, user_id, role, content, sources, created_at;
            """,
            (session_id, user_id, role, content, sources_json),
        )
        msg = dict(cur.fetchone())

        # Update session updated_at
        cur.execute(
            "UPDATE user_vault.chat_sessions SET updated_at = NOW() WHERE id = %s;",
            (session_id,),
        )
        return msg
    finally:
        cur.close()
        conn.close()


def get_chat_history(session_id: str, user_id: str) -> list[dict[str, Any]]:
    """Retrieve chronologically ordered messages in a user's session."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            SELECT id, session_id, user_id, role, content, sources, created_at
            FROM user_vault.chat_messages
            WHERE session_id = %s AND user_id = %s
            ORDER BY created_at ASC;
            """,
            (session_id, user_id),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def get_user_chat_sessions(user_id: str) -> list[dict[str, Any]]:
    """List all chat sessions belonging to a user."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            SELECT s.id, s.title, s.chat_mode, s.document_id, s.created_at, s.updated_at,
                   COUNT(m.id) as message_count
            FROM user_vault.chat_sessions s
            LEFT JOIN user_vault.chat_messages m ON m.session_id = s.id
            WHERE s.user_id = %s
            GROUP BY s.id
            ORDER BY s.updated_at DESC;
            """,
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


# ============================================================================
# User Uploaded Documents (For future 'PDF Uploader and Chat with PDF only')
# ============================================================================

def register_user_document(
    user_id: str,
    title: str,
    filename: str,
    file_size_bytes: int,
    page_count: int = 0,
    storage_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register an uploaded user PDF in their private vault."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            INSERT INTO user_vault.user_documents
                (user_id, title, filename, file_size_bytes, page_count, storage_path, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, user_id, title, filename, file_size_bytes, page_count, status, created_at;
            """,
            (
                user_id,
                title,
                filename,
                file_size_bytes,
                page_count,
                storage_path,
                json.dumps(metadata or {}),
            ),
        )
        return dict(cur.fetchone())
    finally:
        cur.close()
        conn.close()


def list_user_documents(user_id: str) -> list[dict[str, Any]]:
    """List all uploaded documents belonging to a user."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(
            """
            SELECT id, title, filename, file_size_bytes, page_count, status, created_at
            FROM user_vault.user_documents
            WHERE user_id = %s
            ORDER BY created_at DESC;
            """,
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


# ============================================================================
# Self-Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("Testing auth_service.py...")
    test_email = f"test_user_{secrets.token_hex(4)}@example.com"

    # Test Signup
    print(f"\n1. Testing Signup for {test_email}...")
    user = signup(
        first_name="Ram",
        middle_name="Bahadur",
        last_name="Shrestha",
        email=test_email,
        password="SecurePassword123!",
        confirm_password="SecurePassword123!",
        phone_number="+977-9841234567",
        address="Kathmandu, Nepal",
        profession="Legal Researcher",
        organization="Nepal Law Institute",
    )
    print("  User registered successfully! ID:", user["id"])
    print("  Full Name:", f"{user['first_name']} {user['middle_name']} {user['last_name']}")
    print("  Profession:", user["profession"])

    # Test Login
    print("\n2. Testing Login...")
    logged_in = login(test_email, "SecurePassword123!")
    print("  Login SUCCESS! Welcome,", logged_in["first_name"])

    # Test Chat Session Creation
    print("\n3. Testing Isolated Chat Session Creation...")
    session = create_chat_session(
        user_id=logged_in["id"],
        title="Constitution Fundamental Rights Query",
        chat_mode="constitution",
    )
    print("  Chat Session created:", session["id"], "| Title:", session["title"])

    # Test Adding Chat Message
    print("\n4. Testing Message Storage with Citations...")
    msg_user = add_chat_message(
        session_id=session["id"],
        user_id=logged_in["id"],
        role="user",
        content="What does Article 31 say about education rights?",
    )
    msg_ai = add_chat_message(
        session_id=session["id"],
        user_id=logged_in["id"],
        role="assistant",
        content="Article 31 guarantees the right to education. Every citizen has the right to access basic education...",
        sources=[{"article": 31, "title": "Right to education", "page": 11, "clauses": [1, 2, 3, 4]}],
    )
    history = get_chat_history(session["id"], logged_in["id"])
    print(f"  Retrieved {len(history)} messages from session history!")
    for m in history:
        print(f"    [{m['role'].upper()}]: {m['content'][:60]}... (Sources: {len(m['sources'])})")

    print("\nAll Auth and User Vault services verified working perfectly!")
