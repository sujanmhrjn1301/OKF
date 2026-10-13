"""
document_uploader.py -- Universal Legal Document Classifier, Chunker, and Supabase Ingestion

Architecture & Pipeline:
  1. Document Decision Engine:
     - Automatically inspects the uploaded PDF text and structural markers.
     - Decides document category:
       * 'constitution' : The Constitution of Nepal (Parts, Articles, Clauses, Schedules)
       * 'act'          : Acts & Statutory Codes of Nepal (Chapters, Sections, Subsections, Clauses, Provisos, Schedules)
       * 'general'      : Rules, Directives, Regulations, or general legal PDFs
  2. Domain-Aware Chunker:
     - Applies the specialized legal chunking algorithm based on the detected document type.
  3. Batch Vectorizer:
     - Computes 1536-dimensional embeddings via OpenRouter (openai/text-embedding-3-small).
  4. Supabase Knowledge Base Storage:
     - Registers document metadata in knowledge.documents.
     - Ingests chunk embeddings with structural references into knowledge.document_chunks.
     - Optional User Vault synchronization for private user document collections.

Usage:
  python document_uploader.py <path_to_pdf>
  python document_uploader.py data/raw_pdf/privacy_act.pdf --yes
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("ERROR: PyMuPDF is required. Install with: pip install PyMuPDF")

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv

from auth_service import get_db_connection

# Load environment
load_dotenv(Path(__file__).with_name(".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL_NAME = "openai/text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 40


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class DocumentProfile:
    doc_id: str
    title: str
    short_title: str
    doc_category: str  # 'constitution' | 'act' | 'general'
    year_bs: int | None = None
    year_ad: int | None = None
    act_number: str | None = None
    jurisdiction: str = "Nepal"
    language: str = "en"
    total_pages: int = 0
    confidence_score: float = 1.0
    detected_markers: list[str] = field(default_factory=list)


@dataclass
class ParsedChunk:
    chunk_index: int
    doc_type: str            # 'article' | 'section' | 'schedule' | 'preamble' | 'general'
    structural_ref: str      # e.g., 'Section 14', 'Article 20', 'Schedule 1'
    unit_number: int | None
    parent_number: int | None
    parent_title: str
    title: str
    content: str
    page: int
    clause_count: int = 0
    sub_clause_count: int = 0
    proviso_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Decision Engine: Document Classifier
# ============================================================================

def detect_document_type(pdf_path: Path) -> DocumentProfile:
    """Analyze the uploaded PDF to decide if it is the Constitution, an Act, or a general document."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # Extract text from first 3 pages for classification
    sample_text = ""
    for i in range(min(3, total_pages)):
        sample_text += f"\n--- Page {i+1} ---\n" + doc[i].get_text()

    cleaned_sample = re.sub(r"\s+", " ", sample_text).strip()
    detected_markers = []

    # 1. Constitution Check
    is_const = bool(
        re.search(r"Constitution of Nepal", sample_text, re.IGNORECASE)
        and (re.search(r"Part\s*[-–]?\s*1\s+Preliminary", sample_text, re.IGNORECASE)
             or re.search(r"Article\s+\d+", sample_text, re.IGNORECASE))
    )

    if is_const:
        detected_markers.append("Constitutional Title & Part/Article markers")
        doc.close()
        return DocumentProfile(
            doc_id="constitution",
            title="The Constitution of Nepal, 2015",
            short_title="Constitution of Nepal",
            doc_category="constitution",
            year_bs=2072,
            year_ad=2015,
            total_pages=total_pages,
            confidence_score=0.98,
            detected_markers=detected_markers,
        )

    # 2. Statutory Act Check
    is_act = bool(
        re.search(r"\bAct\b", sample_text, re.IGNORECASE)
        and (
            re.search(r"Act\s+(?:Number|No\.?)\s*\d+", sample_text, re.IGNORECASE)
            or re.search(r"An Act (?:Made|Formulated|Promulgated)\b", sample_text, re.IGNORECASE)
            or re.search(r"Preamble\s*:", sample_text, re.IGNORECASE)
            or re.search(r"Date of Authentication", sample_text, re.IGNORECASE)
            or re.search(r"Chapter\s*[-–]?\s*\d+", sample_text, re.IGNORECASE)
            or re.search(r"Short title and commencement", sample_text, re.IGNORECASE)
        )
    )

    if is_act:
        detected_markers.append("Statutory enactment markers (Preamble, Act Number, Chapters, Sections)")
        
        # Extract Act Title
        title_match = re.search(
            r"(?:^|\n)\s*((?:The\s+)?(?:National\s+)?[A-Z][A-Za-z\s,\-–\(\)]+Act,\s*\d{4}(?:\s*\(\d{4}\))?)",
            sample_text
        )
        if not title_match:
            # Fallback title search
            title_match = re.search(r"(?:^|\n)\s*([A-Z][A-Za-z\s,\-–]+Act[^\n]*)", sample_text)

        title = title_match.group(1).strip() if title_match else pdf_path.stem.replace("_", " ").title()
        
        # Slug / doc_id
        clean_slug = re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_")
        
        # Extract Act Number
        act_no_match = re.search(r"Act\s+(?:Number|No\.?)\s*(\d+[^\n]*)", sample_text, re.IGNORECASE)
        act_number = act_no_match.group(1).strip() if act_no_match else None

        # Extract Years (BS and AD)
        year_bs = None
        year_ad = None
        years_found = [int(y) for y in re.findall(r"\b(20\d\d)\b", sample_text)]
        for y in years_found:
            if y >= 2040 and not year_bs:
                year_bs = y
            elif y < 2040 and not year_ad:
                year_ad = y

        doc.close()
        return DocumentProfile(
            doc_id=clean_slug,
            title=title,
            short_title=title.replace("The ", "").split(",")[0].strip(),
            doc_category="act",
            year_bs=year_bs,
            year_ad=year_ad,
            act_number=act_number,
            total_pages=total_pages,
            confidence_score=0.95,
            detected_markers=detected_markers,
        )

    # 3. Fallback: General Legal / Policy Document
    detected_markers.append("General legal document without standard Act/Constitutional layout")
    clean_slug = re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_")
    title = pdf_path.stem.replace("_", " ").title()

    doc.close()
    return DocumentProfile(
        doc_id=clean_slug,
        title=title,
        short_title=title,
        doc_category="general",
        total_pages=total_pages,
        confidence_score=0.80,
        detected_markers=detected_markers,
    )


# ============================================================================
# Chunker 1: Generic Act / Statutory Code Parser
# ============================================================================

def clean_statutory_text(text: str) -> str:
    """Normalize extracted PDF text, removing watermarks and header/footer noise."""
    text = re.sub(r"www\.lawcommission\.gov\.np", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_act_document(pdf_path: Path, profile: DocumentProfile) -> list[ParsedChunk]:
    """Parse any Act of Nepal into structural units (Preamble, Sections, Schedules)."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # 1. Extract per-page text
    page_texts: list[tuple[int, str]] = []
    for pno in range(total_pages):
        raw = doc[pno].get_text()
        page_texts.append((pno + 1, clean_statutory_text(raw)))
    doc.close()

    # 2. Build full combined document with page boundaries tracked
    chunks: list[ParsedChunk] = []
    chunk_index = 0

    # Track Chapters
    current_chapter_num = 1
    current_chapter_title = "Preliminary"

    # Match Sections: e.g. "1. Short title and commencement:", "14. Arrest:" or "Section 14"
    # Section pattern: start of line followed by number. Title
    sec_regex = re.compile(
        r"(?:^|\n)\s*(?:(?:Section\s+)?(\d+)\.\s+([A-Z][A-Za-z0-9\s,\-–\(\)\/]+?)(?::|\.\s|\n))",
        re.MULTILINE
    )

    # Schedule pattern
    sched_regex = re.compile(
        r"(?:^|\n)\s*(Schedule\s*[-–]?\s*(\d+|[A-Z]))\b",
        re.IGNORECASE | re.MULTILINE
    )

    # Process page by page
    current_unit_text = []
    current_unit_ref = "Preamble"
    current_unit_num = None
    current_unit_title = "Preamble and Preliminary"
    current_unit_page = 1
    current_unit_type = "preamble"

    pending_sec_num: int | None = None
    pending_page: int = 1

    def flush_unit():
        nonlocal chunk_index
        if current_unit_text:
            body = "\n".join(current_unit_text).strip()
            if body:
                cl_count = len(re.findall(r"^\s*\([a-z]\)", body, re.MULTILINE))
                sub_count = len(re.findall(r"^\s*\(\d+\)", body, re.MULTILINE))
                prov_count = len(re.findall(r"Provided that", body, re.IGNORECASE))

                chunks.append(ParsedChunk(
                    chunk_index=chunk_index,
                    doc_type=current_unit_type,
                    structural_ref=current_unit_ref,
                    unit_number=current_unit_num,
                    parent_number=current_chapter_num,
                    parent_title=current_chapter_title,
                    title=current_unit_title,
                    content=f"## {current_unit_ref}: {current_unit_title}\n\n{body}",
                    page=current_unit_page,
                    clause_count=cl_count,
                    sub_clause_count=sub_count,
                    proviso_count=prov_count,
                    metadata={
                        "doc_id": profile.doc_id,
                        "page": current_unit_page,
                        "chapter": current_chapter_num,
                    },
                ))
                chunk_index += 1

    for page_num, text in page_texts:
        lines = text.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check Chapter Header
            chap_match = re.match(r"^Chapter\s*[-–]?\s*(\d+)\s*(.*)", line_str, re.IGNORECASE)
            if chap_match:
                current_chapter_num = int(chap_match.group(1))
                if chap_match.group(2).strip():
                    current_chapter_title = chap_match.group(2).strip()
                continue

            # Check Schedule Header
            sched_match = sched_regex.match(line_str)
            if sched_match:
                flush_unit()
                current_unit_ref = sched_match.group(1).title()
                current_unit_num = int(sched_match.group(2)) if sched_match.group(2).isdigit() else None
                current_unit_title = f"{current_unit_ref} of {profile.title}"
                current_unit_page = page_num
                current_unit_type = "schedule"
                current_unit_text = [line_str]
                pending_sec_num = None
                continue

            # Case B: Pending section number waiting for title on the next line
            if pending_sec_num is not None:
                m_title = re.match(r"^([A-Z][A-Za-z0-9\s,\-–\(\)\/]+?)(?::|\.|\s*\(1\)|\n|$)", line_str)
                if m_title and len(m_title.group(1).strip()) < 80 and not m_title.group(1).strip().startswith("Act"):
                    flush_unit()
                    s_num = pending_sec_num
                    s_title = m_title.group(1).strip()
                    current_unit_ref = f"Section {s_num}"
                    current_unit_num = s_num
                    current_unit_title = s_title
                    current_unit_page = pending_page
                    current_unit_type = "section"
                    current_unit_text = [f"{s_num}. {line_str}"]
                    pending_sec_num = None
                    continue
                else:
                    pending_sec_num = None

            # Case B check: Isolated section number e.g. "1." or "Section 1."
            m_isolated = re.match(r"^(?:Section\s+)?(\d+)\.\s*$", line_str, re.IGNORECASE)
            if m_isolated:
                pending_sec_num = int(m_isolated.group(1))
                pending_page = page_num
                continue

            # Case A check: Section Number and Title on the same line
            m_same = re.match(r"^(?:Section\s+)?(\d+)\.\s+([A-Z][A-Za-z0-9\s,\-–\(\)\/]+?)(?::|\.|\s*\(1\)|\n|$)", line_str)
            if m_same and len(m_same.group(2).strip()) < 80 and not m_same.group(2).strip().startswith("Act"):
                flush_unit()
                s_num = int(m_same.group(1))
                s_title = m_same.group(2).strip()
                current_unit_ref = f"Section {s_num}"
                current_unit_num = s_num
                current_unit_title = s_title
                current_unit_page = page_num
                current_unit_type = "section"
                current_unit_text = [line_str]
                pending_sec_num = None
                continue

            current_unit_text.append(line_str)

    flush_unit()
    return chunks


# ============================================================================
# Chunker 2: Generic / Fallback Legal Document Parser
# ============================================================================

def parse_general_document(pdf_path: Path, profile: DocumentProfile) -> list[ParsedChunk]:
    """Parse general legal or policy PDF by page and semantic boundary chunks."""
    doc = fitz.open(pdf_path)
    chunks: list[ParsedChunk] = []
    chunk_index = 0

    for page_idx, page in enumerate(doc):
        pno = page_idx + 1
        text = clean_statutory_text(page.get_text())
        if not text:
            continue

        # Split into blocks of max ~800 characters
        paragraphs = text.split("\n\n")
        current_block = []
        current_len = 0

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if current_len + len(p) > 800 and current_block:
                block_content = "\n\n".join(current_block)
                chunks.append(ParsedChunk(
                    chunk_index=chunk_index,
                    doc_type="general",
                    structural_ref=f"Page {pno} (Part {chunk_index + 1})",
                    unit_number=pno,
                    parent_number=pno,
                    parent_title=f"Page {pno}",
                    title=f"{profile.short_title} - Page {pno}",
                    content=block_content,
                    page=pno,
                    metadata={"doc_id": profile.doc_id, "page": pno},
                ))
                chunk_index += 1
                current_block = [p]
                current_len = len(p)
            else:
                current_block.append(p)
                current_len += len(p)

        if current_block:
            block_content = "\n\n".join(current_block)
            chunks.append(ParsedChunk(
                chunk_index=chunk_index,
                doc_type="general",
                structural_ref=f"Page {pno} (Part {chunk_index + 1})",
                unit_number=pno,
                parent_number=pno,
                parent_title=f"Page {pno}",
                title=f"{profile.short_title} - Page {pno}",
                content=block_content,
                page=pno,
                metadata={"doc_id": profile.doc_id, "page": pno},
            ))
            chunk_index += 1

    doc.close()
    return chunks


# ============================================================================
# Vectorization Pipeline
# ============================================================================

def embed_chunks(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """Batch embed text chunks via OpenRouter (openai/text-embedding-3-small)."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing in .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[list[float]] = []
    total = len(texts)
    print(f"\n[EMBED] Generating 1536-dim embeddings for {total} chunks...")

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "model": MODEL_NAME,
            "input": batch,
        }

        for attempt in range(3):
            try:
                resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    data_sorted = sorted(data, key=lambda x: x["index"])
                    all_embeddings.extend([item["embedding"] for item in data_sorted])
                    done = min(i + batch_size, total)
                    print(f"        [{done}/{total}] chunks embedded...")
                    break
                elif resp.status_code == 429:
                    print(f"        Rate limited, waiting 3s (attempt {attempt+1})...")
                    time.sleep(3)
                else:
                    raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2)

    return all_embeddings


# ============================================================================
# Citation Graph Extraction (Legal Documents)
# ============================================================================

def extract_legal_citation_edges(
    chunks: list[ParsedChunk],
    profile: DocumentProfile,
) -> list[dict[str, Any]]:
    """Extract internal section references, schedules, and cross-act/constitutional citations."""
    edges: list[dict[str, Any]] = []
    unit_numbers = [c.unit_number for c in chunks if c.unit_number is not None]
    if not unit_numbers:
        return edges

    max_sec = max(unit_numbers)

    for c in chunks:
        if c.unit_number is None:
            continue
        src_num = c.unit_number
        text = c.content

        # 1. Intra-document section matches (e.g., "Section 16", "Sections 14 and 15")
        sec_matches = re.findall(
            r"(?:Section|Sections|Sec\.)\s*(\d{1,4}(?:\s*(?:,|or|and|/)\s*\d{1,4})*)",
            text,
            re.IGNORECASE,
        )
        target_secs = set()
        for m in sec_matches:
            for n in re.findall(r"\d{1,4}", m):
                tgt = int(n)
                if tgt != src_num and 1 <= tgt <= max_sec:
                    target_secs.add(tgt)

        # 2. Schedule references
        sched_matches = re.findall(
            r"(?:Schedule|Schedules|Sched\.)\s*-?\s*(\d{1,2}(?:\s*(?:,|or|and|/)\s*\d{1,2})*)",
            text,
            re.IGNORECASE,
        )
        schedules = sorted({
            int(n) for m in sched_matches for n in re.findall(r"\d{1,2}", m)
            if 1 <= int(n) <= 50
        })

        # 3. Pinpoint citations (e.g., "sub-section (5) of Section 16")
        pinpoints = []
        for pm in re.finditer(
            r"sub-sections?\s*((?:\(\d{1,2}\)\s*(?:,|or|and)?\s*)+)of\s+Section\s+(\d{1,4})",
            text,
            re.IGNORECASE,
        ):
            target_sec_num = int(pm.group(2))
            for sub in re.findall(r"\(\d{1,2}\)", pm.group(1)):
                pinpoints.append(f"{target_sec_num}{sub}")

        # 4. Constitutional Article cross-references
        const_matches = re.findall(
            r"(?:Article|Art\.)\s*(\d{1,3})[^\n\.\;]*?(?:Constitution)",
            text,
            re.IGNORECASE,
        )
        target_const = {int(a) for a in const_matches}

        for tgt in sorted(target_secs):
            sec_pinpoints = [p for p in pinpoints if p.startswith(f"{tgt}(")]
            edges.append({
                "source_unit_number": src_num,
                "target_document_id": profile.doc_id,
                "target_unit_number": tgt,
                "relation_type": "refers_to_section",
                "scope": "INTRA_DOCUMENT",
                "metadata": {
                    "scope": "INTRA_DOCUMENT",
                    "relationship_label": f"Section {src_num} -> Section {tgt} (Within Same Document)",
                    "relationship_description": f"Section {src_num} refers internally within {profile.title} to Section {tgt}",
                    "source_document_title": profile.title,
                    "target_document_title": profile.title,
                    "pinpoint": sec_pinpoints,
                    "schedules": schedules,
                },
            })

        for tgt in sorted(target_const):
            edges.append({
                "source_unit_number": src_num,
                "target_document_id": "constitution",
                "target_unit_number": tgt,
                "relation_type": "cites_constitutional_article",
                "scope": "CROSS_DOCUMENT",
                "metadata": {
                    "scope": "CROSS_DOCUMENT",
                    "relationship_label": f"Section {src_num} -> Article {tgt} (Constitution of Nepal)",
                    "relationship_description": f"Section {src_num} of {profile.title} cites externally to Article {tgt} of The Constitution of Nepal, 2015",
                    "source_document_title": profile.title,
                    "target_document_title": "The Constitution of Nepal, 2015",
                },
            })

    return edges


# ============================================================================
# User Vault Storage (100% User-to-User Isolated & Separate from Knowledge Base)
# ============================================================================

def save_to_user_vault(
    user_id: str,
    profile: DocumentProfile,
    chunks: list[ParsedChunk],
    embeddings: list[list[float]],
    citation_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Save document, vector chunks, and legal citation graph into user_vault (isolated per user)."""
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        # 1. Clean previous uploaded document with same filename/doc_id for this user
        cur.execute(
            """
            DELETE FROM user_vault.user_documents
            WHERE user_id = %s AND filename = %s;
            """,
            (user_id, f"{profile.doc_id}.pdf"),
        )

        # 2. Register Document in user_vault.user_documents
        print(f"\n[USER VAULT] Registering document '{profile.title}' for user {user_id}...")
        cur.execute(
            """
            INSERT INTO user_vault.user_documents (
                user_id, title, filename, file_size_bytes, storage_path, page_count, status, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                user_id,
                profile.title,
                f"{profile.doc_id}.pdf",
                0,
                f"vault/{user_id}/{profile.doc_id}.pdf",
                profile.total_pages,
                "ready",
                json.dumps({
                    "doc_category": profile.doc_category,
                    "doc_id": profile.doc_id,
                    "act_number": profile.act_number,
                    "year_bs": profile.year_bs,
                    "year_ad": profile.year_ad,
                    "total_chunks": len(chunks),
                    "total_citation_edges": len(citation_edges),
                    "detected_markers": profile.detected_markers,
                }),
            ),
        )
        doc_uuid = cur.fetchone()["id"]

        # 3. Insert Chunks into user_vault.user_document_chunks (VECTOR 1536)
        print(f"[USER VAULT] Ingesting {len(chunks)} vector chunks into user_vault.user_document_chunks...")
        insert_chunk_sql = """
            INSERT INTO user_vault.user_document_chunks (
                document_id, user_id, chunk_index, page_number, content, metadata, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        chunk_rows = []
        for c, emb in zip(chunks, embeddings):
            meta = {
                "structural_ref": c.structural_ref,
                "unit_number": c.unit_number,
                "parent_number": c.parent_number,
                "parent_title": c.parent_title,
                "title": c.title,
                "doc_type": c.doc_type,
                "clause_count": c.clause_count,
                "sub_clause_count": c.sub_clause_count,
                "proviso_count": c.proviso_count,
            }
            chunk_rows.append((
                doc_uuid,
                user_id,
                c.chunk_index,
                c.page,
                c.content,
                json.dumps(meta),
                str(emb),
            ))
        psycopg2.extras.execute_batch(cur, insert_chunk_sql, chunk_rows, page_size=100)

        # 4. Insert User-Isolated Legal Citation Graph into user_vault.user_citation_edges
        if citation_edges:
            print(f"[USER VAULT] Ingesting {len(citation_edges)} isolated legal citation edges into user_vault.user_citation_edges...")
            insert_edge_sql = """
                INSERT INTO user_vault.user_citation_edges (
                    user_id, document_id, source_unit_number, target_document_id,
                    target_unit_number, relation_type, scope, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            edge_rows = []
            for e in citation_edges:
                edge_rows.append((
                    user_id,
                    doc_uuid,
                    e["source_unit_number"],
                    e["target_document_id"],
                    e["target_unit_number"],
                    e.get("relation_type", "refers_to_section"),
                    e.get("scope", "INTRA_DOCUMENT"),
                    json.dumps(e.get("metadata", {})),
                ))
            psycopg2.extras.execute_batch(cur, insert_edge_sql, edge_rows, page_size=100)
            print(f"[USER VAULT] Citation graph successfully stored! (Isolated to user {user_id})")

        return {
            "status": "success",
            "storage_target": "user_vault",
            "document_id": str(doc_uuid),
            "title": profile.title,
            "chunks_count": len(chunks),
            "citation_edges_count": len(citation_edges),
        }
    finally:
        cur.close()
        conn.close()


def resolve_default_user_id() -> str:
    """Find the first active user ID or raise an error."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM app_users.users WHERE is_active = TRUE ORDER BY created_at ASC LIMIT 1;")
            row = cur.fetchone()
            if row:
                return str(row[0])
            raise ValueError("No active user found in app_users.users. Please create a user account first.")
    finally:
        conn.close()


# ============================================================================
# Main High-Level Upload Workflow
# ============================================================================

def upload_legal_document(
    pdf_path: str | Path,
    auto_confirm: bool = False,
    user_id: str | None = None,
) -> dict[str, Any]:
    """End-to-end legal document upload, classification, chunking, citation graph extraction, and user-vault storage."""
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Resolve target user
    target_user_id = user_id or resolve_default_user_id()

    print("\n" + "=" * 65)
    print("      USER VAULT LEGAL DOCUMENT UPLOADER & GRAPH BUILDER")
    print("=" * 65)
    print(f"Target File : {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"Target User : {target_user_id} (Isolated in user_vault)")

    # 1. Step 1: Decision Engine
    print("\n[Step 1/5] Classifying document structure...")
    profile = detect_document_type(path)
    print(f"  * Detected Category : {profile.doc_category.upper()}")
    print(f"  * Document Title    : {profile.title}")
    print(f"  * Document Slug ID  : {profile.doc_id}")
    if profile.act_number:
        print(f"  * Act Number        : {profile.act_number}")
    if profile.year_bs or profile.year_ad:
        print(f"  * Enactment Year    : BS {profile.year_bs or 'N/A'} / AD {profile.year_ad or 'N/A'}")
    print(f"  * Classification    : {profile.confidence_score * 100:.0f}% confidence ({', '.join(profile.detected_markers)})")

    if not auto_confirm:
        confirm = input("\nProceed with chunking, graph extraction, and Supabase vectorization? (y/N): ").strip().lower()
        if confirm != "y":
            print("Operation cancelled by user.")
            return {"status": "cancelled"}

    # 2. Step 2: Tailored Legal Chunker
    print(f"\n[Step 2/5] Applying {profile.doc_category.upper()} chunking algorithm...")
    if profile.doc_category in ("constitution", "act"):
        chunks = parse_act_document(path, profile)
    else:
        chunks = parse_general_document(path, profile)

    print(f"  * Extracted {len(chunks)} structural units (Sections / Articles / Schedules)")
    if not chunks:
        raise ValueError("Parser yielded 0 chunks. Document structure could not be parsed.")

    # 3. Step 3: Citation Graph Extraction (for legal documents)
    citation_edges: list[dict[str, Any]] = []
    if profile.doc_category in ("act", "constitution"):
        print(f"\n[Step 3/5] Extracting legal citation graph edges...")
        citation_edges = extract_legal_citation_edges(chunks, profile)
        intra_count = sum(1 for e in citation_edges if e["scope"] == "INTRA_DOCUMENT")
        cross_count = sum(1 for e in citation_edges if e["scope"] == "CROSS_DOCUMENT")
        print(f"  * Extracted {len(citation_edges)} legal connections ({intra_count} intra-Act, {cross_count} cross-Act/Constitutional)")
    else:
        print("\n[Step 3/5] Non-statutory document; skipping statutory citation graph.")

    # 4. Step 4: Batch Vectorization
    print(f"\n[Step 4/5] Vectorizing with {MODEL_NAME}...")
    chunk_texts = [f"{c.title}\n\n{c.content}" for c in chunks]
    embeddings = embed_chunks(chunk_texts)

    # 5. Step 5: User Vault Ingestion (Separated from Global Knowledge Base)
    print(f"\n[Step 5/5] Ingesting into user_vault tables in Supabase...")
    result = save_to_user_vault(
        user_id=target_user_id,
        profile=profile,
        chunks=chunks,
        embeddings=embeddings,
        citation_edges=citation_edges,
    )

    print("\n" + "=" * 65)
    print("  USER VAULT UPLOAD, VECTORIZATION & GRAPH COMPLETE!")
    print(f"  Document '{profile.title}' is stored privately for user {target_user_id}.")
    print(f"  Chunks: {len(chunks)} | Citation Edges: {len(citation_edges)}")
    print("  (Stored in user_vault, zero mixing with knowledge schema)")
    print("=" * 65 + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Upload, classify, chunk, graph, and vectorize legal documents into user_vault.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to upload")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm without prompting")
    parser.add_argument("--user-id", type=str, default=None, help="User ID for isolated User Vault association")
    args = parser.parse_args()

    upload_legal_document(args.pdf_path, auto_confirm=args.yes, user_id=args.user_id)


if __name__ == "__main__":
    main()

