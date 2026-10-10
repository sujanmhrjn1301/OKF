"""
chat_cli.py -- Interactive Terminal Legal AI Chat Engine (v3 - Multi-Document & Two-Tier RAG)

5-Layer Scenario-Aware Retrieval Architecture:
  Layer 1: Query Intent Classifier (CONSTITUTIONAL_DIRECT / SCENARIO / STATUTORY_CRIMINAL / MIXED)
  Layer 2: Multi-Query Decomposition (Rights, Remedies, Procedural angles)
  Layer 3: Multi-Document Vector Retrieval across Two-Tier Supabase (Constitution + Criminal Procedure)
  Layer 4: Scenario-Aware Cross-Encoder Reranking
  Layer 5: Domain-Aware Dynamic Prompt Assembly (FIRAC framework & statutory analysis)

Features:
  - Supports All Laws, Criminal Procedure Code (2017), and Constitution of Nepal (2015)
  - Uses OpenRouter openai/text-embedding-3-small (1536-dim vectors, zero PyTorch load time)
  - User Authentication & isolated Chat Sessions (user_vault schema)
  - Real-time streaming via OpenRouter (Llama 3.3 70B)
  - Verified Article, Section, Clause, and Page Number Citations
"""

from __future__ import annotations

import getpass
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

# Suppress HuggingFace / PyTorch console noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv

import auth_service

# Load environment
load_dotenv(Path(__file__).with_name(".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Global model caches
_RERANK_MODEL = None
_ARTICLE_GRAPH: dict[str, list[int]] | None = None

BASE_DIR = Path(__file__).resolve().parent
CONSTITUTION_DIR = BASE_DIR / "data" / "okf" / "constitution"
CRIMINAL_PROC_DIR = BASE_DIR / "data" / "okf" / "criminal_procedure"
ARTICLE_GRAPH_PATH = CONSTITUTION_DIR / "article_graph.json"


def get_query_embedding(text: str) -> list[float]:
    """Generate 1536-dim embedding for query via OpenRouter API."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing in .env file.")

    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    if resp.status_code == 200:
        return resp.json()["data"][0]["embedding"]
    raise RuntimeError(f"Embedding failed ({resp.status_code}): {resp.text}")


def get_rerank_model():
    """Lazy-load cross-encoder reranker."""
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            _RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            # If cross-encoder fails to load, gracefully fall back to similarity ranking
            _RERANK_MODEL = None
    return _RERANK_MODEL


def get_article_graph() -> dict[str, list[int]]:
    """Load the constitutional article cross-reference graph from Supabase (with disk fallback)."""
    global _ARTICLE_GRAPH
    if _ARTICLE_GRAPH is not None:
        return _ARTICLE_GRAPH

    # 1. Try loading directly from Supabase cloud
    try:
        conn = auth_service.get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_unit_number, target_unit_number 
                FROM knowledge.citation_edges 
                WHERE source_document_id = 'constitution'
                ORDER BY source_unit_number, target_unit_number;
            """)
            rows = cur.fetchall()
        conn.close()

        if rows:
            graph: dict[str, list[int]] = {}
            for src, tgt in rows:
                graph.setdefault(str(src), []).append(tgt)
            _ARTICLE_GRAPH = graph
            return _ARTICLE_GRAPH
    except Exception:
        pass

    # 2. Local fallback if DB is unreachable and local file exists
    if ARTICLE_GRAPH_PATH.exists():
        try:
            _ARTICLE_GRAPH = json.loads(ARTICLE_GRAPH_PATH.read_text(encoding="utf-8"))
            return _ARTICLE_GRAPH
        except Exception:
            pass

    _ARTICLE_GRAPH = {}
    return _ARTICLE_GRAPH


# ============================================================================
# LAYER 1: Query Intent Classifier
# ============================================================================

INTENT_CATEGORIES = {
    "CONSTITUTIONAL_DIRECT",
    "CONSTITUTIONAL_SCENARIO",
    "STATUTORY_CRIMINAL",
    "STATUTORY_CIVIL",
    "MIXED",
}


def classify_query_intent(query: str) -> dict[str, Any]:
    """Classify the query's legal domain and intent using OpenRouter."""
    default_result = {
        "intent": "MIXED",
        "reasoning": "General legal inquiry",
        "suggested_provisions": [],
    }

    if not OPENROUTER_API_KEY:
        return default_result

    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        prompt = (
            "You are a Senior Legal Domain Classifier for Nepal Law (Constitution of Nepal, 2015 and National Criminal Procedure Code, 2017).\n"
            "Classify the user's query into exactly ONE category:\n\n"
            "- CONSTITUTIONAL_DIRECT: Direct question about the Constitution of Nepal.\n"
            "- CONSTITUTIONAL_SCENARIO: Real-world scenario where Constitutional Fundamental Rights (Part 3) are primary.\n"
            "- STATUTORY_CRIMINAL: Criminal procedural matter (arrest, detention, remand, search, seizure, bail, charge-sheet, trial, evidence, appeal).\n"
            "- STATUTORY_CIVIL: Private civil dispute (property, contract, family, tenancy).\n"
            "- MIXED: Involves BOTH constitutional rights (e.g., Article 20 justice, Article 23 preventive detention) AND criminal procedure (e.g., Section 14, 18, 19, 72).\n\n"
            "Respond in this EXACT JSON format:\n"
            '{"intent": "CATEGORY_NAME", "reasoning": "brief why", "suggested_provisions": ["Section 14", "Article 20"]}\n\n'
            f'User query: "{query}"'
        )
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
            "temperature": 0.0,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("intent") not in INTENT_CATEGORIES:
                    result["intent"] = "MIXED"
                return result
    except Exception:
        pass

    return default_result


# ============================================================================
# LAYER 2: Multi-Query Decomposition
# ============================================================================

def decompose_query(query: str, intent: dict[str, Any]) -> list[str]:
    """Generate 3 diverse sub-queries targeting different legal angles."""
    intent_type = intent.get("intent", "MIXED")

    if not OPENROUTER_API_KEY:
        return [query]

    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        prompt = (
            "You are a Nepali Legal Search Query Decomposer for vector search over Nepali statutes.\n"
            f"Query type: {intent_type}\n\n"
            "Generate EXACTLY 3 diverse search sub-queries covering all legal angles:\n"
            "1. STATUTORY/PROCEDURAL QUERY: Target specific statutory rules (e.g., 'Section 14 24-hour presentation detention remand' or 'Section 18 19 search residence witness').\n"
            "2. RIGHTS/CONSTITUTIONAL QUERY: Target governing rights/safeguards (e.g., 'Article 20 right to justice right against arbitrary detention').\n"
            "3. LEGAL DOCTRINE/PRACTICE QUERY: Target practical legal requirements and procedural defects.\n\n"
            "Respond as a JSON array of exactly 3 strings (10-15 keywords each), nothing else.\n"
            f'User query: "{query}"'
        )
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.0,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if arr_match:
                sub_queries = json.loads(arr_match.group())
                if isinstance(sub_queries, list) and len(sub_queries) >= 2:
                    return [query] + [str(sq) for sq in sub_queries[:3]]
    except Exception:
        pass

    return [query]


# ============================================================================
# LAYER 3: Two-Tier Multi-Document Retrieval
# ============================================================================

# ============================================================================
# LAYER 3: Two-Tier Multi-Document Retrieval (Hybrid + Targeted Metadata)
# ============================================================================

def extract_explicit_mentions(text: str, suggested: list[str]) -> tuple[list[int], list[int]]:
    """Extract explicit Section and Article numbers from prompt and classifier output."""
    combined = text + " " + " ".join(suggested)
    sec_nums = sorted(list(set(map(int, re.findall(r"(?:Section|Sec\.)\s*(\d+)", combined, re.IGNORECASE)))))
    art_nums = sorted(list(set(map(int, re.findall(r"(?:Article|Art\.)\s*(\d+)", combined, re.IGNORECASE)))))
    return sec_nums, art_nums


def retrieve_exact_provisions(sections: list[int], articles: list[int]) -> list[dict[str, Any]]:
    """Directly fetch explicit Section and Article chunks with top priority."""
    if not sections and not articles:
        return []

    conn = auth_service.get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    results = []
    try:
        conditions = []
        params: list[Any] = []
        if sections:
            conditions.append("(c.document_id = 'criminal_procedure' AND c.unit_number = ANY(%s))")
            params.append(sections)
        if articles:
            conditions.append("(c.document_id = 'constitution' AND c.unit_number = ANY(%s))")
            params.append(articles)

        where_clause = " OR ".join(conditions)
        sql = f"""
            SELECT
                c.id, c.document_id, d.title AS document_title, d.doc_category,
                c.structural_ref, c.unit_number, c.parent_number, c.parent_title,
                c.doc_type, c.title, c.content, c.file_path, c.page,
                c.clause_count, c.sub_clause_count, c.proviso_count, c.tags,
                1.0::FLOAT AS similarity,
                1.0::FLOAT AS rrf_score,
                TRUE AS is_exact_match
            FROM knowledge.document_chunks c
            JOIN knowledge.documents d ON c.document_id = d.id
            WHERE {where_clause}
            ORDER BY c.document_id, c.unit_number, c.chunk_index;
        """
        cur.execute(sql, params)
        results = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"  [EXACT PROVISION RETRIEVAL ERROR] {e}")
    finally:
        cur.close()
        conn.close()

    return results


def retrieve_hybrid_candidates(
    query: str,
    top_k: int = 15,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve candidates combining 1536-dim dense cosine similarity and lexical token matching."""
    query_emb = get_query_embedding(query)

    conn = auth_service.get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Extract target number if query specifically references Section/Article X
    tokens = re.findall(r"(?:Section|Article|Sec\.|Art\.)?\s*(\d+)", query, re.IGNORECASE)
    sec_num = int(tokens[0]) if tokens and tokens[0].isdigit() else None

    sql = """
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
            c.clause_count,
            c.sub_clause_count,
            c.proviso_count,
            c.tags,
            (1 - (c.embedding <=> %s::vector))::FLOAT AS similarity,
            (
                (1 - (c.embedding <=> %s::vector)) +
                CASE 
                    WHEN %s::INT IS NOT NULL AND c.unit_number = %s::INT THEN 0.35
                    ELSE 0.0
                END
            )::FLOAT AS rrf_score
        FROM knowledge.document_chunks c
        JOIN knowledge.documents d ON c.document_id = d.id
        WHERE (1 - (c.embedding <=> %s::vector)) > 0.18
           OR (%s::INT IS NOT NULL AND c.unit_number = %s::INT)
    """
    params: list[Any] = [str(query_emb), str(query_emb), sec_num, sec_num, str(query_emb), sec_num, sec_num]

    if document_id and document_id != "all_laws":
        sql += " AND c.document_id = %s"
        params.append(document_id)

    sql += " ORDER BY rrf_score DESC LIMIT %s;"
    params.append(top_k)

    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
        results = [dict(r) for r in rows]
    except Exception as e:
        print(f"  [RETRIEVAL ERROR] {e}")
        results = []
    finally:
        cur.close()
        conn.close()

    return results


def retrieve_multi_query(
    sub_queries: list[str],
    top_k_per_query: int = 10,
    document_id: str | None = None,
    user_prompt: str = "",
    suggested_provisions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run retrieval with targeted exact-provision pre-fetching and sub-query deduplication."""
    all_candidates: dict[int, dict[str, Any]] = {}

    # 1. Direct Metadata Provision Retrieval (Targeted Injection)
    sec_nums, art_nums = extract_explicit_mentions(user_prompt, suggested_provisions or [])
    if sec_nums or art_nums:
        exact_chunks = retrieve_exact_provisions(sec_nums, art_nums)
        for ec in exact_chunks:
            all_candidates[ec["id"]] = ec

    # 2. Hybrid vector search across decomposed sub-queries
    for sq in sub_queries:
        candidates = retrieve_hybrid_candidates(sq, top_k=top_k_per_query, document_id=document_id)
        for c in candidates:
            doc_id = c.get("id")
            if doc_id is not None:
                existing = all_candidates.get(doc_id)
                if existing is None:
                    c["retrieval_count"] = 1
                    all_candidates[doc_id] = c
                else:
                    existing["retrieval_count"] = existing.get("retrieval_count", 1) + 1
                    existing_score = existing.get("rrf_score", 0)
                    new_score = c.get("rrf_score", 0)
                    if new_score > existing_score:
                        existing["rrf_score"] = new_score

    # Sort with exact matches guaranteed at top
    sorted_candidates = sorted(
        all_candidates.values(),
        key=lambda x: (
            1 if x.get("is_exact_match") else 0,
            x.get("retrieval_count", 1),
            x.get("rrf_score", 0),
        ),
        reverse=True,
    )
    return sorted_candidates


def expand_with_cross_references(candidates: list[dict[str, Any]], intent: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand retrieved Constitutional provisions with their cross-referenced articles."""
    graph = get_article_graph()
    if not graph:
        return candidates

    retrieved_articles = set()
    for c in candidates:
        if c.get("document_id") == "constitution":
            art_num = c.get("unit_number")
            if art_num:
                retrieved_articles.add(art_num)

    articles_to_fetch = set()
    for art in retrieved_articles:
        neighbors = graph.get(str(art), [])
        for neighbor in neighbors[:2]:
            if neighbor not in retrieved_articles:
                articles_to_fetch.add(neighbor)

    if not articles_to_fetch:
        return candidates

    conn = auth_service.get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        placeholders = ",".join(["%s"] * len(articles_to_fetch))
        sql = f"""
            SELECT DISTINCT ON (c.unit_number)
                c.id, c.document_id, d.title AS document_title, d.doc_category,
                c.file_path, c.chunk_index, c.doc_type, c.structural_ref,
                c.title, c.content, c.parent_number, c.parent_title,
                c.unit_number, c.page, c.clause_count, c.sub_clause_count,
                c.proviso_count, c.tags
            FROM knowledge.document_chunks c
            JOIN knowledge.documents d ON c.document_id = d.id
            WHERE c.document_id = 'constitution' AND c.unit_number IN ({placeholders})
            ORDER BY c.unit_number, c.chunk_index;
        """
        cur.execute(sql, list(articles_to_fetch))
        cross_ref_rows = cur.fetchall()

        for row in cross_ref_rows:
            entry = dict(row)
            entry["rrf_score"] = 0.005
            entry["retrieval_count"] = 0
            entry["is_cross_ref"] = True
            candidates.append(entry)
    except Exception:
        pass
    finally:
        cur.close()
        conn.close()

    return candidates


# ============================================================================
# LAYER 4: Scenario-Aware Reranking (Source Diversity & Quota Enforcement)
# ============================================================================

FUNDAMENTAL_RIGHTS_PART = 3
REMEDY_ARTICLES = {46, 133, 144}


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    intent: dict[str, Any],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Rerank candidates using Cross-Encoder with legal boosting and source diversity quotas."""
    if not candidates:
        return []

    intent_type = intent.get("intent", "MIXED")
    reranker = get_rerank_model()

    if reranker is not None:
        try:
            pairs = [(query, c.get("content", "")) for c in candidates]
            scores = reranker.predict(pairs)

            for c, score in zip(candidates, scores):
                base_score = float(score)

                # 1. Exact provision matches get highest priority
                if c.get("is_exact_match"):
                    base_score += 4.0

                # 2. Boost procedural criminal code chunks when query is statutory/mixed
                if intent_type in ("STATUTORY_CRIMINAL", "MIXED") and c.get("document_id") == "criminal_procedure":
                    base_score += 2.0

                # 3. Boost constitutional fundamental rights
                if c.get("document_id") == "constitution" and c.get("parent_number") == FUNDAMENTAL_RIGHTS_PART:
                    base_score += 1.0

                # 4. Boost constitutional remedies
                if c.get("unit_number") in REMEDY_ARTICLES and c.get("document_id") == "constitution":
                    base_score += 1.5

                # 5. Multi-query agreement boost
                if c.get("retrieval_count", 1) >= 2:
                    base_score += 1.0

                c["rerank_score"] = base_score
        except Exception:
            for c in candidates:
                c["rerank_score"] = c.get("rrf_score", 0) + (4.0 if c.get("is_exact_match") else 0.0)
    else:
        for c in candidates:
            c["rerank_score"] = c.get("rrf_score", 0) + (4.0 if c.get("is_exact_match") else 0.0)

    # Sort all candidates
    ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

    # Deduplicate provisions by (document_id, unit_number) so one section's chunks don't starve others
    def deduplicate_provisions(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen = set()
        for item in pool:
            u = item.get("unit_number")
            if u is not None and u in seen:
                continue
            if u is not None:
                seen.add(u)
            unique.append(item)
        return unique

    # Enforce Source Diversity / Quota
    # When query is statutory criminal or mixed, guarantee at least 50% statutory slots
    if intent_type in ("STATUTORY_CRIMINAL", "MIXED"):
        statutory_pool = deduplicate_provisions([c for c in ranked if c.get("document_id") == "criminal_procedure"])
        constitutional_pool = deduplicate_provisions([c for c in ranked if c.get("document_id") == "constitution"])

        selected: list[dict[str, Any]] = []
        # Take top 5 distinct statutory provisions (e.g. Sections 14, 18, 19...)
        selected.extend(statutory_pool[:5])
        # Take top 3 distinct constitutional provisions (e.g. Articles 20, 23...)
        selected.extend(constitutional_pool[:3])

        # Fill any remaining slots up to top_k with remaining unique candidates
        seen_keys = {f"{c.get('document_id')}:{c.get('unit_number')}" for c in selected}
        for c in ranked:
            key = f"{c.get('document_id')}:{c.get('unit_number')}"
            if key not in seen_keys and len(selected) < top_k:
                selected.append(c)
                seen_keys.add(key)

        return selected[:top_k]
    else:
        unique_ranked = deduplicate_provisions(ranked)
        return unique_ranked[:top_k]


# ============================================================================
# Parent-Child (Hierarchical) Context Hydration
# ============================================================================

def hydrate_parent_context(child_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hydrate child chunks with full markdown provision if available."""
    hydrated = []
    seen_units = set()

    for chunk in child_chunks:
        doc_id = chunk.get("document_id")
        unit_num = chunk.get("unit_number")
        key = f"{doc_id}:{unit_num}"

        if unit_num and key not in seen_units:
            seen_units.add(key)
            parent_file = None

            if doc_id == "constitution":
                parent_file = CONSTITUTION_DIR / "articles" / f"article-{unit_num:03d}.md"
            elif doc_id == "criminal_procedure":
                parent_file = CRIMINAL_PROC_DIR / "sections" / f"section-{unit_num:03d}.md"

            if parent_file and parent_file.exists():
                try:
                    full_text = parent_file.read_text(encoding="utf-8")
                    m = re.match(r"^---\s*\n.*?\n---\s*\n", full_text, re.DOTALL)
                    body = full_text[m.end():].strip() if m else full_text

                    chunk_copy = dict(chunk)
                    chunk_copy["content"] = body
                    chunk_copy["is_parent_hydrated"] = True
                    hydrated.append(chunk_copy)
                    continue
                except Exception:
                    pass

        if not unit_num or key not in seen_units:
            hydrated.append(chunk)

    return hydrated


# ============================================================================
# LAYER 5: Domain-Aware Dynamic Prompt Assembly
# ============================================================================

BASE_SYSTEM_PROMPT = """You are "Samvidhan & Nyaya AI", an authoritative and expert Legal AI Assistant specialized in the Laws of Nepal:
1. The Constitution of Nepal (2015 / 2072)
2. The National Criminal Procedure (Code) Act, 2017 (Muluki Faujdari Karyavidhi Samhita, 2074)

Your mission is to provide rigorous, clear, and highly accurate statutory and constitutional legal analysis under Nepali jurisprudence.

STRICT LEGAL CITATION RULES:
1. CITATION NOTATION:
   - For Constitution: Cite as "Article {N}({clause}) (Constitution of Nepal, Part {P}, Page {Page})" (e.g., "Article 20(3) (Page 11)").
   - For Criminal Procedure: Cite as "Section {N}({sub-section}) (National Criminal Procedure Code, Chapter {C}, Page {Page})" (e.g., "Section 14(1) (Page 20)").
2. GROUNDING: Cite only provisions supported by the retrieved statutory and constitutional context.
3. CLEAR SEPARATION: Explicitly distinguish between constitutional fundamental rights and statutory procedural requirements.
"""

DOMAIN_INSTRUCTIONS = {
    "CONSTITUTIONAL_DIRECT": """
FOR THIS DIRECT CONSTITUTIONAL QUESTION:
- Provide a direct 1-2 sentence executive answer first.
- Break down the text of the governing Article and Clauses.
- Include a summary table: | Article & Clause | Requirement | Page |
""",

    "STATUTORY_CRIMINAL": """
FOR THIS CRIMINAL PROCEDURE QUERY / SCENARIO:
Structure your analysis clearly:
1. **EXECUTIVE LEGAL CONCLUSION**: Direct 2-3 sentence answer on legality and procedural compliance.
2. **STATUTORY PROVISIONS APPLIED**:
   - Quote and analyze each governing Section (e.g., Section 14 for 24-hr detention/remand, Section 18 for search powers, Section 19 for witness presence/deed execution).
3. **PROCEDURAL DEFECTS & IRREGULARITIES**:
   - Identify every procedural violation committed (e.g., unlawful nighttime search without emergency exception, absence of local representatives/witnesses, detention beyond 24 hours without remand).
4. **LEGAL CONSEQUENCES & REMEDIES**:
   - What happens to evidence seized in violation of mandatory procedural safeguards.
   - Habeas Corpus or bail remedy if detention is unlawful.
""",

    "CONSTITUTIONAL_SCENARIO": """
FOR THIS CONSTITUTIONAL SCENARIO / FACT PATTERN:
Structure your analysis using the FIRAC framework:
1. **FACTS SUMMARY**: Restate the key facts in 2 sentences.
2. **LEGAL ISSUES**: State the specific constitutional questions.
3. **GOVERNING CONSTITUTIONAL PROVISIONS**: Cite and break down the relevant fundamental rights with clause precision.
4. **LEGAL ANALYSIS**: Apply each provision to the facts.
5. **REMEDY PATHWAY**: Specify the remedy chain (Article 46 -> Article 133 / 144 writ petition).
""",

    "MIXED": """
FOR THIS MIXED CONSTITUTIONAL AND STATUTORY MATTER:
Structure your answer into distinct sections:
1. **PART A - STATUTORY PROCEDURAL ANALYSIS (Criminal Procedure Code, 2017)**:
   - Detailed statutory analysis under the relevant Sections (e.g., Sections 14, 18, 19, 72).
   - Analysis of procedural legality, detention time limits, and search defects.
2. **PART B - CONSTITUTIONAL PROTECTIONS & FUNDAMENTAL RIGHTS (Constitution of Nepal, 2015)**:
   - Fundamental rights of the accused under Article 20 (Right relating to justice), Article 23 (Preventive detention), Article 17.
   - Constitutional guarantees for 24-hour presentation before a judicial authority.
3. **PART C - REMEDIES & ACTIONS**:
   - Court remedies (remand contest, bail application, writ of Habeas Corpus under Article 133/144).
""",
}


def build_system_prompt(intent: dict[str, Any]) -> str:
    """Assemble system prompt based on query intent."""
    intent_type = intent.get("intent", "MIXED")
    domain_block = DOMAIN_INSTRUCTIONS.get(intent_type, DOMAIN_INSTRUCTIONS["MIXED"])
    return BASE_SYSTEM_PROMPT + "\n\n" + domain_block


def stream_openrouter(messages: list[dict[str, str]]) -> str:
    """Stream response from OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing in .env file.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.1,
    }

    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")

    full_response = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                full_response.append(delta)
        except Exception:
            continue

    print()
    return "".join(full_response)


# ============================================================================
# Session & User Interface
# ============================================================================

def print_banner():
    banner = """
================================================================================
           NEPAL LEGAL AI ASSISTANT (SAMVIDHAN & NYAYA)
       Constitution (2015) + Criminal Procedure Code (2017)
             Two-Tier pgvector Store | OpenRouter LLM
================================================================================
"""
    print(banner)


def select_or_create_session(user: dict[str, Any]) -> dict[str, Any]:
    """Allows user to resume an existing conversation or start a new one."""
    sessions = auth_service.get_user_chat_sessions(user["id"])

    print("\n------------------------------------------------------------")
    print("CONVERSATION SESSIONS:")
    print("  [0] Start New Conversation")
    for idx, s in enumerate(sessions, 1):
        dt = s["created_at"].strftime("%Y-%m-%d %H:%M") if hasattr(s["created_at"], "strftime") else str(s["created_at"])[:16]
        msgs = s.get("message_count", 0)
        mode = s.get("chat_mode", "all_laws")
        print(f"  [{idx}] {s['title']} ({msgs} msgs, mode: {mode}, created: {dt})")
    print("------------------------------------------------------------")

    choice = input("Select a session (or 0 for new): ").strip()
    if choice and choice != "0" and choice.isdigit():
        sel_idx = int(choice) - 1
        if 0 <= sel_idx < len(sessions):
            return sessions[sel_idx]

    # Create new session
    title = input("Enter a title for this chat (press Enter for 'Legal Consultation'): ").strip()
    if not title:
        title = "Legal Consultation"

    print("\nSelect Legal Scope:")
    print("  [1] All Laws (Constitution + Criminal Procedure Code) [Recommended]")
    print("  [2] The National Criminal Procedure (Code) Act, 2017")
    print("  [3] Constitution of Nepal, 2015")
    mode_choice = input("Select scope (1-3, default 1): ").strip()

    if mode_choice == "2":
        chat_mode = "criminal_procedure"
    elif mode_choice == "3":
        chat_mode = "constitution"
    else:
        chat_mode = "all_laws"

    return auth_service.create_chat_session(user["id"], title=title, chat_mode=chat_mode)


def chat_loop(user: dict[str, Any], session: dict[str, Any]):
    chat_mode = session.get("chat_mode", "all_laws")
    mode_desc = {
        "all_laws": "All Nepali Laws (Constitution + Criminal Procedure Code)",
        "criminal_procedure": "The National Criminal Procedure (Code) Act, 2017",
        "constitution": "The Constitution of Nepal, 2015",
    }.get(chat_mode, "All Laws")

    print("\n" + "=" * 70)
    print(f"SESSION: {session['title']}")
    print(f"SCOPE:   {mode_desc}")
    print("Ask any question regarding Nepali constitutional or criminal procedural law.")
    print("Commands: /history, /sessions, /sources, /exit")
    print("=" * 70)

    last_retrieved_sources = []

    while True:
        try:
            prompt = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session...")
            break

        if not prompt:
            continue

        if prompt.lower() in ("/exit", "/quit", "/back"):
            break
        elif prompt.lower() == "/history":
            history = auth_service.get_chat_history(session["id"], user["id"])
            print(f"\n--- SESSION HISTORY ({len(history)} messages) ---")
            for msg in history:
                speaker = "YOU" if msg["role"] == "user" else "LEGAL AI"
                print(f"\n[{speaker}]:")
                print(msg["content"])
            print("--- END HISTORY ---\n")
            continue
        elif prompt.lower() == "/sources":
            if not last_retrieved_sources:
                print("\nNo sources retrieved yet for this session.\n")
            else:
                print("\n--- RETRIEVED SOURCES FOR LAST ANSWER ---")
                for s in last_retrieved_sources:
                    doc = s.get("document_title") or s.get("document_id", "")
                    ref = s.get("structural_ref") or s.get("title", "")
                    xref = " [CROSS-REF]" if s.get("is_cross_ref") else ""
                    print(f"- {doc} | {ref}: {s.get('title')} (Page {s.get('page')}){xref}")
                print("------------------------------------------\n")
            continue
        elif prompt.lower() == "/sessions":
            session = select_or_create_session(user)
            chat_mode = session.get("chat_mode", "all_laws")
            print(f"\nSwitched to session: {session['title']} (Mode: {chat_mode})\n")
            continue

        # 1. Intent Classification
        print("\n[1/5] Classifying query intent...")
        intent = classify_query_intent(prompt)
        intent_type = intent.get("intent", "MIXED")
        print(f"       Domain: {intent_type}")
        if intent.get("suggested_provisions"):
            print(f"       Suggested Provisions: {intent['suggested_provisions']}")

        # 2. Query Decomposition
        print("[2/5] Decomposing into sub-queries...")
        sub_queries = decompose_query(prompt, intent)
        print(f"       Generated {len(sub_queries)} search angles")

        # 3. Two-Tier Multi-Document Retrieval
        print("[3/5] Retrieving from Two-Tier Supabase Knowledge Base...")
        doc_filter = None if chat_mode == "all_laws" else chat_mode
        candidates = retrieve_multi_query(
            sub_queries,
            top_k_per_query=10,
            document_id=doc_filter,
            user_prompt=prompt,
            suggested_provisions=intent.get("suggested_provisions", []),
        )
        print(f"       Retrieved {len(candidates)} candidate provisions")

        # Expand constitutional cross-references
        candidates = expand_with_cross_references(candidates, intent)

        # 4. Scenario-Aware Reranking (with Statutory & Constitutional Quota)
        print("[4/5] Reranking candidates...")
        reranked = rerank_candidates(prompt, candidates, intent, top_k=8)
        print(f"       Top {len(reranked)} provisions selected")

        # Hydrate full parent text
        hydrated = hydrate_parent_context(reranked)
        last_retrieved_sources = hydrated

        # 5. Domain-Aware Dynamic Prompt Assembly
        print("[5/5] Assembling statutory & constitutional prompt...")
        context_blocks = []
        citations_metadata = []

        for doc in hydrated:
            doc_name = doc.get("document_title") or doc.get("document_id", "Nepal Law")
            ref_name = doc.get("structural_ref") or doc.get("title", "")
            parent_title = doc.get("parent_title", "")
            parent_num = doc.get("parent_number")
            page_num = doc.get("page")
            content = doc.get("content", "")
            xref_tag = " [Cross-Referenced]" if doc.get("is_cross_ref") else ""

            block = (
                f"--- SOURCE: {doc_name} | {ref_name}{xref_tag} ---\n"
                f"Chapter/Part: {parent_title} (No. {parent_num}) | Citation: {ref_name} | Page: {page_num}\n"
                f"{content}\n"
            )
            context_blocks.append(block)

            citations_metadata.append({
                "document_id": doc.get("document_id"),
                "document_title": doc_name,
                "structural_ref": ref_name,
                "unit_number": doc.get("unit_number"),
                "page": page_num,
                "title": doc.get("title"),
                "is_cross_ref": doc.get("is_cross_ref", False),
            })

        combined_context = "\n".join(context_blocks)
        system_prompt = build_system_prompt(intent)

        # Multi-turn history
        history = auth_service.get_chat_history(session["id"], user["id"])
        recent_history = history[-6:]

        messages = [{"role": "system", "content": system_prompt}]
        for h in recent_history:
            messages.append({"role": h["role"], "content": h["content"]})

        user_turn_content = f"""QUERY INTENT: {intent_type}
{f"ANALYSIS FOCUS: {intent.get('reasoning', '')}" if intent.get('reasoning') else ""}

RETRIEVED LEGAL CONTEXT (CONSTITUTION & STATUTES):
{combined_context}

USER QUESTION / SCENARIO:
{prompt}

Please answer the user's question directly and thoroughly with exact Article / Section citations from the context above."""

        messages.append({"role": "user", "content": user_turn_content})

        # Stream LLM Response
        print("\n" + "=" * 70)
        print("LEGAL AI:")
        print("=" * 70)
        try:
            ai_response = stream_openrouter(messages)
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")
            continue

        # Save conversation turn
        auth_service.add_chat_message(
            session_id=session["id"],
            user_id=user["id"],
            role="user",
            content=prompt,
        )
        auth_service.add_chat_message(
            session_id=session["id"],
            user_id=user["id"],
            role="assistant",
            content=ai_response,
            sources=citations_metadata,
        )

        # Display Citations Footer
        if citations_metadata:
            print("\n------------------------------------------------------------")
            print("PRIMARY LEGAL PROVISIONS APPLIED:")
            seen_refs = set()
            for c in citations_metadata:
                ref_key = f"{c.get('document_id')}:{c.get('structural_ref')}"
                if ref_key not in seen_refs:
                    seen_refs.add(ref_key)
                    doc_title = c.get("document_title", "")
                    ref = c.get("structural_ref", "")
                    page = c.get("page")
                    xref = " [via Cross-Reference]" if c.get("is_cross_ref") else ""
                    print(f"  * {doc_title} -- {ref}: {c.get('title')} (Page {page}){xref}")
            print("------------------------------------------------------------")


# ============================================================================
# Main Entry Point & Authentication
# ============================================================================

def handle_login() -> dict[str, Any] | None:
    print("\n--- USER LOGIN ---")
    email = input("Email: ").strip()
    password = getpass.getpass("Password: ")
    try:
        user = auth_service.login(email, password)
        print(f"\n[SUCCESS] Login successful! Welcome back, {user['first_name']} {user['last_name']}.")
        return user
    except Exception as e:
        print(f"\n[ERROR] Login failed: {e}")
        return None


def handle_signup() -> dict[str, Any] | None:
    print("\n--- NEW USER SIGNUP ---")
    first_name = input("First Name: ").strip()
    last_name = input("Last Name: ").strip()
    email = input("Email: ").strip()
    password = getpass.getpass("Password (min 8 chars, 1 upper, 1 digit, 1 special): ")
    try:
        user = auth_service.signup(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password,
            auto_verify=True,
        )
        print(f"\n[SUCCESS] Account created! Welcome, {user['first_name']}.")
        return user
    except Exception as e:
        print(f"\n[ERROR] Signup failed: {e}")
        return None


def main():
    print_banner()

    graph = get_article_graph()
    if graph:
        print(f"[INIT] Constitution cross-reference graph loaded: {len(graph)} articles")

    current_user = None
    while current_user is None:
        print("\nMAIN MENU:")
        print("  [1] Log In")
        print("  [2] Sign Up")
        print("  [3] Exit")
        choice = input("\nSelect an option (1-3): ").strip()
        if choice == "1":
            current_user = handle_login()
        elif choice == "2":
            current_user = handle_signup()
        elif choice == "3":
            print("\nGoodbye!")
            sys.exit(0)

    session = select_or_create_session(current_user)
    chat_loop(current_user, session)


if __name__ == "__main__":
    main()
