"""
chat_cli.py -- Interactive Terminal Legal AI Chat Engine (v2)

5-Layer Scenario-Aware Retrieval Architecture:
  Layer 1: Query Intent Classifier (CONSTITUTIONAL_DIRECT / SCENARIO / STATUTORY / MIXED)
  Layer 2: Multi-Query Decomposition (3 diverse sub-queries per scenario)
  Layer 3: Enhanced Hybrid Search + Article Cross-Reference Graph Expansion
  Layer 4: Scenario-Aware Reranking (intent-boosted cross-encoder)
  Layer 5: Domain-Aware Dynamic Prompt Assembly

Also includes:
  - User Signup & Login with password hashing & field validation
  - Isolated Chat Sessions & History stored in user_vault schema
  - Real-time streaming via OpenRouter (Llama 3.3 70B)
  - Verified Article, Clause, and Page Number Citations
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

hf_token = os.getenv("HF_TOKEN", "")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

# Global model caches
_EMBED_MODEL = None
_RERANK_MODEL = None
_ARTICLE_GRAPH: dict[str, list[int]] | None = None

OKF_DIR = Path(__file__).resolve().parent / "data" / "okf" / "constitution"
if not OKF_DIR.exists():
    OKF_DIR = Path(__file__).with_name("okf_output")
ARTICLE_GRAPH_PATH = OKF_DIR / "article_graph.json"


def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def get_rerank_model():
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        from sentence_transformers import CrossEncoder
        _RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANK_MODEL


def get_article_graph() -> dict[str, list[int]]:
    """Load the pre-built article cross-reference graph from JSON."""
    global _ARTICLE_GRAPH
    if _ARTICLE_GRAPH is None:
        if ARTICLE_GRAPH_PATH.exists():
            try:
                _ARTICLE_GRAPH = json.loads(ARTICLE_GRAPH_PATH.read_text(encoding="utf-8"))
            except Exception:
                _ARTICLE_GRAPH = {}
        else:
            _ARTICLE_GRAPH = {}
    return _ARTICLE_GRAPH


# ============================================================================
# LAYER 1: Query Intent Classifier
# ============================================================================

# Valid intent categories
INTENT_CATEGORIES = {
    "CONSTITUTIONAL_DIRECT",
    "CONSTITUTIONAL_SCENARIO",
    "STATUTORY_CIVIL",
    "STATUTORY_CRIMINAL",
    "MIXED",
}


def classify_query_intent(query: str) -> dict[str, Any]:
    """Classify the query's legal domain and intent using a fast LLM call.

    Returns:
        dict with keys: intent, reasoning, suggested_articles
    """
    default_result = {
        "intent": "CONSTITUTIONAL_DIRECT",
        "reasoning": "Default classification",
        "suggested_articles": [],
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
            "You are a Nepali Legal Domain Classifier.\n"
            "Classify the user's query into exactly ONE of these categories:\n\n"
            "- CONSTITUTIONAL_DIRECT: Direct question about the Constitution of Nepal "
            "(eligibility, term limits, structure, powers, fundamental rights definitions).\n"
            "- CONSTITUTIONAL_SCENARIO: A real-world scenario/story where constitutional "
            "fundamental rights are the PRIMARY governing law (e.g., discrimination by state, "
            "denial of citizenship, state censorship, forced labor by state).\n"
            "- STATUTORY_CIVIL: A private dispute between citizens best resolved under statutory "
            "law like Muluki Civil Code, 2074 (property boundary disputes, contract disputes, "
            "inheritance, adverse possession, tenancy, private trespass).\n"
            "- STATUTORY_CRIMINAL: A criminal matter under Muluki Criminal Code "
            "(theft, assault, fraud, murder, domestic violence).\n"
            "- MIXED: Has BOTH constitutional AND statutory dimensions "
            "(e.g., state-involved land acquisition, police brutality + criminal complaint).\n\n"
            "Also suggest 2-4 specific Article numbers from Nepal's Constitution that are most relevant.\n\n"
            "Respond in this EXACT JSON format and nothing else:\n"
            '{"intent": "CATEGORY_NAME", "reasoning": "one sentence why", "suggested_articles": [25, 46]}\n\n'
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
            # Extract JSON from response (handle markdown code fences)
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Validate intent
                if result.get("intent") not in INTENT_CATEGORIES:
                    result["intent"] = "CONSTITUTIONAL_DIRECT"
                return result
    except Exception:
        pass

    return default_result


# ============================================================================
# LAYER 2: Multi-Query Decomposition
# ============================================================================

def decompose_query(query: str, intent: dict[str, Any]) -> list[str]:
    """Generate 3 diverse sub-queries targeting different retrieval angles.

    For direct questions, returns the original query plus 1 expansion.
    For scenarios, returns 3 distinct sub-queries.
    """
    if not OPENROUTER_API_KEY:
        return [query]

    intent_type = intent.get("intent", "CONSTITUTIONAL_DIRECT")

    # Direct questions don't need heavy decomposition
    if intent_type == "CONSTITUTIONAL_DIRECT":
        return _simple_expand(query)

    # Scenarios and mixed queries get full 3-angle decomposition
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        suggested_arts = intent.get("suggested_articles", [])
        arts_hint = f"Potentially relevant articles: {suggested_arts}" if suggested_arts else ""

        prompt = (
            "You are a Nepali Legal Search Query Decomposer.\n"
            f"Query type: {intent_type}\n"
            f"{arts_hint}\n\n"
            "Given the user's legal query/scenario below, generate EXACTLY 3 diverse search sub-queries "
            "that together cover all retrieval angles:\n\n"
            "1. RIGHTS QUERY: Target the specific constitutional fundamental rights/articles "
            "(e.g., 'Article 25 right to property protection from unlawful encroachment').\n"
            "2. REMEDY QUERY: Target the constitutional remedy pathway "
            "(e.g., 'Article 46 constitutional remedies Article 133 Supreme Court writ jurisdiction').\n"
            "3. PROCEDURAL QUERY: Target the practical/procedural dimension "
            "(e.g., 'land registration Kitta boundary demarcation survey District Court civil suit').\n\n"
            "Each sub-query should be 10-20 words of formal legal keywords.\n"
            "Respond as a JSON array of exactly 3 strings, nothing else.\n\n"
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
            # Extract JSON array
            arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if arr_match:
                sub_queries = json.loads(arr_match.group())
                if isinstance(sub_queries, list) and len(sub_queries) >= 2:
                    # Always include original query as well
                    return [query] + [str(sq) for sq in sub_queries[:3]]
    except Exception:
        pass

    # Fallback: simple expansion
    return _simple_expand(query)


def _simple_expand(query: str) -> list[str]:
    """Fallback: single expanded query (equivalent to old expand_query)."""
    if not OPENROUTER_API_KEY:
        return [query]
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        prompt = (
            "You are a Senior Constitutional & Legal Analyst for Nepal.\n"
            "Analyze the user's input. It may be a direct constitutional question OR a real-world factual scenario.\n"
            "TASK: Identify the core legal issues, the governing Constitutional Articles/Rights, and statutory legal doctrines under Nepali law.\n"
            "Output 4 to 6 formal legal search terms and governing Articles.\n"
            "Output ONLY the comma-separated legal terms on one line, nothing else.\n"
            f"User input: '{query}'"
        )
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 60,
            "temperature": 0.0,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=6)
        if resp.status_code == 200:
            expanded = resp.json()["choices"][0]["message"]["content"].strip()
            expanded = re.sub(r'["\n]', ' ', expanded).strip()
            return [query, f"{query} {expanded}"]
    except Exception:
        pass
    return [query]


# ============================================================================
# LAYER 3: Hybrid Retrieval + Article Cross-Reference Graph Expansion
# ============================================================================

def retrieve_hybrid_candidates(query: str, expanded_terms: str, top_k: int = 15) -> list[dict[str, Any]]:
    """Retrieve top candidates using PostgreSQL Full-Text Search (BM25) + Vector Cosine with RRF."""
    model = get_embed_model()
    query_emb = model.encode([query], normalize_embeddings=True).tolist()[0]

    conn = auth_service.get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    search_text = f"{query} {expanded_terms}".replace("'", "").strip()

    sql = """
        SELECT *
        FROM knowledge.hybrid_search_okf(
            query_text => %s,
            query_embedding => %s::vector,
            match_count => %s,
            rrf_k => 60
        );
    """
    try:
        cur.execute(sql, (search_text, str(query_emb), top_k))
        rows = cur.fetchall()
        results = [dict(r) for r in rows]
    except Exception:
        # Fallback to standard vector search if stored procedure is unavailable
        fallback_sql = """
            SELECT *, (1 - (embedding <=> %s::vector)) AS rrf_score
            FROM knowledge.okf_documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        cur.execute(fallback_sql, (str(query_emb), str(query_emb), top_k))
        results = [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()

    return results


def retrieve_multi_query(sub_queries: list[str], top_k_per_query: int = 10) -> list[dict[str, Any]]:
    """Run hybrid retrieval for each sub-query, then merge and deduplicate by document ID."""
    all_candidates: dict[int, dict[str, Any]] = {}

    for sq in sub_queries:
        candidates = retrieve_hybrid_candidates(sq, sq, top_k=top_k_per_query)
        for c in candidates:
            doc_id = c.get("id")
            if doc_id is not None:
                existing = all_candidates.get(doc_id)
                if existing is None:
                    c["retrieval_count"] = 1
                    all_candidates[doc_id] = c
                else:
                    # Boost score for documents found by multiple sub-queries
                    existing["retrieval_count"] = existing.get("retrieval_count", 1) + 1
                    existing_score = existing.get("rrf_score", 0)
                    new_score = c.get("rrf_score", 0)
                    if new_score > existing_score:
                        existing["rrf_score"] = new_score

    # Sort by (retrieval_count DESC, rrf_score DESC)
    merged = sorted(
        all_candidates.values(),
        key=lambda x: (x.get("retrieval_count", 1), x.get("rrf_score", 0)),
        reverse=True,
    )
    return merged


def expand_with_cross_references(
    candidates: list[dict[str, Any]],
    intent: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand retrieved candidates by pulling in cross-referenced articles from the graph.

    Also pulls in articles suggested by the intent classifier.
    """
    graph = get_article_graph()

    # Collect already-retrieved article numbers
    retrieved_articles = set()
    for c in candidates:
        art = c.get("article_number")
        if art:
            retrieved_articles.add(art)

    # Find cross-referenced articles not yet retrieved
    articles_to_fetch: set[int] = set()

    # From graph (articles referenced inside the text of retrieved articles)
    for art_num in list(retrieved_articles):
        refs = graph.get(str(art_num), [])
        for ref in refs:
            if ref not in retrieved_articles:
                articles_to_fetch.add(ref)

    # From intent classifier's suggested articles
    for art_num in intent.get("suggested_articles", []):
        if isinstance(art_num, int) and art_num not in retrieved_articles:
            articles_to_fetch.add(art_num)

    if not articles_to_fetch:
        return candidates

    # Fetch these articles from the database
    conn = auth_service.get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Get the first chunk (chunk_index=0) for each cross-referenced article
        placeholders = ",".join(["%s"] * len(articles_to_fetch))
        sql = f"""
            SELECT DISTINCT ON (article_number)
                id, file_path, chunk_index, doc_type, title, content,
                part_number, part_title, article_number, page,
                clause_count, sub_clause_count, proviso_count, tags
            FROM knowledge.okf_documents
            WHERE article_number IN ({placeholders})
            ORDER BY article_number, chunk_index
        """
        cur.execute(sql, list(articles_to_fetch))
        cross_ref_rows = cur.fetchall()

        for row in cross_ref_rows:
            entry = dict(row)
            entry["rrf_score"] = 0.005  # Low score -- will be reranked
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
# LAYER 4: Scenario-Aware Reranking
# ============================================================================

# Part 3 articles are fundamental rights (most relevant for scenarios)
FUNDAMENTAL_RIGHTS_PART = 3
REMEDY_ARTICLES = {46, 133, 144}


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    intent: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rerank candidates using a Cross-Encoder with intent-aware scoring boosts.

    For scenarios: boosts Part 3 (fundamental rights) and remedy articles.
    For direct questions: uses top_k=3 (tighter).
    """
    if not candidates:
        return []

    intent_type = intent.get("intent", "CONSTITUTIONAL_DIRECT")

    # Adjust top_k based on query type
    if intent_type == "CONSTITUTIONAL_DIRECT":
        top_k = min(top_k, 3)
    elif intent_type in ("CONSTITUTIONAL_SCENARIO", "MIXED"):
        top_k = max(top_k, 5)
    elif intent_type in ("STATUTORY_CIVIL", "STATUTORY_CRIMINAL"):
        top_k = max(top_k, 4)

    if len(candidates) <= top_k:
        return candidates

    try:
        reranker = get_rerank_model()
        pairs = [(query, c.get("content", "")) for c in candidates]
        scores = reranker.predict(pairs)

        for c, score in zip(candidates, scores):
            base_score = float(score)

            # Intent-aware boosting
            if intent_type in ("CONSTITUTIONAL_SCENARIO", "MIXED"):
                art_num = c.get("article_number")
                part_num = c.get("part_number")

                # Boost fundamental rights articles
                if part_num == FUNDAMENTAL_RIGHTS_PART:
                    base_score += 1.5

                # Boost remedy chain articles
                if art_num in REMEDY_ARTICLES:
                    base_score += 2.0

                # Boost articles found by multiple sub-queries
                retrieval_count = c.get("retrieval_count", 1)
                if retrieval_count >= 2:
                    base_score += 1.0

                # Boost cross-referenced articles (they were pulled in for a reason)
                if c.get("is_cross_ref"):
                    base_score += 0.5

            c["rerank_score"] = base_score

        ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return ranked[:top_k]
    except Exception:
        return candidates[:top_k]


# ============================================================================
# Parent-Child (Hierarchical) Context Hydration
# ============================================================================

def hydrate_parent_context(child_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hydrate child chunks with their full Parent Article markdown for comprehensive legal analysis."""
    hydrated = []
    seen_articles = set()

    for chunk in child_chunks:
        art_num = chunk.get("article_number")

        if art_num and art_num not in seen_articles:
            seen_articles.add(art_num)
            parent_file = OKF_DIR / "articles" / f"article-{art_num:03d}.md"
            if parent_file.exists():
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

        if not art_num or art_num not in seen_articles:
            hydrated.append(chunk)

    return hydrated


# ============================================================================
# LAYER 5: Domain-Aware Prompt Assembly
# ============================================================================

BASE_SYSTEM_PROMPT = """You are "Samvidhan AI", an authoritative and expert Legal AI Assistant specialized in the Constitution of Nepal (2072 / 2015) and Nepali jurisprudence.

Your mission is to provide rigorous, clear, and highly accurate legal and constitutional analysis.

STRICT LEGAL CITATION RULES:
1. CITATION NOTATION:
   - Always use standard constitutional legal notation: "Article {N}({clause})({subclause})" (e.g., "Article 25(1)" or "Article 64(1)(b)").
   - State the PDF Page number in parentheses when introducing the provision (e.g., "Article 25 (Page 13)").
"""

# Domain-specific instruction blocks injected based on query intent
DOMAIN_INSTRUCTIONS = {
    "CONSTITUTIONAL_DIRECT": """
FOR THIS DIRECT CONSTITUTIONAL QUESTION:
- Provide a direct 1-2 sentence executive answer first.
- Break down the text of the governing Article and Clauses.
- Include a summary table: | Article & Clause | Constitutional Requirement | PDF Page |
- Cite verbatim clause text where possible.
""",

    "CONSTITUTIONAL_SCENARIO": """
FOR THIS CONSTITUTIONAL SCENARIO / FACT PATTERN:
Structure your analysis using the FIRAC legal framework:

1. **FACTS SUMMARY**: Restate the key facts in 2-3 sentences.
2. **LEGAL ISSUES SPOTTING**: State the specific questions of law raised by the facts.
3. **GOVERNING CONSTITUTIONAL PROVISIONS**: Cite and break down the relevant fundamental rights with clause-level precision. Include the full text of governing clauses.
4. **LEGAL ANALYSIS (Application to Facts)**: Apply each constitutional provision to the specific facts. Explain WHY each provision applies or does not apply.
5. **REMEDY PATHWAY**: Specify the exact constitutional remedy chain:
   - Article 46 (Right to Constitutional Remedies)
   - Article 133 (Supreme Court extraordinary writ jurisdiction) OR Article 144 (High Court writ jurisdiction)
   - Specify which type of writ applies (certiorari, mandamus, habeas corpus, prohibition, quo warranto)
6. **ACTIONABLE LEGAL NEXT STEPS**: Concrete lawful procedure the person should follow.

ANTI-HALLUCINATION GUARDRAILS:
- ONLY cite articles that appear in the RETRIEVED CONSTITUTIONAL CONTEXT below.
- If the scenario involves a dimension that the Constitution does not directly address (private civil disputes), explicitly state that the Constitution provides the foundational right but enforcement is through statutory law.
""",

    "STATUTORY_CIVIL": """
FOR THIS PRIVATE CIVIL LAW DISPUTE:

CRITICAL INSTRUCTION: This dispute is primarily governed by STATUTORY law, not constitutional writs.

Structure your response as follows:
1. **CONSTITUTIONAL FOUNDATION**: Identify the fundamental right that provides the constitutional backdrop (e.g., Article 25 - Right to Property). Quote the relevant clause text.
2. **STATUTORY JURISDICTION NOTICE**: State clearly and prominently:
   > "This dispute is primarily adjudicated under the **Muluki Civil Code, 2074 (muulukii devaanii sanhitaa)** in the **District Court**, not through a constitutional writ petition."
3. **APPLICABLE CONSTITUTIONAL PROVISIONS**: Break down which constitutional rights are foundational guarantees (right to property, right to justice, etc.).
4. **PRACTICAL LEGAL PATHWAY**:
   - Administrative remedies first (Land Revenue Office / Survey Office / Napi-Malpot for boundary demarcation)
   - Filing a civil suit in District Court (injunction, eviction, damages, specific performance)
   - Constitutional writ ONLY if a fundamental right is violated by the STATE
5. **WHAT THE CONSTITUTION DOES AND DOES NOT COVER**: Explicitly distinguish between the constitutional guarantee and the statutory enforcement mechanism.

NEVER invent or hallucinate non-existent constitutional clauses for everyday civil matters.
""",

    "STATUTORY_CRIMINAL": """
FOR THIS CRIMINAL LAW MATTER:

CRITICAL INSTRUCTION: Criminal matters are prosecuted under the **Muluki Criminal Code, 2074 (muulukii phoujdaarii sanhitaa)**.

Structure your response as follows:
1. **CONSTITUTIONAL RIGHTS OF THE ACCUSED/VICTIM**: Identify fundamental rights at stake (e.g., Article 20 - Right relating to justice, Article 22 - Right against torture, Article 17 - Right to freedom).
2. **CRIMINAL JURISDICTION**: State that investigation and prosecution occur through Nepal Police and the District Attorney's Office under the Muluki Criminal Code.
3. **CONSTITUTIONAL PROTECTIONS**: Detail constitutional safeguards (right to fair trial, presumption of innocence, right against self-incrimination, right to legal counsel).
4. **PRACTICAL STEPS**: Filing an FIR, investigation process, court jurisdiction.
""",

    "MIXED": """
FOR THIS MIXED (CONSTITUTIONAL + STATUTORY) MATTER:

This query has BOTH constitutional and statutory dimensions. Address them separately:

**PART A - CONSTITUTIONAL DIMENSION:**
- Identify which fundamental rights are at stake.
- Use FIRAC framework for the constitutional analysis.
- Specify the remedy pathway (Article 46 -> Article 133/144 writs).

**PART B - STATUTORY DIMENSION:**
- Identify which statutory law governs the non-constitutional aspect.
- Specify the court jurisdiction and procedural pathway.
- Distinguish clearly between what requires a constitutional writ vs. a regular civil/criminal suit.

ANTI-HALLUCINATION: Only cite constitutional articles present in the context. State explicitly when statutory law (Muluki Civil/Criminal Code) governs instead of the Constitution.
""",
}


def build_system_prompt(intent: dict[str, Any]) -> str:
    """Assemble the system prompt dynamically based on query intent classification."""
    intent_type = intent.get("intent", "CONSTITUTIONAL_DIRECT")
    domain_block = DOMAIN_INSTRUCTIONS.get(intent_type, DOMAIN_INSTRUCTIONS["CONSTITUTIONAL_DIRECT"])
    return BASE_SYSTEM_PROMPT + domain_block


# ============================================================================
# LLM Generation via OpenRouter (Streaming)
# ============================================================================

def stream_openrouter(messages: list[dict[str, str]]) -> str:
    """Stream OpenRouter chat completions token by token to stdout in real time."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing in .env file.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://okf-samvidhan.local",
        "X-Title": "Nepal Samvidhan Legal AI",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1500,
        "stream": True,
    }

    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API Error ({resp.status_code}): {resp.text}")

    full_chunks = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            chunk_data = line[6:].strip()
            if chunk_data == "[DONE]":
                break
            try:
                chunk_json = json.loads(chunk_data)
                delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                content_chunk = delta.get("content", "")
                if content_chunk:
                    full_chunks.append(content_chunk)
                    sys.stdout.write(content_chunk)
                    sys.stdout.flush()
            except json.JSONDecodeError:
                continue

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(full_chunks)


# ============================================================================
# Terminal UI Helpers
# ============================================================================

def print_banner():
    banner = """
================================================================================
           NEPAL SAMVIDHAN (CONSTITUTION) LEGAL AI ASSISTANT
            5-Layer Scenario-Aware RAG | Multi-Tenant OKF
================================================================================
"""
    print(banner)


def handle_signup() -> dict[str, Any] | None:
    print("\n--- NEW USER SIGNUP ---")
    try:
        first_name = input("First Name *: ").strip()
        middle_name = input("Middle Name (optional): ").strip() or None
        last_name = input("Last Name *: ").strip()
        phone_number = input("Phone Number (optional): ").strip() or None
        address = input("Address (optional): ").strip() or None
        profession = input("Profession (optional, e.g. Lawyer, Student): ").strip() or None
        organization = input("Organization (optional): ").strip() or None
        email = input("Email *: ").strip()

        # Secure password input
        password = getpass.getpass("Password * (min 6 chars): ")
        confirm_password = getpass.getpass("Confirm Password *: ")

        # Step 1: Initiate signup & send OTP to email
        print(f"\n[INFO] Sending 6-digit verification code to {email}...")
        signup_res = auth_service.initiate_signup(
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
        )

        clean_email = signup_res["email"]
        print("\n" + "=" * 65)
        print(f"  VERIFICATION CODE SENT")
        print(f"  A 6-digit OTP code has been dispatched to: {clean_email}")
        print("  Please check your inbox (and spam folder). Code valid for 10 min.")
        print("=" * 65)

        # Step 2: Loop to verify OTP
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            otp_input = input("\nEnter 6-digit OTP [or 'r' to resend, 'c' to cancel]: ").strip()

            if otp_input.lower() in ("c", "cancel", "q", "quit"):
                print("[INFO] Signup cancelled. You can sign up again at any time.")
                return None

            if otp_input.lower() in ("r", "resend"):
                try:
                    auth_service.resend_signup_otp(clean_email)
                    print(f"[SUCCESS] Fresh verification code sent to {clean_email}.")
                except Exception as ex:
                    print(f"[ERROR] Could not resend code: {ex}")
                continue

            if not otp_input:
                continue

            try:
                user = auth_service.complete_signup(clean_email, otp_input)
                print(f"\n[SUCCESS] Email verified! Welcome, {user['first_name']} {user['last_name']}.")
                return user
            except Exception as e:
                attempts += 1
                remaining = max_attempts - attempts
                print(f"[ERROR] {e} ({remaining} attempts remaining)")

        print("\n[ERROR] Maximum verification attempts reached. Please try signing up again.")
        return None

    except Exception as e:
        print(f"\n[ERROR] Signup failed: {e}")
        return None


def handle_login() -> dict[str, Any] | None:
    print("\n--- USER LOGIN ---")
    try:
        email = input("Email: ").strip()
        password = getpass.getpass("Password: ")
        
        try:
            user = auth_service.login(email, password)
            print(f"\n[SUCCESS] Login successful! Welcome back, {user['first_name']} {user['last_name']}.")
            return user
        except ValueError as ve:
            if "not verified" in str(ve).lower():
                print(f"\n[NOTICE] {ve}")
                verify_choice = input("Would you like to verify your email now? (y/n): ").strip().lower()
                if verify_choice == "y":
                    try:
                        auth_service.resend_signup_otp(email)
                        otp_input = input("Enter 6-digit OTP sent to your email: ").strip()
                        user = auth_service.complete_signup(email, otp_input)
                        print(f"\n[SUCCESS] Email verified! Welcome, {user['first_name']} {user['last_name']}.")
                        return user
                    except Exception as otp_err:
                        print(f"[ERROR] Verification failed: {otp_err}")
                        return None
            else:
                raise ve

    except Exception as e:
        print(f"\n[ERROR] Login failed: {e}")
        return None



def select_or_create_session(user: dict[str, Any]) -> dict[str, Any]:
    """Allows user to resume an existing conversation or start a new one."""
    sessions = auth_service.get_user_chat_sessions(user["id"])

    print("\n------------------------------------------------------------")
    print("CONVERSATION SESSIONS:")
    print("  [0] Start New Conversation")
    for idx, s in enumerate(sessions, 1):
        dt = s["created_at"].strftime("%Y-%m-%d %H:%M") if hasattr(s["created_at"], "strftime") else str(s["created_at"])[:16]
        msgs = s.get("message_count", 0)
        print(f"  [{idx}] {s['title']} ({msgs} msgs, created: {dt})")
    print("------------------------------------------------------------")

    choice = input("Select a session (or 0 for new): ").strip()
    if choice and choice != "0" and choice.isdigit():
        sel_idx = int(choice) - 1
        if 0 <= sel_idx < len(sessions):
            return sessions[sel_idx]

    # Create new session
    title = input("Enter a title for this chat (press Enter for 'General Inquiry'): ").strip()
    if not title:
        title = "General Inquiry"
    return auth_service.create_chat_session(user["id"], title=title, chat_mode="constitution")


# ============================================================================
# Main Interactive Chat Loop (5-Layer Pipeline)
# ============================================================================

def chat_loop(user: dict[str, Any], session: dict[str, Any]):
    print("\n" + "=" * 70)
    print(f"SESSION: {session['title']} (Mode: {session.get('chat_mode', 'constitution')})")
    print("Ask any question regarding the Constitution of Nepal.")
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

        # Command handling
        if prompt.lower() in ("/exit", "/quit", "/back"):
            break
        elif prompt.lower() == "/history":
            history = auth_service.get_chat_history(session["id"], user["id"])
            print(f"\n--- SESSION HISTORY ({len(history)} messages) ---")
            for msg in history:
                speaker = "YOU" if msg["role"] == "user" else "SAMVIDHAN AI"
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
                    xref_tag = " [CROSS-REF]" if s.get("is_cross_ref") else ""
                    print(f"- {s.get('title')} | Page: {s.get('page')} | Score: {s.get('rerank_score', 0):.4f}{xref_tag}")
                print("------------------------------------------\n")
            continue
        elif prompt.lower() == "/sessions":
            session = select_or_create_session(user)
            print(f"\nSwitched to session: {session['title']}\n")
            continue

        # ================================================================
        # LAYER 1: Query Intent Classification
        # ================================================================
        print("\n[1/5] Classifying query intent...")
        intent = classify_query_intent(prompt)
        intent_type = intent.get("intent", "CONSTITUTIONAL_DIRECT")
        print(f"       Domain: {intent_type}")
        if intent.get("suggested_articles"):
            print(f"       Suggested Articles: {intent['suggested_articles']}")

        # ================================================================
        # LAYER 2: Multi-Query Decomposition
        # ================================================================
        print("[2/5] Decomposing into sub-queries...")
        sub_queries = decompose_query(prompt, intent)
        print(f"       Generated {len(sub_queries)} search queries")

        # ================================================================
        # LAYER 3: Multi-Query Hybrid Retrieval + Graph Expansion
        # ================================================================
        print("[3/5] Hybrid retrieval + cross-reference expansion...")
        candidates = retrieve_multi_query(sub_queries, top_k_per_query=10)
        print(f"       Retrieved {len(candidates)} unique candidates")

        # Expand with cross-referenced articles
        candidates = expand_with_cross_references(candidates, intent)
        print(f"       After graph expansion: {len(candidates)} candidates")

        # ================================================================
        # LAYER 4: Scenario-Aware Reranking
        # ================================================================
        print("[4/5] Reranking with intent-aware scoring...")
        reranked = rerank_candidates(prompt, candidates, intent, top_k=5)
        print(f"       Top {len(reranked)} candidates selected")

        # Parent-Child Context Hydration
        hydrated = hydrate_parent_context(reranked)
        last_retrieved_sources = hydrated

        # ================================================================
        # LAYER 5: Domain-Aware Prompt Assembly
        # ================================================================
        print("[5/5] Assembling domain-specific prompt...")

        # Format context for LLM
        context_blocks = []
        citations_metadata = []
        for doc in hydrated:
            art_num = doc.get("article_number")
            part_num = doc.get("part_number")
            page_num = doc.get("page")
            title = doc.get("title", "")
            content = doc.get("content", "")
            xref_tag = " [Cross-Referenced]" if doc.get("is_cross_ref") else ""

            block = f"--- SOURCE: {title}{xref_tag} ---\nPart: {part_num} | Article: {art_num} | Page: {page_num}\n{content}\n"
            context_blocks.append(block)

            citations_metadata.append({
                "article_number": art_num,
                "part_number": part_num,
                "page": page_num,
                "title": title,
                "clause_count": doc.get("clause_count", 0),
                "is_cross_ref": doc.get("is_cross_ref", False),
            })

        combined_context = "\n".join(context_blocks)

        # Build domain-aware system prompt
        system_prompt = build_system_prompt(intent)

        # Fetch recent conversation history for multi-turn context
        history = auth_service.get_chat_history(session["id"], user["id"])
        recent_history = history[-6:]  # Last 3 turns

        messages = [{"role": "system", "content": system_prompt}]

        for h in recent_history:
            messages.append({"role": h["role"], "content": h["content"]})

        # User turn with augmented context
        user_turn_content = f"""QUERY CLASSIFICATION: {intent_type}
{f"REASONING: {intent.get('reasoning', '')}" if intent.get('reasoning') else ""}

RETRIEVED CONSTITUTIONAL CONTEXT:
{combined_context}

USER QUESTION:
{prompt}

Please answer the user's question with precise Article and Clause citations from the context above."""

        messages.append({"role": "user", "content": user_turn_content})

        # Stream Response Live
        print("\n" + "=" * 70)
        print("SAMVIDHAN AI:")
        print("=" * 70)
        try:
            ai_response = stream_openrouter(messages)
        except Exception as e:
            print(f"\n[ERROR] Could not generate response: {e}")
            continue

        # Filter citations: Only retain articles actually referenced in the AI response
        mentioned_arts = set(map(int, re.findall(r"(?:Article|Art\.)\s*(\d+)", ai_response, re.IGNORECASE)))
        verified_sources = [
            c for c in citations_metadata
            if c.get("article_number") in mentioned_arts
        ]
        # Fallback to top retrieved document if no specific article number was matched
        if not verified_sources and citations_metadata:
            verified_sources = citations_metadata[:1]

        # Save to database (Isolated per user and session)
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
            sources=verified_sources,
        )

        # Display Clean Citations Footer
        if verified_sources:
            print("\n------------------------------------------------------------")
            print("PRIMARY CONSTITUTIONAL PROVISIONS APPLIED:")
            seen_articles = set()
            for c in verified_sources:
                art = c.get("article_number")
                if art and art not in seen_articles:
                    seen_articles.add(art)
                    xref = " [via Cross-Reference]" if c.get("is_cross_ref") else ""
                    print(f"  * Article {art}: {c.get('title')} (PDF Page {c.get('page')}){xref}")
            print("------------------------------------------------------------")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    print_banner()

    # Pre-load the article graph at startup
    graph = get_article_graph()
    if graph:
        print(f"[INIT] Article cross-reference graph loaded: {len(graph)} articles with outgoing refs")
    else:
        print("[INIT] WARNING: Article graph not found. Run 'python build_article_graph.py' first.")

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
        else:
            print("Invalid option. Please choose 1, 2, or 3.")

    # User is logged in
    while True:
        session = select_or_create_session(current_user)
        chat_loop(current_user, session)

        print("\nOptions: [1] Open Another Session  [2] Logout / Exit")
        after_choice = input("Select (1 or 2): ").strip()
        if after_choice != "1":
            print(f"\nGoodbye, {current_user['first_name']}!")
            break


if __name__ == "__main__":
    main()
