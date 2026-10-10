Role: Senior Application Security Engineer & Backend Architect.

Task:
Perform a comprehensive security audit and hardening of the backend codebase for a RAG (Retrieval-Augmented Generation) application backed by Supabase (pgvector) and FastAPI/Python.

Core Security Directives:

1. Environment & Secrets Sanitization:
   - Audit all files for hardcoded API keys, bearer tokens, Supabase service keys, and database passwords.
   - Refactor secret ingestion using Pydantic `BaseSettings` reading exclusively from system environment variables.
   - Enforce strict separation of Supabase keys: the frontend/public client must strictly use the `anon` key, while the `service_role` key must only exist in backend ingestion/admin jobs and never leak to client responses or serialized objects.

2. Database, RPC & Row-Level Security (RLS):
   - Verify that all tables under the `knowledge` schema have Row-Level Security enabled (`ALTER TABLE ... ENABLE ROW LEVEL SECURITY;`).
   - Implement strict SELECT-only policies for public/authenticated users, ensuring insert, update, and delete privileges remain restricted to backend admin roles.
   - Audit all PostgreSQL/RPC stored procedures (e.g., vector similarity search functions) to ensure they are defined with `SECURITY INVOKER` and include an explicit `SET search_path = knowledge, public, extensions` to prevent schema search-path hijacking.
   - Guarantee zero dynamic string formatting in database queries; mandate parameterized queries or typed Supabase RPC calls to prevent SQL injection.

3. API Endpoint Hardening & Compute/Wallet Protection:
   - Apply rate-limiting (e.g., SlowAPI or Redis token bucket) on vector search, question answering, and ingestion endpoints to defend against Denial-of-Service (DoS) and API wallet draining.
   - Prevent "Embedding Asymmetry / Denial-of-Wallet" attacks by enforcing strict character length limits on user search inputs (e.g., max 500–800 characters) before sending text to the embedding provider.
   - Restrict CORS middleware to verified, explicit domain origins—eliminate `allow_origins=["*"]` on all stateful, authenticated, or vector query routes.
   - Enforce rigorous payload validation with Pydantic models (strict string length bounds, valid UUID formats, type checks, and sanitized string inputs).
   - Disable automatic API documentation (`/docs`, `/redoc`, and `/openapi.json`) in production environments.

4. RAG, Prompt Injection & Model Defenses:
   - Sanitize user query inputs (strip null bytes, control characters, and abnormal unicode) before passing them to the embedding and completion models.
   - Structure LLM system prompts with unambiguous delimiter boundaries (e.g., `<context></context>` XML tags) separating system directives from retrieved document chunks.
   - Guard against Indirect Prompt Injection by explicitly instructing the model to treat all retrieved chunks as untrusted passive data rather than instructions.
   - Implement strict token generation limits (`max_tokens`) on LLM completions to avoid token-flooding and excessive cost overruns.

5. Ingestion Pipeline & Supply Chain Hardening:
   - If document ingestion accepts file uploads or external URLs, validate file extensions, MIME types, and implement strict SSRF (Server-Side Request Forgery) protections against internal IP ranges (e.g., `127.0.0.1`, `169.254.169.254`).
   - Audit dependencies against known vulnerabilities and pin package versions.

6. Logging & Error Handling:
   - Mask all authorization headers, API keys, tokens, and personally identifiable information (PII) from application logs.
   - Suppress raw stack traces, database schema details, and internal file paths in API responses; implement global exception handlers that log the error internally with a unique UUID tracking ID and return a generic HTTP 500 error payload.

Output:
1. Review all backend files and explicitly list every identified vulnerability, code smell, and credential risk ranked by severity (Critical, High, Medium).
2. Provide production-ready, refactored replacement code adhering strictly to these specifications.