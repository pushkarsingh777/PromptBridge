================================================================================
  PROMPTBRIDGE — SYSTEM ARCHITECTURE
  Last Updated: 2026-09-29
================================================================================

PromptBridge is a Hybrid RAG (Retrieval-Augmented Generation) API that turns a
user's vague, incomplete request into a fully structured, domain-aware,
optimized prompt — ready to paste into any AI model (ChatGPT, Claude, Gemini).

--------------------------------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------------------------------

Purpose:
  User submits vague request → PromptBridge returns a structured, expert-level
  prompt the user can paste into any LLM to get dramatically better responses.

Core Value Proposition:
  - Intent classification routes the request to a domain-specific prompt template
  - Hybrid retrieval (Dense + BM25) surfaces the best prompt-engineering examples
  - Cross-encoder reranking selects the top 5 most relevant examples
  - LLM (Groq / GPT-OSS) synthesizes a structured, optimized prompt
  - Multi-layer caching (L1/L2/L3) minimizes latency and API costs
  - Persistent conversation memory enables coherent multi-turn sessions

--------------------------------------------------------------------------------
2. REPOSITORY STRUCTURE
--------------------------------------------------------------------------------

promptbridge/
|
+-- system_architecture.txt          <- THIS FILE (updated with every new feature)
+-- prd.txt                          <- Product Requirements Document
+-- README.md                        <- Project overview
+-- requirements.txt                 <- Python dependencies
+-- .env                             <- Secrets (API keys, Redis, R2 -- never commit)
+-- .env.example                     <- Template for environment variables
+-- .gitignore

|
+-- data/
|   +-- promptbridge.db              <- SQLite DB (conversation memory -- legacy path)
|
+-- push/                            <- Python virtual environment (venv)
|   +-- Include/
|   +-- Lib/
|   +-- Scripts/
|   +-- pyvenv.cfg
|
+-- rag_data_collection/             <- Offline data pipeline (run once to build KB)
|   +-- run_pipeline.ps1             <- PowerShell orchestrator: scrape -> generate -> index
|   +-- validate_corpus.py           <- Corpus quality checker (pre-indexing)
|   +-- embed_and_index.py           <- Embeds corpus.jsonl -> Qdrant vector DB
|   |
|   +-- scrapers/
|   |   +-- web_scraper.py           <- Crawls 4 prompt-engineering sites
|   |
|   +-- pdf_ingestor/
|   |   +-- pdf_ingestor.py          <- Ingests local PDFs + arXiv papers
|   |
|   +-- synthetic_pairs/
|   |   +-- generate_pairs.py        <- v1 synthetic pair generator (deprecated)
|   |   +-- generate_pairs_2.py      <- v3 synthetic pair generator (current)
|   |
|   +-- data/
|       +-- raw_docs/                <- Scraped/ingested documents (JSON per page)
|       |   +-- promptingguide/
|       |   +-- learnprompting/
|       |   +-- anthropic_docs/
|       |   +-- openai_cookbook/
|       |   +-- pdf_local/
|       |   +-- pdf_arxiv/
|       +-- training_pairs/
|       |   +-- corpus.jsonl         <- Canonical corpus: flat JSONL of all chunks
|       +-- qdrant_db/               <- Qdrant local vector database (on-disk)
|
+-- rag_service/                     <- Online inference service (FastAPI)
    +-- app.py                       <- FastAPI server -- entrypoint
    +-- pipeline.py                  <- Core PromptBridge pipeline orchestrator
    +-- retriever.py                 <- Dual retrieval: Dense (Qdrant) + Sparse (BM25)
    +-- reranker.py                  <- Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
    +-- memory.py                    <- Conversation memory (RAM + SQLite)
    +-- cache.py                     <- Three-layer Redis cache with in-memory fallback
    +-- inspect_memory.py            <- Debug utility to inspect SQLite memory DB
    +-- data/
        +-- memory.db                <- SQLite database for conversation memory
        +-- qdrant_db/               <- Qdrant DB (symlink or copy)

--------------------------------------------------------------------------------
3. DATA PIPELINE (OFFLINE -- run once to build the knowledge base)
--------------------------------------------------------------------------------

STEP 1 -- Web Scraping                        [rag_data_collection/scrapers/web_scraper.py]
  Target sites:
    * promptingguide.ai        (up to 300 pages)
    * learnprompting.org       (up to 300 pages)
    * docs.anthropic.com       (up to 150 pages)
    * cookbook.openai.com      (up to 150 pages)

  Process:
    BFS crawler -> trafilatura text extraction -> BeautifulSoup fallback
    -> per-page JSON saved locally -> optional upload to Cloudflare R2

  Output format (per doc JSON):
    { id, url, source, title, text, scraped_at, char_count }

  CLI:
    python web_scraper.py --sites all --output ./raw_docs [--upload_r2]

---------------------------------------------------------------------------------

STEP 2 -- PDF Ingestion                    [rag_data_collection/pdf_ingestor/pdf_ingestor.py]
  Sources:
    * Local PDFs from user-specified folder
    * arXiv papers (7 curated queries, up to 20 papers each)

  arXiv Queries:
    - "prompt engineering large language models"
    - "chain of thought reasoning LLM"
    - "retrieval augmented generation RAG"
    - "instruction tuning language models"
    - "few shot learning prompting"
    - "LLM alignment RLHF reward model"
    - "hallucination mitigation language models"

  Process:
    PyMuPDF text extraction -> section detection -> cleaning
    -> per-doc JSON (local/arxiv) -> optional R2/S3 upload

  Output format (per doc JSON):
    { id, source, title, text, pages, sections, char_count, filename,
      ingested_at, arxiv_id, url, authors, published, abstract, categories }

  CLI:
    python pdf_ingestor.py --mode both --query "prompt engineering" --max 30

---------------------------------------------------------------------------------

STEP 3 -- Synthetic Pair Generation    [rag_data_collection/synthetic_pairs/generate_pairs_2.py]
  Reads raw_docs/ -> chunks each document -> sends batches to Groq LLM
  -> generates (query, positive_chunk, hard_negative) training triplets
  -> saves corpus.jsonl with all chunks + training pair metadata

  Key features:
    * Batch size: 3 chunks per LLM call (reduces truncation)
    * Max tokens: 4096
    * JSON mode via Groq (response_format={"type": "json_object"})
    * Checkpoint after every batch (resumable on interruption)
    * Hard negatives: true negative examples for contrastive training
    * Deduplication via SHA-256 hash

  Output:
    data/training_pairs/corpus.jsonl   <- flat JSONL, one chunk per line

  CLI:
    python synthetic_pairs/generate_pairs_2.py \
      --input ./data/raw_docs \
      --output ./data/training_pairs \
      --pairs_per_chunk 2 \
      --batch_size 3 \
      --hard_negatives

---------------------------------------------------------------------------------

STEP 3.5 -- Corpus Validation               [rag_data_collection/validate_corpus.py]
  Pre-indexing quality check on corpus.jsonl.

  Checks performed:
    * Valid JSON on every line
    * Non-empty chunk_id and text fields
    * Duplicate chunk IDs
    * Duplicate text (exact-match after whitespace normalization)
    * Too-short chunks (< 20 tokens)
    * Repeated boilerplate (low vocabulary diversity)

  CLI:
    python validate_corpus.py --corpus data/training_pairs/corpus.jsonl [--strict]

---------------------------------------------------------------------------------

STEP 4 -- Embed & Index                        [rag_data_collection/embed_and_index.py]
  Reads corpus.jsonl -> embeds with BAAI/bge-small-en-v1.5
  -> upserts into local Qdrant vector DB

  Model: BAAI/bge-small-en-v1.5 (384-dim, cosine similarity)
  Distance metric: Cosine
  Batch size: 128 (configurable)
  Device: auto-detects CUDA, falls back to CPU

  CLI:
    python embed_and_index.py \
      --corpus data/training_pairs/corpus.jsonl \
      --qdrant-path data/qdrant_db \
      --recreate

  Orchestrator (PowerShell):
    rag_data_collection/run_pipeline.ps1   <- runs steps 1-4 in sequence

--------------------------------------------------------------------------------
4. INFERENCE SERVICE -- RAG PIPELINE (ONLINE)
--------------------------------------------------------------------------------

Entry Point: rag_service/app.py  (FastAPI, Uvicorn)
  Run: uvicorn app:app --host 0.0.0.0 --port 8080 --reload

Pipeline Orchestrator: rag_service/pipeline.py  [class PromptBridgePipeline]

  Full request flow (one user query):

  +------------------------------------------------------------------+
  |  User Query (vague/incomplete request)                           |
  +-----------------------------+------------------------------------+
                                |
                   +------------v-----------+
                   |  L3 Answer Cache Check | <-- HIT? Return immediately
                   |  (Redis / in-memory)   |     (skip all steps below)
                   +------------+-----------+
                                | MISS
             +------------------v------------------+
             |  Step 1: Classify Intent            |
             |  + Detect Domain                    |
             |  (keyword-based, no LLM call)       |
             +------------------+------------------+
                                |
                   +------------v-----------+
                   |  L2 Retrieval Cache    | <-- HIT? Reuse cached chunks
                   |  Check (Redis/memory)  |     (skip Steps 2 & 3)
                   +------------+-----------+
                                | MISS
             +------------------v------------------+
             |  Step 2: Dual Retrieval             |
             |  Dense (Qdrant, top 10)             |
             |  + BM25 keyword (top 10)            |
             |  -> RRF Fusion -> 20 chunks         |
             +------------------+------------------+
                                |
             +------------------v------------------+
             |  Step 3: Cross-Encoder Rerank       |
             |  20 -> top 5 chunks                 |
             |  ms-marco-MiniLM-L-6-v2             |
             +------------------+------------------+
                                |
             +------------------v------------------+
             |  Step 4: Load Memory Context        |
             |  Last 3 conversation turns          |
             |  (SQLite via Memory class)          |
             +------------------+------------------+
                                |
             +------------------v------------------+
             |  Step 5: Assemble Prompt            |
             |  System: intent-specific template   |
             |  User:   memory + examples +        |
             |          intent/domain + user query |
             +------------------+------------------+
                                |
             +------------------v------------------+
             |  Step 6: LLM Generation             |
             |  Groq API (gpt-oss-20b)             |
             |  temperature=0.4, max_tokens=1200   |
             +------------------+------------------+
                                |
             +------------------v------------------+
             |  Step 7: Save to Memory             |
             |  (RAM + SQLite)                     |
             |  Populate L3 Answer Cache           |
             +------------------+------------------+
                                |
  +------------------------------------------------------------------+
  |  Response: optimized_prompt, intent, domain, sources,           |
  |            latency_s, chunks (top 5 previews), cached flag      |
  +------------------------------------------------------------------+

---------------------------------------------------------------------------------
4a. INTENT CLASSIFICATION                             [pipeline.py: classify_intent()]
---------------------------------------------------------------------------------

  Keyword-based, ordered priority (no LLM call):

  Intent     | Keywords (sample)
  -----------|--------------------------------------------------------------
  ml         | yolo, model, train, dataset, neural, cnn, transformer,
             | classification, detection, machine learning, deep learning,
             | computer vision, nlp, fine-tune, embedding, llm, rag, gan
  code       | code, python, function, script, implement, debug, api,
             | build, flask, fastapi, django, react, typescript, sql
  research   | research, paper, study, literature, survey, academic,
             | hypothesis, experiment, methodology
  creative   | write, story, poem, creative, draft, essay, blog post,
             | article, content, marketing, social media
  analysis   | analyze, compare, difference, evaluate, pros and cons,
             | versus, assess, review, audit, benchmark
  task       | how to, steps to, guide, tutorial, help me, plan,
             | strategy, roadmap, checklist, workflow, process
  question   | (default / catch-all)

---------------------------------------------------------------------------------
4b. DOMAIN DETECTION                                  [pipeline.py: detect_domain()]
---------------------------------------------------------------------------------

  Domains detected:
    Computer Vision     | vision, image, yolo, detection, segmentation, opencv
    NLP                 | text, nlp, language, sentiment, summarize, bert, gpt, llm
    Data Science        | data, dataset, pandas, analysis, csv, statistics, plot
    Software Engineering| api, backend, frontend, database, sql, rest, docker
    Machine Learning    | train, model, neural, deep learning, epoch, accuracy
    Agriculture         | weed, crop, farm, plant, soil, yield, harvest
    Cybersecurity       | security, hack, vulnerability, encryption, pentest
    Research            | paper, research, academic, literature, hypothesis
    General             | (default fallback)

---------------------------------------------------------------------------------
4c. INTENT-SPECIFIC SYSTEM PROMPTS                    [pipeline.py: SYSTEM_PROMPTS]
---------------------------------------------------------------------------------

  Each intent maps to a dedicated system prompt instructing the LLM to:
    1. NOT answer the user's question directly
    2. Generate a structured, optimized prompt for the user to paste into any AI

  Output prompt sections (tailored per intent):
    code:     Role, Task, Context, Requirements, Input, Output Format, Constraints, Examples
    ml:       Role, Task, Domain, Dataset, Model Requirements, Evaluation Metrics,
              Output Format, Constraints, Assumptions
    research: Role, Task, Background, Scope, Methodology, Output Format, Constraints, Evaluation
    creative: Role, Task, Style & Tone, Audience, Length, Key Elements, Output Format, Constraints
    analysis: Role, Task, Data, Analysis Type, Output Format, Constraints, Evaluation
    task:     Role, Task, Context, Steps, Output Format, Constraints, Success Criteria
    question: Role, Task, Context, Output Format, Constraints, Assumptions

--------------------------------------------------------------------------------
5. RETRIEVAL MODULE                                              [rag_service/retriever.py]
--------------------------------------------------------------------------------

Class: DualRetriever

  Dense Search (Qdrant + BGE):
    Model:    BAAI/bge-small-en-v1.5 (384-dim, normalize_embeddings=True)
    Backend:  Qdrant local on-disk (cosine similarity)
    Top-K:    10 chunks
    L1 Cache: Checks Redis for pre-computed embedding before calling model
              (saves ~50-200 ms per repeated query)

  Sparse Search (BM25):
    Library:  rank-bm25 (BM25Okapi)
    Index:    Built in-memory from corpus.jsonl on startup
    Top-K:    10 chunks
    Tokenizer: Simple whitespace lowercase split

  RRF Fusion (Reciprocal Rank Fusion):
    Formula:  score(d) = SUM 1 / (k + rank(d))  where k=60 (standard)
    Input:    10 dense + 10 BM25 results
    Output:   Up to 20 fused, deduplicated chunks (ranked by RRF score)

  Chunk payload fields:
    chunk_id, text, source, url, doc_title, score, method, rrf_score

  Data files:
    CORPUS_PATH: rag_data_collection/data/training_pairs/corpus.jsonl
    QDRANT_PATH: rag_data_collection/data/qdrant_db

--------------------------------------------------------------------------------
6. RERANKER MODULE                                               [rag_service/reranker.py]
--------------------------------------------------------------------------------

Class: Reranker

  Model:      cross-encoder/ms-marco-MiniLM-L-6-v2  (~85 MB)
  Input:      20 fused chunks + query
  Output:     Top 5 chunks by cross-encoder relevance score
  Device:     Auto-detects CUDA, falls back to CPU
  Max length: 512 tokens per (query, chunk) pair

  Process:
    Build (query, chunk_text) pairs -> CrossEncoder.predict() -> attach scores
    -> sort descending -> return top_n=5

  Added field per chunk: rerank_score (float, 4 decimal places)

--------------------------------------------------------------------------------
7. CACHE MODULE                                                    [rag_service/cache.py]
--------------------------------------------------------------------------------

Class: RAGCache

  Three cache layers:

  Layer | Key prefix  | Stores               | TTL  | What it saves
  ------|-------------|----------------------|------|---------------------
  L1    | pb:emb:     | query -> embedding   | 24h  | ~50-200ms model.encode
  L2    | pb:chunks:  | query -> top-5 chunks|  6h  | ~500ms retrieval+rerank
  L3    | pb:ans:     | query -> full result |  1h  | ~1-3s LLM call + all

  Key design:
    * All keys SHA-256 hashed (16 hex chars) for tiny key sizes
    * Connection pool (max 20) shared across process
    * Sliding-window TTL: refreshed on every cache HIT
    * Graceful fallback to in-memory dict when Redis is unavailable
    * Thread-safe stats counters (threading.Lock)
    * L3 strips chunk blobs before caching (keeps entry lean)

  Backend detection (env vars):
    REDIS_HOST, REDIS_PORT (default 6379), REDIS_DB, REDIS_PASSWORD
    CACHE_TTL_EMBEDDING, CACHE_TTL_RETRIEVAL, CACHE_TTL_ANSWER

  Stats endpoint:
    GET /cache/stats -> hits, misses, hit_rate, backend,
                        redis_used_memory_human, redis_peak_memory_human,
                        keys (embeddings / chunks / answers)

--------------------------------------------------------------------------------
8. MEMORY MODULE                                                  [rag_service/memory.py]
--------------------------------------------------------------------------------

Class: Memory

  Two-level persistence:

  Level 1 -- In-session (RAM):
    Python list in self.session_history
    Instant access, no I/O

  Level 2 -- Cross-session (SQLite):
    File:  rag_service/data/memory.db
    Table: conversations (id, session_id, role, content, ts)
    Index: (session_id, ts) for fast per-session queries
    Persists across server restarts

  Key methods:
    add(role, content)        -> save message to RAM + SQLite
    get_context(max_turns=3)  -> last N turns as formatted string for LLM
    get_history()             -> full session history as list of dicts
    clear_session()           -> delete session from RAM + DB
    list_sessions()           -> all sessions with message counts (limit 20)
    turn_count (property)     -> number of complete user/assistant turns

  Context injection:
    Last 3 turns (up to 6 messages) prepended to each LLM call
    Each message truncated to 300 chars to control context length

--------------------------------------------------------------------------------
9. API ENDPOINTS                                                  [rag_service/app.py]
--------------------------------------------------------------------------------

  FastAPI server -- run with:
    uvicorn app:app --host 0.0.0.0 --port 8080 --reload
    Swagger docs: http://localhost:8080/docs

  Session management:
    Per-session PromptBridgePipeline instances stored in process memory
    "default" session pre-loaded on startup lifespan event

  Endpoints:

  POST  /generate-prompt
    Request:  { query: str, session_id: str = "default" }
    Response: { query, optimized_prompt, intent, domain, sources[],
                latency_s, chunks[], cached }
    Behavior: Full pipeline -- vague request -> optimized prompt

  POST  /chat
    Alias for /generate-prompt (backward compatibility)

  POST  /session/clear
    Request:  { session_id: str = "default" }
    Response: { status: "cleared", session_id }
    Behavior: Clears conversation memory for a session

  GET   /session/{session_id}
    Response: { session_id, turns, history[] (last 10) }

  GET   /health
    Response: { status: "ok", active_sessions, model }

  GET   /cache/stats
    Response: { hits, misses, errors, hit_rate, backend,
                redis_used_memory_human, redis_peak_memory_human, keys }

  POST  /cache/flush
    Response: { status: "flushed" }
    Behavior: Deletes all pb:emb:*, pb:chunks:*, pb:ans:* keys from Redis

  CORS: allow_origins=["*"]  (open for local dev -- restrict in production)

--------------------------------------------------------------------------------
10. STORAGE SYSTEMS
--------------------------------------------------------------------------------

  System          | Technology        | Purpose
  ----------------|-------------------|------------------------------------------
  Vector DB       | Qdrant (on-disk)  | Dense retrieval of prompt examples
  BM25 index      | rank-bm25 (RAM)   | Sparse keyword retrieval
  Cache           | Redis + fallback  | L1/L2/L3 caching (embeddings/chunks/answers)
  Memory          | SQLite            | Persistent conversation history
  Raw docs        | Local JSON files  | Scraped web pages + PDF content
  Corpus          | JSONL file        | Canonical flat chunk store for indexing
  Cloud storage   | Cloudflare R2     | Optional backup of raw_docs (S3-compatible)

--------------------------------------------------------------------------------
11. MODELS & EXTERNAL SERVICES
--------------------------------------------------------------------------------

  Model / Service                        | Used for
  ---------------------------------------|--------------------------------------
  BAAI/bge-small-en-v1.5 (local)         | Query + corpus embedding (384-dim)
  cross-encoder/ms-marco-MiniLM-L-6-v2   | Cross-encoder reranking (20 -> 5)
  Groq API -- gpt-oss-20b                | LLM: optimized prompt generation
                                         | + synthetic training pair generation
  Cloudflare R2 (optional)               | Raw doc backup storage (S3-compatible)
  Redis (optional)                       | Three-layer cache backend

--------------------------------------------------------------------------------
12. KEY PYTHON DEPENDENCIES                                      [requirements.txt]
--------------------------------------------------------------------------------

  Category             | Packages
  ---------------------|--------------------------------------------------------
  Web scraping         | requests, beautifulsoup4, trafilatura, markdownify
  PDF ingestion        | pymupdf (fitz), arxiv
  LLM / AI             | groq >= 0.11.0
  Retrieval            | sentence-transformers >= 3.0.1, qdrant-client >= 1.11.0,
                       | rank-bm25 >= 0.2.2
  API framework        | fastapi >= 0.115.0, uvicorn[standard] >= 0.30.0,
                       | pydantic >= 2.8.0
  Caching              | redis[hiredis] >= 5.0.0
  Cloud storage        | boto3 == 1.35.0
  Utilities            | python-dotenv >= 1.0.0, tqdm == 4.66.5

--------------------------------------------------------------------------------
13. ENVIRONMENT VARIABLES                                    [.env / .env.example]
--------------------------------------------------------------------------------

  Variable              | Description
  ----------------------|--------------------------------------------------------
  GROQ_API_KEY          | Groq API key (required for LLM + pair generation)
  REDIS_HOST            | Redis host (default: localhost)
  REDIS_PORT            | Redis port (default: 6379)
  REDIS_DB              | Redis database number (default: 0)
  REDIS_PASSWORD        | Redis password (optional)
  CACHE_TTL_EMBEDDING   | L1 TTL in seconds (default: 86400 = 24h)
  CACHE_TTL_RETRIEVAL   | L2 TTL in seconds (default: 21600 = 6h)
  CACHE_TTL_ANSWER      | L3 TTL in seconds (default: 3600 = 1h)
  R2_ACCOUNT_ID         | Cloudflare account ID (for R2 upload)
  R2_ACCESS_KEY_ID      | Cloudflare R2 access key
  R2_SECRET_ACCESS_KEY  | Cloudflare R2 secret key
  R2_BUCKET_NAME        | Cloudflare R2 bucket name

--------------------------------------------------------------------------------
14. QUICK START
--------------------------------------------------------------------------------

  # 1. Setup
  cd promptbridge
  python -m venv push
  push\Scripts\activate
  pip install -r requirements.txt

  # 2. Build knowledge base (one-time)
  cd rag_data_collection
  python scrapers/web_scraper.py --sites all --output ./data/raw_docs
  python pdf_ingestor/pdf_ingestor.py --mode arxiv --output ./data/raw_docs
  python synthetic_pairs/generate_pairs_2.py --input ./data/raw_docs --output ./data/training_pairs
  python validate_corpus.py --corpus ./data/training_pairs/corpus.jsonl
  python embed_and_index.py --recreate

  # 3. Start inference service
  cd ..\rag_service
  uvicorn app:app --host 0.0.0.0 --port 8080 --reload

  # 4. Test
  curl -X POST http://localhost:8080/generate-prompt \
       -H "Content-Type: application/json" \
       -d '{"query": "help me train a YOLO model on my custom dataset"}'

--------------------------------------------------------------------------------
15. FEATURE CHANGELOG
--------------------------------------------------------------------------------

  [2026-09-29] -- Initial architecture captured
    [x] Web scraper (4 prompt-engineering sites, BFS crawler, R2 upload)
    [x] PDF ingestor (local PDFs + arXiv multi-query, PyMuPDF)
    [x] Synthetic pair generator v3 (Groq JSON mode, checkpointing, hard negatives)
    [x] Corpus validator (dedup, quality checks)
    [x] Embed & index pipeline (BGE + Qdrant, batch upsert, --recreate flag)
    [x] Dual retriever (Dense Qdrant + BM25 RRF fusion, L1 embedding cache)
    [x] Cross-encoder reranker (ms-marco-MiniLM-L-6-v2, 20->5 chunks)
    [x] Three-layer Redis cache (L1 embedding / L2 retrieval / L3 answer)
        with in-memory fallback, sliding TTL, thread-safe stats
    [x] Conversation memory (dual-level: RAM + SQLite, multi-session)
    [x] Intent classifier (7 intents: ml, code, research, creative, analysis, task, question)
    [x] Domain detector (8 domains + General fallback)
    [x] Intent-specific system prompts (7 prompt templates)
    [x] FastAPI server (7 endpoints: /generate-prompt, /chat, /session/clear,
        /session/{id}, /health, /cache/stats, /cache/flush)
    [x] Session-scoped pipeline instances (per-user isolation)
    [x] CORS middleware (open for development)
    [x] PowerShell pipeline orchestrator (run_pipeline.ps1)

  [FUTURE -- add entries here when new features are implemented]
    [ ] Authentication / API key middleware
    [ ] Rate limiting per session/IP
    [ ] Streaming response (SSE/WebSocket) for LLM output
    [ ] Frontend UI (web interface for PromptBridge)
    [ ] Fine-tuned embedding model on PromptBridge corpus
    [ ] Prompt quality scoring / feedback loop
    [ ] Multi-language support
    [ ] Docker Compose deployment (Redis + API service)
    [ ] Monitoring / observability (Prometheus metrics endpoint)

================================================================================
  END OF SYSTEM ARCHITECTURE
  Update this file whenever a new feature, module, or endpoint is added.
================================================================================
