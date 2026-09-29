# PromptBridge — System Architecture

> **Last Updated:** 2026-09-29
>
> PromptBridge is a **Hybrid RAG (Retrieval-Augmented Generation) API** that turns a user's vague, incomplete request into a fully structured, domain-aware, optimized prompt — ready to paste into any AI model (ChatGPT, Claude, Gemini).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Pipeline (Offline)](#3-data-pipeline-offline)
4. [Inference Service — RAG Pipeline](#4-inference-service--rag-pipeline-online)
   - [4a. Intent Classification](#4a-intent-classification)
   - [4b. Domain Detection](#4b-domain-detection)
   - [4c. Intent-Specific System Prompts](#4c-intent-specific-system-prompts)
5. [Retrieval Module](#5-retrieval-module)
6. [Reranker Module](#6-reranker-module)
7. [Cache Module](#7-cache-module)
8. [Memory Module](#8-memory-module)
9. [API Endpoints](#9-api-endpoints)
10. [Storage Systems](#10-storage-systems)
11. [Models & External Services](#11-models--external-services)
12. [Python Dependencies](#12-python-dependencies)
13. [Environment Variables](#13-environment-variables)
14. [Quick Start](#14-quick-start)
15. [Feature Changelog](#15-feature-changelog)

---

## 1. Project Overview

**Purpose:** User submits a vague request → PromptBridge returns a structured, expert-level prompt the user can paste into any LLM to get dramatically better responses.

### Core Value Proposition

| Feature | Description |
|---|---|
| 🎯 Intent Classification | Routes each request to a domain-specific prompt template |
| 🔍 Hybrid Retrieval | Dense (Qdrant) + BM25 surfaces the best prompt-engineering examples |
| 📊 Cross-Encoder Reranking | Selects the top 5 most relevant examples from 20 candidates |
| 🤖 LLM Synthesis | Groq / GPT-OSS generates a fully structured, optimized prompt |
| ⚡ Multi-Layer Caching | L1/L2/L3 Redis cache minimizes latency and API costs |
| 🧠 Conversation Memory | Persistent SQLite memory enables coherent multi-turn sessions |

---

## 2. Repository Structure

```
promptbridge/
│
├── system_architecture.md          ← THIS FILE (updated with every new feature)
├── prd.txt                         ← Product Requirements Document
├── README.md                       ← Project overview
├── requirements.txt                ← Python dependencies
├── .env                            ← Secrets (API keys, Redis, R2 — never commit)
├── .env.example                    ← Template for environment variables
├── .gitignore
├── dump.rdb                        ← Redis persistence snapshot
│
├── data/
│   └── promptbridge.db             ← SQLite DB (conversation memory — legacy path)
│
├── push/                           ← Python virtual environment (venv)
│
├── rag_data_collection/            ← Offline data pipeline (run once to build KB)
│   ├── run_pipeline.ps1            ← PowerShell orchestrator: scrape → generate → index
│   ├── validate_corpus.py          ← Corpus quality checker (pre-indexing)
│   ├── embed_and_index.py          ← Embeds corpus.jsonl → Qdrant vector DB
│   │
│   ├── scrapers/
│   │   └── web_scraper.py          ← Crawls 4 prompt-engineering sites
│   │
│   ├── pdf_ingestor/
│   │   └── pdf_ingestor.py         ← Ingests local PDFs + arXiv papers
│   │
│   ├── synthetic_pairs/
│   │   ├── generate_pairs.py       ← v1 generator (deprecated)
│   │   └── generate_pairs_2.py     ← v3 generator (current)
│   │
│   └── data/
│       ├── raw_docs/               ← Scraped/ingested documents (JSON per page)
│       │   ├── promptingguide/
│       │   ├── learnprompting/
│       │   ├── anthropic_docs/
│       │   ├── openai_cookbook/
│       │   ├── pdf_local/
│       │   └── pdf_arxiv/
│       ├── training_pairs/
│       │   └── corpus.jsonl        ← Canonical corpus: flat JSONL of all chunks
│       └── qdrant_db/              ← Qdrant local vector database (on-disk)
│
└── rag_service/                    ← Online inference service (FastAPI)
    ├── app.py                      ← FastAPI server — entrypoint
    ├── pipeline.py                 ← Core PromptBridge pipeline orchestrator
    ├── retriever.py                ← Dual retrieval: Dense (Qdrant) + Sparse (BM25)
    ├── reranker.py                 ← Cross-encoder reranker
    ├── memory.py                   ← Conversation memory (RAM + SQLite)
    ├── cache.py                    ← Three-layer Redis cache with in-memory fallback
    ├── inspect_memory.py           ← Debug utility to inspect SQLite memory DB
    └── data/
        ├── memory.db               ← SQLite database for conversation memory
        └── qdrant_db/              ← Qdrant DB (symlink or copy)
```

---

## 3. Data Pipeline (Offline)

> Run once to build the knowledge base. Orchestrated via `run_pipeline.ps1`.

### Step 1 — Web Scraping
**File:** `rag_data_collection/scrapers/web_scraper.py`

| Site | Max Pages |
|---|---|
| promptingguide.ai | 300 |
| learnprompting.org | 300 |
| docs.anthropic.com | 150 |
| cookbook.openai.com | 150 |

**Process:** BFS crawler → `trafilatura` text extraction → BeautifulSoup fallback → per-page JSON saved locally → optional Cloudflare R2 upload

**Output schema per doc:**
```json
{ "id": "", "url": "", "source": "", "title": "", "text": "", "scraped_at": "", "char_count": 0 }
```

```bash
python web_scraper.py --sites all --output ./raw_docs [--upload_r2]
```

---

### Step 2 — PDF Ingestion
**File:** `rag_data_collection/pdf_ingestor/pdf_ingestor.py`

**Sources:**
- Local PDFs from a user-specified folder
- arXiv papers via 7 curated queries (up to 20 papers each):

| arXiv Query |
|---|
| `prompt engineering large language models` |
| `chain of thought reasoning LLM` |
| `retrieval augmented generation RAG` |
| `instruction tuning language models` |
| `few shot learning prompting` |
| `LLM alignment RLHF reward model` |
| `hallucination mitigation language models` |

**Process:** PyMuPDF text extraction → section detection → cleaning → per-doc JSON → optional R2/S3 upload

```bash
python pdf_ingestor.py --mode both --query "prompt engineering" --max 30
```

---

### Step 3 — Synthetic Pair Generation
**File:** `rag_data_collection/synthetic_pairs/generate_pairs_2.py`

Reads `raw_docs/` → chunks each document → sends batches to Groq LLM → generates `(query, positive_chunk, hard_negative)` training triplets → outputs `corpus.jsonl`.

**Key features:**
- Batch size: 3 chunks per LLM call (reduces truncation)
- Max tokens: 4096
- JSON mode via Groq (`response_format={"type": "json_object"}`)
- Checkpoint after every batch (resumable on interruption)
- Hard negatives for contrastive training
- Deduplication via SHA-256 hash

```bash
python synthetic_pairs/generate_pairs_2.py \
  --input ./data/raw_docs \
  --output ./data/training_pairs \
  --pairs_per_chunk 2 \
  --batch_size 3 \
  --hard_negatives
```

---

### Step 3.5 — Corpus Validation
**File:** `rag_data_collection/validate_corpus.py`

Pre-indexing quality check on `corpus.jsonl`.

**Checks performed:**
- ✅ Valid JSON on every line
- ✅ Non-empty `chunk_id` and `text` fields
- ✅ Duplicate chunk IDs
- ✅ Duplicate text (exact-match after whitespace normalization)
- ✅ Too-short chunks (< 20 tokens)
- ✅ Repeated boilerplate (low vocabulary diversity)

```bash
python validate_corpus.py --corpus data/training_pairs/corpus.jsonl [--strict]
```

---

### Step 4 — Embed & Index
**File:** `rag_data_collection/embed_and_index.py`

Reads `corpus.jsonl` → embeds with `BAAI/bge-small-en-v1.5` → upserts into local Qdrant vector DB.

| Parameter | Value |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) |
| Distance metric | Cosine |
| Batch size | 128 (configurable) |
| Device | Auto-detects CUDA, falls back to CPU |

```bash
python embed_and_index.py \
  --corpus data/training_pairs/corpus.jsonl \
  --qdrant-path data/qdrant_db \
  --recreate
```

---

## 4. Inference Service — RAG Pipeline (Online)

**Entry point:** `rag_service/app.py` (FastAPI + Uvicorn)

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
# Swagger docs: http://localhost:8080/docs
```

**Pipeline class:** `PromptBridgePipeline` in `pipeline.py`

### Request Flow

```
User Query (vague/incomplete request)
        │
        ▼
┌───────────────────────┐
│  L3 Answer Cache      │──── HIT ──▶  Return cached optimized prompt immediately
│  (Redis / in-memory)  │
└───────────┬───────────┘
            │ MISS
            ▼
┌───────────────────────┐
│  Step 1               │
│  Classify Intent      │  keyword-based, no LLM call
│  + Detect Domain      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  L2 Retrieval Cache   │──── HIT ──▶  Reuse cached chunks (skip Steps 2 & 3)
│  (Redis / in-memory)  │
└───────────┬───────────┘
            │ MISS
            ▼
┌───────────────────────┐
│  Step 2               │
│  Dual Retrieval       │  Dense (Qdrant, top 10) + BM25 (top 10)
│  → RRF Fusion         │  → 20 fused chunks
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Step 3               │
│  Cross-Encoder Rerank │  20 → top 5 chunks
│  ms-marco-MiniLM-L-6  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Step 4               │
│  Load Memory Context  │  Last 3 conversation turns (SQLite)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Step 5               │  System: intent-specific template
│  Assemble Prompt      │  User: memory + examples + intent/domain + raw query
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Step 6               │
│  LLM Generation       │  Groq API (gpt-oss-20b), temp=0.4, max_tokens=1200
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Step 7               │
│  Save to Memory       │  RAM + SQLite
│  Populate L3 Cache    │
└───────────┬───────────┘
            │
            ▼
Response: { optimized_prompt, intent, domain, sources, latency_s, chunks, cached }
```

---

### 4a. Intent Classification

**File:** `pipeline.py` → `classify_intent()`

Keyword-based, ordered priority — **no LLM call required**.

| Intent | Sample Keywords |
|---|---|
| `ml` | yolo, model, train, dataset, neural, cnn, transformer, classification, detection, deep learning, nlp, fine-tune, embedding, llm, rag, gan |
| `code` | code, python, function, script, implement, debug, api, build, flask, fastapi, django, react, typescript, sql |
| `research` | research, paper, study, literature, survey, academic, hypothesis, experiment, methodology |
| `creative` | write, story, poem, creative, draft, essay, blog post, article, content, marketing, social media |
| `analysis` | analyze, compare, difference, evaluate, pros and cons, versus, assess, review, audit, benchmark |
| `task` | how to, steps to, guide, tutorial, help me, plan, strategy, roadmap, checklist, workflow, process |
| `question` | *(default / catch-all)* |

---

### 4b. Domain Detection

**File:** `pipeline.py` → `detect_domain()`

| Domain | Keywords |
|---|---|
| Computer Vision | vision, image, yolo, detection, segmentation, opencv, pixel |
| NLP | text, nlp, language, sentiment, summarize, translate, bert, gpt, llm |
| Data Science | data, dataset, pandas, analysis, csv, statistics, visualization, plot |
| Software Engineering | api, backend, frontend, database, sql, rest, microservice, docker |
| Machine Learning | train, model, neural, deep learning, epoch, loss, accuracy, classification |
| Agriculture | weed, crop, farm, plant, soil, yield, harvest, irrigation |
| Cybersecurity | security, hack, vulnerability, encryption, firewall, pentest, threat |
| Research | paper, research, academic, literature, hypothesis, experiment, survey |
| **General** | *(default fallback)* |

---

### 4c. Intent-Specific System Prompts

**File:** `pipeline.py` → `SYSTEM_PROMPTS`

Each intent maps to a dedicated system prompt that instructs the LLM to:
1. **NOT** answer the user's question directly
2. **Generate** a structured, optimized prompt the user can paste into any AI

| Intent | Output Prompt Sections |
|---|---|
| `code` | Role, Task, Context, Requirements, Input, Output Format, Constraints, Examples |
| `ml` | Role, Task, Domain, Dataset, Model Requirements, Evaluation Metrics, Output Format, Constraints, Assumptions |
| `research` | Role, Task, Background, Scope, Methodology, Output Format, Constraints, Evaluation |
| `creative` | Role, Task, Style & Tone, Audience, Length, Key Elements, Output Format, Constraints |
| `analysis` | Role, Task, Data, Analysis Type, Output Format, Constraints, Evaluation |
| `task` | Role, Task, Context, Steps, Output Format, Constraints, Success Criteria |
| `question` | Role, Task, Context, Output Format, Constraints, Assumptions |

---

## 5. Retrieval Module

**File:** `rag_service/retriever.py` — Class: `DualRetriever`

### Dense Search (Qdrant + BGE)

| Parameter | Value |
|---|---|
| Model | `BAAI/bge-small-en-v1.5` (384-dim, `normalize_embeddings=True`) |
| Backend | Qdrant local on-disk (cosine similarity) |
| Top-K | 10 chunks |
| L1 Cache | Checks Redis for pre-computed embedding before calling the model (~50–200 ms saved per repeated query) |

### Sparse Search (BM25)

| Parameter | Value |
|---|---|
| Library | `rank-bm25` (BM25Okapi) |
| Index | Built in-memory from `corpus.jsonl` on startup |
| Top-K | 10 chunks |
| Tokenizer | Simple whitespace lowercase split |

### RRF Fusion

**Formula:** `score(d) = Σ 1 / (k + rank(d))` where `k = 60` (standard)

- **Input:** 10 dense + 10 BM25 results
- **Output:** Up to 20 fused, deduplicated chunks ranked by RRF score

### Chunk Payload Fields

`chunk_id` · `text` · `source` · `url` · `doc_title` · `score` · `method` · `rrf_score`

---

## 6. Reranker Module

**File:** `rag_service/reranker.py` — Class: `Reranker`

| Parameter | Value |
|---|---|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` (~85 MB) |
| Input | 20 fused chunks + query |
| Output | Top 5 chunks by cross-encoder relevance score |
| Device | Auto-detects CUDA, falls back to CPU |
| Max length | 512 tokens per (query, chunk) pair |

**Process:** Build `(query, chunk_text)` pairs → `CrossEncoder.predict()` → attach scores → sort descending → return top 5

Each returned chunk gains a `rerank_score` field (float, 4 decimal places).

---

## 7. Cache Module

**File:** `rag_service/cache.py` — Class: `RAGCache`

### Three Cache Layers

| Layer | Key Prefix | Stores | TTL | Latency Saved |
|---|---|---|---|---|
| **L1** | `pb:emb:` | query → embedding vector | 24h | ~50–200 ms (model.encode) |
| **L2** | `pb:chunks:` | query → top-5 chunks | 6h | ~500 ms (retrieval + rerank) |
| **L3** | `pb:ans:` | query → full result dict | 1h | ~1–3 s (entire pipeline) |

### Design Decisions

- All keys are **SHA-256 hashed** (16 hex chars) to keep key sizes tiny
- **Connection pool** (max 20) shared across the entire process
- **Sliding-window TTL** — refreshed on every cache HIT so hot queries never expire mid-session
- **Graceful fallback** to in-memory `dict` when Redis is unavailable — pipeline always works
- **Thread-safe** stats counters via `threading.Lock`
- L3 strips chunk blobs before caching to keep entries lean

### Cache Stats Endpoint

`GET /cache/stats` returns: `hits`, `misses`, `errors`, `hit_rate`, `backend`, `redis_used_memory_human`, `redis_peak_memory_human`, `keys` (per layer)

---

## 8. Memory Module

**File:** `rag_service/memory.py` — Class: `Memory`

### Two-Level Persistence

| Level | Storage | Purpose |
|---|---|---|
| **Level 1** | Python list in RAM (`session_history`) | Instant access, no I/O, lost on restart |
| **Level 2** | SQLite (`rag_service/data/memory.db`) | Persists forever across server restarts |

**DB Schema:**
```sql
CREATE TABLE conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,   -- 'user' or 'assistant'
    content    TEXT NOT NULL,
    ts         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_session ON conversations(session_id, ts);
```

### Key Methods

| Method | Description |
|---|---|
| `add(role, content)` | Save message to RAM + SQLite |
| `get_context(max_turns=3)` | Last N turns as formatted string for LLM injection |
| `get_history()` | Full session history as `list[dict]` |
| `clear_session()` | Delete session from RAM + DB |
| `list_sessions()` | All sessions with message counts (limit 20) |
| `turn_count` *(property)* | Number of complete user/assistant turns |

> **Context injection:** Last 3 turns (≤ 6 messages) prepended to each LLM call. Each message truncated to 300 chars.

---

## 9. API Endpoints

**File:** `rag_service/app.py` — FastAPI + Uvicorn

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
# Interactive docs: http://localhost:8080/docs
```

> Session-scoped `PromptBridgePipeline` instances are stored in process memory. The `"default"` session is pre-loaded on startup.

### Endpoint Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate-prompt` | Main endpoint — vague request → optimized prompt |
| `POST` | `/chat` | Alias for `/generate-prompt` (backward compat) |
| `POST` | `/session/clear` | Clear conversation memory for a session |
| `GET` | `/session/{session_id}` | Get session info + last 10 history messages |
| `GET` | `/health` | Health check — status, active sessions, model |
| `GET` | `/cache/stats` | Cache hit/miss statistics |
| `POST` | `/cache/flush` | Flush all `pb:emb:*`, `pb:chunks:*`, `pb:ans:*` keys |

### Request / Response Schemas

**`POST /generate-prompt`**
```json
// Request
{ "query": "help me train a YOLO model", "session_id": "default" }

// Response
{
  "query": "...",
  "optimized_prompt": "...",
  "intent": "ml",
  "domain": "Computer Vision",
  "sources": ["promptingguide", "arxiv"],
  "latency_s": 1.23,
  "chunks": [{ "text": "...", "source": "...", "doc_title": "...", "rerank_score": 0.9 }],
  "cached": false
}
```

> **CORS:** `allow_origins=["*"]` — open for local dev. Restrict in production.

---

## 10. Storage Systems

| System | Technology | Purpose |
|---|---|---|
| Vector DB | Qdrant (on-disk) | Dense retrieval of prompt examples |
| BM25 Index | rank-bm25 (RAM) | Sparse keyword retrieval |
| Cache | Redis + in-memory fallback | L1/L2/L3 caching |
| Memory | SQLite | Persistent conversation history |
| Raw Docs | Local JSON files | Scraped web pages + PDF content |
| Corpus | JSONL file | Canonical flat chunk store for indexing |
| Cloud Storage | Cloudflare R2 (optional) | Backup of raw_docs (S3-compatible API) |

---

## 11. Models & External Services

| Model / Service | Used For |
|---|---|
| `BAAI/bge-small-en-v1.5` (local) | Query + corpus embedding (384-dim) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) | Cross-encoder reranking (20 → 5) |
| Groq API — `gpt-oss-20b` | LLM: optimized prompt generation + synthetic pair generation |
| Cloudflare R2 *(optional)* | Raw doc backup storage (S3-compatible) |
| Redis *(optional)* | Three-layer cache backend |

---

## 12. Python Dependencies

**File:** `requirements.txt`

| Category | Packages |
|---|---|
| Web scraping | `requests`, `beautifulsoup4`, `trafilatura`, `markdownify` |
| PDF ingestion | `pymupdf` (fitz), `arxiv` |
| LLM / AI | `groq >= 0.11.0` |
| Retrieval | `sentence-transformers >= 3.0.1`, `qdrant-client >= 1.11.0`, `rank-bm25 >= 0.2.2` |
| API framework | `fastapi >= 0.115.0`, `uvicorn[standard] >= 0.30.0`, `pydantic >= 2.8.0` |
| Caching | `redis[hiredis] >= 5.0.0` |
| Cloud storage | `boto3 == 1.35.0` |
| Utilities | `python-dotenv >= 1.0.0`, `tqdm == 4.66.5` |

---

## 13. Environment Variables

**Files:** `.env` / `.env.example`

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key (**required**) | — |
| `REDIS_HOST` | Redis hostname | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_DB` | Redis database number | `0` |
| `REDIS_PASSWORD` | Redis password | *(none)* |
| `CACHE_TTL_EMBEDDING` | L1 TTL in seconds | `86400` (24h) |
| `CACHE_TTL_RETRIEVAL` | L2 TTL in seconds | `21600` (6h) |
| `CACHE_TTL_ANSWER` | L3 TTL in seconds | `3600` (1h) |
| `R2_ACCOUNT_ID` | Cloudflare account ID | — |
| `R2_ACCESS_KEY_ID` | Cloudflare R2 access key | — |
| `R2_SECRET_ACCESS_KEY` | Cloudflare R2 secret key | — |
| `R2_BUCKET_NAME` | Cloudflare R2 bucket name | — |

---

## 14. Quick Start

```bash
# 1. Setup
cd promptbridge
python -m venv push
push\Scripts\activate
pip install -r requirements.txt
```

```bash
# 2. Build knowledge base (one-time)
cd rag_data_collection
python scrapers/web_scraper.py --sites all --output ./data/raw_docs
python pdf_ingestor/pdf_ingestor.py --mode arxiv --output ./data/raw_docs
python synthetic_pairs/generate_pairs_2.py --input ./data/raw_docs --output ./data/training_pairs
python validate_corpus.py --corpus ./data/training_pairs/corpus.jsonl
python embed_and_index.py --recreate
```

```bash
# 3. Start inference service
cd ..\rag_service
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

```bash
# 4. Test
curl -X POST http://localhost:8080/generate-prompt \
     -H "Content-Type: application/json" \
     -d '{"query": "help me train a YOLO model on my custom dataset"}'
```

---

## 15. Feature Changelog

### ✅ 2026-09-29 — Initial Implementation

- [x] Web scraper (4 prompt-engineering sites, BFS crawler, Cloudflare R2 upload)
- [x] PDF ingestor (local PDFs + arXiv multi-query, PyMuPDF)
- [x] Synthetic pair generator v3 (Groq JSON mode, checkpointing, hard negatives)
- [x] Corpus validator (dedup, quality checks, strict mode)
- [x] Embed & index pipeline (BGE + Qdrant, batch upsert, `--recreate` flag)
- [x] Dual retriever (Dense Qdrant + BM25 + RRF fusion, L1 embedding cache)
- [x] Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`, 20 → 5 chunks)
- [x] Three-layer Redis cache (L1 embedding / L2 retrieval / L3 answer) with in-memory fallback, sliding TTL, thread-safe stats
- [x] Conversation memory (dual-level: RAM + SQLite, multi-session)
- [x] Intent classifier (7 intents: `ml`, `code`, `research`, `creative`, `analysis`, `task`, `question`)
- [x] Domain detector (8 domains + General fallback)
- [x] Intent-specific system prompts (7 prompt templates)
- [x] FastAPI server (7 endpoints: `/generate-prompt`, `/chat`, `/session/clear`, `/session/{id}`, `/health`, `/cache/stats`, `/cache/flush`)
- [x] Session-scoped pipeline instances (per-user isolation)
- [x] CORS middleware (open for development)
- [x] PowerShell pipeline orchestrator (`run_pipeline.ps1`)

---

### 🔮 Future Roadmap

- [ ] Authentication / API key middleware
- [ ] Rate limiting per session / IP
- [ ] Streaming response (SSE / WebSocket) for LLM output
- [ ] Frontend UI (web interface for PromptBridge)
- [ ] Fine-tuned embedding model on PromptBridge corpus
- [ ] Prompt quality scoring / feedback loop
- [ ] Multi-language support
- [ ] Docker Compose deployment (Redis + API service)
- [ ] Monitoring / observability (Prometheus metrics endpoint)

---

> **Reminder:** Update this file whenever a new feature, module, or endpoint is added.
