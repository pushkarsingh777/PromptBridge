# PromptBridge — project reference

**Project status:** Completed for now. This document captures the repository's implemented state as of 2026-10-10. It is a consolidated reference for the product, code, data, runtime, operations, and known boundaries.

## 1. Summary

PromptBridge is a local-first hybrid retrieval-augmented generation (RAG) application for prompt engineering. It takes a user's requirement and returns a structured prompt that can be copied into another AI model. PromptBridge's generation step constructs the prompt; it does not perform the task described by the user.

The implemented path is:

```text
Natural-language request
  → keyword intent and domain detection
  → dense vector retrieval + sparse BM25 retrieval
  → reciprocal-rank fusion
  → cross-encoder reranking
  → recent SQLite-backed conversation context
  → Groq chat completion
  → structured prompt plus source and runtime metadata
```

The codebase includes the FastAPI service, data collection/index tools, local persistent data, a Docker Compose deployment, Prometheus/Grafana configuration, and mocked API smoke tests.

## 2. Product behavior

### Inputs and outputs

The main input is a non-empty text `query`; an optional `session_id` groups conversation history. The service returns:

- Original query.
- `optimized_prompt`: a prompt intended to be pasted into another AI model.
- `intent` and `domain` labels.
- Source identifiers for retrieved examples.
- Up to five selected chunk summaries (text is abbreviated in the API output).
- `latency_s` and a `cached` flag.

### Intent and domain

Intent classification is deterministic keyword matching, not a trained model. Current intents are `code`, `ml`, `research`, `creative`, `analysis`, `task`, and `question`. Domain detection is another rule list and can return Computer Vision, NLP, Data Science, Software Engineering, Machine Learning, Agriculture, Cybersecurity, Research, or General. Since each rule list checks keywords in a fixed order, overlapping topics can resolve to the first matching rule.

### Generation instructions

The pipeline has one system instruction per intent family. These direct the LLM to create a structured, ready-to-use prompt and to avoid answering/implementing the user's underlying task. Sections vary by intent, and commonly cover role, objective, context, inputs, constraints, output format, evaluation, and assumptions. The returned text is model-generated and is not formally validated for truth or completeness.

## 3. Retrieval and generation stack

| Stage | Current implementation |
|---|---|
| Query embedding | `BAAI/bge-small-en-v1.5` via Sentence Transformers |
| Dense store/search | Qdrant collection `promptbridge`; local persistence by default, HTTP service if `QDRANT_URL` is set |
| Sparse search | BM25 over the configured corpus text |
| Candidate depth | Ten dense and ten sparse candidates by default |
| Rank fusion | Reciprocal-rank fusion with constant 60 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation context | Up to five reranked prompt-engineering examples plus up to three previous conversation turns |
| LLM provider/model | Groq API, model identifier `openai/gpt-oss-20b` in `rag_service/pipeline.py` |

An index rebuild should use the same embedding model and collection configuration as online retrieval. The local Qdrant files are in `rag_data_collection/data/qdrant_db`; the index corpus is `rag_data_collection/data/training_pairs/corpus.jsonl`.

## 4. API reference

The FastAPI app is `rag_service/app.py`. Interactive API docs appear at `/docs` when the server runs.

| Method and path | Behavior |
|---|---|
| `POST /generate-prompt` | Generate prompt; blank/whitespace query returns HTTP 400 |
| `POST /chat` | Alias for `/generate-prompt` |
| `GET /session/{session_id}` | Return turn count and the last ten messages for a session present in this app process; unknown session returns 404 |
| `POST /session/clear` | Clear the session history from SQLite and its active memory |
| `GET /health` | Return `ok`, active session count, and configured model name; does not probe upstream dependencies |
| `GET /cache/stats` | Return counters for the default pipeline cache |
| `POST /cache/flush` | Clear PromptBridge cache keys for the default pipeline |
| `GET /metrics` | Prometheus text exposition endpoint |

Example request body:

```json
{
  "query": "Create a prompt for debugging a Python API",
  "session_id": "demo"
}
```

The response schema is defined by `PromptResponse` in `rag_service/app.py`. It includes `query`, `optimized_prompt`, `intent`, `domain`, `sources`, `latency_s`, `chunks`, and `cached`.

## 5. Memory and cache

Conversation records are written to SQLite at `./data/memory.db` relative to the service working directory. Run the API from `rag_service/` to use the checked-in service data location. The active app stores pipeline instances by session ID in process memory. The most recent three turns are eligible for context injection.

`rag_service/cache.py` implements:

| Cache layer | Value | Default lifetime |
|---|---|---:|
| L1 | Query embedding | 24 hours |
| L2 | Retrieved/reranked chunks | 6 hours |
| L3 | Full generated result | 1 hour |

The cache connects to Redis when the Redis dependency and server are available. Otherwise it uses an in-process TTL dictionary. Query keys are hashes of stripped, lowercased query strings. Answer cache is shared by identical query text across sessions; it is not a session-specific conversation cache. The in-memory fallback is neither shared across workers nor persistent.

## 6. Data collection and indexing

The scripts in `rag_data_collection/` support an offline corpus build:

1. `scrapers/web_scraper.py` scrapes configured prompt-engineering resources (Prompting Guide, Learn Prompting, Anthropic documentation, and OpenAI documentation are listed as sources).
2. `pdf_ingestor/pdf_ingestor.py` can ingest local PDFs or download a bounded arXiv result set and extract PDF text.
3. `synthetic_pairs/generate_pairs.py` creates synthetic query/document prompt examples and corpus artifacts from raw source documents.
4. `validate_corpus.py` checks corpus records and reports malformed, empty, duplicate, or otherwise questionable entries; strictness is configurable.
5. `embed_and_index.py` embeds corpus entries and writes them into Qdrant.

`run_pipeline.ps1` runs the stages in order. It defaults to local output, requests at most 100 pages per selected site and 15 arXiv papers in its current invocation, and accepts `-RecreateIndex` to replace the collection. Optional R2 upload is enabled by its switch/environment setting and requires valid R2 credentials.

Main data locations:

- `rag_data_collection/data/raw_docs/`: collected raw source records.
- `rag_data_collection/data/training_pairs/corpus.jsonl`: primary retrieval corpus.
- `rag_data_collection/data/training_pairs/train_pairs.jsonl` and `val_pairs.jsonl`: generated train/validation pair sets.
- `rag_data_collection/data/qdrant_db/`: local Qdrant index.
- `rag_service/data/memory.db`: local conversation store created at runtime.

Corpus quality validation is not equivalent to human source review or a retrieval benchmark. Recreating an index is a destructive replacement operation for the named local collection; keep backups if the current index matters.

## 7. Repository map

| Path | Role |
|---|---|
| `README.md` | Quick setup, API usage, index build, Compose launch |
| `prd.txt` | Completed requirements, acceptance boundary, deferred work |
| `system_architecture.md` | Component and deployment architecture |
| `PROJECT_DETAILS.md` | Consolidated project reference (this file) |
| `rag_service/app.py` | FastAPI routes, schemas, session registry, metrics endpoint |
| `rag_service/pipeline.py` | Rules, prompt templates, retrieval orchestration, Groq generation |
| `rag_service/retriever.py` | Dense and sparse search, Qdrant client, ranking fusion |
| `rag_service/reranker.py` | Cross-encoder scoring and top-k selection |
| `rag_service/memory.py` | SQLite-backed conversation memory |
| `rag_service/cache.py` | Redis/in-memory three-layer cache |
| `rag_service/metrics.py` | Prometheus metric definitions |
| `rag_service/Dockerfile` | Container image for the API |
| `rag_data_collection/scrapers/` | Website extraction |
| `rag_data_collection/pdf_ingestor/` | PDF/arXiv ingestion |
| `rag_data_collection/synthetic_pairs/` | Synthetic example generation |
| `rag_data_collection/validate_corpus.py` | JSONL corpus validation |
| `rag_data_collection/embed_and_index.py` | Qdrant collection build/update |
| `rag_data_collection/run_pipeline.ps1` | End-to-end data refresh orchestration |
| `monitoring/docker-compose.yml` | Local multi-service deployment |
| `monitoring/prometheus.yml` | Prometheus scraping configuration |
| `monitoring/grafana/` | Provisioned datasource and dashboard |
| `tests/test_api.py` | Mocked API smoke tests |
| `requirements.txt`, `requirements-api.txt` | Collection/development and service dependencies |
| `.env.example` | Configuration template; values are placeholders |

## 8. Local setup and run

Use Python 3.11 or a compatible installation. From the repository root:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` to a valid key in `.env`. Then start the API from its service directory:

```powershell
Set-Location rag_service
python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/docs`. The retriever expects the local index under `rag_data_collection/data/qdrant_db` unless `QDRANT_URL` points to a Qdrant server. If the index needs to be built, from repository root:

```powershell
python rag_data_collection/validate_corpus.py --corpus rag_data_collection/data/training_pairs/corpus.jsonl
python rag_data_collection/embed_and_index.py --corpus rag_data_collection/data/training_pairs/corpus.jsonl
```

Add `--recreate` to the indexing command only when replacing the existing `promptbridge` collection is intended.

## 9. Docker and monitoring

From `monitoring/`, provide `GROQ_API_KEY` in the Compose environment and run:

```powershell
docker compose up --build
```

| Service | Host port | Purpose |
|---|---:|---|
| PromptBridge API | 8080 | HTTP API |
| Qdrant | 6333 (REST), 6334 (gRPC) | Vector database |
| Redis | 6379 | Optional persistent cache backend |
| Prometheus | 9090 | Metrics scraping and query UI |
| Grafana | 3000 | Dashboard UI |

The Compose file currently contains Grafana development credentials (`admin` / `promptbridge`) and permissive CORS defaults. Those settings are suitable only for a trusted local environment unless changed. Prometheus retains seven days of metrics as configured in Compose; its scrape interval is 15 seconds.

## 10. Configuration and secrets

Important runtime settings:

- `GROQ_API_KEY`: required for cache misses that need generation.
- `QDRANT_URL`: optional remote Qdrant endpoint; local persistent Qdrant is default.
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`: optional cache backend settings.
- `CACHE_TTL_EMBEDDING`, `CACHE_TTL_RETRIEVAL`, `CACHE_TTL_ANSWER`: optional cache lifetimes in seconds.
- `CORS_ORIGINS`: comma-separated allowed origins; the application default is `*`.
- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`: optional ingestion upload credentials.

`.env.example` also has placeholders for Gemini, Anthropic, OpenAI, Hugging Face, Cohere, AWS, R2, and Kafka. These are not all runtime requirements; inspect the specific ingestion command before supplying any optional key. Never commit a populated `.env` file or expose credentials in logs.

## 11. Tests and verification

`tests/test_api.py` uses mocked pipeline behavior and checks HTTP behavior for health, prompt generation, blank input, alias, session lookup/clear, cache endpoints, and metrics. It does not test live Groq behavior, embedding downloads, Qdrant retrieval quality, or end-to-end source collection. The documentation update itself does not imply that any live integration or quality benchmark was run.

## 12. Known limitations and explicit non-goals

- No web frontend is included; clients use the API or its Swagger UI.
- No trained intent classifier or confidence output exists.
- No API authentication, rate limiting, user account system, or tenant isolation exists.
- Sessions and pipeline instances are process-local; multi-worker/replica coordination is not provided.
- `/health` is a basic app status response, not a dependency readiness probe.
- Generated prompts are not automatically fact-checked or guaranteed to avoid unsupported assumptions.
- No feedback/rating product flow, MLflow registry, Airflow scheduling, automated retraining, drift detection, quality gate, or alerting policy is implemented.
- No production SLO or load benchmark is defined.
- The global API exception handler currently includes traceback details in its response; reconsider this before exposing the API to untrusted users.

The completion milestone means the scoped MVP and its supporting repository artifacts are in place. It does not claim that all items from the original broader vision have been built or that the system has passed a production-readiness review.

## 13. Related documents

- [`README.md`](README.md): concise installation and command reference.
- [`prd.txt`](prd.txt): scope and requirements status.
- [`system_architecture.md`](system_architecture.md): runtime architecture and service interactions.
