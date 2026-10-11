# PromptBridge

PromptBridge is a completed, local-first RAG application that turns a short natural-language request into a structured, reusable prompt. It detects intent and domain, retrieves prompt-engineering examples, reranks them, adds recent conversation context, and asks Groq to generate the optimized prompt.

This repository records the project's completed state as of October 2026. It is a prototype/MVP, not a hosted production service. The code and documentation are the source of truth for implemented behavior; proposed capabilities are labeled as future work.

## What is included

- FastAPI API in `rag_service/` for prompt generation, session history, cache controls, health, and Prometheus metrics.
- Hybrid retrieval using local Qdrant dense search and BM25, reciprocal-rank fusion, and a cross-encoder reranker.
- A corpus collection pipeline in `rag_data_collection/` for web/PDF ingestion, synthetic prompt-example generation, corpus validation, and Qdrant indexing.
- SQLite conversation memory and a three-level cache that uses Redis when available and otherwise falls back to process memory.
- Docker Compose monitoring stack with Qdrant, the API, Redis, Prometheus, and Grafana.
- Smoke tests in `tests/test_api.py` that mock the generation pipeline.

See [system_architecture.md](system_architecture.md) for implementation details, [prd.txt](prd.txt) for the completed product requirements, and [PROJECT_DETAILS.md](PROJECT_DETAILS.md) for the full project reference.

## Run locally

Use Python 3.11 or a compatible Python installation. From the repository root:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set a valid `GROQ_API_KEY` in `.env`. The API constructs its pipeline from `rag_service/`; run it from that directory so the configured relative memory path resolves as intended:

```powershell
Set-Location rag_service
python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/docs` for the interactive API reference. The API also needs the existing local Qdrant index under `rag_data_collection/data/qdrant_db`. If that index is absent, build it using the instructions below.

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/generate-prompt -ContentType 'application/json' -Body '{"query":"Create a prompt for debugging a Python API","session_id":"demo"}'
```

The response includes `optimized_prompt`, detected `intent` and `domain`, source names, selected chunks, cache status, and latency.

## Build or refresh the local index

The source corpus is `rag_data_collection/data/training_pairs/corpus.jsonl`. To validate and index the corpus from the repository root:

```powershell
python rag_data_collection/validate_corpus.py --corpus rag_data_collection/data/training_pairs/corpus.jsonl
python rag_data_collection/embed_and_index.py --corpus rag_data_collection/data/training_pairs/corpus.jsonl
```

To replace the existing collection, explicitly add `--recreate`; this deletes/recreates the named local collection. The end-to-end PowerShell data collection process is `rag_data_collection/run_pipeline.ps1`; it collects web and arXiv material, generates synthetic examples, validates the corpus, and indexes it. Review its output and corpus quality before using a refreshed index.

## Run with Docker Compose

From `monitoring/`, configure `GROQ_API_KEY` in the environment (or a Compose-readable `.env`) and start the stack:

```powershell
docker compose up --build
```

The API listens on port 8080, Qdrant on 6333, Prometheus on 9090, Grafana on 3000, and Redis on 6379. The Compose file currently sets Grafana's default admin credentials to `admin` / `promptbridge`; change them before exposing that service beyond a trusted local environment. For the exact configuration, see `monitoring/docker-compose.yml`.

## API routes

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/generate-prompt` | Generate a structured prompt from a request |
| `POST` | `/chat` | Backward-compatible alias for prompt generation |
| `GET` | `/session/{session_id}` | Read the in-memory session's recent history |
| `POST` | `/session/clear` | Clear a session's persisted conversation history |
| `GET` | `/health` | Basic service status and configured LLM model |
| `GET` | `/cache/stats` | Cache statistics for the default pipeline |
| `POST` | `/cache/flush` | Flush PromptBridge cache entries for the default pipeline |
| `GET` | `/metrics` | Prometheus metrics endpoint |

## Project documents

- [README](README.md): setup and common commands.
- [PRD](prd.txt): implemented requirements, known limits, and deferred scope.
- [System architecture](system_architecture.md): components, runtime flow, and deployment.
- [Project details](PROJECT_DETAILS.md): consolidated reference and repository map.
