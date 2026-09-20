# PromptBridge

PromptBridge is a local hybrid-RAG service that turns a user requirement into a structured prompt. The current MVP uses BGE embeddings and Qdrant dense retrieval, BM25 sparse retrieval, reciprocal-rank fusion, SQLite short-term memory, and Groq for final prompt generation.

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`. It is required only for generation, not for indexing.

## Build the existing index

The checked-in local index can be used as-is. To rebuild it from the corpus, use:

```powershell
cd rag_data_collection
python validate_corpus.py --corpus .\data\training_pairs\corpus.jsonl
python embed_and_index.py --recreate
```

`--recreate` is deliberately required because rebuilding replaces the local collection.

## Run the API

```powershell
python -m uvicorn rag_service.main:app --reload
```

Send a request to `POST /v1/prompts/generate`:

```json
{"query":"Create a prompt for debugging a Python API", "session_id":"optional-session-id"}
```

The API returns the detected baseline intent, generated structured prompt, session ID, and the five retrieved sources.

## Refresh source data

```powershell
cd rag_data_collection
.\run_pipeline.ps1 -RecreateIndex
```

The pipeline collects data, creates synthetic pairs through Groq, reports corpus-quality issues, then indexes the corpus. Corpus validation currently reports issues but does not discard documents; review its output before treating a refresh as production-ready.
