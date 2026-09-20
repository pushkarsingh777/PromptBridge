"""
PromptBridge - Synthetic RAG Training Pair Generator (v3)
===========================================================

Fixes applied vs v2:
    - Smaller batch size (default 3) to reduce truncation risk
    - Higher max output tokens (4096)
    - Uses Groq's response_format={"type": "json_object"} with a
      wrapper key "pairs" (Groq's JSON mode requires an object,
      not a bare array)
    - Concise-answer system prompt to reduce token usage per pair
    - On a malformed/truncated JSON response, does NOT immediately
      resend the identical request — backs off and mutates the
      request slightly (fewer chunks) before retrying
    - Checkpoints after every batch (resumable)
    - Handles Ctrl+C safely

Expected project structure:

promptbridge/
├── .env                (GROQ_API_KEY=gsk_...)
└── rag_data_collection/
    ├── data/
    │   ├── raw_docs/
    │   └── training_pairs/
    └── synthetic_pairs/
        └── generate_pairs_3.py

Example:

python synthetic_pairs/generate_pairs_3.py `
    --input ./data/raw_docs `
    --output ./data/training_pairs `
    --pairs_per_chunk 2 `
    --max_chunks_per_doc 15 `
    --batch_size 3 `
    --hard_negatives
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

# pyrefly: ignore [missing-import]
from groq import Groq
# pyrefly: ignore [missing-import]
from groq import APIError
# pyrefly: ignore [missing-import]
from groq import RateLimitError


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-20b"

DEFAULT_PAIRS_PER_CHUNK = 2
DEFAULT_MAX_CHUNKS_PER_DOC = 15

# Smaller batches = less chance of the model running out of
# output tokens mid-JSON.
DEFAULT_BATCH_SIZE = 3

DEFAULT_TRAIN_RATIO = 0.90

DEFAULT_MAX_RETRIES = 3

# Maximum amount of text sent from a chunk to the model.
MAX_CHUNK_CHARS = 6500

# Raised from 1800 -> 4096. Truncation was happening because
# the model ran out of room mid-answer.
MAX_OUTPUT_TOKENS = 4096

TEMPERATURE = 0.2

RANDOM_SEED = 42

SYSTEM_PROMPT = """You generate high-quality synthetic RAG training pairs.

For each input chunk, generate exactly the requested number of
question-answer pairs.

Rules:
1. Questions must be answerable directly from the provided chunk.
2. Answers must contain only information supported by the chunk.
3. Keep answers concise: 1-2 sentences maximum.
4. Do not explain your reasoning.
5. Do not add commentary.
6. Do not use markdown.
7. Return ONLY valid JSON matching the requested schema.
8. Never truncate the JSON. Every object you start must be finished.
9. Output exactly the requested number of pairs per chunk.
10. If you are running low on room, generate fewer chunks worth of
    pairs completely rather than truncating a pair halfway."""


# ============================================================
# GLOBAL STATE
# ============================================================

STOP_REQUESTED = False


def signal_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n" + "=" * 65)
    print("Interrupt requested. Finishing current operation and saving progress...")
    print("=" * 65)


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# PATH / ENV UTILITIES
# ============================================================

def find_project_root() -> Path:
    script_dir = Path(__file__).resolve()
    candidates = [
        script_dir.parent.parent.parent,
        script_dir.parent.parent,
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / ".env").exists():
            return candidate
    return Path.cwd()


def load_environment() -> Path:
    project_root = find_project_root()
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from: {env_file}")
    else:
        print(f"WARNING: .env not found at: {env_file}")
        print("Trying environment variables directly.")
    return project_root


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    first = int(max_chars * 0.75)
    last = max_chars - first
    return text[:first] + "\n\n[...content truncated...]\n\n" + text[-last:]


def stable_id(*values: str) -> str:
    raw = "||".join(values)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


# ============================================================
# DOCUMENT LOADING
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".json",
    ".yaml", ".yml", ".csv", ".xml", ".html", ".htm",
}


def extract_text_from_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return ""


def load_documents(input_dir: Path) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for path in tqdm(files, desc="Loading documents", unit="doc"):
        raw = extract_text_from_file(path)
        if path.suffix.lower() == ".json":
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                record = {}
        else:
            record = {}
        text = clean_text(record.get("text", raw))
        if not text:
            continue
        documents.append({
            "doc_id": str(record.get("id") or stable_id(str(path.resolve()))),
            "source": str(record.get("source") or path.relative_to(input_dir)),
            "doc_title": str(record.get("title") or path.stem),
            "url": str(record.get("url") or ""),
            "path": str(path),
            "text": text,
        })

    return documents


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text: str, chunk_size: int = 3500, overlap: int = 350) -> List[str]:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = start + chunk_size
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start = end - overlap
            continue

        candidate = current + "\n\n" + paragraph if current else paragraph

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            if overlap > 0 and current:
                overlap_text = current[-overlap:]
                current = overlap_text + "\n\n" + paragraph
            else:
                current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def is_usable_chunk(text: str) -> bool:
    """Remove short and repeated loading/error boilerplate before it reaches the index."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return len(tokens) >= 20 and len(set(tokens)) > 2


def build_chunks(documents: List[Dict[str, Any]], max_chunks_per_doc: int) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []

    for doc in tqdm(documents, desc="Chunking documents", unit="doc"):
        chunks = chunk_text(doc["text"])
        if max_chunks_per_doc > 0:
            chunks = chunks[:max_chunks_per_doc]

        for index, chunk in enumerate(chunks):
            if not is_usable_chunk(chunk):
                continue
            chunk_id = stable_id(doc["doc_id"], str(index), chunk)
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "doc_title": doc.get("doc_title", ""),
                "url": doc.get("url", ""),
                "chunk_index": index,
                "text": chunk,
            })

    return all_chunks


# ============================================================
# GROQ CLIENT
# ============================================================

def create_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found.\n\n"
            "Make sure your .env contains:\n"
            "GROQ_API_KEY=gsk_..."
        )
    return Groq(api_key=api_key)


def test_groq(client: Groq, model: str) -> bool:
    print("\nTesting Groq model...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK"}],
            temperature=0,
            max_tokens=10,
        )
        content = response.choices[0].message.content or ""
        print(f"Model response: {content}")
        print(f"Model: {model}")
        print("Groq connection OK.\n")
        return True
    except Exception as exc:
        print("Groq connection FAILED.")
        print(exc)
        return False


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_pairs_json(text: str) -> List[Dict[str, Any]]:
    """
    Expects an object like: {"pairs": [ {...}, {...} ]}
    Falls back to locating a bare array if the model ignores
    the object wrapper.
    """
    if not text:
        raise ValueError("Empty model response")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # Preferred: object with "pairs" key
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("pairs"), list):
            return data["pairs"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: find first {...} containing "pairs"
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and isinstance(data.get("pairs"), list):
                return data["pairs"]
        except json.JSONDecodeError:
            pass

    # Fallback: find a bare array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON from model response:\n" + text[:1000])


# ============================================================
# PROMPT
# ============================================================

def build_generation_prompt(chunks: List[Dict[str, Any]], pairs_per_chunk: int) -> str:
    chunk_blocks = []
    for chunk in chunks:
        content = truncate_text(chunk["text"], MAX_CHUNK_CHARS)
        chunk_blocks.append(f'CHUNK_ID: {chunk["chunk_id"]}\nTEXT:\n{content}'.strip())

    chunks_text = "\n\n".join(chunk_blocks)
    total_pairs = len(chunks) * pairs_per_chunk

    prompt = f"""
Generate exactly {total_pairs} synthetic RAG training pairs
({pairs_per_chunk} per chunk, for {len(chunks)} chunks).

Return a JSON object of this exact shape:

{{
  "pairs": [
    {{
      "chunk_id": "CHUNK_ID",
      "query": "question",
      "answer": "concise answer, 1-2 sentences"
    }}
  ]
}}

INPUT CHUNKS:

{chunks_text}
"""
    return prompt.strip()


# ============================================================
# GENERATE ONE BATCH
# ============================================================

def call_model(
    client: Groq,
    model: str,
    chunks: List[Dict[str, Any]],
    pairs_per_chunk: int,
) -> List[Dict[str, Any]]:
    """
    Single API call. Raises on failure/malformed JSON so the
    caller can decide how to retry.
    """
    prompt = build_generation_prompt(chunks, pairs_per_chunk)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )

    finish_reason = response.choices[0].finish_reason
    raw = response.choices[0].message.content or ""

    if finish_reason == "length":
        # The model ran out of tokens - the JSON is guaranteed
        # truncated. Treat this as a hard failure so the caller
        # shrinks the batch rather than retrying identically.
        raise ValueError("Response was cut off (finish_reason=length)")

    parsed = extract_pairs_json(raw)

    valid_chunk_ids = {c["chunk_id"] for c in chunks}
    valid_pairs: List[Dict[str, Any]] = []

    for item in parsed:
        if not isinstance(item, dict):
            continue

        chunk_id = str(item.get("chunk_id", "")).strip()
        query = clean_text(str(item.get("query", "")))
        answer = clean_text(str(item.get("answer", "")))

        if chunk_id not in valid_chunk_ids:
            continue
        if not query or not answer:
            continue
        if len(query) < 5 or len(answer) < 2:
            continue

        valid_pairs.append({"chunk_id": chunk_id, "query": query, "answer": answer})

    # Deduplicate within batch
    seen = set()
    unique = []
    for pair in valid_pairs:
        key = (pair["chunk_id"], pair["query"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)

    return unique


def generate_batch(
    client: Groq,
    model: str,
    chunks: List[Dict[str, Any]],
    pairs_per_chunk: int,
    max_retries: int,
) -> List[Dict[str, Any]]:
    """
    Generate one batch, with backoff that SHRINKS the batch on
    truncation/parse failures instead of resending the identical
    request.
    """
    working_chunks = list(chunks)

    for attempt in range(1, max_retries + 1):
        try:
            return call_model(client, model, working_chunks, pairs_per_chunk)

        except RateLimitError as exc:
            error_text = str(exc)
            print(f"\nRate limit on attempt {attempt}/{max_retries}")
            print(error_text)

            wait_seconds = parse_retry_seconds(error_text)
            if wait_seconds is None:
                wait_seconds = min(60 * attempt, 180)

            if wait_seconds > 300:
                print(f"Rate limit requires ~{wait_seconds / 60:.1f} minutes.")
                raise

            print(f"Waiting {wait_seconds:.1f} seconds...")
            time.sleep(wait_seconds)

        except APIError as exc:
            print(f"\nGroq API error (attempt {attempt}/{max_retries}):")
            print(exc)
            if attempt < max_retries:
                time.sleep(min(5 * attempt, 30))
            else:
                raise

        except (ValueError, json.JSONDecodeError) as exc:
            # Truncated or malformed JSON. Don't resend the same
            # request - shrink it so it's more likely to fit.
            print(f"\nGeneration error (attempt {attempt}/{max_retries}):")
            print(exc)

            if attempt < max_retries and len(working_chunks) > 1:
                new_size = max(1, len(working_chunks) // 2)
                print(f"Shrinking batch from {len(working_chunks)} to {new_size} chunk(s) and retrying.")
                working_chunks = working_chunks[:new_size]
                time.sleep(2 * attempt)
            elif attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                print("Giving up on this batch after max retries.")
                return []

        except Exception as exc:
            print(f"\nUnexpected error (attempt {attempt}/{max_retries}):")
            print(exc)
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                raise

    return []


def parse_retry_seconds(error_text: str) -> Optional[int]:
    match = re.search(
        r"try again in\s+(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?",
        error_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    minutes = int(match.group(1) or 0)
    seconds = float(match.group(2) or 0)
    return int(minutes * 60 + seconds + 2)


# ============================================================
# CHECKPOINT
# ============================================================

def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "generation_checkpoint.json"


def save_checkpoint(output_dir: Path, completed_batches: int, pairs: List[Dict[str, Any]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "completed_batches": completed_batches,
        "total_pairs": len(pairs),
        "pairs": pairs,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = checkpoint_path(output_dir)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    temp_path.replace(path)


def load_checkpoint(output_dir: Path) -> Tuple[int, List[Dict[str, Any]]]:
    path = checkpoint_path(output_dir)
    if not path.exists():
        return 0, []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        completed_batches = int(data.get("completed_batches", 0))
        pairs = data.get("pairs", [])
        if not isinstance(pairs, list):
            pairs = []

        print("\nResuming generation.")
        print(f"Existing pairs: {len(pairs)}")
        print(f"Starting batch: {completed_batches}")

        return completed_batches, pairs

    except Exception as exc:
        print(f"WARNING: Could not load checkpoint: {exc}")
        return 0, []


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_pairs(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = []
    seen = set()
    for pair in pairs:
        chunk_id = pair.get("chunk_id", "")
        query = clean_text(pair.get("query", ""))
        if not chunk_id or not query:
            continue
        key = (chunk_id, query.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


# ============================================================
# HARD NEGATIVES (local, lexical)
# ============================================================

def tokenize(text: str) -> set:
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "have", "has", "had", "into", "about", "what", "which",
        "when", "where", "how", "why", "can", "does", "not", "but", "you",
        "your", "its", "their", "they", "then", "than",
    }
    return {w for w in words if w not in stopwords}


def similarity_score(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def find_hard_negative(
    query: str,
    positive_chunk_id: str,
    chunks_by_id: Dict[str, Dict[str, Any]],
    token_cache: Dict[str, set],
) -> Optional[str]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return None

    best_id = None
    best_score = -1.0

    candidates = list(chunks_by_id.keys())
    if len(candidates) > 1000:
        candidates = random.sample(candidates, 1000)

    for chunk_id in candidates:
        if chunk_id == positive_chunk_id:
            continue
        tokens = token_cache.get(chunk_id, set())
        score = similarity_score(query_tokens, tokens)
        if score > best_score:
            best_score = score
            best_id = chunk_id

    return best_id


# ============================================================
# OUTPUT
# ============================================================

def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_corpus(output_dir: Path, chunks: List[Dict[str, Any]]):
    records = [
        {
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "doc_id": chunk["doc_id"],
            "doc_title": chunk.get("doc_title", ""),
            "source": chunk["source"],
            "url": chunk.get("url", ""),
        }
        for chunk in chunks
    ]
    write_jsonl(output_dir / "corpus.jsonl", records)


def build_training_records(
    pairs: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    hard_negatives: bool,
) -> List[Dict[str, Any]]:
    token_cache = {}
    if hard_negatives:
        for chunk_id, chunk in chunks_by_id.items():
            token_cache[chunk_id] = tokenize(chunk["text"])

    records = []
    for pair in pairs:
        chunk_id = pair["chunk_id"]
        positive = chunks_by_id.get(chunk_id)
        if positive is None:
            continue

        record = {
            "id": stable_id(chunk_id, pair["query"]),
            "query": pair["query"],
            "positive": positive["text"],
            "positive_chunk_id": chunk_id,
            "source": positive["source"],
        }

        if hard_negatives:
            negative_id = find_hard_negative(pair["query"], chunk_id, chunks_by_id, token_cache)
            if negative_id:
                negative = chunks_by_id.get(negative_id)
                if negative:
                    record["hard_negative"] = negative["text"]
                    record["hard_negative_chunk_id"] = negative_id

        records.append(record)

    return records


def split_pairs(records: List[Dict[str, Any]], train_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = list(records)
    random.Random(RANDOM_SEED).shuffle(records)
    split_index = int(len(records) * train_ratio)
    return records[:split_index], records[split_index:]


# ============================================================
# MAIN GENERATION
# ============================================================

def generate_all_pairs(args):
    load_environment()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = create_groq_client()

    if not test_groq(client, args.model):
        raise RuntimeError("Groq model test failed.")

    documents = load_documents(input_dir)
    print(f"\nLoaded {len(documents)} documents")

    chunks = build_chunks(documents, args.max_chunks_per_doc)
    print(f"Total chunks: {len(chunks)}")

    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    start_batch, existing_pairs = load_checkpoint(output_dir)
    existing_pairs = deduplicate_pairs(existing_pairs)

    batch_size = args.batch_size
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    total_batches = len(batches)
    expected_pairs = len(chunks) * args.pairs_per_chunk

    print("\n" + "=" * 65)
    print("Synthetic Pair Generation")
    print("=" * 65)
    print(f"Documents          : {len(documents)}")
    print(f"Chunks             : {len(chunks)}")
    print(f"Pairs per chunk    : {args.pairs_per_chunk}")
    print(f"Batch size         : {batch_size}")
    print(f"Total batches      : {total_batches}")
    print(f"Model              : {args.model}")
    print(f"Expected pairs     : ~{expected_pairs}")
    print(f"Existing pairs     : {len(existing_pairs)}")
    print(f"Starting batch     : {start_batch}")
    print("=" * 65)

    all_pairs = list(existing_pairs)

    progress_total = max(total_batches - start_batch, 0)
    progress = tqdm(range(start_batch, total_batches), total=progress_total, desc="Generating pairs")

    for batch_index in progress:
        if STOP_REQUESTED:
            print("\nStopping before next batch.")
            save_checkpoint(output_dir, batch_index, all_pairs)
            break

        batch_chunks = batches[batch_index]

        try:
            generated = generate_batch(
                client=client,
                model=args.model,
                chunks=batch_chunks,
                pairs_per_chunk=args.pairs_per_chunk,
                max_retries=args.max_retries,
            )

        except RateLimitError as exc:
            print("\n" + "=" * 65)
            print("GROQ RATE LIMIT REACHED")
            print("=" * 65)
            print(exc)
            print("\nProgress has been saved. Resume later using the same command.")
            print("=" * 65)
            save_checkpoint(output_dir, batch_index, all_pairs)
            break

        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            save_checkpoint(output_dir, batch_index, all_pairs)
            print("Progress has been checkpointed. Run the same command again to resume.")
            break

        except Exception as exc:
            print(f"\nBatch {batch_index}: unrecoverable API error")
            print(exc)
            print("\nSaving checkpoint.")
            save_checkpoint(output_dir, batch_index, all_pairs)
            break

        before = len(all_pairs)
        all_pairs.extend(generated)
        all_pairs = deduplicate_pairs(all_pairs)
        added = len(all_pairs) - before

        if added == 0:
            print(f"\nWARNING: Batch {batch_index} generated 0 pairs.")

        save_checkpoint(output_dir, batch_index + 1, all_pairs)
        progress.set_postfix(pairs=len(all_pairs), added=added)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("Writing final dataset")
    print("=" * 65)

    all_pairs = deduplicate_pairs(all_pairs)

    write_corpus(output_dir, chunks)

    records = build_training_records(
        pairs=all_pairs,
        chunks_by_id=chunks_by_id,
        hard_negatives=args.hard_negatives,
    )

    train_pairs, val_pairs = split_pairs(records, args.train_ratio)

    write_jsonl(output_dir / "train_pairs.jsonl", train_pairs)
    write_jsonl(output_dir / "val_pairs.jsonl", val_pairs)

    metadata = {
        "total_docs": len(documents),
        "total_chunks": len(chunks),
        "total_pairs": len(all_pairs),
        "training_records": len(records),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "pairs_per_chunk": args.pairs_per_chunk,
        "batch_size": args.batch_size,
        "max_chunks_per_doc": args.max_chunks_per_doc,
        "hard_negatives": args.hard_negatives,
        "model": args.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with (output_dir / "generation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nGeneration Summary")
    print("─" * 50)
    print(f"  Documents   : {len(documents)}")
    print(f"  Chunks      : {len(chunks)}")
    print(f"  Total pairs : {len(all_pairs)}")
    print(f"  Train / Val : {len(train_pairs)} / {len(val_pairs)}")
    print(f"\n  Saved to: {output_dir}")
    print("    corpus.jsonl")
    print("    train_pairs.jsonl")
    print("    val_pairs.jsonl")
    print("    generation_metadata.json")
    print("    generation_checkpoint.json")
    print("\nDone.")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic RAG training pairs using Groq.")

    parser.add_argument("--input", required=True, help="Directory containing raw documents.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model ID. Default: {DEFAULT_MODEL}")
    parser.add_argument("--pairs_per_chunk", type=int, default=DEFAULT_PAIRS_PER_CHUNK)
    parser.add_argument("--max_chunks_per_doc", type=int, default=DEFAULT_MAX_CHUNKS_PER_DOC)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--hard_negatives", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.pairs_per_chunk < 1:
        raise ValueError("--pairs_per_chunk must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if not (0.5 <= args.train_ratio <= 0.99):
        raise ValueError("--train_ratio must be between 0.5 and 0.99")

    try:
        generate_all_pairs(args)
    except KeyboardInterrupt:
        print("\nGeneration interrupted.")
    except Exception as exc:
        print("\nFATAL ERROR:")
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
