"""
Synthetic Query-Document Pair Generator
Uses Groq (free, very fast) + batch processing (10 chunks per call).
~20-30 minutes for 690 docs vs 8+ hours with Gemini one-by-one.

Install:
  pip install groq boto3 tqdm tiktoken python-dotenv

Get free Groq key at: https://console.groq.com  (free, no card needed)
Add to .env: GROQ_API_KEY=your_key_here

Run:
  python generate_pairs.py --input ./data/raw_docs --output ./data/training_pairs
"""

import os
import re
import json
import time
import random
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import tiktoken
import boto3
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from groq import Groq

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
TOKENIZER      = tiktoken.get_encoding("cl100k_base")
MODEL          = "openai/gpt-oss-20b"   # fastest free Groq model
BATCH_SIZE     = 10                        # chunks per API call

# ── Groq client ───────────────────────────────────────────────────────────────
def get_groq_client():
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set in .env\n"
            "Get your free key (no card) at: https://console.groq.com\n"
            "Then add to .env: GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=GROQ_API_KEY)

# ── R2 client ─────────────────────────────────────────────────────────────────
def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )

R2_BUCKET = os.getenv("R2_BUCKET_NAME", "promptbridge-data")

# ── Iterative chunker ─────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, disallowed_special=()))

def chunk_text(text: str, chunk_tokens: int = 400, overlap_tokens: int = 60) -> list[dict]:
    if not text or not text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    chunks, buf, buf_tok = [], [], 0

    for sentence in sentences:
        sent_tok = count_tokens(sentence)

        if sent_tok > chunk_tokens:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_tok = [], 0
            words, word_buf, word_tok = sentence.split(), [], 0
            for word in words:
                w_tok = count_tokens(word + " ")
                if word_tok + w_tok > chunk_tokens and word_buf:
                    chunks.append(" ".join(word_buf))
                    word_buf, word_tok = [], 0
                word_buf.append(word)
                word_tok += w_tok
            if word_buf:
                chunks.append(" ".join(word_buf))
            continue

        if buf_tok + sent_tok <= chunk_tokens:
            buf.append(sentence)
            buf_tok += sent_tok
        else:
            if buf:
                chunks.append(" ".join(buf))
            if overlap_tokens > 0 and buf:
                overlap_buf, overlap_used = [], 0
                for s in reversed(buf):
                    s_tok = count_tokens(s)
                    if overlap_used + s_tok <= overlap_tokens:
                        overlap_buf.insert(0, s)
                        overlap_used += s_tok
                    else:
                        break
                buf     = overlap_buf + [sentence]
                buf_tok = overlap_used + sent_tok
            else:
                buf, buf_tok = [sentence], sent_tok

    if buf:
        chunks.append(" ".join(buf))

    return [
        {
            "chunk_id":    hashlib.md5(c.encode()).hexdigest()[:10],
            "chunk_index": i,
            "text":        c,
            "token_count": count_tokens(c),
        }
        for i, c in enumerate(chunks) if c.strip()
    ]

# ── Batch query generation ────────────────────────────────────────────────────
# Send 10 chunks in one call → get 10 sets of queries back
# 10x fewer API calls = 10x faster

BATCH_PROMPT = """You are building training data for a RAG system about prompt engineering.

Below are {n} numbered document chunks. For EACH chunk generate {q} diverse user queries that chunk would perfectly answer.

Rules for queries:
- Write queries a real user would actually type
- Mix short (5-8 words) and longer (15-25 words) queries  
- Use different formats: questions, "how do I", "what is", "explain"

{chunks}

Respond ONLY with a JSON object mapping chunk number to array of queries.
No explanation, no markdown:
{{"1": ["query", "query"], "2": ["query", "query"], ...}}"""

def generate_queries_batch(client: Groq, chunks: list[str], queries_per_chunk: int) -> dict[int, list[str]]:
    """Send multiple chunks in one API call. Returns {chunk_index: [queries]}."""
    numbered = "\n\n".join(
        f"CHUNK {i+1}:\n---\n{chunk[:800]}\n---"
        for i, chunk in enumerate(chunks)
    )

    prompt = BATCH_PROMPT.format(
        n=len(chunks),
        q=queries_per_chunk,
        chunks=numbered
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)

        # Normalize keys to int
        return {
            int(k): [q for q in v if isinstance(q, str) and len(q) > 5]
            for k, v in result.items()
        }

    except json.JSONDecodeError:
        # Try to extract JSON object from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                return {int(k): v for k, v in result.items()}
            except Exception:
                pass
        return {}

    except Exception as e:
        err = str(e)
        if "429" in err or "rate" in err.lower() or "limit" in err.lower():
            print(f"  Rate limit — waiting 30s...")
            time.sleep(30)
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000,
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"```json|```", "", raw).strip()
                return {int(k): v for k, v in json.loads(raw).items()}
            except Exception:
                return {}
        print(f"  Batch error: {e}")
        return {}

# ── Load docs ─────────────────────────────────────────────────────────────────
def load_docs(input_dir: str) -> list[dict]:
    skip = {"summary.json", "pdf_summary.json", "generation_summary.json"}
    docs = []
    for f in Path(input_dir).rglob("*.json"):
        if f.name in skip:
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if "text" in doc and len(doc["text"]) > 200:
                docs.append(doc)
        except Exception:
            continue
    print(f"  Loaded {len(docs)} documents")
    return docs

# ── Upload to R2 ──────────────────────────────────────────────────────────────
def upload_to_r2(output_dir: Path):
    missing = [k for k in ["R2_ACCOUNT_ID","R2_ACCESS_KEY_ID",
                            "R2_SECRET_ACCESS_KEY","R2_BUCKET_NAME"]
               if not os.getenv(k)]
    if missing:
        print(f"[R2] Missing in .env: {missing} — skipping upload")
        return
    r2    = get_r2_client()
    files = ["corpus.jsonl","train_pairs.jsonl","val_pairs.jsonl","generation_summary.json"]
    print(f"\nUploading to R2 bucket: {R2_BUCKET}")
    for filename in files:
        local = output_dir / filename
        if not local.exists():
            continue
        key = f"training_data/{filename}"
        r2.upload_file(str(local), R2_BUCKET, key)
        print(f"  uploaded {filename} -> r2://{R2_BUCKET}/{key}")

# ── Main ──────────────────────────────────────────────────────────────────────
def generate_all_pairs(args):
    client     = get_groq_client()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(args.input)
    if not docs:
        print("No documents found. Run web_scraper.py first.")
        return

    # Step 1: Chunk all documents first
    print("Chunking documents...")
    all_chunks_with_meta = []
    for doc in docs:
        text = doc.get("text", "")
        if not text.strip():
            continue
        chunks = chunk_text(text, args.chunk_tokens, args.overlap_tokens)
        if not chunks:
            continue
        if len(chunks) > args.max_chunks_per_doc:
            chunks = random.sample(chunks, args.max_chunks_per_doc)
        for chunk in chunks:
            all_chunks_with_meta.append({
                **chunk,
                "doc_id":    doc.get("id", ""),
                "doc_title": doc.get("title", ""),
                "source":    doc.get("source", ""),
                "url":       doc.get("url", ""),
            })

    print(f"  Total chunks: {len(all_chunks_with_meta)}")
    estimated_calls = len(all_chunks_with_meta) // BATCH_SIZE + 1
    print(f"  API calls needed: ~{estimated_calls} (batches of {BATCH_SIZE})")
    print(f"  Estimated time: ~{estimated_calls * 2 // 60} minutes\n")

    # Step 2: Generate queries in batches
    all_pairs = []
    batches   = [
        all_chunks_with_meta[i:i + BATCH_SIZE]
        for i in range(0, len(all_chunks_with_meta), BATCH_SIZE)
    ]

    for batch in tqdm(batches, desc="Generating pairs (batched)"):
        chunk_texts = [c["text"] for c in batch]
        results     = generate_queries_batch(client, chunk_texts, args.pairs_per_chunk)

        for i, chunk_meta in enumerate(batch):
            queries = results.get(i + 1, [])  # keys are 1-indexed
            for query in queries:
                all_pairs.append({
                    "id":           hashlib.md5((query + chunk_meta["chunk_id"]).encode()).hexdigest()[:12],
                    "query":        query,
                    "positive":     chunk_meta["text"],
                    "chunk_id":     chunk_meta["chunk_id"],
                    "doc_id":       chunk_meta["doc_id"],
                    "doc_title":    chunk_meta["doc_title"],
                    "source":       chunk_meta["source"],
                    "generated_at": datetime.utcnow().isoformat(),
                })

        time.sleep(1)   # Groq is fast — just 1s between batch calls

    # Step 3: Save outputs
    with (output_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for chunk in all_chunks_with_meta:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    random.shuffle(all_pairs)
    split = int(len(all_pairs) * 0.85)

    (output_dir / "train_pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in all_pairs[:split]),
        encoding="utf-8"
    )
    (output_dir / "val_pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in all_pairs[split:]),
        encoding="utf-8"
    )

    summary = {
        "total_docs":    len(docs),
        "total_chunks":  len(all_chunks_with_meta),
        "total_pairs":   len(all_pairs),
        "train_pairs":   len(all_pairs[:split]),
        "val_pairs":     len(all_pairs[split:]),
        "generated_at":  datetime.utcnow().isoformat(),
        "model":         MODEL,
        "batch_size":    BATCH_SIZE,
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n── Generation Summary ──────────────────────────────")
    print(f"  Documents   : {len(docs)}")
    print(f"  Chunks      : {len(all_chunks_with_meta)}")
    print(f"  Total pairs : {len(all_pairs)}")
    print(f"  Train / Val : {len(all_pairs[:split])} / {len(all_pairs[split:])}")
    print(f"\n  Saved to: {output_dir}/")
    print(f"    corpus.jsonl       <- vector DB")
    print(f"    train_pairs.jsonl  <- embedder fine-tuning")
    print(f"    val_pairs.jsonl    <- evaluation")

    if args.upload_r2:
        upload_to_r2(output_dir)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",              default="./data/raw_docs")
    parser.add_argument("--output",             default="./data/training_pairs")
    parser.add_argument("--chunk_tokens",       type=int, default=400)
    parser.add_argument("--overlap_tokens",     type=int, default=60)
    parser.add_argument("--pairs_per_chunk",    type=int, default=3)
    parser.add_argument("--max_chunks_per_doc", type=int, default=15)
    parser.add_argument("--upload_r2",          action="store_true")
    args = parser.parse_args()
    generate_all_pairs(args)

if __name__ == "__main__":
    main()