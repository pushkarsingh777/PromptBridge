"""
Synthetic Query-Document Pair Generator
Uses Google Gemini Flash (free tier) instead of Claude.

Install:
  pip install google-generativeai boto3 tqdm tiktoken

Run:
  python generate_pairs.py --input ./raw_docs --output ./training_pairs --pairs_per_chunk 3
"""

import os
import re
import json
import time
import random
import hashlib
import argparse
import boto3
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import tiktoken
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TOKENIZER      = tiktoken.get_encoding("cl100k_base")

# ── R2 client ─────────────────────────────────────────────────────────────────
def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        region_name="auto",     # always "auto" for R2
    )

R2_BUCKET = os.getenv("R2_BUCKET_NAME", "promptbridge-data")

# ── Gemini setup ──────────────────────────────────────────────────────────────
def get_gemini_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=500,
        )
    )

# ── Chunking ──────────────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

def chunk_text(text: str, chunk_tokens: int = 400, overlap_tokens: int = 60) -> list[dict]:
    """
    Recursive character splitter with token-aware sizing.
    Tries paragraph -> sentence -> word boundaries. Never cuts mid-sentence.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def split(t: str, sep_idx: int = 0) -> list[str]:
        if count_tokens(t) <= chunk_tokens:
            return [t]
        if sep_idx >= len(separators):
            words = t.split()
            mid   = len(words) // 2
            return split(" ".join(words[:mid])) + split(" ".join(words[mid:]))
        sep    = separators[sep_idx]
        parts  = t.split(sep)
        result, buf = [], ""
        for part in parts:
            candidate = buf + sep + part if buf else part
            if count_tokens(candidate) <= chunk_tokens:
                buf = candidate
            else:
                if buf:
                    result.extend(split(buf, sep_idx + 1))
                buf = part
        if buf:
            result.extend(split(buf, sep_idx + 1))
        return result

    raw_chunks = split(text)

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap_tokens > 0:
            prev_words  = raw_chunks[i - 1].split()
            overlap_txt = " ".join(prev_words[-overlap_tokens:])
            chunk       = overlap_txt + " " + chunk
        chunks.append({
            "chunk_id":    hashlib.md5(chunk.encode()).hexdigest()[:10],
            "chunk_index": i,
            "text":        chunk.strip(),
            "token_count": count_tokens(chunk),
        })
    return chunks

# ── Gemini prompts ────────────────────────────────────────────────────────────
QUERY_PROMPT = """You are building training data for a RAG system about prompt engineering and AI.

Given this document chunk, generate {n} diverse user queries this chunk would perfectly answer.

Rules:
- Write queries a real user would actually type
- Mix short queries (5-8 words) and longer ones (15-25 words)
- Use different formats: questions, instructions, "how do I", "what is", "explain"
- Cover different aspects of the chunk

Document chunk:
---
{chunk}
---

Respond ONLY with a JSON array of strings. No explanation, no markdown backticks:
["query 1", "query 2", "query 3"]"""

HARD_NEGATIVE_PROMPT = """Given this query and the correct document that answers it, write a DIFFERENT document that:
- Uses similar keywords and sounds related
- Could fool a search system into thinking it is relevant
- But does NOT actually answer the query

Query: {query}

Correct document:
---
{positive}
---

Write only the hard negative text. No labels, no explanation. 2-3 paragraphs."""

def generate_queries(model, chunk: str, n: int) -> list[str]:
    try:
        response = model.generate_content(
            QUERY_PROMPT.format(chunk=chunk[:2000], n=n)
        )
        text = re.sub(r"```json|```", "", response.text).strip()
        queries = json.loads(text)
        return [q for q in queries if isinstance(q, str) and len(q) > 5]
    except json.JSONDecodeError:
        match = re.search(r'\[.*?\]', response.text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return []
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            print("  Rate limit hit -- waiting 60s...")
            time.sleep(60)
            try:
                response = model.generate_content(
                    QUERY_PROMPT.format(chunk=chunk[:2000], n=n)
                )
                text = re.sub(r"```json|```", "", response.text).strip()
                return json.loads(text)
            except Exception:
                return []
        return []

def generate_hard_negative(model, query: str, positive: str) -> str | None:
    try:
        response = model.generate_content(
            HARD_NEGATIVE_PROMPT.format(query=query, positive=positive[:1000])
        )
        return response.text.strip()
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            time.sleep(60)
        return None

# ── Load raw docs ─────────────────────────────────────────────────────────────
def load_docs(input_dir: str) -> list[dict]:
    docs = []
    skip = {"summary.json", "pdf_summary.json", "generation_summary.json"}
    for f in Path(input_dir).rglob("*.json"):
        if f.name in skip:
            continue
        try:
            doc = json.loads(f.read_text())
            if "text" in doc and len(doc["text"]) > 200:
                docs.append(doc)
        except Exception:
            continue
    print(f"  Loaded {len(docs)} documents")
    return docs

# ── Upload to R2 ──────────────────────────────────────────────────────────────
def upload_to_r2(output_dir: Path):
    r2    = get_r2_client()
    files = ["corpus.jsonl", "train_pairs.jsonl", "val_pairs.jsonl", "generation_summary.json"]
    print(f"\nUploading to R2 bucket: {R2_BUCKET}")
    for filename in files:
        local_path = output_dir / filename
        if not local_path.exists():
            continue
        r2_key = f"training_data/{filename}"
        r2.upload_file(str(local_path), R2_BUCKET, r2_key)
        print(f"  uploaded {filename} --> r2://{R2_BUCKET}/{r2_key}")

# ── Main pipeline ─────────────────────────────────────────────────────────────
def generate_all_pairs(args):
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable.")
        print("Get your free key at: https://aistudio.google.com/apikey")
        return

    model      = get_gemini_model()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_docs(args.input)
    if not docs:
        print("No documents found. Run web_scraper.py and pdf_ingestor.py first.")
        return

    all_pairs  = []
    all_chunks = []
    skipped    = 0

    for doc in tqdm(docs, desc="Generating pairs"):
        chunks = chunk_text(
            doc["text"],
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
        )
        if not chunks:
            skipped += 1
            continue

        if len(chunks) > args.max_chunks_per_doc:
            chunks = random.sample(chunks, args.max_chunks_per_doc)

        for chunk in chunks:
            chunk_meta = {
                **chunk,
                "doc_id":    doc["id"],
                "doc_title": doc.get("title", ""),
                "source":    doc.get("source", ""),
                "url":       doc.get("url", ""),
            }
            all_chunks.append(chunk_meta)

            queries = generate_queries(model, chunk["text"], n=args.pairs_per_chunk)
            time.sleep(4)   # Gemini free tier: 15 req/min

            for query in queries:
                pair = {
                    "id":           hashlib.md5((query + chunk["chunk_id"]).encode()).hexdigest()[:12],
                    "query":        query,
                    "positive":     chunk["text"],
                    "chunk_id":     chunk["chunk_id"],
                    "doc_id":       doc["id"],
                    "doc_title":    doc.get("title", ""),
                    "source":       doc.get("source", ""),
                    "generated_at": datetime.utcnow().isoformat(),
                }

                if args.hard_negatives and random.random() < 0.4:
                    neg = generate_hard_negative(model, query, chunk["text"])
                    if neg:
                        pair["hard_negative"] = neg
                    time.sleep(4)

                all_pairs.append(pair)

    # Save corpus
    corpus_file = output_dir / "corpus.jsonl"
    with corpus_file.open("w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Save train/val split
    random.shuffle(all_pairs)
    split       = int(len(all_pairs) * 0.85)
    train_pairs = all_pairs[:split]
    val_pairs   = all_pairs[split:]

    (output_dir / "train_pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in train_pairs)
    )
    (output_dir / "val_pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in val_pairs)
    )

    # Save summary
    summary = {
        "total_docs":          len(docs),
        "skipped_docs":        skipped,
        "total_chunks":        len(all_chunks),
        "total_pairs":         len(all_pairs),
        "train_pairs":         len(train_pairs),
        "val_pairs":           len(val_pairs),
        "with_hard_negatives": sum(1 for p in all_pairs if "hard_negative" in p),
        "generated_at":        datetime.utcnow().isoformat(),
        "model":               "gemini-1.5-flash",
        "config": {
            "chunk_tokens":    args.chunk_tokens,
            "pairs_per_chunk": args.pairs_per_chunk,
            "hard_negatives":  args.hard_negatives,
        }
    }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n-- Generation Summary --")
    print(f"  Documents processed  : {len(docs)}")
    print(f"  Total chunks         : {len(all_chunks)}")
    print(f"  Total pairs          : {len(all_pairs)}")
    print(f"  Train / Val          : {len(train_pairs)} / {len(val_pairs)}")
    print(f"  With hard negatives  : {summary['with_hard_negatives']}")
    print(f"\n  Files saved to {output_dir}/")
    print(f"    corpus.jsonl        <- load into Qdrant vector DB")
    print(f"    train_pairs.jsonl   <- fine-tune your embedder")
    print(f"    val_pairs.jsonl     <- evaluate retrieval quality")

    if args.upload_r2:
        upload_to_r2(output_dir)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",              default="./raw_docs")
    parser.add_argument("--output",             default="./training_pairs")
    parser.add_argument("--chunk_tokens",       type=int, default=400)
    parser.add_argument("--overlap_tokens",     type=int, default=60)
    parser.add_argument("--pairs_per_chunk",    type=int, default=3)
    parser.add_argument("--max_chunks_per_doc", type=int, default=20)
    parser.add_argument("--hard_negatives",     action="store_true")
    parser.add_argument("--upload_r2",          action="store_true",
                        help="Upload outputs to Cloudflare R2 after generation")
    args = parser.parse_args()
    generate_all_pairs(args)

if __name__ == "__main__":
    main()