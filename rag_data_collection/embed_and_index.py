"""
Stage 2: Embedding + Indexing
Converts every chunk in corpus.jsonl into a vector and stores in Qdrant.

Runs fully LOCAL — no API needed, no cost.
Uses sentence-transformers (free HuggingFace model).

Install:
  pip install sentence-transformers qdrant-client tqdm python-dotenv

Run:
  python embed_and_index.py --corpus ./data/training_pairs/corpus.jsonl
"""

import os
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, PayloadSchemaType
)

load_dotenv(dotenv_path="../.env")

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"   # free, fast, 384-dim, ~130MB
COLLECTION     = "promptbridge"
QDRANT_PATH    = "./data/qdrant_db"          # local folder — no server needed
BATCH_SIZE     = 256                           # chunks to embed at once

# ── Setup ─────────────────────────────────────────────────────────────────────
def load_chunks(corpus_path: str) -> list[dict]:
    chunks = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks)} chunks from corpus")
    return chunks

def setup_qdrant(client: QdrantClient, vector_size: int):
    """Create collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"  Collection '{COLLECTION}' already exists — deleting and recreating")
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,   # cosine similarity for text
        )
    )
    print(f"  Created Qdrant collection '{COLLECTION}' (vector size: {vector_size})")

def embed_and_index(args):
    # ── Load chunks ───────────────────────────────────────────────────────────
    chunks = load_chunks(args.corpus)
    if not chunks:
        print("No chunks found. Run generate_pairs.py first.")
        return

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("  (downloading ~130MB on first run — cached after that)")
    model = SentenceTransformer(EMBED_MODEL,device="cuda")
    vector_size = model.get_sentence_embedding_dimension()
    print(f"  Model loaded. Vector size: {vector_size}")

    # ── Setup Qdrant (local, no server needed) ────────────────────────────────
    print(f"\nSetting up Qdrant at: {QDRANT_PATH}")
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)
    setup_qdrant(client, vector_size)

    # ── Embed + index in batches ──────────────────────────────────────────────
    print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    print("  This runs on CPU — estimated 10-20 minutes for 4,534 chunks\n")

    total_indexed = 0
    batches = [
        chunks[i:i + BATCH_SIZE]
        for i in range(0, len(chunks), BATCH_SIZE)
    ]

    for batch in tqdm(batches, desc="Embedding + indexing"):
        texts = [c["text"] for c in batch]

        # Convert text → vectors
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,   # important for cosine similarity
        )

        # Build Qdrant points
        points = []
        for i, (chunk, vector) in enumerate(zip(batch, vectors)):
            points.append(PointStruct(
                id=total_indexed + i,    # unique int ID
                vector=vector.tolist(),
                payload={                # metadata stored alongside vector
                    "chunk_id":    chunk.get("chunk_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text":        chunk["text"],
                    "doc_id":      chunk.get("doc_id", ""),
                    "doc_title":   chunk.get("doc_title", ""),
                    "source":      chunk.get("source", ""),
                    "url":         chunk.get("url", ""),
                    "token_count": chunk.get("token_count", 0),
                }
            ))

        # Upload batch to Qdrant
        client.upsert(
            collection_name=COLLECTION,
            points=points,
        )
        total_indexed += len(batch)

    # ── Verify ────────────────────────────────────────────────────────────────
    info = client.get_collection(COLLECTION)
    print(f"\n── Indexing Complete ───────────────────────────────")
    print(f"  Chunks indexed  : {info.points_count}")
    print(f"  Vector size     : {vector_size}")
    print(f"  Distance metric : cosine")
    print(f"  Saved to        : {QDRANT_PATH}/")

    # ── Quick test search ─────────────────────────────────────────────────────
    print(f"\nRunning test search...")
    test_query   = "how does chain of thought prompting work"
    query_vector = model.encode(test_query, normalize_embeddings=True).tolist()

    results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=3,
    ).points

    print(f"  Query: '{test_query}'")
    print(f"  Top 3 results:")
    for i, r in enumerate(results):
        print(f"\n  [{i+1}] Score: {r.score:.4f}")
        print(f"       Source: {r.payload.get('source','')}")
        print(f"       Title : {r.payload.get('doc_title','')[:60]}")
        print(f"       Text  : {r.payload['text'][:120]}...")

    print(f"\n  Qdrant index is ready for RAG pipeline!")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="./data/training_pairs/corpus.jsonl",
                        help="Path to corpus.jsonl from generate_pairs.py")
    args = parser.parse_args()
    embed_and_index(args)

if __name__ == "__main__":
    main()