"""
retriever.py — Dual retrieval with RRF fusion

Runs two searches simultaneously:
  1. Dense (Traditional RAG) — BGE embedder + Qdrant vector search
  2. Sparse (BM25 / Hybrid RAG) — keyword matching

Both return 10 chunks each → RRF fusion → 20 combined chunks

Install:
  pip install qdrant-client sentence-transformers rank-bm25
"""

import json
import os
from pathlib import Path

from cache import RAGCache
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient

# pyrefly: ignore [missing-import]
from qdrant_client.models import SearchParams

# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path="../.env")

# Derive data directory from env var (CI/Docker friendly) or fall back to the
# canonical repo-relative location so the service runs on any machine.
_DATA_DIR = Path(
    os.getenv(
        "PROMPTBRIDGE_DATA_DIR",
        str(Path(__file__).resolve().parent.parent / "rag_data_collection" / "data"),
    )
)

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_PATH    = str(_DATA_DIR / "qdrant_db")
QDRANT_URL     = os.getenv("QDRANT_URL", "")   # e.g. http://qdrant:6333 (Docker)
CORPUS_PATH    = str(_DATA_DIR / "training_pairs" / "corpus.jsonl")
COLLECTION     = "promptbridge"
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
DENSE_TOP_K    = 10       # chunks from dense search
SPARSE_TOP_K   = 10       # chunks from BM25 search
RRF_K          = 60       # RRF constant (standard value)

# ── HNSW query-time config ──────────────────────────────────────────
# ef is the search beam width: how many candidate neighbours Qdrant
# evaluates before returning the final top-k results.
# • Qdrant default: max(limit, 20) → only 20 candidates for a top-10 query.
# • 128 gives meaningfully better recall with <10ms overhead per query.
# Raise to 256 if you observe recall degradation at the cost of more latency.
HNSW_EF        = 128

# ── Qdrant singleton — only one file lock per process ────────────────────────
# In Docker: QDRANT_URL=http://qdrant:6333 → HTTP client (no file lock, safe
# for concurrent processes like API + worker sharing the same volume).
# In local dev without the env var: falls back to local on-disk path client.
_qdrant_client: "QdrantClient | None" = None

def _get_qdrant_client() -> "QdrantClient":
    global _qdrant_client
    if _qdrant_client is None:
        if QDRANT_URL:
            _qdrant_client = QdrantClient(url=QDRANT_URL)
            print(f"  Qdrant mode : HTTP  ({QDRANT_URL})")
        else:
            _qdrant_client = QdrantClient(path=QDRANT_PATH)
            print(f"  Qdrant mode : local ({QDRANT_PATH})")
    return _qdrant_client

class DualRetriever:
    def __init__(self, cache: RAGCache | None = None):
        print("Initializing retriever...")

        # L1 embedding cache — shared with the pipeline's RAGCache instance
        self._cache: RAGCache = cache or RAGCache()

        # Dense search — shared Qdrant client (singleton, one file lock per process)
        self.qdrant   = _get_qdrant_client()
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cuda"
                        if self._cuda_available() else "cpu")

        # Sparse search — BM25 index built from corpus
        self.corpus_chunks = self._load_corpus()
        self.bm25, self.bm25_chunks = self._build_bm25_index()

        print(f"  Dense index : {self.qdrant.get_collection(COLLECTION).points_count} vectors")
        print(f"  BM25 index  : {len(self.bm25_chunks)} chunks")
        print(f"  HNSW ef     : {HNSW_EF} (query-time beam width)")
        print("  Retriever ready.\n")

    def _cuda_available(self):
        try:
            # pyrefly: ignore [missing-import]
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_corpus(self) -> list[dict]:
        chunks = []
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def _build_bm25_index(self):
        """Tokenize all chunks and build BM25 index."""
        print("  Building BM25 index...")
        tokenized = [
            chunk["text"].lower().split()
            for chunk in self.corpus_chunks
        ]
        bm25 = BM25Okapi(tokenized)
        return bm25, self.corpus_chunks

    # ── Dense retrieval ───────────────────────────────────────────────────────
    def dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> list[dict]:
        """Embed query and search Qdrant vector index.

        L1 cache: checks Redis for a pre-computed embedding before calling
        SentenceTransformer.encode — saves ~50–200 ms per repeated query.
        """
        # ── L1: try embedding cache first ─────────────────────────────────
        query_vector = self._cache.get_embedding(query)
        if query_vector is None:
            query_vector = self.embedder.encode(
                query,
                normalize_embeddings=True,
            ).tolist()
            self._cache.set_embedding(query, query_vector)

        results = self.qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=HNSW_EF),
        ).points

        return [
            {
                "chunk_id":  r.payload.get("chunk_id", ""),
                "text":      r.payload["text"],
                "source":    r.payload.get("source", ""),
                "url":       r.payload.get("url", ""),
                "doc_title": r.payload.get("doc_title", ""),
                "score":     r.score,
                "method":    "dense",
            }
            for r in results
        ]

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────
    def sparse_search(self, query: str, top_k: int = SPARSE_TOP_K) -> list[dict]:
        """BM25 keyword search over corpus."""
        tokenized_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.bm25_chunks[idx]
            results.append({
                "chunk_id":  chunk.get("chunk_id", str(idx)),
                "text":      chunk["text"],
                "source":    chunk.get("source", ""),
                "url":       chunk.get("url", ""),
                "doc_title": chunk.get("doc_title", ""),
                "score":     float(scores[idx]),
                "method":    "bm25",
            })

        return results

    # ── RRF Fusion ────────────────────────────────────────────────────────────
    def rrf_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        k: int = RRF_K,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.
        Combines rankings from both retrieval methods.
        Score = 1/(k + rank) summed across all lists.
        """
        rrf_scores = {}   # chunk_id → rrf_score
        chunk_map  = {}   # chunk_id → chunk dict

        for rank, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
            chunk_map[cid]  = chunk

        for rank, chunk in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        fused = []
        for cid in sorted_ids:
            chunk = chunk_map[cid].copy()
            chunk["rrf_score"] = round(rrf_scores[cid], 6)
            chunk["method"]    = "rrf_fused"
            fused.append(chunk)

        return fused

    # ── Main retrieve method ──────────────────────────────────────────────────
    def retrieve(self, query: str) -> list[dict]:
        """
        Full dual retrieval:
        Dense (10) + BM25 (10) → RRF fusion → 20 chunks
        """
        dense_results  = self.dense_search(query)
        sparse_results = self.sparse_search(query)
        fused_results  = self.rrf_fusion(dense_results, sparse_results)

        print(f"  Dense: {len(dense_results)} | BM25: {len(sparse_results)} | "
              f"Fused: {len(fused_results)} chunks")

        return fused_results   # up to 20 chunks