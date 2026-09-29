"""
reranker.py — Cross-encoder reranker

Takes 20 fused chunks → scores each one carefully against the query
→ returns top 5 most relevant chunks

Uses: cross-encoder/ms-marco-MiniLM-L-6-v2
- Free, ~85MB, runs on CPU or GPU
- Much more accurate than cosine similarity
- Reads (query, chunk) pair together and gives a relevance score
"""

# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder

# ── Config ────────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N          = 5     # chunks to return after reranking

class Reranker:
    def __init__(self):
        print(f"Loading reranker: {RERANKER_MODEL}")
        self.model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,
            device="cuda" if self._cuda_available() else "cpu",
        )
        print("  Reranker ready.\n")

    def _cuda_available(self):
        try:
            # pyrefly: ignore [missing-import]
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def rerank(self, query: str, chunks: list[dict], top_n: int = TOP_N) -> list[dict]:
        """
        Score every (query, chunk) pair using cross-encoder.
        Returns top_n chunks sorted by relevance score.

        Cross-encoder reads BOTH query and chunk together
        → much more accurate than embedding similarity alone
        """
        if not chunks:
            return []

        # Build (query, chunk_text) pairs for cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs — cross-encoder reads each pair fully
        scores = self.model.predict(pairs)

        # Attach scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)

        # Sort by rerank score descending
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        top_chunks = reranked[:top_n]

        print(f"  Reranked {len(chunks)} → top {len(top_chunks)} chunks")
        for i, c in enumerate(top_chunks):
            print(f"    [{i+1}] score={c['rerank_score']:.3f} | "
                  f"{c['source']} | {c['text'][:60]}...")

        return top_chunks