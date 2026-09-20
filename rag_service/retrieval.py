"""Hybrid retrieval: dense Qdrant retrieval plus local BM25 with RRF fusion."""

import json
import re
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def normalise_record(record: dict) -> dict:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    return {
        "chunk_id": str(record.get("chunk_id") or record.get("id") or ""),
        "chunk_index": record.get("chunk_index", metadata.get("chunk_index", 0)),
        "text": str(record.get("text", "")).strip(),
        "doc_id": record.get("doc_id", metadata.get("doc_id", "")),
        "doc_title": record.get("doc_title", metadata.get("doc_title", "")),
        "source": record.get("source", metadata.get("source", "")),
        "url": record.get("url", metadata.get("url", "")),
    }


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\\-]+", text.lower())


class HybridRetriever:
    def __init__(self, corpus_path: Path, qdrant_path: Path, collection: str, model_name: str):
        self.collection = collection
        self.client = QdrantClient(path=str(qdrant_path))
        self.model = SentenceTransformer(model_name)
        self.chunks = self._load_corpus(corpus_path)
        self.by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.bm25 = BM25Okapi([tokenise(chunk["text"]) for chunk in self.chunks])

    @staticmethod
    def _load_corpus(path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {path}")
        chunks = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = normalise_record(json.loads(line))
                    if chunk["chunk_id"] and chunk["text"]:
                        chunks.append(chunk)
        if not chunks:
            raise ValueError("Corpus has no usable chunks.")
        return chunks

    def _dense(self, query: str, limit: int) -> list[dict]:
        vector = self.model.encode(query, normalize_embeddings=True).tolist()
        points = self.client.query_points(collection_name=self.collection, query=vector, limit=limit).points
        return [{"chunk_id": str(point.payload["chunk_id"]), "score": point.score} for point in points]

    def _sparse(self, query: str, limit: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenise(query))
        indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:limit]
        return [{"chunk_id": self.chunks[index]["chunk_id"], "score": float(scores[index])} for index in indices]

    def search(self, query: str, candidate_limit: int = 10, result_limit: int = 5) -> list[dict]:
        fused = defaultdict(float)
        for ranking in (self._dense(query, candidate_limit), self._sparse(query, candidate_limit)):
            for rank, item in enumerate(ranking, 1):
                fused[item["chunk_id"]] += 1 / (60 + rank)
        ranked = sorted(fused, key=fused.get, reverse=True)[:result_limit]
        return [{**self.by_id[chunk_id], "retrieval_score": round(fused[chunk_id], 6)} for chunk_id in ranked]
