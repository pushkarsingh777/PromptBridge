"""Build the local PromptBridge Qdrant index from a canonical corpus.jsonl."""

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "promptbridge"
DEFAULT_BATCH_SIZE = 128


def project_root() -> Path:
    return Path(__file__).resolve().parent


def normalise_chunk(record: dict) -> dict:
    """Accept the legacy corpus and the newer nested-metadata corpus."""
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    text = record.get("text", "")
    chunk_id = record.get("chunk_id") or record.get("id")
    if not isinstance(text, str) or not text.strip() or not chunk_id:
        raise ValueError("Every corpus record requires non-empty text and chunk_id (or id).")
    return {
        "chunk_id": str(chunk_id),
        "chunk_index": record.get("chunk_index", metadata.get("chunk_index", 0)),
        "text": text.strip(),
        "doc_id": record.get("doc_id", metadata.get("doc_id", "")),
        "doc_title": record.get("doc_title", metadata.get("doc_title", "")),
        "source": record.get("source", metadata.get("source", "")),
        "url": record.get("url", metadata.get("url", "")),
        "token_count": record.get("token_count", 0),
    }


def load_chunks(corpus_path: Path) -> list[dict]:
    chunks, seen = [], set()
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                chunk = normalise_chunk(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid corpus row {line_number}: {exc}") from exc
            if chunk["chunk_id"] not in seen:
                chunks.append(chunk)
                seen.add(chunk["chunk_id"])
    if not chunks:
        raise ValueError("Corpus contains no valid chunks.")
    return chunks


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_index(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus).resolve()
    db_path = Path(args.qdrant_path).resolve()
    chunks = load_chunks(corpus_path)
    print(f"Loaded {len(chunks)} valid chunks from {corpus_path}")

    device = select_device(args.device)
    print(f"Loading {args.model} on {device}")
    model = SentenceTransformer(args.model, device=device)
    vector_size = model.get_sentence_embedding_dimension()
    client = QdrantClient(path=str(db_path))

    existing = {item.name for item in client.get_collections().collections}
    if args.collection in existing:
        if not args.recreate:
            raise RuntimeError(
                f"Collection '{args.collection}' already exists. Re-run with --recreate to replace it."
            )
        client.delete_collection(args.collection)

    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    for start in tqdm(range(0, len(chunks), args.batch_size), desc="Embedding and indexing"):
        batch = chunks[start : start + args.batch_size]
        vectors = model.encode(
            [item["text"] for item in batch],
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        client.upsert(
            collection_name=args.collection,
            points=[
                PointStruct(id=start + offset, vector=vector.tolist(), payload=chunk)
                for offset, (chunk, vector) in enumerate(zip(batch, vectors))
            ],
        )

    count = client.get_collection(args.collection).points_count
    print(f"Indexed {count} chunks into '{args.collection}' at {db_path}")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Embed PromptBridge corpus into local Qdrant.")
    parser.add_argument("--corpus", default=str(root / "data/training_pairs/corpus.jsonl"))
    parser.add_argument("--qdrant-path", default=str(root / "data/qdrant_db"))
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--model", default=EMBED_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild an existing collection.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    build_index(parse_args())
