import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(ROOT / "rag_data_collection/data/qdrant_db")))
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", str(ROOT / "rag_data_collection/data/training_pairs/corpus.jsonl")))
COLLECTION = os.getenv("QDRANT_COLLECTION", "promptbridge")
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(ROOT / "data/promptbridge.db")))
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
