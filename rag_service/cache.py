"""
cache.py — Three-layer Redis cache for PromptBridge
=====================================================

Layers
------
  L1 — Embedding cache  : query text → vector (list[float])   TTL: 24 h
  L2 — Retrieval cache  : query text → top-K chunks (list[dict])  TTL:  6 h
  L3 — Answer cache     : query text → full LLM answer (str)      TTL:  1 h

Design decisions
----------------
* All keys are SHA-256 hashed (16 hex chars) to keep key sizes tiny.
* Uses a connection pool so every thread reuses connections instead of
  opening a new socket per call.
* Optional password auth via REDIS_PASSWORD env var.
* Graceful fallback to an in-memory dict when Redis is absent — the
  pipeline always works, even in dev without a Redis server.
* TTL is refreshed on every cache *hit* (sliding-window expiry) so
  frequently-asked queries never expire mid-session.
* Thread-safe stats counters via a threading.Lock.

Requires
--------
  pip install "redis[hiredis]>=5.0.0"

Usage
-----
  cache = RAGCache()
  cache.set_embedding("what is RAG?", vector)
  vec = cache.get_embedding("what is RAG?")   # None on miss
"""

import hashlib
import json
import os
import threading
import time

try:
    import redis                         # type: ignore
    from redis import ConnectionPool     # type: ignore
    _REDIS_OK = True
except ImportError:
    _REDIS_OK = False

# ── Config (override via env vars) ────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST",     "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB       = int(os.getenv("REDIS_DB",   "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None) or None   # None if blank

TTL_EMBEDDING  = int(os.getenv("CACHE_TTL_EMBEDDING", str(60 * 60 * 24)))  # 24 h
TTL_RETRIEVAL  = int(os.getenv("CACHE_TTL_RETRIEVAL", str(60 * 60 * 6)))   #  6 h
TTL_ANSWER     = int(os.getenv("CACHE_TTL_ANSWER",    str(60 * 60 * 1)))   #  1 h

PREFIX_EMBED   = "pb:emb:"
PREFIX_CHUNKS  = "pb:chunks:"
PREFIX_ANSWER  = "pb:ans:"

# Connection pool shared across all RAGCache instances in this process
_pool: "ConnectionPool | None" = None
_pool_lock = threading.Lock()


def _get_pool() -> "ConnectionPool | None":
    """Lazily create a shared connection pool (thread-safe)."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        if not _REDIS_OK:
            return None
        try:
            p = ConnectionPool(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                max_connections=20,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=False,   # we handle encoding ourselves
            )
            # Verify the pool works by pinging via a temporary connection
            r = redis.Redis(connection_pool=p)
            r.ping()
            _pool = p
            print(f"  [Cache] Redis pool ready → {REDIS_HOST}:{REDIS_PORT} db={REDIS_DB}")
        except Exception as exc:
            print(f"  [Cache] Redis unavailable ({exc}) — using in-memory fallback.")
            _pool = None
    return _pool


def _sha(text: str) -> str:
    """Stable 16-char hash of a string (for cache keys)."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


class RAGCache:
    """
    Three-layer Redis cache with in-memory fallback.

    Falls back to a TTL-aware in-memory dict if Redis is unavailable,
    so the pipeline always works even without a Redis server running.
    """

    def __init__(self):
        self._pool = _get_pool()
        self._redis: "redis.Redis | None" = (
            redis.Redis(connection_pool=self._pool) if self._pool else None
        )
        self._fallback: dict = {}
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "errors": 0}

    # ── Internal get/set ──────────────────────────────────────────────────────
    def _get(self, key: str, ttl: int | None = None):
        """
        Fetch a cached value.  Returns None on miss.
        If `ttl` is provided and Redis is live, refreshes the key's TTL
        on a cache hit (sliding-window expiry).
        """
        try:
            if self._redis:
                raw = self._redis.get(key)
                if raw is None:
                    with self._lock:
                        self._stats["misses"] += 1
                    return None
                with self._lock:
                    self._stats["hits"] += 1
                # Refresh TTL on hit so hot keys never expire mid-session
                if ttl:
                    self._redis.expire(key, ttl)
                return json.loads(raw)

            # ── in-memory fallback ────────────────────────────────────────
            entry = self._fallback.get(key)
            if entry is None or (entry["exp"] and time.time() > entry["exp"]):
                with self._lock:
                    self._stats["misses"] += 1
                return None
            with self._lock:
                self._stats["hits"] += 1
            # Refresh expiry on hit
            if ttl and entry["exp"]:
                entry["exp"] = time.time() + ttl
            return entry["val"]

        except Exception as exc:
            with self._lock:
                self._stats["errors"] += 1
            print(f"  [Cache] GET error: {exc}")
            return None

    def _set(self, key: str, value, ttl: int) -> None:
        try:
            if self._redis:
                self._redis.setex(key, ttl, json.dumps(value))
                return
            # in-memory fallback
            self._fallback[key] = {
                "val": value,
                "exp": time.time() + ttl if ttl else None,
            }
        except Exception as exc:
            with self._lock:
                self._stats["errors"] += 1
            print(f"  [Cache] SET error: {exc}")

    # ── L1 — Embedding cache ──────────────────────────────────────────────────
    def get_embedding(self, query: str) -> "list[float] | None":
        """Return cached query embedding vector, or None on miss."""
        return self._get(PREFIX_EMBED + _sha(query), ttl=TTL_EMBEDDING)

    def set_embedding(self, query: str, vector: "list[float]") -> None:
        self._set(PREFIX_EMBED + _sha(query), vector, TTL_EMBEDDING)

    # ── L2 — Retrieval cache ──────────────────────────────────────────────────
    def get_chunks(self, query: str) -> "list[dict] | None":
        """Return cached retrieval chunks, or None on miss."""
        return self._get(PREFIX_CHUNKS + _sha(query), ttl=TTL_RETRIEVAL)

    def set_chunks(self, query: str, chunks: "list[dict]") -> None:
        self._set(PREFIX_CHUNKS + _sha(query), chunks, TTL_RETRIEVAL)

    # ── L3 — Answer cache ─────────────────────────────────────────────────────
    def get_answer(self, query: str) -> "dict | None":
        """Return full cached answer dict, or None on miss."""
        return self._get(PREFIX_ANSWER + _sha(query), ttl=TTL_ANSWER)

    def set_answer(self, query: str, result: dict) -> None:
        """Cache answer; strips large chunk blobs to keep the entry lean."""
        slim = {k: v for k, v in result.items() if k != "chunks"}
        self._set(PREFIX_ANSWER + _sha(query), slim, TTL_ANSWER)

    # ── Utilities ─────────────────────────────────────────────────────────────
    def invalidate(self, query: str) -> None:
        """Remove all cached data for a specific query (all three layers)."""
        for prefix in (PREFIX_EMBED, PREFIX_CHUNKS, PREFIX_ANSWER):
            key = prefix + _sha(query)
            try:
                if self._redis:
                    self._redis.delete(key)
                else:
                    self._fallback.pop(key, None)
            except Exception:
                pass

    def flush_all(self) -> None:
        """Flush every PromptBridge key from Redis (pattern delete)."""
        if self._redis:
            for prefix in (PREFIX_EMBED, PREFIX_CHUNKS, PREFIX_ANSWER):
                keys = self._redis.keys(f"{prefix}*")
                if keys:
                    self._redis.delete(*keys)
        else:
            self._fallback.clear()

    @property
    def stats(self) -> dict:
        """Return cache hit/miss stats plus live Redis memory info."""
        with self._lock:
            s = dict(self._stats)
        total = s["hits"] + s["misses"]
        s["hit_rate"] = round(s["hits"] / total, 3) if total else 0.0
        s["backend"]  = "redis" if self._redis else "in-memory"

        if self._redis:
            try:
                info = self._redis.info("memory")
                s["redis_used_memory_human"] = info.get("used_memory_human", "?")
                s["redis_peak_memory_human"] = info.get("used_memory_peak_human", "?")
                # Count keys per layer
                s["keys"] = {
                    "embeddings": self._redis.dbsize()
                        if not any([PREFIX_EMBED, PREFIX_CHUNKS, PREFIX_ANSWER])
                        else len(self._redis.keys(f"{PREFIX_EMBED}*")),
                    "chunks":  len(self._redis.keys(f"{PREFIX_CHUNKS}*")),
                    "answers": len(self._redis.keys(f"{PREFIX_ANSWER}*")),
                }
            except Exception:
                pass
        return s
