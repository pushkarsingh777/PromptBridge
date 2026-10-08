"""
metrics.py — Prometheus metrics for PromptBridge
=================================================

Custom metrics exposed at GET /metrics (scraped by Prometheus → Grafana).

Metrics:
  promptbridge_requests_total          Counter   requests by endpoint/intent/domain/status
  promptbridge_request_latency_seconds Histogram end-to-end latency per request
  promptbridge_llm_latency_seconds     Histogram LLM-only generation latency
  promptbridge_cache_hits_total        Counter   cache hits by layer (L1/L2/L3)
  promptbridge_cache_misses_total      Counter   cache misses by layer
  promptbridge_active_sessions         Gauge     number of live pipeline sessions
  promptbridge_retrieval_chunks        Histogram chunks returned after RRF fusion
  promptbridge_rerank_chunks           Histogram chunks returned after reranking

Usage:
  from metrics import (
      REQUEST_COUNTER, REQUEST_LATENCY, LLM_LATENCY,
      CACHE_HITS, CACHE_MISSES, ACTIVE_SESSIONS,
      RETRIEVAL_CHUNKS, RERANK_CHUNKS
  )
"""

from prometheus_client import Counter, Gauge, Histogram  # type: ignore

# ── Request metrics ───────────────────────────────────────────────────────────
REQUEST_COUNTER = Counter(
    "promptbridge_requests_total",
    "Total number of requests processed",
    ["endpoint", "intent", "domain", "status"],
)

REQUEST_LATENCY = Histogram(
    "promptbridge_request_latency_seconds",
    "End-to-end request latency in seconds",
    ["endpoint", "intent", "cached"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)

# ── LLM metrics ───────────────────────────────────────────────────────────────
LLM_LATENCY = Histogram(
    "promptbridge_llm_latency_seconds",
    "Time spent waiting for Groq LLM response",
    ["intent", "domain"],
    buckets=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
)

# ── Cache metrics ─────────────────────────────────────────────────────────────
CACHE_HITS = Counter(
    "promptbridge_cache_hits_total",
    "Total cache hits by layer",
    ["layer"],
)

CACHE_MISSES = Counter(
    "promptbridge_cache_misses_total",
    "Total cache misses by layer",
    ["layer"],
)

# ── Session metrics ───────────────────────────────────────────────────────────
ACTIVE_SESSIONS = Gauge(
    "promptbridge_active_sessions",
    "Number of active pipeline sessions in memory",
)

# ── Retrieval metrics ─────────────────────────────────────────────────────────
RETRIEVAL_CHUNKS = Histogram(
    "promptbridge_retrieval_chunks",
    "Number of chunks after RRF fusion (pre-rerank)",
    buckets=[5, 10, 15, 20, 25],
)

RERANK_CHUNKS = Histogram(
    "promptbridge_rerank_chunks_returned",
    "Number of chunks returned after cross-encoder reranking",
    buckets=[1, 2, 3, 4, 5, 6],
)
