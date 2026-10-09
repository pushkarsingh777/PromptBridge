"""
kafka_worker.py — Kafka consumer / pipeline worker for PromptBridge
====================================================================

Runs as a SEPARATE PROCESS alongside the FastAPI server.
Consumes messages from the `promptbridge.requests` Kafka topic,
executes the full PromptBridge RAG pipeline, and writes the result
to Redis so GET /job/{job_id} can serve it.

Redis result key schema:
  pb:job:<job_id>   →  JSON-encoded PromptResponse dict
  TTL               →  JOB_TTL_SECONDS (default 10 min)

Job status values stored under pb:job:<job_id>:
  {"status": "pending"}   — immediately after publish (set by producer)
  {"status": "running"}   — worker has picked up the job
  {"status": "done", ...} — pipeline complete, full result included
  {"status": "error", "detail": "..."}  — pipeline raised an exception

Run:
  python kafka_worker.py

Docker:
  See monitoring/docker-compose.yml — 'worker' service.
"""

import json
import logging
import os
import signal
import sys
import time

# pyrefly: ignore [missing-import]
from kafka import KafkaConsumer
# pyrefly: ignore [missing-import]
from kafka.errors import KafkaError

# ── Import PromptBridge pipeline ───────────────────────────────────────────────
# Ensure the rag_service directory is on the path when run standalone.
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import PromptBridgePipeline
from metrics  import REQUEST_COUNTER, REQUEST_LATENCY, LLM_LATENCY

# ── Lazy Redis import (same connection logic as cache.py) ──────────────────────
# pyrefly: ignore [missing-import]
import redis as _redis_lib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
REQUEST_TOPIC    = "promptbridge.requests"
CONSUMER_GROUP   = "promptbridge-workers"
REDIS_HOST       = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT       = int(os.getenv("REDIS_PORT", "6379"))
JOB_TTL_SECONDS  = int(os.getenv("JOB_TTL_SECONDS", "600"))   # 10 min
JOB_KEY_PREFIX   = "pb:job:"

# ── Redis client ──────────────────────────────────────────────────────────────
def _get_redis() -> _redis_lib.Redis:
    return _redis_lib.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=3,
    )


def _set_job(redis: _redis_lib.Redis, job_id: str, payload: dict) -> None:
    key = f"{JOB_KEY_PREFIX}{job_id}"
    redis.set(key, json.dumps(payload), ex=JOB_TTL_SECONDS)


# ── Per-session pipeline cache (avoid re-loading models on every message) ─────
_pipelines: dict[str, PromptBridgePipeline] = {}

def _get_pipeline(session_id: str) -> PromptBridgePipeline:
    if session_id not in _pipelines:
        _pipelines[session_id] = PromptBridgePipeline(session_id=session_id)
    return _pipelines[session_id]


# ── Message processor ─────────────────────────────────────────────────────────
def process_message(redis: _redis_lib.Redis, msg: dict) -> None:
    job_id     = msg["job_id"]
    query      = msg["query"]
    session_id = msg.get("session_id", "default")

    logger.info("Processing job %s | session=%s", job_id, session_id)

    # Mark as running so the poller knows the worker has picked it up.
    _set_job(redis, job_id, {"status": "running"})

    t_start  = time.time()
    pipeline = _get_pipeline(session_id)

    try:
        result  = pipeline.query(query, verbose=False)
        elapsed = time.time() - t_start

        intent = result.get("intent", "unknown")
        domain = result.get("domain",  "General")
        cached = str(result.get("cached", False)).lower()

        # ── Prometheus metrics (mirrors app.py instrumentation) ──────────
        REQUEST_COUNTER.labels(
            endpoint="/generate-prompt", intent=intent, domain=domain, status="success"
        ).inc()
        REQUEST_LATENCY.labels(
            endpoint="/generate-prompt", intent=intent, cached=cached
        ).observe(elapsed)

        # Persist result so the API can serve it.
        job_result = {
            "status":           "done",
            "job_id":           job_id,
            "query":            result["query"],
            "optimized_prompt": result["optimized_prompt"],
            "intent":           intent,
            "domain":           domain,
            "sources":          result["sources"],
            "latency_s":        round(elapsed, 3),
            "cached":           result.get("cached", False),
            "chunks": [
                {
                    "text":         c["text"][:200],
                    "source":       c.get("source", ""),
                    "doc_title":    c.get("doc_title", ""),
                    "rerank_score": c.get("rerank_score", 0),
                }
                for c in result.get("chunks", [])
            ],
        }
        _set_job(redis, job_id, job_result)
        logger.info("Job %s done in %.2fs (intent=%s)", job_id, elapsed, intent)

    except Exception as exc:
        elapsed = time.time() - t_start
        logger.exception("Job %s failed after %.2fs: %s", job_id, elapsed, exc)
        REQUEST_COUNTER.labels(
            endpoint="/generate-prompt", intent="unknown", domain="unknown", status="error"
        ).inc()
        _set_job(redis, job_id, {"status": "error", "detail": str(exc)})


# ── Main consumer loop ────────────────────────────────────────────────────────
def run_worker() -> None:
    logger.info("Starting PromptBridge Kafka worker")
    logger.info("  Bootstrap : %s", KAFKA_BOOTSTRAP)
    logger.info("  Topic     : %s", REQUEST_TOPIC)
    logger.info("  Group     : %s", CONSUMER_GROUP)

    redis = _get_redis()

    consumer = KafkaConsumer(
        REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=CONSUMER_GROUP,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="earliest",   # process messages missed while down
        enable_auto_commit=False,        # manual commit after successful processing
        session_timeout_ms=30_000,
        heartbeat_interval_ms=10_000,
        max_poll_interval_ms=300_000,    # up to 5 min per message (LLM can be slow)
        max_poll_records=1,              # one message at a time — keeps processing simple
    )
    logger.info("Consumer connected. Waiting for messages…")

    # Graceful shutdown on SIGTERM / SIGINT
    _running = [True]
    def _stop(sig, frame):
        logger.info("Shutdown signal received — draining…")
        _running[0] = False
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT,  _stop)

    try:
        while _running[0]:
            records = consumer.poll(timeout_ms=1_000)
            for tp, messages in records.items():
                for message in messages:
                    try:
                        process_message(redis, message.value)
                        # Commit only after successful processing.
                        consumer.commit()
                    except Exception:
                        # Log and commit anyway — the error is already stored
                        # in Redis as {"status": "error"}.
                        logger.exception("Unhandled error in process_message, committing offset.")
                        consumer.commit()
    finally:
        consumer.close()
        logger.info("Kafka worker shut down cleanly.")


if __name__ == "__main__":
    run_worker()
