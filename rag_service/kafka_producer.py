"""
kafka_producer.py — Kafka producer for PromptBridge
====================================================

Publishes incoming generate-prompt requests to the Kafka topic
`promptbridge.requests`.  The companion kafka_worker.py consumes
these messages, runs the full RAG pipeline, and stores the result
in Redis so the caller can poll GET /job/{job_id}.

Topic schema (JSON):
  {
    "job_id":    str,   # UUID4, also used as the Redis result key
    "query":     str,   # user's raw request
    "session_id": str   # pipeline session (for memory continuity)
  }

Usage:
  from kafka_producer import get_producer, publish_request
  job_id = publish_request(query="...", session_id="default")
"""

import json
import logging
import os
import uuid

# pyrefly: ignore [missing-import]
from kafka import KafkaProducer
# pyrefly: ignore [missing-import]
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
REQUEST_TOPIC   = "promptbridge.requests"

# ── Singleton producer ────────────────────────────────────────────────────────
_producer: "KafkaProducer | None" = None


def get_producer() -> KafkaProducer:
    """Return the shared KafkaProducer, creating it on first call."""
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            # Durability: wait for leader + 1 replica ack before returning.
            acks="all",
            # Retry on transient errors (network blip, leader election).
            retries=3,
            retry_backoff_ms=200,
            # Compression reduces network + broker storage cost.
            compression_type="gzip",
            # Linger 5 ms so nearby messages are batched together.
            linger_ms=5,
            request_timeout_ms=10_000,
        )
        logger.info("KafkaProducer connected to %s", KAFKA_BOOTSTRAP)
    return _producer


def publish_request(query: str, session_id: str = "default") -> str:
    """
    Publish a generate-prompt job to Kafka.

    Returns the job_id (UUID4 string) the caller should use to poll
    GET /job/{job_id} for the result.

    Raises RuntimeError if Kafka is unavailable so the caller can fall
    back to synchronous pipeline execution.
    """
    job_id  = str(uuid.uuid4())
    payload = {
        "job_id":     job_id,
        "query":      query,
        "session_id": session_id,
    }

    try:
        future = get_producer().send(REQUEST_TOPIC, value=payload)
        # Block briefly to confirm delivery; raises KafkaError on failure.
        future.get(timeout=5)
        logger.info("Published job %s (session=%s)", job_id, session_id)
    except KafkaError as exc:
        logger.error("Kafka publish failed for job %s: %s", job_id, exc)
        raise RuntimeError(f"Kafka unavailable: {exc}") from exc

    return job_id


def close_producer() -> None:
    """Flush and close the producer — call on application shutdown."""
    global _producer
    if _producer is not None:
        _producer.flush(timeout=10)
        _producer.close()
        _producer = None
        logger.info("KafkaProducer closed.")
