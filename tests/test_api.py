"""
tests/test_api.py — Smoke tests for the PromptBridge FastAPI app.

These tests run WITHOUT a real pipeline (pipeline.query is mocked) so
they execute quickly in CI with no Groq API calls or model downloads.
"""

import sys, os

# Make rag_service importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_service"))

# pyrefly: ignore [missing-import]
import pytest
from unittest.mock import MagicMock, patch
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient


# ── Patch heavy deps before importing app ────────────────────────────────────
MOCK_RESULT = {
    "query":            "test query",
    "optimized_prompt": "Optimized: test query",
    "intent":           "research",
    "domain":           "General",
    "sources":          ["doc1.md"],
    "latency_s":        0.5,
    "cached":           False,
    "chunks":           [],
}


@pytest.fixture(scope="module")
def client():
    """TestClient with the pipeline mocked out."""
    with patch("pipeline.PromptBridgePipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.query.return_value = MOCK_RESULT
        instance.memory.turn_count = 0
        instance.memory.get_history.return_value = []
        instance.memory.clear_session = MagicMock()
        instance.cache.stats = {"hits": 0, "misses": 0}
        instance.cache.flush_all = MagicMock()

        # pyrefly: ignore [missing-import]
        from app import app
        yield TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generate_prompt_success(client):
    r = client.post("/generate-prompt", json={"query": "test query"})
    assert r.status_code == 200
    body = r.json()
    assert "optimized_prompt" in body
    assert body["intent"] == "research"


def test_generate_prompt_empty_query(client):
    r = client.post("/generate-prompt", json={"query": "   "})
    assert r.status_code == 400


def test_chat_alias(client):
    r = client.post("/chat", json={"query": "test query"})
    assert r.status_code == 200


def test_session_not_found(client):
    r = client.get("/session/nonexistent-session-xyz")
    assert r.status_code == 404


def test_session_clear(client):
    r = client.post("/session/clear", json={"session_id": "default"})
    assert r.status_code == 200


def test_cache_stats(client):
    r = client.get("/cache/stats")
    assert r.status_code == 200


def test_cache_flush(client):
    r = client.post("/cache/flush")
    assert r.status_code == 200


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"HELP" in r.content  # Prometheus text format
