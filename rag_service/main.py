# python -m uvicorn rag_service.main:app --reload --app-dir ..

import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from groq import APIError
from pydantic import BaseModel, Field

from .memory import ConversationStore
from .retrieval import HybridRetriever
from .settings import COLLECTION, CORPUS_PATH, EMBED_MODEL, GROQ_MODEL, QDRANT_PATH, SQLITE_PATH

SYSTEM_PROMPT = """You improve a user's request into a reusable, structured prompt.
Return the prompt itself, not a plan for asking the user questions. Preserve the user's intent and
use supplied sources only as supporting context; do not invent facts. When a request is usable with
ordinary defaults, make those defaults explicit in Constraints and produce the prompt immediately.
Ask one focused clarification only when a missing detail materially changes correctness, safety, or
the requested output. Do not ask for clarification merely because multiple reasonable implementations
exist. For a coding request, make the optimized prompt ask the target model to provide the requested
code, a short explanation, and appropriate edge-case handling.
Return sections: Role, Objective, Context, Task, Constraints, Expected output."""

KEYWORDS = {
    "code": {"code", "python", "api", "bug", "debug", "database", "javascript"},
    "analysis": {"analyze", "analysis", "compare", "research", "rag", "data"},
    "creative": {"write", "story", "poem", "creative", "marketing"},
    "task": {"create", "build", "design", "make", "plan"},
}


class GenerateRequest(BaseModel):
    query: str = Field(min_length=3)
    session_id: str | None = Field(default=None, max_length=128)


class RetrievedChunk(BaseModel):
    chunk_id: str
    title: str = ""
    source: str = ""
    url: str = ""
    score: float


class GenerateResponse(BaseModel):
    session_id: str
    intent: str
    prompt: str
    sources: list[RetrievedChunk]


def classify_intent(query: str) -> str:
    terms = set(query.lower().split())
    scores = {name: len(terms & keywords) for name, keywords in KEYWORDS.items()}
    return max(scores, key=scores.get) if max(scores.values()) else "question"


def build_messages(query: str, history: list[dict], chunks: list[dict]) -> list[dict]:
    history_text = "\n".join(f"User: {item['query']}\nAssistant: {item['response']}" for item in history)
    context = "\n\n".join(
        f"[{index}] {chunk['doc_title'] or chunk['source']}\n{chunk['text']}"
        for index, chunk in enumerate(chunks, 1)
    )
    content = f"Conversation history:\n{history_text or '(none)'}\n\nRetrieved context:\n{context}\n\nUser request:\n{query}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def generate_with_groq(messages: list[dict]) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    from groq import Groq
    response = Groq(api_key=key).chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.2, max_tokens=900
    )
    return response.choices[0].message.content.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = HybridRetriever(CORPUS_PATH, QDRANT_PATH, COLLECTION, EMBED_MODEL)
    app.state.memory = ConversationStore(SQLITE_PATH)
    yield


app = FastAPI(title="PromptBridge API", version="0.1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/prompts/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be blank.")
    session_id = request.session_id or str(uuid.uuid4())
    try:
        chunks = await asyncio.to_thread(app.state.retriever.search, query)
        history = await asyncio.to_thread(app.state.memory.recent, session_id)
        prompt = await asyncio.to_thread(generate_with_groq, build_messages(query, history, chunks))
        await asyncio.to_thread(app.state.memory.save, session_id, query, prompt)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except APIError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Groq generation failed for model '{GROQ_MODEL}': {exc}",
        ) from exc
    return GenerateResponse(
        session_id=session_id,
        intent=classify_intent(query),
        prompt=prompt,
        sources=[
            RetrievedChunk(chunk_id=item["chunk_id"], title=item["doc_title"], source=item["source"], url=item["url"], score=item["retrieval_score"])
            for item in chunks
        ],
    )
