"""
pipeline.py — PromptBridge RAG Prompt Engineering Pipeline
===========================================================

Correct product behavior:
  User submits a vague/incomplete request
      → Intent + Domain classify
      → Dual retrieval (Dense + BM25 → 20 chunks of prompt engineering examples)
      → Cross-encoder rerank (20 → 5 best examples)
      → Memory inject (last 3 turns for context)
      → LLM uses examples to GENERATE an optimized, structured prompt
      → Returns the OPTIMIZED PROMPT (not an answer to the question)

Output prompt structure (per FR-08):
  - Role / Persona
  - Task Objective
  - Background Context
  - Detailed Requirements
  - Inputs & Constraints
  - Expected Output Format
  - Evaluation Criteria
  - Assumptions

Install:
  pip install groq rank-bm25 sentence-transformers qdrant-client python-dotenv

Run:
  python pipeline.py
"""

import os
import time
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from groq import Groq

from retriever import DualRetriever
from reranker  import Reranker
from memory    import Memory
from cache     import RAGCache

load_dotenv(dotenv_path="../.env")

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "openai/gpt-oss-20b"

# ── Intent → domain-aware system prompt ───────────────────────────────────────
# Each system prompt instructs the LLM to GENERATE a structured prompt,
# NOT to answer the user's question directly.
SYSTEM_PROMPTS = {

    "code": """You are an expert prompt engineer specializing in software development and coding tasks.

Your ONLY job is to transform the user's vague coding request into a fully structured, optimized prompt that they can paste directly into an AI model like ChatGPT, Claude, or Gemini.

DO NOT answer the user's question. DO NOT write the actual code. ONLY generate the optimized prompt.

Use the retrieved examples from the knowledge base to understand best practices for structuring coding prompts.

Your output prompt must include these sections where relevant:
**Role**: [Define the AI's persona, e.g., "You are a senior Python engineer..."]
**Task**: [Clear, specific task description]
**Context**: [Relevant background, tech stack, constraints]
**Requirements**: [Detailed functional and technical requirements]
**Input**: [What the user will provide to the model]
**Output Format**: [Exact format expected: code block, explanation, tests, etc.]
**Constraints**: [Language version, libraries, performance, style guide]
**Examples**: [If applicable, example input/output pairs]""",

    "ml": """You are an expert prompt engineer specializing in machine learning and AI systems.

Your ONLY job is to transform the user's vague ML/AI request into a fully structured, optimized prompt that they can paste directly into an AI model.

DO NOT explain ML concepts. DO NOT build the model. ONLY generate the optimized prompt.

Use the retrieved examples to understand how to structure ML-focused prompts effectively.

Your output prompt must include:
**Role**: [e.g., "You are an ML engineer specializing in computer vision, specializing in natural language processing, specializing in MLOPs, etc."]
**Task**: [Specific ML task with measurable objective]
**Domain**: [Application domain and problem context]
**Dataset**: [Data description, format, size, labels]
**Model Requirements**: [Architecture preferences, frameworks, constraints]
**Evaluation Metrics**: [How success is measured: accuracy, F1, mAP, etc.]
**Output Format**: [Code, explanation, training script, results table]
**Constraints**: [Hardware limits, latency requirements, interpretability needs]
**Assumptions**: [State any assumptions made]""",

    "research": """You are an expert prompt engineer specializing in academic research and technical analysis.

Your ONLY job is to transform the user's vague research request into a fully structured, optimized prompt that they can paste directly into an AI model.

DO NOT conduct the research. DO NOT write the paper. ONLY generate the optimized prompt.

Use the retrieved examples to model well-structured research prompts.

Your output prompt must include:
**Role**: [e.g., "You are a research scientist specializing in..."]
**Task**: [Research question or analysis objective]
**Background**: [Relevant context, prior work references]
**Scope**: [What is in/out of scope for this research task]
**Methodology**: [Preferred approach or method]
**Output Format**: [Literature review, structured analysis, comparison table, etc.]
**Constraints**: [Time period, publication sources, technical depth]
**Evaluation**: [How the output will be judged for quality]""",

    "creative": """You are an expert prompt engineer specializing in creative writing and content generation.

Your ONLY job is to transform the user's vague creative request into a fully structured, optimized prompt that they can paste directly into an AI model.

DO NOT write the content yourself. ONLY generate the optimized prompt.

Your output prompt must include:
**Role**: [e.g., "You are a creative writing expert with a talent for..."]
**Task**: [Specific creative task]
**Style & Tone**: [Writing style, voice, tone, genre]
**Audience**: [Who will read this content]
**Length**: [Word count, sections, depth]
**Key Elements**: [Themes, characters, structure to include]
**Output Format**: [Blog post, story, poem, script, etc.]
**Constraints**: [What to avoid, content restrictions]""",

    "analysis": """You are an expert prompt engineer specializing in data analysis and strategic evaluation.

Your ONLY job is to transform the user's vague analysis request into a fully structured, optimized prompt that they can paste directly into an AI model.

DO NOT perform the analysis yourself. ONLY generate the optimized prompt.

Your output prompt must include:
**Role**: [e.g., "You are a senior data analyst specializing in..."]
**Task**: [Specific analytical objective]
**Data**: [Data source, format, variables of interest]
**Analysis Type**: [Comparative, causal, descriptive, predictive]
**Output Format**: [Tables, charts description, written summary, recommendations]
**Constraints**: [Scope, assumptions, limitations to acknowledge]
**Evaluation**: [How the analysis quality will be judged]""",

    "task": """You are an expert prompt engineer helping users structure task-oriented requests.

Your ONLY job is to transform the user's vague task request into a fully structured, optimized prompt that they can paste directly into an AI model.

DO NOT complete the task yourself. ONLY generate the optimized prompt.

Your output prompt must include:
**Role**: [Appropriate AI persona for this task]
**Task**: [Specific, actionable task description]
**Context**: [Background and why this task matters]
**Steps**: [If multi-step, outline the expected process]
**Output Format**: [Deliverable format: list, plan, document, etc.]
**Constraints**: [Time, resources, style, length]
**Success Criteria**: [How to know the task is done well]""",

    "question": """You are an expert prompt engineer.

Your ONLY job is to transform the user's vague question or request into a fully structured, optimized prompt that they can paste directly into an AI model like ChatGPT, Claude, or Gemini to get a much better response.

DO NOT answer the question yourself. ONLY generate the optimized prompt.

Use the retrieved knowledge-base examples to understand how to structure prompts for this domain.

Your output prompt must include:
**Role**: [The AI persona best suited to answer this]
**Task**: [The refined, specific version of the user's question]
**Context**: [Relevant background the AI needs to know]
**Output Format**: [How the answer should be structured]
**Constraints**: [Depth, length, technical level, what to avoid]
**Assumptions**: [Any assumptions made in refining the request]""",
}

# ── Intent + Domain classifier ────────────────────────────────────────────────
def classify_intent(query: str) -> str:
    """
    Classify the user's request intent to select the right prompt template.
    Returns one of: code, ml, research, creative, analysis, task, question
    """
    q = query.lower()

    ml_words       = ["yolo", "model", "train", "dataset", "neural", "cnn", "rnn",
                      "transformer", "classification", "detection", "segmentation",
                      "machine learning", "deep learning", "computer vision", "nlp",
                      "fine-tune", "embedding", "llm", "rag", "diffusion", "gan",
                      "accuracy", "f1", "precision", "recall", "epoch", "batch"]
    code_words     = ["code", "python", "function", "script", "implement", "debug",
                      "error", "program", "class", "def ", "sql", "javascript",
                      "api", "build", "write a function", "typescript", "react",
                      "flask", "fastapi", "django", "bug", "fix"]
    research_words = ["research", "paper", "study", "literature", "survey", "review",
                      "analyze literature", "academic", "journal", "findings",
                      "hypothesis", "experiment", "methodology"]
    creative_words = ["write", "story", "poem", "creative", "generate text", "draft",
                      "compose", "essay", "blog post", "article", "content", "copy",
                      "marketing", "social media", "caption"]
    analysis_words = ["analyze", "compare", "difference", "evaluate", "pros and cons",
                      "versus", "vs ", "explain the difference", "which is better",
                      "assess", "review", "audit", "benchmark"]
    task_words     = ["how to", "steps to", "guide", "tutorial", "help me",
                      "what should i", "plan", "strategy", "roadmap", "checklist",
                      "workflow", "process"]

    if any(w in q for w in ml_words):       return "ml"
    if any(w in q for w in code_words):     return "code"
    if any(w in q for w in research_words): return "research"
    if any(w in q for w in creative_words): return "creative"
    if any(w in q for w in analysis_words): return "analysis"
    if any(w in q for w in task_words):     return "task"
    return "question"


def detect_domain(query: str) -> str:
    """
    Detect the application domain from the user's request (FR-04).
    """
    q = query.lower()
    domains = {
        "Computer Vision":      ["vision", "image", "yolo", "detection", "segmentation",
                                  "object", "camera", "visual", "opencv", "pixel"],
        "NLP":                  ["text", "nlp", "language", "sentiment", "summarize",
                                  "translate", "token", "bert", "gpt", "llm", "rag"],
        "Data Science":         ["data", "dataset", "pandas", "analysis", "csv",
                                  "statistics", "visualization", "plot", "chart"],
        "Software Engineering": ["api", "backend", "frontend", "database", "sql",
                                  "rest", "microservice", "deployment", "docker"],
        "Machine Learning":     ["train", "model", "neural", "deep learning", "epoch",
                                  "loss", "accuracy", "classification", "regression"],
        "Agriculture":          ["weed", "crop", "farm", "plant", "soil", "yield",
                                  "harvest", "agriculture", "irrigation"],
        "Cybersecurity":        ["security", "hack", "vulnerability", "encryption",
                                  "firewall", "pentest", "threat", "auth"],
        "Research":             ["paper", "research", "academic", "literature",
                                  "hypothesis", "experiment", "survey"],
    }
    for domain, keywords in domains.items():
        if any(k in q for k in keywords):
            return domain
    return "General"


# ── Prompt assembler ──────────────────────────────────────────────────────────
def assemble_prompt(
    query:    str,
    chunks:   list[dict],
    memory:   str,
    intent:   str,
    domain:   str,
) -> list[dict]:
    """
    Builds the messages list for the LLM.

    The LLM's job: use retrieved prompt engineering examples to generate
    an optimized, structured prompt from the user's vague request.

    Structure:
      [System]  intent-specific prompt-engineering instructions
      [User]    memory + retrieved examples + user's raw request
    """
    system = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["question"])

    # Format retrieved knowledge-base examples
    example_parts = []
    for i, chunk in enumerate(chunks):
        source    = chunk.get("source", "unknown")
        doc_title = chunk.get("doc_title", "")[:60]
        text      = chunk["text"]
        example_parts.append(
            f"[Example {i+1}] Source: {source} | {doc_title}\n{text}"
        )
    examples_block = "\n\n".join(example_parts)

    # Build user message
    user_parts = []

    if memory:
        user_parts.append(f"PREVIOUS CONVERSATION (for context continuity):\n{memory}")

    user_parts.append(
        f"RETRIEVED PROMPT ENGINEERING EXAMPLES FROM KNOWLEDGE BASE:\n{examples_block}"
    )
    user_parts.append(
        f"DETECTED INTENT: {intent.upper()}\n"
        f"DETECTED DOMAIN: {domain}"
    )
    user_parts.append(
        f"USER'S RAW REQUEST:\n{query}"
    )
    user_parts.append(
        "Using the retrieved examples above as reference, generate a fully structured, "
        "optimized prompt for the user's request. "
        "The output should be a ready-to-use prompt the user can paste into an AI model."
    )

    user_message = "\n\n" + "\n\n---\n\n".join(user_parts)

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_message},
    ]


# ── Main pipeline class ───────────────────────────────────────────────────────
class PromptBridgePipeline:
    def __init__(self, session_id: str = "default"):
        print("\n" + "="*50)
        print("  PromptBridge — Prompt Engineering Pipeline")
        print("="*50)

        self.cache     = RAGCache()                       # shared across all layers
        self.retriever = DualRetriever(cache=self.cache)  # L1 embedding cache shared
        self.reranker  = Reranker()
        self.memory    = Memory(session_id=session_id)
        self.groq      = Groq(api_key=GROQ_API_KEY)

        print(f"  Session: {session_id}")
        print(f"  Past turns in memory: {self.memory.turn_count}")
        print(f"  Cache backend: {self.cache.stats['backend']}")
        print("="*50 + "\n")

    def query(self, user_input: str, verbose: bool = True) -> dict:
        """
        Full PromptBridge pipeline for one user request.

        Input:  user's vague/incomplete requirement (str)
        Output: dict containing the OPTIMIZED PROMPT + metadata

        Cache layers:
          L3 (answer)    → full optimized prompt, skips everything on hit
          L2 (retrieval) → reuses chunks, skips dense+BM25+rerank on hit
          L1 (embedding) → handled inside DualRetriever.dense_search
        """
        start = time.time()

        if verbose:
            print(f"\nUser Request: {user_input}")
            print("-" * 40)

        # ── L3: Answer cache hit → return immediately ─────────────────────
        cached = self.cache.get_answer(user_input)
        if cached:
            if verbose:
                print("  [Cache] L3 HIT — returning cached optimized prompt")
            cached["latency_s"] = round(time.time() - start, 3)
            cached["cached"]    = True
            return cached

        # Step 1: Classify intent + detect domain
        intent = classify_intent(user_input)
        domain = detect_domain(user_input)
        if verbose:
            print(f"  Intent: {intent} | Domain: {domain}")

        # ── L2: Retrieval cache hit → skip retrieval + rerank ─────────────
        cached_chunks = self.cache.get_chunks(user_input)
        if cached_chunks:
            if verbose:
                print("  [Cache] L2 HIT — reusing cached chunks")
            chunks_5 = cached_chunks
        else:
            # Step 2: Dual retrieval → 20 chunks (prompt engineering examples)
            chunks_20 = self.retriever.retrieve(user_input)

            # Step 3: Rerank → top 5 most relevant examples
            chunks_5 = self.reranker.rerank(user_input, chunks_20, top_n=5)

            # Populate L2 cache
            self.cache.set_chunks(user_input, chunks_5)

        # Step 4: Load memory context (last 3 turns)
        memory_context = self.memory.get_context(max_turns=3)

        # Step 5: Assemble prompt-generation request
        messages = assemble_prompt(
            query=user_input,
            chunks=chunks_5,
            memory=memory_context,
            intent=intent,
            domain=domain,
        )

        # Step 6: LLM generates the optimized prompt
        if verbose:
            print("  Generating optimized prompt...")

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.4,    # slight creativity for prompt variation
            max_tokens=1200,
        )
        optimized_prompt = response.choices[0].message.content.strip()

        # Step 7: Save to memory (so follow-up requests have context)
        self.memory.add("user",      user_input)
        self.memory.add("assistant", optimized_prompt)

        elapsed = round(time.time() - start, 2)

        result = {
            "query":            user_input,
            "intent":           intent,
            "domain":           domain,
            "optimized_prompt": optimized_prompt,
            "chunks":           chunks_5,
            "latency_s":        elapsed,
            "sources":          list({c["source"] for c in chunks_5}),
            "cached":           False,
        }

        # Populate L3 answer cache
        self.cache.set_answer(user_input, result)

        if verbose:
            print(f"\n{'='*50}")
            print("OPTIMIZED PROMPT:")
            print('='*50)
            print(optimized_prompt)
            print(f"{'='*50}")
            print(f"Time: {elapsed}s | Intent: {intent} | Domain: {domain}")

        return result

    def chat(self):
        """Interactive terminal mode for testing the pipeline."""
        print("\nPromptBridge — Interactive Mode")
        print("Enter your vague request and get an optimized prompt back.")
        print("Commands: 'quit' | 'clear' (reset memory) | 'history'\n")

        while True:
            try:
                user_input = input("Your request: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                self.memory.clear_session()
                print("Memory cleared.\n")
                continue
            if user_input.lower() == "history":
                history = self.memory.get_history()
                if not history:
                    print("No history yet.\n")
                else:
                    for msg in history:
                        role = "You" if msg["role"] == "user" else "PromptBridge"
                        print(f"{role}: {msg['content'][:300]}")
                    print()
                continue

            result = self.query(user_input, verbose=True)
            print(f"\nSources: {result['sources']} | Domain: {result['domain']} | Time: {result['latency_s']}s\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PromptBridge — Prompt Engineering Pipeline")
    parser.add_argument("--session", default="default", help="Session ID for memory")
    parser.add_argument("--query",   default="",        help="Single request (non-interactive)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    pipeline = PromptBridgePipeline(session_id=args.session)

    if args.query:
        pipeline.query(args.query, verbose=True)
    else:
        pipeline.chat()