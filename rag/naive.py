"""Phase 4 — Naive RAG baseline.

Retrieve → stuff into prompt → generate. No LangGraph, no MCP, no hybrid
retrieval, no reranking, no guardrails. The dumbest thing that works, so
Phase 6's RAGAS scores have a real baseline to compare against.

Pipeline:
  1. dense_search(query, k=5)         →   top-K chunks by cosine similarity
  2. build_prompt(query, hits)        →   context stuffed in with [S1]...[Sn]
                                          citation markers per chunk
  3. call Ollama (llama3.1:8b, local) →   grounded answer with inline citations

Local, free, no API key. Requires Ollama running (`ollama serve`) and
`ollama pull llama3.1:8b` done at least once. Default host is
http://localhost:11434 (override with OLLAMA_HOST env var).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import ollama

from retrieval.dense import RetrievedChunk, dense_search


_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------- CONFIG ------------------------------------------------------

# Local model served by Ollama. llama3.1:8b runs comfortably on ~6GB VRAM
# (RTX 4050 tier). Swap to `qwen2.5:7b` or `mistral:7b` for alternatives.
LLM_MODEL = "llama3.1:8b"

# Ollama host — read at call time so tests can point at a different port.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# How many chunks we retrieve and stuff into the prompt.
DEFAULT_K = 5

# Max tokens the model can emit. Naive baseline answers rarely need more.
MAX_OUTPUT_TOKENS = 1024


# ---------- Prompt template --------------------------------------------
# Kept deliberately blunt: use the context, cite it, and refuse when the
# context doesn't cover the question. Later phases (guardrails, hybrid
# retrieval) will make this less fragile — but the naive baseline should
# rely on the prompt alone.

SYSTEM_PROMPT = """You are a documentation assistant for LangChain and LangGraph.

You will be given a QUESTION and a set of CONTEXT snippets pulled from the
official docs. Answer the question using ONLY the information in the CONTEXT.

Rules:
- If the CONTEXT does not contain enough information to answer, say exactly:
  "I don't have enough information in the retrieved docs to answer that."
  Do not guess or use outside knowledge.
- Cite the source file for every non-trivial claim, in this format: [S1], [S2].
  The bracketed number matches the "[S<n>]" label on each CONTEXT snippet.
- Prefer short, direct answers over long ones. Show code when the context has code.
- If two snippets say different things, prefer the value stated by MORE snippets,
  and say explicitly that the sources disagree. Never present a fact asserted by a
  single snippet as settled when other snippets contradict it. If you cannot tell
  which is correct, say so rather than picking one.
"""


@dataclass
class RagAnswer:
    """Result bundle so callers can render / evaluate / cache the pieces."""
    query: str
    answer: str
    hits: list[RetrievedChunk]
    prompt_user: str
    model: str


def _format_context(hits: Sequence[RetrievedChunk]) -> str:
    """Turn the retrieved chunks into a numbered CONTEXT block."""
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        header = f"[S{i}] source={h.source_file}   section={h.section!r}"
        blocks.append(f"{header}\n{h.text.strip()}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, hits: Sequence[RetrievedChunk]) -> str:
    """Build the user-message content — CONTEXT block + QUESTION."""
    return (
        "CONTEXT:\n\n"
        f"{_format_context(hits)}\n\n"
        "---\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER (cite [S1], [S2], ... where appropriate):"
    )


def _get_client() -> ollama.Client:
    """Return an Ollama client pointed at OLLAMA_HOST. Cheap to instantiate,
    but callers often reuse across many questions in an eval loop."""
    return ollama.Client(host=OLLAMA_HOST)


def answer(query: str, k: int = DEFAULT_K) -> RagAnswer:
    """End-to-end: retrieve → build prompt → generate → return answer bundle."""
    hits = dense_search(query, k=k)
    if not hits:
        return RagAnswer(
            query=query,
            answer="I don't have enough information in the retrieved docs to answer that.",
            hits=[],
            prompt_user="",
            model=LLM_MODEL,
        )

    user_prompt = build_prompt(query, hits)
    client = _get_client()
    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        options={"num_predict": MAX_OUTPUT_TOKENS, "temperature": 0.0},
    )
    generated = (response.get("message", {}).get("content") or "").strip()

    return RagAnswer(
        query=query,
        answer=generated,
        hits=hits,
        prompt_user=user_prompt,
        model=LLM_MODEL,
    )


if __name__ == "__main__":
    # Smoke test — asks the Phase 4 DoD question.
    result = answer("how do I use a memory checkpoint in LangGraph?", k=5)
    print("=" * 70)
    print(f"QUERY: {result.query}")
    print(f"MODEL: {result.model}")
    print("=" * 70)
    print()
    print("Retrieved sources:")
    for i, h in enumerate(result.hits, 1):
        print(f"  [S{i}] sim={h.similarity:.3f}  {h.source_file}  ({h.section})")
    print()
    print("ANSWER:")
    print(result.answer)
