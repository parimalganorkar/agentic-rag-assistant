"""Phase 8 — the LangGraph agent's shared state.

`messages` is the DURABLE conversation history (persisted by MemorySaver across
turns, keyed by thread_id) — this is what gives the agent multi-turn memory.
Everything else is per-turn scratch that each turn overwrites: the router's
decision, this turn's retrieved chunks / tool result, and the final answer.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from retrieval.dense import RetrievedChunk


class AgentState(TypedDict, total=False):
    # Durable across turns (add_messages appends each turn's messages).
    messages: Annotated[list, add_messages]

    # Per-turn scratch.
    query: str                       # the current user turn, extracted for nodes
    answer_cache_key: str            # full-answer cache key (query + history)
    cache_hit: bool                  # precheck served a cached answer this turn
    route: list[str]                 # router decision(s): retrieve / <tool> / clarify
    route_args: dict                 # router-extracted tool args (package / topic / live_query)
    retrieved: list[RetrievedChunk]  # this turn's retrieval hits
    retrieval_confidence: float      # top rerank score — feeds the live-fetch gate
    tool_result: dict                # this turn's MCP tool result (parsed)
    tool_name: str                   # which tool was called (for the generator/citation)
    answer: str                      # final answer text
    clarification: str               # clarifying question (clarify path)
    escalated: bool                  # retrieval was weak -> escalated to fetch_live_doc

    # --- Guardrails (Phase 9) ---------------------------------------------
    # INPUT stage (agent.guards.check_input)
    pii_redacted: list[str]          # credential/PII kinds scrubbed from the query
    dropped_chunks: list[dict]       # chunks removed by the injection filter
    injection_flags: list[str]       # injection patterns seen this turn
    source_conflicts: list[dict]     # facts where retrieved sources disagreed
    outlier_chunks: list[dict]       # chunks dropped for being outvoted on a fact

    # OUTPUT stage (agent.guards.check_output)
    guard_action: str                # pass | repair | refuse
    guard_reason: str                # why the answer was refused
    guard_repairs: list[str]         # what was fixed rather than refused
    guard_degraded: bool             # an LLM guard failed open (verifier unavailable)
    relevance: float                 # question<->answer cosine similarity
    grounded: bool                   # did the answer pass the fact check
    unsupported: list[str]           # claims the fact checker couldn't support
    verifier_backend: str            # which verifier ran (ollama / gemini / fallback)
    policy_violation: bool           # answer contains content not serving the question
    policy_reason: str               # why the output policy check failed
    policy_artifacts: list[str]      # the offending text (injected tokens, unsafe patterns, ...)
    policy_backend: str              # regex / ollama / gemini
