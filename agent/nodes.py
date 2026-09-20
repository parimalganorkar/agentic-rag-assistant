"""Phase 8 — LangGraph node implementations.

Nodes are async so the async MCP tools (Step 3) drop in without reshaping the
graph; the sync pieces (hybrid retrieval, the Ollama call) run in a worker
thread via `asyncio.to_thread` so they don't block the event loop.

This module holds the retrieve + generate nodes (Step 2). The router, tool
dispatch, and clarify nodes are added in later steps.
"""

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.messages import AIMessage, HumanMessage

from agent import guardrails, guards, llm
from agent.mcp_client import parse_tool_result
from agent.state import AgentState
from rag.naive import (
    MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT,
    _format_context,
    build_prompt,
)
from retrieval.pipeline import retrieve as hybrid_retrieve

RETRIEVE_K = 5
# Cross-encoder rerank scores are logits, not [0,1]. Measured on the 100-question
# eval set (2026-09-20, top-1 logit after hybrid+rerank):
#   legit questions (91):   min 0.62  p10 4.06  median 6.26  max 9.31
#   adversarial (9):        min -9.57 median -3.39           max 0.82
# The gate used to sit at 0.0, and "why is sky blue" scored +1.15 because the
# docs' own streaming example uses that exact prompt — one coincidental hit
# above a sea of -9s, and the generator answered a physics question from it.
# 2.0 sits in the empty band between the adversarial max and the legit p10:
# it catches 9/9 adversarial plus that case, and escalates only the 2 vaguest
# legit questions (0.62, 0.94) to a live fetch rather than refusing them.
RETRIEVAL_CONFIDENCE_MIN = 2.0

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the retrieved docs to answer that."
)
# Router-level refusal for questions that are not about LangChain/LangGraph at
# all. Ends with the NO_CONTEXT_ANSWER phrase on purpose: both eval harnesses
# recognise a refusal by "enough information in the retrieved docs", so this
# stays countable without touching eval code.
OUT_OF_SCOPE_ANSWER = (
    "That's outside what I cover. I answer questions about LangChain and LangGraph "
    "from their documentation, so I don't have enough information in the retrieved "
    "docs to answer that."
)

_PKG_RE = re.compile(r"\b(langchain[-\w]*|langgraph[-\w]*|langsmith)\b", re.I)

# For phrasing answers off a structured tool result (version / corpus status).
TOOL_SYSTEM = """You are a LangChain/LangGraph documentation assistant. The user asked a
question and a TOOL returned data as JSON. Answer the user's question using ONLY that tool
data — be concise and factual. If the data has an "error" field or "matched": false, say you
couldn't find that information rather than guessing. Never invent version numbers, dates, or
documentation content that isn't in the tool data."""

# For a live-fetched doc page (real prose content) — cite the source URL.
LIVE_DOC_SYSTEM = """You are a LangChain/LangGraph documentation assistant. Answer the user's
question using ONLY the DOC CONTENT below, which was fetched live from the current docs. Be
concise, show code when the content has it, and cite the source URL at the end. If the content
doesn't actually answer the question, say so."""


def _answer_cache_key(query: str, history: list[dict]) -> str:
    """Key a full-answer cache entry on the query AND the prior conversation.

    History MUST be in the key: a follow-up like "how do I install it?" produces a
    different answer depending on what came before, so caching on the query alone
    would serve the wrong answer. First-turn queries have empty history and share
    a key across threads, which is exactly what we want (same question → same
    answer). Retrieval-corpus state is NOT in the key, which is safe here because
    the cache is in-process and every eval run is a fresh process with an empty
    cache — see agent/cache.py.
    """
    from agent.cache import key_of
    sig = "|".join(f"{m['role']}:{m['content']}" for m in history)
    return key_of("answer", query, sig)


async def precheck_node(state: AgentState) -> dict:
    """Runs FIRST, before routing. Scrubs PII from the query (so every downstream
    path — retrieve, tool, clarify — sees the redacted text), then checks the
    full-answer cache. A hit short-circuits the whole graph to finalize, skipping
    the router LLM call, retrieval, generation and all guards — which is safe
    because only CLEAN, already-guarded answers are ever stored (see finalize)."""
    from agent.cache import ANSWER_CACHE, MISSING

    clean_query, pii = guards.scrub_pii(state["query"])
    history = history_messages(state.get("messages"))
    ckey = _answer_cache_key(clean_query, history)

    out: dict = {"query": clean_query, "pii_redacted": pii, "answer_cache_key": ckey}
    cached = ANSWER_CACHE.get(ckey)
    if cached is not MISSING:
        out["answer"] = cached
        out["cache_hit"] = True
    return out


def precheck_gate(state: AgentState) -> str:
    return "cached" if state.get("cache_hit") else "route"


async def retrieve_node(state: AgentState) -> dict:
    """Hybrid retrieval + rerank for the current query. Records the top rerank
    score as `retrieval_confidence` (the live-fetch escalation gate reads it)."""
    query = state["query"]
    hits = await asyncio.to_thread(hybrid_retrieve, query, RETRIEVE_K)
    confidence = hits[0].similarity if hits else 0.0
    return {"retrieved": hits, "retrieval_confidence": confidence}


HISTORY_TURNS = 10  # how many prior messages to feed the model for follow-ups


def history_messages(messages: list | None, limit: int = HISTORY_TURNS) -> list[dict]:
    """Prior turns (excluding the current human message) as chat dicts, so the
    model can resolve follow-ups like 'how do I install it?'.

    Scrubbed on the way out. Redacting only the current query was theatre: the
    RAW message is what gets stored in the durable history, so a key pasted on
    turn 1 was replayed verbatim into the prompt on every later turn.
    """
    prior = (messages or [])[:-1]
    out: list[dict] = []
    for m in prior[-limit:]:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        content = getattr(m, "content", "")
        if content:
            clean, _ = guards.scrub_pii(content) if isinstance(content, str) else (content, [])
            out.append({"role": role, "content": clean})
    return out


def _run_llm(system: str, user: str, history: list[dict] | None = None) -> str:
    """One generation on whichever provider `LLM_PROVIDER` names (agent/llm.py).
    Used by generate_node and clarify_node."""
    return llm.chat(system, user, history, max_tokens=MAX_OUTPUT_TOKENS)


async def guard_input_node(state: AgentState) -> dict:
    """INPUT STAGE — everything that must happen before the LLM sees anything.

    One node, one call: `guards.check_input` scrubs PII from the query, drops
    chunks carrying prompt-injection text, and resolves factual conflicts between
    sources by independent-document majority. Confidence is RECOMPUTED from the
    survivors, so if the poisoned chunk was the top hit the escalation gate sees
    the new top score rather than the one it displaced.
    """
    hits = state.get("retrieved") or []
    verdict = await asyncio.to_thread(guards.check_input, state["query"], hits)
    confidence = verdict.chunks[0].similarity if verdict.chunks else 0.0
    out: dict = {
        "query": verdict.query,
        "retrieved": verdict.chunks,
        "retrieval_confidence": confidence,
        "dropped_chunks": verdict.dropped_chunks,
        "injection_flags": verdict.injection_flags,
    }
    if verdict.source_conflicts:
        out["source_conflicts"] = verdict.source_conflicts
    if verdict.pii_found:
        out["pii_redacted"] = verdict.pii_found
    # The router's out_of_scope call is a prior. If retrieval is confident the
    # docs cover this, the evidence wins: convert to a normal retrieve turn and
    # record that the router was overruled (telemetry / trace).
    if (state.get("route") or [""])[0] == "out_of_scope" and confidence >= RETRIEVAL_CONFIDENCE_MIN:
        out["route"] = ["retrieve"]
        out["scope_overruled"] = True
    return out


async def sanitize_tool_node(state: AgentState) -> dict:
    """INPUT STAGE (tool path) — scan EVERY string a tool returns.

    This used to scan only `fetch_live_doc`, on the reasoning that the other two
    tools return "our own" structured data. That was wrong: `get_package_version`
    passes through the PyPI `summary` field, which is written by whoever
    published the package. Anyone can upload a package whose description carries
    injection text, and it went into the prompt unscanned. Tool output is
    third-party data whatever its shape, so all of it is scanned now.
    """
    result = state.get("tool_result")
    if not isinstance(result, dict):
        return {}

    cleaned: dict = {}
    patterns: list[str] = []
    for key, value in result.items():
        if isinstance(value, str) and value:
            safe, found = guardrails.sanitize_text(value)
            cleaned[key] = safe
            patterns.extend(found)
        else:
            cleaned[key] = value

    if not patterns:
        return {}

    cleaned["injection_filtered"] = True
    return {
        "tool_result": cleaned,
        "injection_flags": sorted(set(patterns)),
        "dropped_chunks": [{"source_file": result.get("source_file", state.get("tool_name", "tool")),
                            "chunk_id": "tool-result", "patterns": sorted(set(patterns))}],
    }


def retrieval_gate(state: AgentState) -> str:
    """After retrieval, decide whether the hits are good enough to answer from.

    Weak retrieval means two different things depending on what the router
    thought: for an in-domain question it means the frozen corpus lacks the
    topic -> escalate to a live GitHub fetch; for a question the router already
    called out_of_scope it is confirmation -> refuse, without paying for a live
    fetch that cannot help. (A confident out_of_scope turn never reaches this
    branch: guard_input_node already converted it to a retrieve turn.)"""
    if (state.get("retrieval_confidence") or 0.0) >= RETRIEVAL_CONFIDENCE_MIN:
        return "generate"
    if (state.get("route") or [""])[0] == "out_of_scope":
        return "out_of_scope"
    return "escalate"


async def escalate_node(state: AgentState) -> dict:
    """Retrieval was weak — redirect this turn to a live-doc fetch. Sets the
    route/args so the shared call_tool node runs fetch_live_doc next."""
    return {
        "route": ["fetch_live_doc"],
        "route_args": {"live_query": state["query"]},
        "escalated": True,
    }


def make_call_tool_node(tools: dict):
    """Factory: a node that dispatches to the routed MCP tool. Tools come from
    the warm MCP session (`agent.mcp_client.warm_tools`)."""

    async def call_tool_node(state: AgentState) -> dict:
        primary = (state.get("route") or ["retrieve"])[0]
        args_in = state.get("route_args") or {}
        tool = tools.get(primary)
        if tool is None:
            return {"tool_name": primary,
                    "tool_result": {"error": f"tool '{primary}' is unavailable"}}

        if primary == "get_package_version":
            pkg = args_in.get("package")
            if not pkg:
                m = _PKG_RE.search(state["query"])
                pkg = m.group(0) if m else ""
            call_args = {"package": pkg}
        elif primary == "get_corpus_status":
            call_args = {"topic": args_in["topic"]} if args_in.get("topic") else {}
        elif primary == "fetch_live_doc":
            call_args = {"query": args_in.get("live_query") or state["query"]}
        else:
            call_args = {}

        raw = await tool.ainvoke(call_args)
        return {"tool_name": primary, "tool_result": parse_tool_result(raw)}

    return call_tool_node


async def out_of_scope_node(state: AgentState) -> dict:
    """The router called the question off-domain AND retrieval agreed (top rerank
    logit below RETRIEVAL_CONFIDENCE_MIN). Nothing is fetched or generated — the
    reply is a fixed string, so it does not go through guard_output (that stage
    exists for LLM text). Guard fields are set explicitly so the UI shows an
    honest "refused" badge and finalize never caches it.
    """
    return {
        "answer": OUT_OF_SCOPE_ANSWER,
        "guard_action": "refuse",
        "guard_reason": "question is outside the LangChain/LangGraph documentation",
        "guard_repairs": [],
        "grounded": True,          # nothing was claimed, so nothing is ungrounded
        "guard_degraded": False,
    }


CLARIFY_SYSTEM = """The user's message is too vague to answer. Reply with ONE short, friendly
clarifying question that would let you help them with LangChain/LangGraph. Ask only the question."""


async def clarify_node(state: AgentState) -> dict:
    """Ask a clarifying question instead of guessing. (Message is appended by
    finalize_node, like every other path.)"""
    answer = await asyncio.to_thread(_run_llm, CLARIFY_SYSTEM, state["query"])
    return {"clarification": answer, "answer": answer}


def _build_tool_generation(query: str, tool_name: str, tool_result: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for answering off a tool result."""
    if tool_name == "fetch_live_doc" and tool_result.get("matched"):
        context = (
            f"SOURCE URL: {tool_result.get('source_url')}\n"
            f"TITLE: {tool_result.get('title')}\n\n"
            f"{tool_result.get('content', '')}"
        )
        user = f"DOC CONTENT:\n\n{context}\n\n---\n\nQUESTION: {query}\n\nANSWER:"
        return LIVE_DOC_SYSTEM, user
    # structured tools (version / corpus status) or an unmatched live fetch
    user = (
        f"TOOL: {tool_name}\nTOOL DATA (JSON):\n{json.dumps(tool_result, indent=2)}\n\n"
        f"---\n\nQUESTION: {query}\n\nANSWER:"
    )
    return TOOL_SYSTEM, user


def _context_for_verification(state: AgentState) -> str | None:
    """Rebuild the exact context the answer was generated from, so the fact
    checker judges against the same evidence. Returns None when there's nothing
    meaningful to check (structured tool data)."""
    tool_result = state.get("tool_result")
    if tool_result is not None:
        # Only live-doc prose is worth fact-checking; a PyPI version or corpus
        # count is a structured passthrough that can't really be hallucinated.
        if state.get("tool_name") == "fetch_live_doc" and tool_result.get("matched"):
            return tool_result.get("content") or ""
        return None
    hits = state.get("retrieved") or []
    # No hits and no tool result = nothing external was in the prompt (the clarify
    # path, or a retrieval that came back empty). Returning "" here made the fact
    # checker judge an answer against an EMPTY context, where every claim is
    # trivially unsupported — it would refuse 100% of those turns.
    return _format_context(hits) if hits else None


async def guard_output_node(state: AgentState) -> dict:
    """OUTPUT STAGE — one call runs every post-generation check, cheap first.

    `guards.check_output` runs the deterministic tier (gibberish, duplicate
    sentences, citation integrity, content safety, symbol allowlist, URL/host
    safety), then the embedding relevance check, and only then the two LLM
    checks. It REPAIRS what can be repaired and refuses only when the answer is
    unsafe or has nothing substantive left.

    `context is None` means the answer came from our own structured tool data
    (a PyPI version, corpus counts) — no external text entered the prompt, so
    the LLM tier is skipped as latency for nothing.
    """
    context = _context_for_verification(state)
    n_sources = len(state.get("retrieved") or [])
    # A clarifying question is not an answer: it deliberately does NOT address the
    # query (it asks about it), so the answer-shaped checks must not judge it.
    kind = "clarification" if (state.get("route") or [""])[0] == "clarify" else "answer"
    injection_seen = bool(state.get("injection_flags"))
    verdict = await asyncio.to_thread(
        guards.check_output, state["query"], context, state.get("answer", ""),
        n_sources, kind, injection_seen,
    )
    return {
        "answer": verdict.answer,
        "guard_action": verdict.action,
        "guard_reason": verdict.refuse_reason,
        "guard_repairs": verdict.repairs,
        "grounded": verdict.grounded if verdict.grounded is not None else True,
        "unsupported": verdict.unsupported,
        "relevance": verdict.relevance,
        "policy_violation": verdict.policy_violation,
        "policy_reason": verdict.policy_reason,
        "policy_artifacts": verdict.safety_flags,
        "verifier_backend": verdict.backend,
        "guard_degraded": verdict.degraded,
    }


def output_gate(state: AgentState) -> str:
    """Ship the (possibly repaired) answer, or replace it with an honest refusal."""
    return "refuse" if state.get("guard_action") == "refuse" else "ok"


async def refuse_node(state: AgentState) -> dict:
    """Replace an ungrounded answer with the standard refusal. We do NOT try to
    rewrite it — the context didn't support the claims, so a retry would be
    guessing again. Saying "I don't know" is the correct output."""
    return {"answer": NO_CONTEXT_ANSWER}


async def finalize_node(state: AgentState) -> dict:
    """Append the FINAL answer to the durable history — exactly once.

    Every path (generated / refused / clarified) funnels through here so a
    rejected hallucination never lands in the conversation memory alongside the
    refusal that replaced it.

    Also stores the answer in the full-answer cache, but ONLY when it is clean:
    a normal answer that passed the output guards, was not degraded (every
    judge unavailable), is not a refusal, and is not a clarify turn. Caching a refusal
    or a degraded verdict would pin a transient failure and replay it for every
    later ask of the same question.
    """
    answer = state.get("answer", "")
    ckey = state.get("answer_cache_key")
    is_clean = (
        not state.get("cache_hit")                    # already cached; don't rewrite
        and ckey
        and answer
        and answer != NO_CONTEXT_ANSWER
        and state.get("guard_action") in ("pass", "repair")
        and not state.get("guard_degraded")
        and (state.get("route") or [""])[0] != "clarify"
    )
    if is_clean:
        from agent.cache import ANSWER_CACHE
        ANSWER_CACHE.set(ckey, answer)
    return {"messages": [AIMessage(content=answer)]}


async def generate_node(state: AgentState) -> dict:
    """Generate the answer from whichever context this turn produced — a tool
    result if a tool ran, otherwise the retrieved chunks."""
    query = state["query"]
    tool_result = state.get("tool_result")

    if tool_result is not None:
        # The sanitizer replaces injected text with a placeholder. Generating from
        # that placeholder produced a confident answer built on nothing; the
        # honest output is the refusal.
        if tool_result.get("injection_filtered"):
            return {"answer": NO_CONTEXT_ANSWER}
        system, user_prompt = _build_tool_generation(query, state.get("tool_name", ""), tool_result)
    else:
        hits = state.get("retrieved") or []
        if not hits:
            return {"answer": NO_CONTEXT_ANSWER}
        system, user_prompt = SYSTEM_PROMPT, build_prompt(query, hits)

    history = history_messages(state.get("messages"))
    answer = await asyncio.to_thread(_run_llm, system, user_prompt, history)
    return {"answer": answer}
