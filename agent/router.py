"""Phase 8 — the router node: pick the path for a query.

Pure-LLM routing (llama3.1:8b) into one of five paths, with argument extraction
in the SAME call (which package? which topic? what to search live for?). A
rule-based fallback catches the cases where the model returns unusable JSON, so
the graph never stalls on a bad route. If the eval (Step 6) shows the LLM router
is weak, the fallback rules can be promoted to a hybrid first-pass.

Routing paths:
  retrieve            answerable from the indexed LangChain/LangGraph docs
  get_package_version current PyPI version of a package (corpus is frozen)
  get_corpus_status   meta about OUR knowledge: freshness, pinned commit, coverage
  fetch_live_doc      current/new/out-of-scope doc CONTENT the frozen corpus lacks
  clarify             too vague to answer without a follow-up
"""

from __future__ import annotations

import asyncio
import json
import re

import ollama

from agent.state import AgentState
from rag.naive import LLM_MODEL, OLLAMA_HOST

ALLOWED_ROUTES = (
    "retrieve",
    "get_package_version",
    "get_corpus_status",
    "fetch_live_doc",
    "clarify",
)

ROUTER_SYSTEM = """You are the ROUTER for a LangChain/LangGraph documentation assistant.
Decide how to handle the user's message. Reply with ONE JSON object, nothing else.

Schema:
{
  "routes": ["<one or more of: retrieve, get_package_version, get_corpus_status, fetch_live_doc, clarify>"],
  "package": "<PyPI package name, or null>",
  "topic": "<short topic to check coverage for, or null>",
  "live_query": "<what to search the live docs for, or null>"
}

How to choose:
- "retrieve": answer a real HOW-TO or CONCEPTUAL question from existing LangChain/LangGraph docs
  ("how do I X?", "what is Y?"). This is the DEFAULT for genuine doc questions.
- "get_package_version": the user wants the CURRENT/LATEST released version of a PACKAGE, or asks
  whether a package is "up to date" / its "newest release". This is a PyPI version question — set
  "package". Do NOT also add get_corpus_status: a package being "up to date" is about the package,
  not about our docs.
- "get_corpus_status": a META question about OUR OWN knowledge base — how fresh/current OUR docs are,
  what commit they're pinned to, or WHETHER WE HAVE / COVER docs on a topic. Asking IF we have
  documentation on something ("do you have docs on X?", "do you cover Y?") is get_corpus_status,
  NOT retrieve. Set "topic" for a coverage check.
- "fetch_live_doc": the user wants to SEE / FETCH / SHOW the CURRENT CONTENT of a documentation page —
  especially one that may be new, recently changed, or OUTSIDE our indexed scope (integration/provider
  pages, "show me the page on X", "get the current docs for Y", "the latest guide", "a feature added
  recently"). If they want the actual page CONTENT and it's current/live/new/out-of-scope, use this.
  Set "live_query".
- "clarify": the message is too vague to act on — NO specific LangChain/LangGraph subject to work with
  ("help", "it's broken", "can you help me fix this?", "what should I use?"). Even if phrased as a
  question, if there is no concrete topic, choose clarify.

Key distinctions:
- "Do you HAVE docs on X?"        -> get_corpus_status (meta: do we cover it)
- "How do I do X?"                -> retrieve (answer from docs we have)
- "Show me / fetch the page on X" -> fetch_live_doc (live page CONTENT)
- Version of a PACKAGE            -> get_package_version ; freshness of OUR DOCS -> get_corpus_status
- Integration / provider pages    -> fetch_live_doc (they're outside our indexed scope)

If the message has two DISTINCT intents (e.g. "what's the latest version AND how do I install it"),
list both routes. Otherwise return exactly one.

Examples:
User: "How do I add a tool to an agent?"             -> {"routes":["retrieve"],"package":null,"topic":null,"live_query":null}
User: "Is my langchain install the newest one?"      -> {"routes":["get_package_version"],"package":"langchain","topic":null,"live_query":null}
User: "How current are your docs?"                   -> {"routes":["get_corpus_status"],"package":null,"topic":null,"live_query":null}
User: "Do you have anything on middleware?"          -> {"routes":["get_corpus_status"],"package":null,"topic":"middleware","live_query":null}
User: "Show me the current page on persistence"      -> {"routes":["fetch_live_doc"],"package":null,"topic":null,"live_query":"persistence"}
User: "Get me the docs for the Pinecone integration" -> {"routes":["fetch_live_doc"],"package":null,"topic":null,"live_query":"Pinecone integration"}
User: "Can you help me fix this?"                    -> {"routes":["clarify"],"package":null,"topic":null,"live_query":null}
"""


def _run_router_llm(user_query: str, history: list[dict] | None = None) -> str:
    client = ollama.Client(host=OLLAMA_HOST)
    messages = [{"role": "system", "content": ROUTER_SYSTEM}]
    messages.extend(history or [])  # prior turns so follow-ups route correctly
    messages.append({"role": "user", "content": user_query})
    resp = client.chat(
        model=LLM_MODEL,
        messages=messages,
        format="json",  # force valid JSON out of the model
        options={"num_predict": 200, "temperature": 0.0},
    )
    return (resp.get("message", {}).get("content") or "").strip()


def _parse_router_json(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    routes = data.get("routes")
    if isinstance(routes, str):
        routes = [routes]
    if not isinstance(routes, list):
        return None
    routes = [r for r in routes if r in ALLOWED_ROUTES]
    if not routes:
        return None
    return {
        "routes": routes,
        "package": data.get("package"),
        "topic": data.get("topic"),
        "live_query": data.get("live_query"),
    }


# --- Rule-based fallback (only used when the LLM JSON is unusable) -----------

_VERSION_RE = re.compile(r"\b(version|release|latest|newest|up[- ]?to[- ]?date|pypi)\b", re.I)
_STATUS_RE = re.compile(r"\b(your docs|how current|how fresh|pinned|commit|do you (have|cover)|coverage)\b", re.I)
_LIVE_RE = re.compile(r"\b(fetch|pull|live|latest .* (guide|doc|page)|from github|newest doc)\b", re.I)
_PKG_RE = re.compile(r"\b(langchain[-\w]*|langgraph[-\w]*|langsmith)\b", re.I)


def _fallback_route(query: str) -> dict:
    q = query.strip()
    if len(q) < 12 and not q.endswith("?"):
        return {"routes": ["clarify"], "package": None, "topic": None, "live_query": None}
    if _VERSION_RE.search(q):
        pkg = _PKG_RE.search(q)
        return {"routes": ["get_package_version"], "package": pkg.group(0) if pkg else None,
                "topic": None, "live_query": None}
    if _STATUS_RE.search(q):
        return {"routes": ["get_corpus_status"], "package": None, "topic": None, "live_query": None}
    if _LIVE_RE.search(q):
        return {"routes": ["fetch_live_doc"], "package": None, "topic": None, "live_query": q}
    return {"routes": ["retrieve"], "package": None, "topic": None, "live_query": None}


async def router_node(state: AgentState) -> dict:
    """Classify the current query (+ extract tool args) into a route list."""
    from agent.nodes import history_messages

    # The query is already PII-scrubbed by precheck_node (which runs first on
    # every path); read it from state rather than scrubbing again.
    query = state["query"]
    history = history_messages(state.get("messages"))
    raw = await asyncio.to_thread(_run_router_llm, query, history)
    decision = _parse_router_json(raw) or _fallback_route(query)
    return {
        "route": decision["routes"],
        "route_args": {
            "package": decision.get("package"),
            "topic": decision.get("topic"),
            "live_query": decision.get("live_query"),
        },
        # Reset per-turn scratch so a prior turn's tool_result/retrieved can't
        # leak into this turn's generation (checkpointed state persists these).
        "tool_result": None,
        "tool_name": None,
        "retrieved": [],
        "escalated": False,
    }


def route_selector(state: AgentState) -> str:
    """Conditional-edge function: map the primary route to the next node."""
    routes = state.get("route") or ["retrieve"]
    primary = routes[0]
    if primary == "retrieve":
        return "retrieve"
    if primary == "clarify":
        return "clarify"
    return "call_tool"  # any of the three MCP tools
