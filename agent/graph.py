"""Phase 8 — assemble the LangGraph agent.

Step 3: router → (retrieve | call_tool | clarify) → generate. The router picks
the path; retrieval and the three MCP tools each feed the generator; clarify
ends the turn with a follow-up question. `build_agent(tools)` needs the warm MCP
tools (from `agent.mcp_client.warm_tools`) for the call_tool node. MemorySaver
and the live-fetch escalation come in later steps.

    START → precheck ─(answer cached)────────────────────────────────────────┐
              │ (miss)                                                        │
              ↓                                                               │
            route ─┬─ "retrieve"  → retrieve → guard_input ┬─(good)→ generate ┤
                   │                                        └─(weak)→ escalate ┤
                   ├─ "call_tool" → call_tool → sanitize_tool ─────→ generate ─┤
                   └─ "clarify"   → clarify ───────────────────────────────────┤
                                                                               ↓
                       generate → guard_output ─(ok / repaired)→ finalize → END
                                        └─(unsafe)→ refuse ↗

`precheck` (Phase 10) scrubs PII and checks the full-answer cache; a hit skips the
router LLM call, retrieval, generation and all guards, because only clean
already-guarded answers are stored. The embedding cache and the two LLM-guard
verdict caches sit inside their respective functions (see agent/cache.py).

Guardrails are organised as exactly TWO stages (see `agent/guards/__init__.py`):

  `guard_input`  — BEFORE the LLM. All deterministic, microseconds. Scrubs PII
      from the query, drops chunks carrying prompt-injection text, and resolves
      factual conflicts between sources by independent-document majority.

  `guard_output` — AFTER the LLM. Ordered cheap → expensive so most answers never
      reach the paid tier: deterministic checks (gibberish, duplicate sentences,
      citation integrity, content safety, symbol allowlist, URL/host safety),
      then embedding relevance, then the two LLM checks (groundedness, and a
      context-blind policy check that catches answers which obey a poisoned
      chunk — those are perfectly *grounded*, so only a context-blind view sees
      that the injected token serves nobody but the attacker).

`guard_output` REPAIRS where it can (dropping duplicate or unsupported sentences)
and refuses only when the answer is unsafe or has nothing substantive left —
refusing a mostly-correct answer over one bad sentence was itself a measured
failure mode.

The escalate branch is the "index-first, live-fallback" pattern: when the top
rerank score is below RETRIEVAL_CONFIDENCE_MIN, we fetch live from GitHub
instead of answering from irrelevant context. Every path funnels through
`finalize` so exactly one final message lands in the durable history.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.router import route_selector, router_node
from agent.state import AgentState


def build_agent(tools: dict | None = None, checkpointer=None):
    """Compile the agent graph. `tools` is the name->BaseTool dict from the warm
    MCP session; required for the call_tool path. `checkpointer` (MemorySaver)
    is wired in Step 5."""
    # Turn on LangSmith tracing if a key is present (no-ops offline). Called here
    # so every entry point that builds the agent gets tracing without extra setup.
    from agent.observability import configure_tracing
    configure_tracing()

    tools = tools or {}
    graph = StateGraph(AgentState)

    graph.add_node("precheck", nodes.precheck_node)             # PII scrub + answer cache
    graph.add_node("route", router_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("guard_input", nodes.guard_input_node)        # INPUT STAGE
    graph.add_node("escalate", nodes.escalate_node)
    graph.add_node("call_tool", nodes.make_call_tool_node(tools))
    graph.add_node("sanitize_tool", nodes.sanitize_tool_node)    # INPUT STAGE (live doc)
    graph.add_node("clarify", nodes.clarify_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("guard_output", nodes.guard_output_node)      # OUTPUT STAGE
    graph.add_node("refuse", nodes.refuse_node)
    graph.add_node("finalize", nodes.finalize_node)

    # Answer cache runs before everything: a hit skips the router LLM call,
    # retrieval, generation and all guards, because only clean already-guarded
    # answers were stored.
    graph.add_edge(START, "precheck")
    graph.add_conditional_edges(
        "precheck",
        nodes.precheck_gate,
        {"cached": "finalize", "route": "route"},
    )
    graph.add_conditional_edges(
        "route",
        route_selector,
        {"retrieve": "retrieve", "call_tool": "call_tool", "clarify": "clarify"},
    )
    # INPUT STAGE runs BEFORE the confidence gate — it recomputes confidence from
    # the surviving chunks, so a dropped top hit is reflected in the gate's view.
    graph.add_edge("retrieve", "guard_input")
    graph.add_conditional_edges(
        "guard_input",
        nodes.retrieval_gate,
        {"generate": "generate", "escalate": "escalate"},
    )
    graph.add_edge("escalate", "call_tool")
    graph.add_edge("call_tool", "sanitize_tool")
    graph.add_edge("sanitize_tool", "generate")

    # OUTPUT STAGE: repair what can be repaired, refuse only what is unsafe or
    # unsupported. Both outcomes funnel to finalize so exactly one final message
    # reaches the durable history.
    graph.add_edge("generate", "guard_output")
    graph.add_conditional_edges(
        "guard_output",
        nodes.output_gate,
        {"ok": "finalize", "refuse": "refuse"},
    )
    graph.add_edge("refuse", "finalize")
    # The clarify path used to run straight to finalize, so an LLM response built
    # from raw user input reached the user with NO output guard at all. Every
    # LLM-generated path now goes through guard_output.
    graph.add_edge("clarify", "guard_output")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    import asyncio

    from langchain_core.messages import HumanMessage

    from agent.mcp_client import warm_tools

    async def _ask(agent, q: str) -> None:
        out = await agent.ainvoke({"query": q, "messages": [HumanMessage(content=q)]})
        print("=" * 70)
        print("Q:", q)
        print("route:", out.get("route"), "| tool:", out.get("tool_name"))
        print("A:", out["answer"][:500])
        print()

    async def _smoke() -> None:
        async with warm_tools() as tools:
            agent = build_agent(tools)
            # one query per path
            await _ask(agent, "How do I add a tool to an agent in LangChain?")   # retrieve
            await _ask(agent, "What's the latest version of langgraph?")          # get_package_version
            await _ask(agent, "How current are your docs and what do they cover?")# get_corpus_status
            await _ask(agent, "Fetch the current LangGraph Studio setup guide.")  # fetch_live_doc
            await _ask(agent, "help")                                             # clarify

    asyncio.run(_smoke())
