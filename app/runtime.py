"""Phase 11 — the bridge between Streamlit's execution model and the async agent.

THE PROBLEM. The agent needs ONE warm MCP session (an async context manager that
spawns a stdio subprocess) and ONE MemorySaver held open for the whole app's life
— see agent/cli.py. Streamlit re-runs the entire script top-to-bottom on every
user interaction, so the naive `async with warm_tools()` would spawn, use, and
tear down a fresh MCP subprocess on every keystroke: painfully slow and it would
drop the conversation memory each time.

THE FIX. Run a dedicated asyncio event loop in a background thread that lives for
the whole process. On that loop we enter `warm_tools()` ONCE and build the agent
with a MemorySaver ONCE. Each user turn is submitted to that loop with
`run_coroutine_threadsafe`, so the MCP subprocess, the tool session and the
conversation memory all persist across Streamlit reruns. Streamlit caches this
whole object with @st.cache_resource, so it's created exactly once per server.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_agent
from agent.mcp_client import warm_tools


@dataclass
class TurnResult:
    """Everything the UI needs to render one agent turn."""
    answer: str
    route: list[str] = field(default_factory=list)
    tool_name: str | None = None
    escalated: bool = False
    sources: list[dict] = field(default_factory=list)   # {source_file, section, title, score}
    guard_action: str = "pass"                            # pass | repair | refuse
    guard_repairs: list[str] = field(default_factory=list)
    grounded: bool | None = None
    pii_redacted: list[str] = field(default_factory=list)
    cache_hit: bool = False
    latency_s: float = 0.0


class AgentRuntime:
    """Owns the background event loop, the warm MCP session, and the agent."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready = threading.Event()
        self._agent = None
        self._tools_cm = None
        # Enter the warm session on the background loop and hold it open.
        self._submit(self._startup()).result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _startup(self) -> None:
        # Manually enter the async context manager and keep it open for the
        # process lifetime (there is no `async with` block to hold it, so we
        # drive __aenter__ directly and stash the CM for shutdown).
        self._tools_cm = warm_tools()
        tools = await self._tools_cm.__aenter__()
        self._agent = build_agent(tools, checkpointer=MemorySaver())
        self._ready.set()

    def ask(self, message: str, conversation_id: str) -> TurnResult:
        """Run one turn synchronously (blocks the caller until the agent answers).

        `conversation_id` becomes the LangGraph thread_id, so each browser
        conversation keeps its own multi-turn memory in the shared MemorySaver.
        """
        import time
        t0 = time.perf_counter()
        out = self._submit(self._ainvoke(message, conversation_id)).result()
        dt = time.perf_counter() - t0
        return _to_result(out, dt)

    async def _ainvoke(self, message: str, conversation_id: str) -> dict:
        return await self._agent.ainvoke(
            {"query": message, "messages": [HumanMessage(content=message)]},
            {"configurable": {"thread_id": conversation_id}},
        )

    def shutdown(self) -> None:
        async def _close():
            if self._tools_cm is not None:
                await self._tools_cm.__aexit__(None, None, None)
        try:
            self._submit(_close()).result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


def _to_result(out: dict, latency_s: float) -> TurnResult:
    hits = out.get("retrieved") or []
    sources = [
        {
            "source_file": getattr(h, "source_file", "?"),
            "section": getattr(h, "section", ""),
            "title": getattr(h, "title", ""),
            "score": round(float(getattr(h, "similarity", 0.0)), 3),
        }
        for h in hits
    ]
    return TurnResult(
        answer=out.get("answer") or "",
        route=out.get("route") or [],
        tool_name=out.get("tool_name"),
        escalated=bool(out.get("escalated")),
        sources=sources,
        guard_action=out.get("guard_action") or "pass",
        guard_repairs=out.get("guard_repairs") or [],
        grounded=out.get("grounded"),
        pii_redacted=out.get("pii_redacted") or [],
        cache_hit=bool(out.get("cache_hit")),
        latency_s=round(latency_s, 2),
    )
