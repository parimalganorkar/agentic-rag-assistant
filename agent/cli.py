"""Phase 8 — interactive CLI for the LangGraph agent (with multi-turn memory).

Holds ONE warm MCP session for the whole conversation and a `MemorySaver`
checkpointer keyed by a thread_id, so follow-up turns ("how do I install it?")
resolve against earlier ones.

    python -m agent.cli            # interactive chat
    python -m agent.cli --demo     # scripted 3-turn conversation that proves memory
"""

from __future__ import annotations

import argparse
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_agent
from agent.mcp_client import warm_tools

THREAD_ID = "cli-session"


async def _turn(agent, query: str, config: dict) -> dict:
    return await agent.ainvoke(
        {"query": query, "messages": [HumanMessage(content=query)]}, config
    )


def _show(query: str, out: dict) -> None:
    tag = out.get("route")
    if out.get("escalated"):
        tag = f"{tag} (escalated→live)"
    print(f"\n\033[1mYou:\033[0m {query}")
    print(f"\033[90m[route: {tag} | tool: {out.get('tool_name')}]\033[0m")
    print(f"\033[1mAgent:\033[0m {out['answer']}\n")


async def _interactive(agent, config: dict) -> None:
    print("Agent ready. Ask about LangChain/LangGraph. Ctrl-C or 'exit' to quit.\n")
    while True:
        try:
            query = (await asyncio.to_thread(input, "You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if query.lower() in {"exit", "quit"}:
            return
        if not query:
            continue
        out = await _turn(agent, query, config)
        print(f"\033[90m[route: {out.get('route')} | tool: {out.get('tool_name')}]\033[0m")
        print(f"Agent: {out['answer']}\n")


# A scripted conversation whose later turns ONLY make sense with memory.
DEMO_TURNS = [
    "What's the latest version of langgraph?",   # get_package_version
    "What about langchain-core?",                 # memory: still a version question
    "How do I install it?",                       # memory: 'it' == langchain-core
]


async def _demo(agent, config: dict) -> None:
    print("=== MEMORY DEMO (each turn depends on the previous) ===")
    for q in DEMO_TURNS:
        out = await _turn(agent, q, config)
        _show(q, out)


async def main() -> None:
    ap = argparse.ArgumentParser(description="LangGraph docs agent")
    ap.add_argument("--demo", action="store_true", help="run the scripted memory demo")
    args = ap.parse_args()

    config = {"configurable": {"thread_id": THREAD_ID}}
    async with warm_tools() as tools:
        agent = build_agent(tools, checkpointer=MemorySaver())
        if args.demo:
            await _demo(agent, config)
        else:
            await _interactive(agent, config)


if __name__ == "__main__":
    asyncio.run(main())
