"""Phase 8 — bridge the LangGraph agent to the Phase 7 MCP server.

Uses `langchain-mcp-adapters` to expose the three MCP tools
(`get_package_version`, `get_corpus_status`, `fetch_live_doc`) as LangChain
tools the agent can call.

Why a PERSISTENT session (not the stateless `client.get_tools()`): the default
adapter path spawns a fresh server subprocess per tool call, and our server
prewarms bge-small on startup (~12s). Paying that on every `fetch_live_doc`
would make the agent painfully slow. `warm_tools()` holds ONE session open for
the agent's lifetime, so the server (and its warm embedder) is reused across
calls. Keep it open with `async with` around the whole conversation loop.

Run this module directly to smoke-test the bridge:  python -m agent.mcp_client
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_NAME = "docs_tools"

# The three tools we expect the server to expose — used to sanity-check the
# bridge and to look tools up by name from the router/dispatcher.
EXPECTED_TOOLS = ("get_package_version", "get_corpus_status", "fetch_live_doc")


def _connections() -> dict:
    """stdio connection spec for our FastMCP server. `cwd` matters: the server
    does `import mcp_server...` / `import ingestion...`, which only resolve from
    the repo root."""
    return {
        SERVER_NAME: {
            "command": "python",
            "args": ["-m", "mcp_server.server"],
            "transport": "stdio",
            "cwd": str(REPO_ROOT),
        }
    }


def make_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(_connections())


def parse_tool_result(raw: Any) -> dict:
    """Normalize an MCP tool result to a dict.

    Our tools return plain dicts, which FastMCP serializes as JSON text; the
    adapter hands that back as a list of content blocks
    ([{"type": "text", "text": "<json>"}]). Unwrap to the parsed dict so graph
    nodes get a clean object regardless of that transport detail.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"text": raw}
    if isinstance(raw, list):
        for block in raw:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return {"text": text}
    return {"raw": raw}


@asynccontextmanager
async def warm_tools() -> AsyncIterator[dict[str, BaseTool]]:
    """Yield {tool_name: BaseTool} bound to a single persistent MCP session.

    The server stays warm for the whole `async with` block, so repeated tool
    calls don't re-spawn it. Returns a name->tool dict for easy dispatch.
    """
    client = make_client()
    async with client.session(SERVER_NAME) as session:
        tools = await load_mcp_tools(session)
        yield {t.name: t for t in tools}


async def _smoke_test() -> None:
    async with warm_tools() as tools:
        print("tools exposed:", sorted(tools))
        missing = [t for t in EXPECTED_TOOLS if t not in tools]
        assert not missing, f"missing expected tools: {missing}"

        print("\n[get_package_version langgraph]")
        d = parse_tool_result(await tools["get_package_version"].ainvoke({"package": "langgraph"}))
        print("  version:", d.get("version"), "| released:", d.get("released"))

        print("\n[get_corpus_status topic=memory]")
        d = parse_tool_result(await tools["get_corpus_status"].ainvoke({"topic": "memory"}))
        print("  total_docs:", d.get("total_docs"), "| by_doc_type:", d.get("by_doc_type"))
        print("  memory match_count:", d.get("match_count"))

        print("\n[fetch_live_doc path=src/oss/langgraph/studio.mdx]")
        d = parse_tool_result(await tools["fetch_live_doc"].ainvoke({"path": "src/oss/langgraph/studio.mdx"}))
        print("  matched:", d.get("matched"), "| title:", d.get("title"),
              "| content chars:", len(d.get("content") or ""))

    print("\nOK — MCP bridge works over a warm session.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_smoke_test())
