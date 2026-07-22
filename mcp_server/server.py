"""Phase 7 — MCP server exposing the LangChain-docs-assistant tools.

Wraps the pure functions in `mcp_server/tools.py` as MCP tools over stdio, so
any MCP client — the LangGraph agent (`agent/mcp_client.py`) or the MCP
Inspector — can discover and call them.

The docstrings below matter: an LLM agent reads a tool's description to decide
WHEN to call it, so they're written as routing hints, not just documentation.

Run:  python -m mcp_server.server     (starts a stdio MCP server)
Test: python -m mcp_server.tools      (smoke-tests the tool logic directly)
      npx @modelcontextprotocol/inspector python -m mcp_server.server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server import tools


app = FastMCP("langchain-docs-assistant-tools")


@app.tool()
def get_package_version(package: str) -> dict:
    """Get the CURRENT released version and release date of a Python package
    from PyPI (live network lookup).

    Use this for version questions — e.g. "what's the latest version of
    langgraph?" or "is langchain-core up to date?". The assistant's documentation
    is frozen at a pinned commit, so it CANNOT answer questions about current
    versions; this tool can. Intended for the LangChain ecosystem (langchain,
    langgraph, langchain-core, langchain-community, ...) but accepts any PyPI
    package name. Returns {version, released, summary} or {error}.
    """
    return tools.get_package_version(package)


@app.tool()
def get_corpus_status(topic: str | None = None) -> dict:
    """Report what the assistant's documentation corpus covers and how fresh it
    is: the pinned source commit, when it was built, total document count, and
    per-product / per-doc-type breakdowns. Read from a local manifest (no
    network).

    Use this for meta questions about the assistant's own knowledge — "how
    current are your docs?", "what version of the docs do you have?" — or pass a
    `topic` to check coverage of a subject ("do you have docs on middleware?"),
    which returns the number of matching docs and a few sample file paths.
    """
    return tools.get_corpus_status(topic)


@app.tool()
def fetch_live_doc(query: str | None = None, path: str | None = None) -> dict:
    """Fetch a CURRENT LangChain/LangGraph documentation page live from GitHub,
    for questions the INDEXED corpus can't answer — new features or docs that
    were added or changed after our pinned snapshot.

    Call this AFTER retrieval/`get_corpus_status` shows the topic isn't in the
    indexed corpus. Pass a natural-language `query` to search the live docs
    (returns the best-matching page's content plus other close candidates), or a
    specific `path` (e.g. from a prior call's candidates) to fetch that exact
    file. The match is a best guess — judge from the returned content whether it
    actually answers, and refetch a candidate via `path` if needed. Returns the
    cleaned page content + its GitHub source URL, or `matched: false` when the
    topic looks out of scope (so you can refuse honestly rather than guess).
    """
    return tools.fetch_live_doc(query=query, path=path)


if __name__ == "__main__":
    # Warm the embedder + live-doc tree BEFORE the stdio protocol starts, so the
    # first fetch_live_doc(query=...) call responds fast instead of paying the
    # model load mid-request (which, over a fresh MCP subprocess, could exceed a
    # client's request timeout). Best-effort — falls back to lazy load on failure.
    tools.prewarm_live_docs()
    app.run(transport="stdio")
