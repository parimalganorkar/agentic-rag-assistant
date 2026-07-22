"""Phase 7 — MCP tool logic (pure functions, no MCP layer).

Two tools that answer questions the frozen docs corpus CANNOT:

  get_package_version(package)  -> the CURRENT released version from PyPI (live).
      The corpus is pinned to a commit (2026-07-04), so it can never answer
      "what's the latest version of X?". A live lookup can.

  get_corpus_status(topic=None) -> what the pinned docs cover + how fresh they
      are, read from our own manifest (fully local). Makes the assistant
      self-aware about its knowledge boundary.

These are kept as plain functions (not MCP-wrapped) so they can be tested in
isolation with `python -m mcp_server.tools` BEFORE server.py wraps them as MCP
tools — debug the tool alone first (a project convention).
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

# Load the embedder from the local HF cache ONLY — never round-trip to the Hub.
# Why this matters here specifically: this module runs inside an MCP stdio server
# that a client spawns as a FRESH subprocess. On every fresh process,
# SentenceTransformer() otherwise pings the HF Hub to revalidate the model; done
# unauthenticated that gets rate-limited with backoff and can stall well past a
# client's request timeout (observed: a query-mode call exceeded 200s over MCP
# while the same call is ~12s once the Hub check is skipped). The model is
# already cached (the whole retrieval pipeline uses bge-small), so offline is
# safe. `setdefault` still lets a caller override if they really need online.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import requests

from ingestion.cleaner.manifest import load_manifest
from ingestion.cleaner.normalizer import clean_mdx


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "cleaned" / "manifest.json"

PYPI_URL = "https://pypi.org/pypi/{package}/json"
PYPI_TIMEOUT_SEC = 8

# --- Live-docs (fetch_live_doc) config --------------------------------------
# The docs corpus is a FROZEN snapshot (pinned commit). These endpoints let us
# reach the CURRENT docs on GitHub for topics the vector DB doesn't cover.
GITHUB_TREE_URL = "https://api.github.com/repos/langchain-ai/docs/git/trees/main?recursive=1"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/langchain-ai/docs/main/{path}"
GITHUB_BLOB_URL = "https://github.com/langchain-ai/docs/blob/main/{path}"
GITHUB_TIMEOUT_SEC = 12
# Same corpus scope as Phase 1 — keep live discovery aligned with what we index.
SCOPE_PREFIXES = ("src/oss/langchain/", "src/oss/langgraph/", "src/oss/concepts/")
TREE_CACHE_TTL_SEC = 3600          # the GitHub API is 60 req/hr unauth; one call covers everything
CONTENT_CHAR_BUDGET = 6000         # truncate a fetched page so the tool response stays manageable

# bge-small — the SAME embedder the retriever uses — ranks which live file best
# matches the query. Loaded independently here so this tool doesn't depend on Chroma.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# LangChain-ecosystem packages this tool is intended for. NOT enforced — any
# valid PyPI name works — but surfaced so the agent knows the intended scope.
KNOWN_PACKAGES = [
    "langchain", "langchain-core", "langgraph", "langchain-community",
    "langchain-text-splitters", "langgraph-checkpoint", "langsmith",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_package_version(package: str) -> dict[str, Any]:
    """Look up the CURRENT released version of a Python package on PyPI (live).

    Use for version questions like "what's the latest langgraph?" — the docs
    corpus is frozen at a pinned commit and can't answer these. Intended for the
    LangChain ecosystem (see KNOWN_PACKAGES) but accepts any PyPI package name.

    Returns a dict with either {version, released, summary, ...} or {error}.
    Never raises — network/lookup failures come back as an `error` field so the
    calling agent can handle them gracefully.
    """
    package = (package or "").strip()
    if not package:
        return {"package": package, "error": "empty package name"}

    try:
        resp = requests.get(PYPI_URL.format(package=package), timeout=PYPI_TIMEOUT_SEC)
    except requests.RequestException as e:
        return {"package": package, "error": f"network error: {type(e).__name__}"}

    if resp.status_code == 404:
        return {"package": package, "error": "not found on PyPI"}
    if resp.status_code != 200:
        return {"package": package, "error": f"PyPI returned HTTP {resp.status_code}"}

    data = resp.json()
    info = data.get("info", {})
    version = info.get("version", "")

    # Release date = upload time of the first distributed file for this version.
    released = None
    files = data.get("releases", {}).get(version, [])
    if files:
        released = files[0].get("upload_time_iso_8601") or files[0].get("upload_time")

    return {
        "package": package,
        "version": version,
        "released": released,
        "summary": info.get("summary"),
        "source": "pypi-live",
        "fetched_at": _now_iso(),
    }


def get_corpus_status(topic: str | None = None) -> dict[str, Any]:
    """Report what the assistant's documentation corpus covers and how fresh it is.

    Reads the local manifest (no network). Use for meta questions about the
    assistant's own knowledge: "how current are your docs?", "what version of
    the docs?", or — with `topic` — "do you have docs on middleware?".

    Returns pinned commit, generation date, total docs, and per-product /
    per-doc_type counts. With `topic`, also returns how many docs match and a
    few sample source files.
    """
    manifest = load_manifest(MANIFEST_PATH)
    ok = {k: e for k, e in manifest.entries.items() if e.status == "ok"}

    by_product: dict[str, int] = {}
    by_doc_type: dict[str, int] = {}
    for e in ok.values():
        prod = e.product or "unknown"
        dtype = e.doc_type or "unknown"
        by_product[prod] = by_product.get(prod, 0) + 1
        by_doc_type[dtype] = by_doc_type.get(dtype, 0) + 1

    result: dict[str, Any] = {
        "source_commit": manifest.source_commit,
        "docs_generated_at": manifest.generated_at,
        "total_docs": len(ok),
        "by_product": dict(sorted(by_product.items())),
        "by_doc_type": dict(sorted(by_doc_type.items())),
    }

    if topic:
        t = topic.strip().lower()
        matches = [
            e.source_rel_path for e in ok.values()
            if t in (e.title or "").lower()
            or t in (e.section or "").lower()
            or t in e.source_rel_path.lower()
        ]
        result["topic"] = topic
        result["match_count"] = len(matches)
        result["sample_files"] = sorted(matches)[:8]

    return result


# ============================================================================
# fetch_live_doc — reach the CURRENT docs for topics the frozen corpus lacks
# ============================================================================

# Module-level cache of the live file tree + its candidate embeddings, so we
# don't re-hit the GitHub API (rate-limited) or re-embed 100+ paths per call.
_tree_cache: dict[str, Any] = {"ts": 0.0, "files": [], "embeds": None}


@lru_cache(maxsize=1)
def _embedder():
    """Lazy-load bge-small once (first call downloads/loads ~130 MB)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


def _path_to_text(path: str) -> str:
    """'src/oss/langchain/human-in-the-loop.mdx' -> 'langchain human in the loop'
    so the path itself becomes something bge-small can rank against a query."""
    p = path[len("src/oss/"):] if path.startswith("src/oss/") else path
    if p.endswith(".mdx"):
        p = p[:-4]
    return p.replace("/", " ").replace("-", " ").replace("_", " ").strip()


def _live_candidates() -> tuple[list[str], Any]:
    """Return (in-scope current .mdx paths, their embeddings), cached ~1h."""
    now = time.time()
    if _tree_cache["files"] and _tree_cache["embeds"] is not None \
            and now - _tree_cache["ts"] < TREE_CACHE_TTL_SEC:
        return _tree_cache["files"], _tree_cache["embeds"]

    resp = requests.get(
        GITHUB_TREE_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "rag-build"},
        timeout=GITHUB_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    tree = resp.json()
    files = [
        n["path"] for n in tree.get("tree", [])
        if n.get("type") == "blob"
        and n["path"].startswith(SCOPE_PREFIXES)
        and n["path"].endswith(".mdx")
    ]
    embeds = _embedder().encode(
        [_path_to_text(f) for f in files],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _tree_cache.update(ts=now, files=files, embeds=embeds)
    return files, embeds


def prewarm_live_docs() -> None:
    """Load bge-small and build the live-tree cache up front (best-effort).

    The server calls this once at startup so the first `fetch_live_doc` query
    doesn't pay the ~12s model load mid-request — and, importantly, so any
    model-loading output happens BEFORE the MCP stdio JSON-RPC protocol starts on
    stdout. Failures (e.g. no network for the tree) are swallowed: the tool falls
    back to lazy loading on first use.
    """
    try:
        _live_candidates()
    except Exception:
        # Best-effort: at minimum warm the model so path mode is instant too.
        try:
            _embedder()
        except Exception:
            pass


@lru_cache(maxsize=512)
def _fetch_github_snippet(github_path: str) -> str | None:
    """Fetch a snippet .mdx from GitHub main (cached). `github_path` is repo-
    relative, e.g. 'src/snippets/code-samples/agents-intro-py.mdx'."""
    try:
        r = requests.get(
            GITHUB_RAW_URL.format(path=github_path),
            headers={"User-Agent": "rag-build"},
            timeout=GITHUB_TIMEOUT_SEC,
        )
        return r.text if r.status_code == 200 else None
    except requests.RequestException:
        return None


def _github_snippet_resolver(doc_path: str):
    """Resolver that maps a live page's snippet import path to its GitHub
    content, so live-fetched docs inline their code the same way the local
    corpus does (the docs keep code examples in `/snippets/...mdx` pulled in via
    `<Component />`). Absolute '/snippets/...' paths are relative to the Mintlify
    content root (the repo's `src/` dir); relative paths resolve against the
    importing page's folder. Fetches are cached; a snippet-heavy page costs a few
    extra requests, acceptable for a fallback tool."""
    doc_dir = doc_path.rsplit("/", 1)[0] if "/" in doc_path else ""

    def resolve(import_path: str) -> str | None:
        github_path = ("src" + import_path) if import_path.startswith("/") \
            else f"{doc_dir}/{import_path}".lstrip("/")
        # The import path comes out of a FETCHED page, i.e. it is remote content
        # deciding what else we fetch. Traversal here would walk off the pinned
        # repo the same way path mode could.
        if not _SAFE_PATH_RE.match(github_path) or ".." in github_path.split("/") \
                or not github_path.startswith("src/"):
            return None
        return _fetch_github_snippet(github_path)

    return resolve


def _fetch_and_clean(path: str, query: str | None, score: float | None) -> dict[str, Any]:
    """Fetch one raw .mdx from GitHub main, clean it, truncate to budget."""
    try:
        raw = requests.get(
            GITHUB_RAW_URL.format(path=path),
            headers={"User-Agent": "rag-build"},
            timeout=GITHUB_TIMEOUT_SEC,
        )
        raw.raise_for_status()
    except requests.RequestException as e:
        return {"query": query, "path": path, "error": f"fetch failed: {type(e).__name__}"}

    # Inline snippet code from GitHub so live pages don't lose their examples.
    cleaned = clean_mdx(raw.text, snippet_resolver=_github_snippet_resolver(path))
    content = cleaned.text
    truncated = len(content) > CONTENT_CHAR_BUDGET
    if truncated:
        content = content[:CONTENT_CHAR_BUDGET].rstrip() + "\n\n[...truncated...]"

    out: dict[str, Any] = {
        "query": query,
        "matched": True,
        "source_file": path,
        "source_url": GITHUB_BLOB_URL.format(path=path),
        "title": cleaned.title,
        "content": content,
        "truncated": truncated,
        "note": "live content from GitHub main — NOT from the indexed corpus",
        "fetched_at": _now_iso(),
    }
    if score is not None:
        out["score"] = round(score, 3)
    return out


_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def _validate_live_path(path: str) -> str | None:
    """Return an error string if `path` is not a safe in-scope docs path.

    `path` is interpolated straight into the raw.githubusercontent URL, so an
    unvalidated value is a request-forgery primitive: `../../` walks out of the
    pinned repo/ref and makes the server fetch ATTACKER-CHOSEN content, which is
    then cleaned, trusted, and fed to the model as documentation. Scope is
    enforced too, so path mode can't reach files the corpus deliberately excludes.
    """
    p = (path or "").strip()
    if not p:
        return "empty path"
    if not _SAFE_PATH_RE.match(p):
        return "path contains illegal characters"
    if p.startswith("/") or ".." in p.split("/") or "//" in p:
        return "path traversal is not allowed"
    if not p.startswith(SCOPE_PREFIXES):
        return f"path is outside the indexed scope {SCOPE_PREFIXES}"
    if not p.endswith(".mdx"):
        return "only .mdx documentation files may be fetched"
    return None


def fetch_live_doc(
    query: str | None = None,
    path: str | None = None,
    min_score: float = 0.62,
    top_k: int = 5,
) -> dict[str, Any]:
    """Fetch a CURRENT documentation page live from GitHub — for questions the
    frozen indexed corpus can't answer (features / docs added or changed since
    our pinned snapshot).

    Two modes:
      - query mode (default): ranks the live in-scope files against `query` with
        bge-small (the same embedder the retriever uses), fetches the top match,
        and ALSO returns the other close candidates. Ranking on short file paths
        is a coarse signal, so the top guess can be imperfect — the caller/agent
        should judge from the returned CONTENT whether it actually answers, and
        can refetch a listed candidate via `path` if not.
      - path mode: pass an exact `path` (e.g. from a prior call's candidates) to
        fetch that specific current file, skipping ranking.

    `min_score` only coarsely gates obvious out-of-scope queries; because every
    doc in a single-domain corpus looks somewhat similar to any dev question,
    the FINAL relevance judgment is the caller's, from the content. Never raises;
    failures come back as `error`.
    """
    # --- path mode: fetch an explicit file, no ranking ---
    if path:
        bad = _validate_live_path(path)
        if bad:
            return {"query": query, "path": path, "matched": False,
                    "error": f"refused: {bad}"}
        return _fetch_and_clean(path, query=query, score=None)

    query = (query or "").strip()
    if not query:
        return {"query": query, "error": "provide a `query` to search, or a `path` to fetch"}

    try:
        files, embeds = _live_candidates()
    except requests.RequestException as e:
        return {"query": query, "error": f"could not list live docs: {type(e).__name__}"}
    if not files:
        return {"query": query, "error": "live doc tree was empty"}

    # Rank: cosine of the query vector against every candidate (all normalized).
    q_vec = _embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = embeds @ q_vec
    ranked_idx = scores.argsort()[::-1][:top_k]
    ranked = [(files[i], float(scores[i])) for i in ranked_idx]
    candidates = [{"path": p, "score": round(s, 3)} for p, s in ranked]

    best_path, best_score = ranked[0]
    if best_score < min_score:
        return {
            "query": query,
            "matched": False,
            "candidates": candidates,
            "note": "no live doc cleared the relevance threshold — likely out of scope",
        }

    result = _fetch_and_clean(best_path, query=query, score=best_score)
    if result.get("matched"):
        result["other_candidates"] = candidates[1:]
        result["note"] = (
            "best-guess match by path similarity — verify against the content; "
            "if it doesn't answer, refetch a listed candidate via the `path` argument"
        )
    return result


if __name__ == "__main__":
    import json

    print("=== get_package_version('langgraph')  [live PyPI] ===")
    print(json.dumps(get_package_version("langgraph"), indent=2))

    print("\n=== get_package_version('nonexistent-pkg-zzz')  [error path] ===")
    print(json.dumps(get_package_version("nonexistent-pkg-zzz"), indent=2))

    print("\n=== get_corpus_status()  [local manifest] ===")
    print(json.dumps(get_corpus_status(), indent=2))

    print("\n=== get_corpus_status(topic='memory')  [coverage check] ===")
    print(json.dumps(get_corpus_status(topic="memory"), indent=2))

    def _preview(d: dict) -> dict:
        # trim the big content field so the smoke test stays readable
        d = dict(d)
        if isinstance(d.get("content"), str) and len(d["content"]) > 300:
            d["content"] = d["content"][:300] + " …[trimmed]"
        return d

    print("\n=== fetch_live_doc('how do I use LangGraph Studio?')  [in-scope, live] ===")
    print(json.dumps(_preview(fetch_live_doc("how do I use LangGraph Studio to debug my graph?")), indent=2))

    print("\n=== fetch_live_doc('kubernetes cluster with helm')  [out-of-scope] ===")
    print(json.dumps(_preview(fetch_live_doc("how do I set up a Kubernetes cluster with Helm?")), indent=2))

    print("\n=== fetch_live_doc(path='src/oss/langgraph/studio.mdx')  [explicit path mode] ===")
    print(json.dumps(_preview(fetch_live_doc(path="src/oss/langgraph/studio.mdx")), indent=2))
