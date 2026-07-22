"""Sparse (BM25) retrieval over the chunks corpus.

BM25 shines where dense retrieval stumbles: exact keyword matches. If a user
asks about `create_agent` or `bind_tools` or `InMemorySaver`, we want the
chunks that literally contain that token to rank high — and a semantic
embedding model doesn't guarantee that.

This module builds an in-memory BM25 index from `data/processed/chunks.jsonl`
and exposes a `sparse_search(query, k)` helper that mirrors the shape of
`dense_search` in dense.py, returning `RetrievedChunk` objects so downstream
code can treat dense and sparse hits uniformly.

The index is built lazily and cached across calls — the first `sparse_search`
in a process pays a ~500 ms tokenization cost, subsequent calls are instant.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from retrieval.dense import RetrievedChunk


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_JSONL = REPO_ROOT / "data" / "processed" / "chunks.jsonl"

# BM25 doesn't score in [0, 1] — it's an unbounded relevance number. Fusion
# with dense (in Phase 5's hybrid.py) uses ranks, not raw scores, so absolute
# BM25 values don't matter for retrieval. Keep them anyway for debugging.


# --- Tokenizer --------------------------------------------------------------
# `\w+` keeps letters, digits, and underscores. That's important: identifiers
# like `create_agent` or `bind_tools` MUST stay as single tokens so BM25 can
# match them intact against a query that also spells them the same way.

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --- Index build ------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_chunks_and_index() -> tuple[list[dict], BM25Okapi]:
    """Load chunks + build the BM25 index. Cached — pays the cost once."""
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_JSONL}. "
            f"Run `python -m ingestion.run` first."
        )

    chunks: list[dict] = []
    with CHUNKS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    tokenized = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return chunks, bm25


# --- Public API -------------------------------------------------------------

def sparse_search(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Return the top-K BM25 hits for `query`, wrapped as `RetrievedChunk`."""
    chunks, bm25 = _load_chunks_and_index()
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)  # numpy array, one score per chunk
    # Take top-K indices by score (descending)
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

    hits: list[RetrievedChunk] = []
    for idx in top_idx:
        c = chunks[idx]
        headers = c.get("headers") or {}
        hits.append(
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                # BM25 scores aren't cosine similarities; expose the raw score
                # in `similarity` and set distance to 0 for compatibility with
                # `RetrievedChunk`. Downstream code should treat sparse hits'
                # `similarity` as a relative BM25 score, not a probability.
                similarity=float(scores[idx]),
                distance=0.0,
                source_file=c.get("source_file", ""),
                section=c.get("section", ""),
                title=c.get("title", ""),
                product=c.get("product", ""),
                doc_type=c.get("doc_type", ""),
                token_count=int(c.get("token_count", 0) or 0),
                content_hash=c.get("content_hash", ""),
            )
        )
    return hits


if __name__ == "__main__":
    # Smoke test — an exact-term query where BM25 should shine.
    print("[sparse] building BM25 index (first-time cost) ...")
    hits = sparse_search("create_agent", k=5)
    print(f"[sparse] returned {len(hits)} hits\n")
    for i, h in enumerate(hits, 1):
        print(f"--- rank {i}   bm25={h.similarity:.2f}")
        print(f"    source  : {h.source_file}")
        print(f"    section : {h.section}")
        print(f"    preview : {h.text[:200].strip()!r}")
        print()
