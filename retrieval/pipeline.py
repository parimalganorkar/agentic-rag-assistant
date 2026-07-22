"""End-to-end retrieval pipeline: dense + sparse → RRF → rerank.

This is the production entry point Phase 4's `rag/naive.py` (and later
phases) should call instead of `dense_search` when they want the best
retrieval quality. Signature deliberately mirrors `dense_search(query, k)`
so callers can swap it in without changing anything else.

Flow:
  1. dense_search(query, k=20)   →   top-20 semantic candidates
  2. sparse_search(query, k=20)  →   top-20 BM25 candidates
  3. RRF-fuse                     →   top-20 by combined rank
  4. cross-encoder rerank         →   top-K (default 5) by relevance
"""

from __future__ import annotations

from retrieval.dense import RetrievedChunk
from retrieval.hybrid import hybrid_search
from retrieval.rerank import rerank


DEFAULT_CANDIDATES = 20  # width of the pool the reranker sees


def retrieve(
    query: str,
    k: int = 5,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[RetrievedChunk]:
    """Retrieve top-K chunks using hybrid retrieval + cross-encoder reranking.

    `candidates` controls how many hybrid results the reranker considers.
    Wider = better recall but slower (each candidate is one cross-encoder
    pass). 20 is a good default for CPU."""
    pool = hybrid_search(query, k_dense=candidates, k_sparse=candidates, k_out=candidates)
    return rerank(query, pool, k=k)


if __name__ == "__main__":
    # Smoke test on the Phase 4 DoD query.
    q = "how do I use a memory checkpoint in LangGraph"
    print(f"QUERY: {q!r}\n")
    hits = retrieve(q, k=5)
    for i, h in enumerate(hits, 1):
        print(f"  {i}. cross-enc={h.similarity:.2f}   {h.source_file}   ({h.section})")
