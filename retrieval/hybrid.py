"""Hybrid retrieval — fuse dense (semantic) and sparse (BM25) rankings.

Uses Reciprocal Rank Fusion (RRF), the standard rank-fusion technique. RRF's
key property: it uses only the *rank* each retriever assigned to a document,
not the raw score. That means we don't need to normalize BM25 scores against
cosine similarities — they live on different scales but their ranks don't.

Formula per document:
    rrf_score(doc) = sum over retrievers of  1 / (K + rank(doc))
where `rank` is 1-indexed and `K` is a smoothing constant (60 is the standard
value from the original RRF paper — smaller K favors top-ranked docs more).
"""

from __future__ import annotations

from retrieval.dense import RetrievedChunk, dense_search
from retrieval.sparse import sparse_search


# Standard RRF smoothing constant. 60 is the value from Cormack et al. 2009.
DEFAULT_RRF_K = 60


def _rrf_fuse(
    rankings: list[list[RetrievedChunk]],
    k: int = DEFAULT_RRF_K,
) -> dict[str, tuple[float, RetrievedChunk]]:
    """Compute RRF scores; return {chunk_id: (rrf_score, canonical_chunk)}.

    The canonical chunk is the first `RetrievedChunk` seen for a given
    `chunk_id` — we prefer the dense-side hit's metadata since its
    `similarity` is a real cosine value the caller can reason about."""
    fused: dict[str, tuple[float, RetrievedChunk]] = {}

    for ranking in rankings:
        for rank_pos, chunk in enumerate(ranking, start=1):
            cid = chunk.chunk_id
            contribution = 1.0 / (k + rank_pos)
            if cid in fused:
                prior_score, canonical = fused[cid]
                fused[cid] = (prior_score + contribution, canonical)
            else:
                fused[cid] = (contribution, chunk)

    return fused


def hybrid_search(
    query: str,
    k_dense: int = 20,
    k_sparse: int = 20,
    k_out: int = 20,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Retrieve dense + sparse candidates, RRF-fuse them, return top-K.

    We pull 20 from each side by default because Phase 5's reranker is the
    layer that trims to a final top-5. Fusing wider gives the reranker more
    to work with; keeping it at 20 caps how many cross-encoder passes the
    reranker has to run (which is where most of the query-time cost lives)."""
    dense_hits = dense_search(query, k=k_dense)
    sparse_hits = sparse_search(query, k=k_sparse)
    fused = _rrf_fuse([dense_hits, sparse_hits], k=rrf_k)

    # Sort by RRF score descending, keep top k_out
    ordered = sorted(fused.items(), key=lambda kv: -kv[1][0])[:k_out]
    return [chunk for _, (_, chunk) in ordered]


if __name__ == "__main__":
    # Smoke test — compare dense-only vs hybrid on a keyword-heavy query.
    q = "bind_tools structured output"
    print(f"QUERY: {q!r}\n")

    print("--- DENSE ONLY (top 5) ---")
    for i, h in enumerate(dense_search(q, k=5), 1):
        print(f"  {i}. {h.source_file}   ({h.section})")

    print("\n--- HYBRID (top 5 after RRF fusion) ---")
    for i, h in enumerate(hybrid_search(q, k_out=5), 1):
        print(f"  {i}. {h.source_file}   ({h.section})")
