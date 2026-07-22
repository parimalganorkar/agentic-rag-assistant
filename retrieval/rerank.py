"""Cross-encoder reranker.

The bi-encoder we use for dense retrieval (bge-small) encodes the query and
each passage *independently* into a vector, then compares. That's cheap but
imprecise — it can't see interactions between query and passage tokens.

A cross-encoder does the opposite: it takes (query, passage) together as one
input and scores the pair with a full transformer pass. Much more accurate,
much more expensive per pair. Perfect for reranking a small candidate set.

We use `cross-encoder/ms-marco-MiniLM-L-6-v2` — trained on MS MARCO's
passage-ranking task, well-benchmarked, and small enough to run on CPU in
sub-second time for a top-20 rerank.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from sentence_transformers import CrossEncoder

from retrieval.dense import RetrievedChunk


RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder once and cache. First call downloads ~90 MB."""
    return CrossEncoder(RERANK_MODEL_NAME)


def rerank(
    query: str,
    candidates: Sequence[RetrievedChunk],
    k: int = 5,
) -> list[RetrievedChunk]:
    """Score every (query, chunk) pair with the cross-encoder, return top-K.

    The returned chunks have their `similarity` field overwritten with the
    cross-encoder relevance score so the caller can see what the reranker
    thought of each hit. That score is NOT a probability — it's a raw model
    logit that can be negative; higher = more relevant."""
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)  # numpy array, shape (len(candidates),)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda cs: -float(cs[1]))

    out: list[RetrievedChunk] = []
    for chunk, score in scored[:k]:
        # Replace the incoming similarity (which may be a cosine-sim from
        # dense OR an RRF score from hybrid — very different scales) with
        # the cross-encoder relevance score. Preserves the RetrievedChunk
        # shape so downstream generation code doesn't care.
        out.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                similarity=float(score),
                distance=chunk.distance,
                source_file=chunk.source_file,
                section=chunk.section,
                title=chunk.title,
                product=chunk.product,
                doc_type=chunk.doc_type,
                token_count=chunk.token_count,
                content_hash=chunk.content_hash,
            )
        )
    return out


if __name__ == "__main__":
    # Smoke test: rerank hybrid's top-20 down to top-5 and see how the order shifts.
    from retrieval.hybrid import hybrid_search

    q = "how do I bind tools to a chat model"
    print(f"QUERY: {q!r}\n")

    print("--- HYBRID (top 5 before rerank) ---")
    candidates = hybrid_search(q, k_out=20)
    for i, h in enumerate(candidates[:5], 1):
        print(f"  {i}. {h.source_file}   ({h.section})")

    print("\n--- HYBRID + RERANK (top 5) ---")
    reranked = rerank(q, candidates, k=5)
    for i, h in enumerate(reranked, 1):
        print(f"  {i}. cross-enc={h.similarity:.2f}  {h.source_file}   ({h.section})")
