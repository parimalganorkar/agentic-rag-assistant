"""Dense (vector-similarity) retrieval from Chroma.

Loads the persistent Chroma collection built in Phase 3 and provides a
`dense_search(query, k)` helper that returns the top-K most similar chunks.

Two important bge-small conventions preserved here:
  1. Queries are prefixed with "Represent this sentence for searching
     relevant passages: " — passages are not.
  2. Query embeddings are L2-normalized, matching the collection's cosine
     distance metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_ROOT = REPO_ROOT / "data" / "chroma"
COLLECTION_NAME = "langchain_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class RetrievedChunk:
    """One retrieval hit: everything the generator or reranker will need."""
    chunk_id: str
    text: str
    similarity: float           # cosine similarity, in [0, 1] for normalized vectors
    distance: float             # cosine distance = 1 - similarity
    source_file: str
    section: str
    title: str
    product: str
    doc_type: str
    token_count: int
    content_hash: str


# --- Lazy-loaded singletons -------------------------------------------------
# The embedding model and Chroma client are expensive to instantiate; we cache
# them so a script doing many queries doesn't pay the load cost repeatedly.

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_collection():
    if not CHROMA_ROOT.exists():
        raise FileNotFoundError(
            f"Chroma folder not found at {CHROMA_ROOT}. "
            f"Run `python -m ingestion.embed_and_store` first."
        )
    client = chromadb.PersistentClient(path=str(CHROMA_ROOT))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# --- Public API -------------------------------------------------------------

def embed_query(query: str) -> list[float]:
    """Encode a query for retrieval — prefix + L2-normalized 384-dim vector.

    Cached: the embedding of a given query text never changes, and a session
    repeats queries (retries, the eval harness, follow-ups), so this skips a
    model forward pass on repeats. Import is local to avoid a hard dependency
    from the retrieval layer on the agent package.
    """
    from agent.cache import EMBED_CACHE, key_of

    def _compute() -> list[float]:
        prefixed = QUERY_PREFIX + query
        vec = _get_model().encode(
            [prefixed], normalize_embeddings=True, show_progress_bar=False,
        )
        return vec[0].tolist()

    return EMBED_CACHE.get_or_compute(key_of("q", query), _compute)


def dense_search(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Return the top-K most similar chunks to `query`."""
    collection = _get_collection()
    q_vec = embed_query(query)
    result = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[RetrievedChunk] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                similarity=1.0 - dist,
                distance=dist,
                source_file=meta.get("source_file", ""),
                section=meta.get("section", ""),
                title=meta.get("title", ""),
                product=meta.get("product", ""),
                doc_type=meta.get("doc_type", ""),
                token_count=int(meta.get("token_count", 0) or 0),
                content_hash=meta.get("content_hash", ""),
            )
        )
    return hits


if __name__ == "__main__":
    # Smoke test
    print("[dense] warming up model + Chroma ...")
    hits = dense_search("how do I use a memory checkpoint in LangGraph", k=3)
    print(f"[dense] returned {len(hits)} hits\n")
    for i, h in enumerate(hits, 1):
        print(f"--- rank {i}   sim={h.similarity:.3f}")
        print(f"    source  : {h.source_file}")
        print(f"    section : {h.section}")
        print(f"    preview : {h.text[:200].strip()!r}")
        print()
