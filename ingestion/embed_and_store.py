"""Phase 3 — Embed chunks with bge-small and upsert into local Chroma.

Reads `data/processed/chunks.jsonl`, computes 384-dim embeddings with
`BAAI/bge-small-en-v1.5` (matches the tokenizer used in Phase 2 chunking),
and stores them in a persistent Chroma collection at `data/chroma/`.

Incremental: a chunk is skipped if the collection already has a document
with the same `chunk_id` AND whose stored `content_hash` matches. So a re-run
after a partial source update only re-embeds the changed chunks — no full
rebuild required. Pass `--force` to wipe and rebuild the collection.

Embeddings are L2-normalized and the collection uses cosine distance; for
normalized vectors cosine and dot product are equivalent, and the smaller
distance number the more similar the result.

Query-side prefix (used by Phase 4 retrieval, NOT here): bge instructs
callers to prepend "Represent this sentence for searching relevant passages: "
to the query only, never to the passages being indexed. We follow that
convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chromadb
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_JSONL = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_ROOT = REPO_ROOT / "data" / "chroma"
COLLECTION_NAME = "langchain_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


@dataclass
class RunStats:
    total_chunks: int = 0
    unchanged_skipped: int = 0
    newly_embedded: int = 0
    orphans_deleted: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)


def load_chunks(path: Path) -> Iterator[dict]:
    """Stream chunks from JSONL."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def flatten_metadata(chunk: dict) -> dict:
    """Chroma metadata must be primitives (str/int/float/bool). The chunker
    stores headers as a nested dict; we flatten to h1/h2/h3 for filterability."""
    headers = chunk.get("headers") or {}
    return {
        "source_file": chunk["source_file"],
        "section": chunk["section"],
        "title": chunk["title"],
        "product": chunk["product"],
        "doc_type": chunk["doc_type"],
        "h1": headers.get("h1", ""),
        "h2": headers.get("h2", ""),
        "h3": headers.get("h3", ""),
        "token_count": int(chunk["token_count"]),
        "content_hash": chunk["content_hash"],
    }


def existing_hashes(collection) -> dict[str, str]:
    """Return {chunk_id: content_hash} for docs already in the collection."""
    if collection.count() == 0:
        return {}
    result = collection.get(include=["metadatas"])
    return {
        cid: (meta or {}).get("content_hash", "")
        for cid, meta in zip(result.get("ids", []), result.get("metadatas") or [])
    }


def run(force: bool = False, dry_run: bool = False) -> RunStats:
    print(f"[embed] chunks   = {CHUNKS_JSONL}")
    print(f"[embed] chroma   = {CHROMA_ROOT}")
    print(f"[embed] model    = {EMBEDDING_MODEL_NAME}")
    print(f"[embed] force    = {force}, dry_run = {dry_run}")
    print()

    stats = RunStats()

    if not CHUNKS_JSONL.exists():
        raise SystemExit(f"chunks file not found: {CHUNKS_JSONL} (run the chunker first)")

    chunks = list(load_chunks(CHUNKS_JSONL))
    stats.total_chunks = len(chunks)
    print(f"[embed] loaded {len(chunks)} chunks from JSONL")

    CHROMA_ROOT.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_ROOT))

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[embed] deleted existing '{COLLECTION_NAME}' (force=True)")
        except Exception as e:
            print(f"[embed] no existing collection to delete ({e.__class__.__name__})")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"[embed] collection has {collection.count()} existing docs")

    prior = {} if force else existing_hashes(collection)

    # Change detection keys on `content_hash`, which is the SHA256 of the raw
    # SOURCE file (set in Phase 1), NOT of the chunk text. So edits to a source
    # file are detected, but changes to *chunker logic* are invisible here —
    # after changing the chunker, re-run with --force to rebuild from scratch.
    to_embed: list[dict] = []
    for c in chunks:
        prior_hash = prior.get(c["chunk_id"])
        if prior_hash and prior_hash == c["content_hash"]:
            stats.unchanged_skipped += 1
        else:
            to_embed.append(c)

    # Orphan reconciliation: any chunk_id currently in Chroma but absent from
    # the freshly generated chunk set is stale and must be deleted. Upsert alone
    # can't handle two cases:
    #   1. An edited file — all its chunks get a new content_hash prefix, so the
    #      new IDs never overwrite the old ones; the old-prefix chunks linger.
    #   2. A deleted file — none of its chunks appear in chunks.jsonl anymore.
    # Without this, incremental re-runs accumulate stale duplicates in the index
    # (only --force avoided it before, which defeats the point of incremental).
    desired_ids = {c["chunk_id"] for c in chunks}
    orphan_ids = [cid for cid in prior if cid not in desired_ids]

    print(f"[embed] plan: {stats.unchanged_skipped} unchanged, "
          f"{len(to_embed)} to (re-)embed, {len(orphan_ids)} orphan(s) to delete")
    print()

    if dry_run:
        print("[embed] dry_run — stopping before any writes")
        return stats

    if orphan_ids:
        collection.delete(ids=orphan_ids)
        stats.orphans_deleted = len(orphan_ids)
        print(f"[embed] deleted {len(orphan_ids)} orphaned chunk(s)")

    if not to_embed:
        msg = "collection is already up to date" if not orphan_ids else "no new chunks to embed"
        print(f"[embed] {msg}")
        print(f"[embed] collection now has {collection.count()} docs")
        return stats

    print(f"[embed] loading '{EMBEDDING_MODEL_NAME}' (first run may download ~130 MB) ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total = len(to_embed)
    done = 0
    while done < total:
        batch = to_embed[done : done + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            normalize_embeddings=True,  # bge recommends L2-normalized embeddings
        ).tolist()
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=texts,
            metadatas=[flatten_metadata(c) for c in batch],
        )
        done += len(batch)
        stats.newly_embedded += len(batch)
        print(f"[embed] progress: {done}/{total}  ({100 * done / total:.0f}%)")

    print()
    print(f"[embed] done — collection now has {collection.count()} docs")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 embedder + Chroma upsert")
    ap.add_argument("--force", action="store_true", help="wipe + rebuild the collection")
    ap.add_argument("--dry-run", action="store_true", help="print plan, embed nothing")
    args = ap.parse_args()
    run(force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
