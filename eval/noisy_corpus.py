"""Phase 9 — inject / remove the deliberately-bad chunks used to stress-test
retrieval and the guardrails.

The noise goes into BOTH indexes — Chroma (dense) AND chunks.jsonl (BM25).
That matters for the test to mean anything: retrieval fuses dense + sparse with
RRF, and a chunk present in only ONE index gets a single rank contribution, so
it is structurally out-ranked by chunks that appear in both. Injecting only into
Chroma would suppress the noise by construction and we'd "prove" retrieval
ignores noise when really the harness rigged it.

Every inserted chunk carries `is_noise=True` in its Chroma metadata and a
`noise_` id prefix in the JSONL, so removal is exact — the real corpus is never
left polluted and the Phase 6 baselines can't silently drift.

    python -m eval.noisy_corpus --add       # mix the noise in
    python -m eval.noisy_corpus --status    # how many noise chunks are present
    python -m eval.noisy_corpus --purge     # take them back out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
NOISE_FILE = REPO_ROOT / "eval" / "noisy_chunks.json"
CHROMA_ROOT = REPO_ROOT / "data" / "chroma"
CHUNKS_JSONL = REPO_ROOT / "data" / "processed" / "chunks.jsonl"  # the BM25 source
COLLECTION_NAME = "langchain_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
NOISE_ID_PREFIX = "noise_"


def _as_chunk_record(c: dict) -> dict:
    """Shape a noise chunk like a real chunks.jsonl record (what BM25 reads)."""
    return {
        "chunk_id": c["id"],
        "text": c["text"],
        "source_file": c["source_file"],
        "section": c["section"],
        "title": c["title"],
        "product": "noise",
        "doc_type": "noise",
        "headers": {},
        "token_count": len(c["text"].split()),
        "content_hash": "noise",
    }


def _rewrite_jsonl_without_noise() -> int:
    """Drop noise lines from chunks.jsonl. Returns how many were removed."""
    if not CHUNKS_JSONL.exists():
        return 0
    kept, removed = [], 0
    with CHUNKS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if json.loads(line).get("chunk_id", "").startswith(NOISE_ID_PREFIX):
                removed += 1
            else:
                kept.append(line.rstrip("\n"))
    if removed:
        CHUNKS_JSONL.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return removed


def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_ROOT))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _load_noise() -> list[dict]:
    return json.loads(NOISE_FILE.read_text(encoding="utf-8"))["chunks"]


def add() -> None:
    chunks = _load_noise()
    collection = _collection()
    before = collection.count()

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,   # match the real ingestion pipeline exactly
        show_progress_bar=False,
    ).tolist()

    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "is_noise": True,                      # <- the exact removal key
            "noise_kind": c["kind"],               # offtopic | poisoned
            "source_file": c["source_file"],
            "section": c["section"],
            "title": c["title"],
            "product": "noise",
            "doc_type": "noise",
            "token_count": len(c["text"].split()),
            "content_hash": "noise",
            "chunk_id": c["id"],
        } for c in chunks],
    )
    # BM25 side: append to chunks.jsonl (idempotent — clear any prior noise first)
    _rewrite_jsonl_without_noise()
    with CHUNKS_JSONL.open("a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(_as_chunk_record(c), ensure_ascii=False) + "\n")

    kinds = {}
    for c in chunks:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"[noise] added {len(chunks)} chunks {kinds}")
    print(f"[noise] chroma {before} -> {collection.count()}  (+ appended to chunks.jsonl for BM25)")


def purge() -> None:
    collection = _collection()
    before = collection.count()
    collection.delete(where={"is_noise": True})
    after = collection.count()
    removed_jsonl = _rewrite_jsonl_without_noise()
    print(f"[noise] purged {before - after} from chroma ({before} -> {after}), "
          f"{removed_jsonl} from chunks.jsonl")


def status() -> None:
    collection = _collection()
    got = collection.get(where={"is_noise": True}, include=["metadatas"])
    ids = got.get("ids") or []
    kinds = {}
    for m in got.get("metadatas") or []:
        k = (m or {}).get("noise_kind", "?")
        kinds[k] = kinds.get(k, 0) + 1
    jsonl_noise = 0
    if CHUNKS_JSONL.exists():
        with CHUNKS_JSONL.open(encoding="utf-8") as f:
            jsonl_noise = sum(
                1 for line in f
                if line.strip() and json.loads(line).get("chunk_id", "").startswith(NOISE_ID_PREFIX)
            )
    print(f"[noise] chroma total     : {collection.count()}")
    print(f"[noise] noise in chroma  : {len(ids)} {kinds}")
    print(f"[noise] noise in jsonl   : {jsonl_noise} (BM25 source)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 9 noisy-corpus injector")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--add", action="store_true", help="insert the noise chunks")
    group.add_argument("--purge", action="store_true", help="remove all noise chunks")
    group.add_argument("--status", action="store_true", help="report what's present")
    args = ap.parse_args()

    if args.add:
        add()
    elif args.purge:
        purge()
    else:
        status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
