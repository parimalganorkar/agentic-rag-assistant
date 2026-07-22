"""Phase 2 orchestrator — load cleaned docs, chunk them, write chunks.jsonl.

Reads `data/cleaned/manifest.json`, streams every OK-status doc through the
chunker, and writes one JSON line per chunk to `data/processed/chunks.jsonl`.

Usage:
    python -m ingestion.run
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from ingestion.chunker import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, chunk_doc
from ingestion.loader import load_cleaned_docs


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
CHUNKS_JSONL = PROCESSED_ROOT / "chunks.jsonl"


def run() -> int:
    print(f"[ingest] chunk_size={CHUNK_SIZE_TOKENS} overlap={CHUNK_OVERLAP_TOKENS}")
    print(f"[ingest] output    = {CHUNKS_JSONL}\n")

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_chunks = 0
    token_hist: list[int] = []
    per_product = Counter()
    per_doc_chunks = Counter()

    with CHUNKS_JSONL.open("w", encoding="utf-8") as out:
        for doc in load_cleaned_docs():
            total_docs += 1
            chunks = chunk_doc(doc)
            per_doc_chunks[doc.source_file] = len(chunks)
            for c in chunks:
                out.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                token_hist.append(c.token_count)
                per_product[c.product] += 1
                total_chunks += 1

    if not total_chunks:
        print("[ingest] no chunks produced — nothing to write")
        return 1

    avg_tokens = sum(token_hist) / len(token_hist)
    sorted_tokens = sorted(token_hist)
    p50 = sorted_tokens[len(sorted_tokens) // 2]
    p95 = sorted_tokens[int(len(sorted_tokens) * 0.95)]
    max_tokens = sorted_tokens[-1]
    empty_or_tiny = sum(1 for t in token_hist if t < 20)

    print("[summary]")
    print(f"  docs read              {total_docs:>6}")
    print(f"  chunks written         {total_chunks:>6}")
    print(f"  avg chunks per doc     {total_chunks / total_docs:>6.1f}")
    print(f"  tokens/chunk avg       {avg_tokens:>6.0f}")
    print(f"  tokens/chunk p50/p95   {p50:>6} / {p95}")
    print(f"  tokens/chunk max       {max_tokens:>6}")
    print(f"  tiny chunks (<20 tok)  {empty_or_tiny:>6}")
    print(f"\n  chunks per product:")
    for product, n in sorted(per_product.items(), key=lambda x: -x[1]):
        print(f"    {product:>12}  {n:>5}")

    # Show top 3 docs by chunk count — sanity check that no single doc is
    # blowing up our chunk store
    top_chunky = per_doc_chunks.most_common(3)
    print(f"\n  top 3 docs by chunk count:")
    for src, n in top_chunky:
        print(f"    {n:>3}  {src}")

    return 0


if __name__ == "__main__":
    sys.exit(run())
