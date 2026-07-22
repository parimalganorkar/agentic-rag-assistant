"""Phase 2 — Load cleaned Markdown docs together with their manifest metadata.

Phase 1 wrote plain-Markdown files to `data/cleaned/` and a `manifest.json` with
per-file metadata (title, product, doc_type, section, content_hash). This module
joins the two into a stream of `LoadedDoc` objects that Phase 2's chunker
consumes.

Nothing here does format detection — Phase 1 already guaranteed everything is
plain Markdown with no frontmatter, no JSX, no imports. If a downstream chunk
looks broken, the fix goes in the cleaner, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ingestion.cleaner.manifest import Manifest, load_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLEANED_ROOT = REPO_ROOT / "data" / "cleaned"
MANIFEST_PATH = CLEANED_ROOT / "manifest.json"


@dataclass
class LoadedDoc:
    """One cleaned Markdown file plus the metadata Phase 1 attached to it."""
    text: str
    source_file: str        # e.g. "src/oss/langchain/agents.mdx" (the ORIGINAL raw path)
    target_file: str        # e.g. "data/cleaned/src/oss/langchain/agents.md"
    title: str
    product: str            # langchain | langgraph | deepagents | shared
    doc_type: str           # concept | how-to | reference | tutorial | get-started | integration | error | unknown
    section: str            # coarse section label
    content_hash: str       # SHA256 of the raw source (used for chunk_id stability)
    source_commit: str | None = None
    warnings: list[str] = field(default_factory=list)


def load_cleaned_docs(
    cleaned_root: Path = CLEANED_ROOT,
    manifest_path: Path = MANIFEST_PATH,
) -> Iterator[LoadedDoc]:
    """Yield one LoadedDoc per successfully-cleaned file listed in the manifest.

    Files marked as dropped or errored in the manifest are skipped. Files whose
    target `.md` is missing on disk are reported as a warning and skipped —
    that indicates the manifest is stale relative to `data/cleaned/`, and the
    fix is to re-run the cleaner."""
    manifest = load_manifest(manifest_path)

    for source_rel, entry in manifest.entries.items():
        if entry.status != "ok":
            continue

        target = REPO_ROOT / entry.target_rel_path
        if not target.exists():
            print(f"  [loader] WARNING: manifest points to missing file {target}")
            continue

        text = target.read_text(encoding="utf-8")
        yield LoadedDoc(
            text=text,
            source_file=source_rel,
            target_file=entry.target_rel_path,
            title=entry.title or "Untitled",
            product=entry.product or "unknown",
            doc_type=entry.doc_type or "unknown",
            section=entry.section or "unknown",
            content_hash=entry.content_hash,
            source_commit=entry.source_commit,
        )


def load_stats(cleaned_root: Path = CLEANED_ROOT, manifest_path: Path = MANIFEST_PATH) -> dict:
    """Summary counts for a quick sanity check."""
    docs = list(load_cleaned_docs(cleaned_root, manifest_path))
    by_product: dict[str, int] = {}
    total_chars = 0
    for d in docs:
        by_product[d.product] = by_product.get(d.product, 0) + 1
        total_chars += len(d.text)
    return {
        "total_docs": len(docs),
        "by_product": by_product,
        "total_chars": total_chars,
        "avg_chars_per_doc": total_chars // max(len(docs), 1),
    }


if __name__ == "__main__":
    stats = load_stats()
    print("[loader] summary:")
    for k, v in stats.items():
        print(f"  {k:20} {v}")
