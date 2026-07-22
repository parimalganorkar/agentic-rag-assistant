"""Phase 1 orchestrator — chains detector → normalizer → filter → manifest.

Reads from data/raw/ and writes plain-Markdown output to data/cleaned/, along
with a manifest.json describing what was processed. Incremental by SHA256:
files whose raw bytes haven't changed since the last run are skipped.

Usage:
    python -m ingestion.cleaner.run                # normal run
    python -m ingestion.cleaner.run --force        # ignore manifest, reprocess all
    python -m ingestion.cleaner.run --dry-run      # show what would happen, write nothing
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.cleaner.detector import (
    CategorizedFile,
    FileCategory,
    categorize_files,
    summarize,
)
from ingestion.cleaner.filter import filter_and_enrich
from ingestion.cleaner.manifest import (
    Manifest,
    ManifestEntry,
    detect_source_commit,
    hash_file,
    load_manifest,
    now_utc_iso,
    save_manifest,
)
from ingestion.cleaner.normalizer import normalize_file


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw"
CLEANED_ROOT = REPO_ROOT / "data" / "cleaned"
MANIFEST_PATH = CLEANED_ROOT / "manifest.json"


@dataclass
class RunStats:
    scanned: int = 0
    included_by_detector: int = 0
    unchanged_skipped: int = 0
    newly_processed: int = 0
    dropped_by_filter: int = 0
    errored: int = 0
    removed: int = 0
    dropped_reasons: dict[str, int] = field(default_factory=dict)


def _target_path_for(source_rel: Path) -> Path:
    """Where the cleaned .md for a given raw source goes.
    Always writes .md regardless of the source extension."""
    return CLEANED_ROOT / source_rel.with_suffix(".md")


def _process_one(
    entry: CategorizedFile,
    content_hash: str,
    source_commit: str | None,
    dry_run: bool,
) -> tuple[ManifestEntry | None, str]:
    """Normalize + filter + write one file. Returns (manifest_entry, status_desc)."""
    try:
        doc = normalize_file(entry.abs_path)
    except Exception as e:  # noqa: BLE001 — we want to catch everything at the boundary
        return None, f"error:normalize:{type(e).__name__}: {e}"

    result = filter_and_enrich(doc, entry.rel_path)
    if result.keep is None:
        # Record dropped files in the manifest too, so re-runs don't retry them
        # unless the source content changed.
        me = ManifestEntry(
            source_rel_path=entry.rel_path.as_posix(),
            target_rel_path="",
            original_format=entry.rel_path.suffix,
            content_hash=content_hash,
            source_commit=source_commit,
            ingested_at=now_utc_iso(),
            status=f"dropped:{result.dropped_reason}",
        )
        return me, f"dropped:{result.dropped_reason}"

    enriched = result.keep
    target = _target_path_for(entry.rel_path)

    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(enriched.text, encoding="utf-8")

    me = ManifestEntry(
        source_rel_path=entry.rel_path.as_posix(),
        target_rel_path=target.relative_to(REPO_ROOT).as_posix(),
        original_format=entry.rel_path.suffix,
        content_hash=content_hash,
        source_commit=source_commit,
        ingested_at=now_utc_iso(),
        status="ok",
        title=enriched.title,
        product=enriched.product,
        doc_type=enriched.doc_type,
        section=enriched.section,
    )
    return me, "ok"


def run(force: bool = False, dry_run: bool = False) -> RunStats:
    stats = RunStats()
    print(f"[cleaner] raw    = {RAW_ROOT}")
    print(f"[cleaner] cleaned = {CLEANED_ROOT}")
    print(f"[cleaner] force  = {force}, dry_run = {dry_run}\n")

    # 1. Categorize every file
    all_files = categorize_files(RAW_ROOT)
    stats.scanned = len(all_files)
    counts = summarize(all_files)
    print("[detector] category counts:")
    for cat, n in counts.items():
        print(f"  {cat.value:28} {n:>6}")
    print()

    included = [f for f in all_files if f.category == FileCategory.INCLUDE]
    stats.included_by_detector = len(included)

    # 2. Load prior manifest & git commit
    prior = Manifest() if force else load_manifest(MANIFEST_PATH)
    source_commit = detect_source_commit(RAW_ROOT)

    new_manifest = Manifest(source_commit=source_commit)
    seen_rel_paths: set[str] = set()

    # 3. Process each included file
    for entry in included:
        rel_posix = entry.rel_path.as_posix()
        seen_rel_paths.add(rel_posix)
        current_hash = hash_file(entry.abs_path)

        prior_entry = prior.entries.get(rel_posix)
        if (
            not force
            and prior_entry is not None
            and prior_entry.content_hash == current_hash
            and prior_entry.status == "ok"
            and (REPO_ROOT / prior_entry.target_rel_path).exists()
        ):
            # unchanged and cleaned file still on disk — reuse manifest entry
            new_manifest.entries[rel_posix] = prior_entry
            stats.unchanged_skipped += 1
            continue

        me, status = _process_one(entry, current_hash, source_commit, dry_run)
        if me is not None:
            new_manifest.entries[rel_posix] = me

        if status == "ok":
            stats.newly_processed += 1
        elif status.startswith("dropped"):
            stats.dropped_by_filter += 1
            reason = status.split(":", 1)[1] if ":" in status else status
            stats.dropped_reasons[reason] = stats.dropped_reasons.get(reason, 0) + 1
        else:
            stats.errored += 1
            print(f"  ERROR {rel_posix}: {status}")

    # 4. Handle removed files (were in prior manifest, gone now)
    for rel_posix, prior_entry in prior.entries.items():
        if rel_posix not in seen_rel_paths:
            stats.removed += 1
            if not dry_run and prior_entry.status == "ok" and prior_entry.target_rel_path:
                target = REPO_ROOT / prior_entry.target_rel_path
                if target.exists():
                    target.unlink()

    # 5. Write manifest
    if not dry_run:
        save_manifest(new_manifest, MANIFEST_PATH)

    # 6. Report
    print("[summary]")
    print(f"  scanned files (all)         {stats.scanned:>6}")
    print(f"  included by detector        {stats.included_by_detector:>6}")
    print(f"  unchanged (skipped)         {stats.unchanged_skipped:>6}")
    print(f"  newly processed             {stats.newly_processed:>6}")
    print(f"  dropped by filter           {stats.dropped_by_filter:>6}")
    print(f"  removed (gone from raw/)    {stats.removed:>6}")
    print(f"  errored                     {stats.errored:>6}")
    if stats.dropped_reasons:
        print("\n  drop reasons:")
        for reason, n in sorted(stats.dropped_reasons.items(), key=lambda x: -x[1]):
            print(f"    {n:>4}  {reason}")

    kept = stats.newly_processed + stats.unchanged_skipped
    print(f"\n[cleaner] cleaned corpus size: {kept} files")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 data cleaner")
    ap.add_argument("--force", action="store_true", help="ignore manifest, reprocess everything")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, write nothing")
    args = ap.parse_args()

    stats = run(force=args.force, dry_run=args.dry_run)
    return 1 if stats.errored else 0


if __name__ == "__main__":
    sys.exit(main())