"""Phase 1d — Manifest & change tracking.

The manifest maps every raw source path to a record of the last time we
processed it: content hash, output path, timestamps, status. On re-runs the
orchestrator compares each raw file's current SHA256 to the manifest and only
reprocesses files whose hash changed or that were newly added. That's what
makes the Phase 3 Chroma upsert incremental instead of a full reindex.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MANIFEST_VERSION = 1
DEFAULT_MANIFEST_NAME = "manifest.json"


@dataclass
class ManifestEntry:
    source_rel_path: str        # POSIX path under data/raw/
    target_rel_path: str        # POSIX path under data/cleaned/
    original_format: str        # ".mdx" | ".md"
    content_hash: str           # SHA256 of the RAW source bytes
    source_commit: str | None   # git commit of langchain-ai/docs at ingest time
    ingested_at: str            # ISO-8601 UTC
    status: str                 # "ok" | "skipped" | "dropped:<reason>" | "error:<msg>"
    title: str | None = None
    product: str | None = None
    doc_type: str | None = None
    section: str | None = None


@dataclass
class Manifest:
    version: int = MANIFEST_VERSION
    generated_at: str = ""
    source_commit: str | None = None
    entries: dict[str, ManifestEntry] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "version": self.version,
            "generated_at": self.generated_at,
            "source_commit": self.source_commit,
            "entries": {k: asdict(v) for k, v in self.entries.items()},
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)


# --- Hashing ----------------------------------------------------------------

def hash_file(path: Path, chunk_size: int = 65536) -> str:
    """SHA256 of the raw file bytes. Streamed so large files don't blow memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Git commit lookup ------------------------------------------------------

def detect_source_commit(raw_root: Path) -> str | None:
    """Return the HEAD commit SHA of the git repo at raw_root, if it is one.
    We use this so citations can be pinned to a specific docs snapshot."""
    if not (raw_root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(raw_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


# --- Manifest I/O -----------------------------------------------------------

def load_manifest(path: Path) -> Manifest:
    """Load an existing manifest, or return an empty one if the file doesn't exist."""
    if not path.exists():
        return Manifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries_raw = data.get("entries", {})
    entries = {k: ManifestEntry(**v) for k, v in entries_raw.items()}
    return Manifest(
        version=data.get("version", MANIFEST_VERSION),
        generated_at=data.get("generated_at", ""),
        source_commit=data.get("source_commit"),
        entries=entries,
    )


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(manifest.to_json(), encoding="utf-8")


# --- Change diffing ---------------------------------------------------------

@dataclass
class ChangeSet:
    added: list[str] = field(default_factory=list)      # new source paths
    modified: list[str] = field(default_factory=list)   # hash changed
    unchanged: list[str] = field(default_factory=list)  # hash matches manifest
    removed: list[str] = field(default_factory=list)    # in manifest, no longer on disk


def diff_against_manifest(
    current: Iterable[tuple[str, str]],
    manifest: Manifest,
) -> ChangeSet:
    """`current` is an iterable of (source_rel_path, content_hash) tuples for the
    files currently on disk. Returns which of them are new/modified/unchanged,
    plus which manifest entries have disappeared."""
    seen: set[str] = set()
    changes = ChangeSet()

    for rel_path, current_hash in current:
        seen.add(rel_path)
        prior = manifest.entries.get(rel_path)
        if prior is None:
            changes.added.append(rel_path)
        elif prior.content_hash != current_hash:
            changes.modified.append(rel_path)
        else:
            changes.unchanged.append(rel_path)

    for rel_path in manifest.entries:
        if rel_path not in seen:
            changes.removed.append(rel_path)

    return changes


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    # Smoke test: hash our own file and print it.
    self_path = Path(__file__).resolve()
    print(f"hash({self_path.name}) = {hash_file(self_path)}")

    raw_root = self_path.parents[2] / "data" / "raw"
    commit = detect_source_commit(raw_root)
    print(f"source_commit = {commit}")