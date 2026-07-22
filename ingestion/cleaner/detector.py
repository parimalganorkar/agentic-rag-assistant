"""Phase 1a — Format detection & categorization.

Walks data/raw/ and decides, for every file, whether it's a documentation source
we want to clean, or noise we should skip. The categorization rules are
scoping-level policy (which parts of the langchain-ai/docs corpus become our
RAG corpus), not per-file heuristics — those live in filter.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class FileCategory(str, Enum):
    INCLUDE = "include"
    SKIP_IRRELEVANT_PATH = "skip_irrelevant_path"
    SKIP_NAMED_COLLISION = "skip_named_collision"
    SKIP_NON_TEXT = "skip_non_text"
    SKIP_OUTSIDE_OSS = "skip_outside_oss"


@dataclass(frozen=True)
class CategorizedFile:
    abs_path: Path
    rel_path: Path
    category: FileCategory
    reason: str


# Path prefixes (under data/raw/) whose MDX/MD files we want in the corpus.
# See docs/roadmap.md Phase 1 for the reasoning behind this scope.
INCLUDE_ROOTS: tuple[tuple[str, ...], ...] = (
    ("src", "oss", "langchain"),
    ("src", "oss", "langgraph"),
    ("src", "oss", "concepts"),
)

CONTENT_EXTENSIONS: frozenset[str] = frozenset({".mdx", ".md"})

# Defense-in-depth: never treat these as documentation, even if they appear
# inside an included root. `data/raw/CLAUDE.md` and `AGENTS.md` are corpus files
# from the source repo but their names collide with our project memory.
COLLISION_NAMES: frozenset[str] = frozenset({"CLAUDE.md", "AGENTS.md"})

# Folder segment names that indicate non-prose content even when nested inside
# an included root (e.g. src/oss/langchain/multi-agent/images/).
NON_PROSE_SEGMENTS: frozenset[str] = frozenset({"images", "mermaid"})


def _starts_with(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(parts) >= len(prefix) and parts[: len(prefix)] == prefix


def _has_segment(parts: tuple[str, ...], segments: frozenset[str]) -> bool:
    return any(part in segments for part in parts)


def categorize_one(rel_path: Path) -> tuple[FileCategory, str]:
    """Classify a single relative path. Pure function — no filesystem access."""
    parts = rel_path.parts
    name = rel_path.name
    suffix = rel_path.suffix

    if name in COLLISION_NAMES:
        return (
            FileCategory.SKIP_NAMED_COLLISION,
            f"filename {name!r} collides with our project memory",
        )

    if not _starts_with(parts, ("src", "oss")):
        return (
            FileCategory.SKIP_OUTSIDE_OSS,
            "outside src/oss/ scope",
        )

    if suffix not in CONTENT_EXTENSIONS:
        return (
            FileCategory.SKIP_NON_TEXT,
            f"non-text extension {suffix!r}",
        )

    if _has_segment(parts, NON_PROSE_SEGMENTS):
        return (
            FileCategory.SKIP_NON_TEXT,
            "inside an images/ or mermaid/ subfolder",
        )

    if not any(_starts_with(parts, root) for root in INCLUDE_ROOTS):
        return (
            FileCategory.SKIP_IRRELEVANT_PATH,
            "under src/oss/ but not in a whitelisted subfolder",
        )

    return (FileCategory.INCLUDE, "matches an INCLUDE_ROOT prefix")


def categorize_files(raw_root: Path) -> list[CategorizedFile]:
    """Walk raw_root and return one CategorizedFile per file found."""
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root does not exist or is not a dir: {raw_root}")

    results: list[CategorizedFile] = []
    for abs_path in raw_root.rglob("*"):
        if not abs_path.is_file():
            continue
        rel_path = abs_path.relative_to(raw_root)
        category, reason = categorize_one(rel_path)
        results.append(
            CategorizedFile(
                abs_path=abs_path,
                rel_path=rel_path,
                category=category,
                reason=reason,
            )
        )
    return results


def summarize(results: Iterable[CategorizedFile]) -> dict[FileCategory, int]:
    counts: dict[FileCategory, int] = {c: 0 for c in FileCategory}
    for r in results:
        counts[r.category] += 1
    return counts


if __name__ == "__main__":
    # Quick manual sanity check when running the file directly.
    import sys

    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    files = categorize_files(raw)
    counts = summarize(files)

    print(f"Scanned {len(files)} files under {raw}\n")
    for cat, n in counts.items():
        print(f"  {cat.value:28} {n:>6}")

    include_count = counts[FileCategory.INCLUDE]
    print(f"\nWill process {include_count} files. Target range: 150–250.")
    if not (100 <= include_count <= 400):
        print("WARNING: outside expected range — review INCLUDE_ROOTS")
        sys.exit(1)