"""Phase 1c — Relevance filter + metadata enrichment.

After normalization, some pages are still worthless as retrieval targets:
  - stub pages with barely any prose
  - index/navigation pages that are 90% link lists
  - pages whose "content" is one code block with no explanation

This module drops those and enriches the survivors with metadata derived
from the source path (product, doc_type, section) so the downstream chunker
and the retrieval eval have real labels to work with.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ingestion.cleaner.normalizer import NormalizedDoc


REPO_ROOT = Path(__file__).resolve().parents[2]
# The Mintlify nav config that ships with the corpus. Its named groups are the
# docs authors' OWN categorization of every page — a far more honest doc_type
# signal than guessing from path or code density (the repo is organized by
# topic, not by doc-type, so most flat pages have no type in their path).
DOCS_NAV_PATH = REPO_ROOT / "data" / "raw" / "src" / "docs.json"

# nav group label -> canonical doc_type. Grounded in the docs' information
# architecture (see docs.json navigation). Groups not listed fall through to
# path/heuristic rules and ultimately "unknown".
_NAV_GROUP_TO_DOC_TYPE = {
    "Get started": "get-started",
    "Core components": "concept",
    "Conceptual overviews": "concept",
    "LangChain": "concept",
    "LangGraph": "concept",
    "Agent development": "concept",
    "Capabilities": "how-to",
    "Advanced usage": "how-to",
    "Patterns": "how-to",
    "Multi-agent": "how-to",
    "Middleware": "how-to",
    "Frontend": "how-to",
    "Graph API": "how-to",
    "Functional API": "how-to",
    "Test": "how-to",
    "Production": "how-to",
    "LangGraph APIs": "reference",
    "Additional resources": "reference",
    "Integrations": "integration",
}


# --- Relevance thresholds ---------------------------------------------------

MIN_TOKENS = 100                 # docs shorter than this are stubs
ERROR_MIN_TOKENS = 15            # error pages are useful even when terse (a support
                                 # assistant wants them), so exempt errors/ from the
                                 # 100-token stub floor — keep only a tiny empty-guard
MAX_LINK_RATIO = 0.60            # if >60% of lines are pure links, it's a nav page

# NOTE: we deliberately do NOT filter by code ratio. Code-heavy pages are
# exactly what retrieval needs to answer code-specific queries. Retrieval
# quality is protected downstream by hybrid retrieval + reranking + guardrails,
# not by dropping documentation up front.


# --- Regex patterns ---------------------------------------------------------

_LINK_ONLY_LINE = re.compile(r"^\s*[-*]?\s*\[[^\]]+\]\([^)]+\)\s*\.?\s*$")


# --- Data classes -----------------------------------------------------------

@dataclass
class EnrichedDoc:
    """A NormalizedDoc plus metadata derived from the source path."""
    text: str
    title: str
    description: str | None
    sidebar_title: str | None
    source_rel_path: str      # POSIX-style path under data/raw/, e.g. "src/oss/langchain/agents.mdx"
    product: str              # langchain | langgraph | deepagents | shared
    doc_type: str             # concept | how-to | reference | tutorial | get-started | integration | error | unknown
    section: str              # coarse section label, derived from the path
    warnings: list[str] = field(default_factory=list)


@dataclass
class FilterResult:
    keep: EnrichedDoc | None
    dropped_reason: str | None


# --- Relevance heuristics ---------------------------------------------------

def _approx_token_count(text: str) -> int:
    """A cheap word-based token proxy — good enough for a min-length filter."""
    return len(text.split())


def _link_only_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    link_lines = sum(1 for ln in lines if _LINK_ONLY_LINE.match(ln))
    return link_lines / len(lines)


# --- Metadata derivation ----------------------------------------------------

def _derive_product(parts: tuple[str, ...]) -> str:
    # parts is relative to data/raw/, e.g. ("src", "oss", "langchain", "agents.mdx")
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "oss":
        sub = parts[2]
        if sub in {"langchain", "langgraph", "deepagents"}:
            return sub
        if sub in {"concepts", "reference", "integrations"}:
            return "shared"
    return "unknown"


@lru_cache(maxsize=1)
def _nav_group_index() -> dict[str, str]:
    """Build {page-stem -> nav group label} from the corpus's docs.json.

    Page stems are normalized to '<product>/<path>' (e.g. 'langchain/agents'),
    matching how we key our own files. The nav uses 'oss/python/langchain/...'
    paths (the Python-language variant of each page); we strip that prefix.
    Returns {} if docs.json is missing so the cleaner still works without it.
    """
    if not DOCS_NAV_PATH.exists():
        return {}
    try:
        data = json.loads(DOCS_NAV_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    index: dict[str, str] = {}

    def _norm(page: str) -> str | None:
        if page.startswith("oss/python/"):
            page = page[len("oss/python/"):]
        elif page.startswith("oss/javascript/"):
            return None  # JS variant — out of our corpus scope
        elif page.startswith("oss/"):
            page = page[len("oss/"):]
        else:
            return None
        return page if page.startswith(("langchain/", "langgraph/", "concepts/")) else None

    def _walk(node, group: str | None) -> None:
        if isinstance(node, dict):
            g = node.get("group", group)
            for value in node.values():
                _walk(value, g)
        elif isinstance(node, list):
            for item in node:
                _walk(item, group)
        elif isinstance(node, str):
            stem = _norm(node)
            if stem and group and stem not in index:
                index[stem] = group

    _walk(data.get("navigation", {}), None)
    return index


def _path_stem(parts: tuple[str, ...]) -> str:
    """('src','oss','langchain','agents.mdx') -> 'langchain/agents'."""
    rest = parts[2:] if len(parts) >= 3 and parts[0] == "src" and parts[1] == "oss" else parts
    stem = "/".join(rest)
    for ext in (".mdx", ".md"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return stem


def _derive_doc_type(parts: tuple[str, ...]) -> str:
    """Classify a page's doc_type, grounded in the docs' OWN navigation.

    The langchain-ai/docs repo organizes files by topic, not by doc-type, so the
    path alone can't tell concept from how-to for the ~60 flat langchain/langgraph
    pages. Instead we map each page to its docs.json nav group (the authors'
    categorization) and translate that to a canonical type. High-precision path
    rules (errors, concepts, integrations) run first as a fallback for pages the
    nav doesn't cover; the two product 'overview' landing pages default to
    get-started."""
    segments = set(parts)
    # Path rules first — errors/ pages aren't in the nav, and these are unambiguous.
    if "errors" in segments:
        return "error"
    if "reference" in segments:
        return "reference"

    # Authoritative: the docs' own nav group.
    group = _nav_group_index().get(_path_stem(parts))
    if group is not None:
        return _NAV_GROUP_TO_DOC_TYPE.get(group, "unknown")

    # Fallbacks for anything the nav didn't cover.
    if "concepts" in segments:
        return "concept"
    if "integrations" in segments:
        return "integration"
    if "tutorials" in segments or "tutorial" in segments:
        return "tutorial"
    if "how-tos" in segments or "how-to" in segments:
        return "how-to"
    stem = _path_stem(parts)
    if stem in ("langchain/overview", "langgraph/overview"):
        return "get-started"
    return "unknown"


def _derive_section(parts: tuple[str, ...]) -> str:
    """A coarse section label — the parent folder relative to src/oss/,
    joined by '/'. e.g. 'langchain/multi-agent' or 'langgraph/frontend'."""
    if len(parts) < 3 or parts[0] != "src" or parts[1] != "oss":
        return "unknown"
    # everything between src/oss/ and the filename, path-joined
    middle = parts[2:-1]
    return "/".join(middle) if middle else parts[2]


# --- Public API -------------------------------------------------------------

def filter_and_enrich(
    doc: NormalizedDoc,
    rel_path: Path,
) -> FilterResult:
    """Decide whether to keep `doc`; if so, enrich it with path-derived metadata."""

    # Error pages are high-value for a support assistant even when short, so they
    # get a much lower length floor than regular docs.
    is_error = "errors" in rel_path.parts
    min_tokens = ERROR_MIN_TOKENS if is_error else MIN_TOKENS

    token_count = _approx_token_count(doc.text)
    if token_count < min_tokens:
        return FilterResult(keep=None, dropped_reason=f"stub: only {token_count} tokens")

    link_r = _link_only_ratio(doc.text)
    if link_r > MAX_LINK_RATIO:
        return FilterResult(keep=None, dropped_reason=f"nav page: {link_r:.0%} link-only lines")

    parts = rel_path.parts
    enriched = EnrichedDoc(
        text=doc.text,
        title=doc.title,
        description=doc.description,
        sidebar_title=doc.sidebar_title,
        source_rel_path=rel_path.as_posix(),
        product=_derive_product(parts),
        doc_type=_derive_doc_type(parts),
        section=_derive_section(parts),
        warnings=list(doc.warnings),
    )
    return FilterResult(keep=enriched, dropped_reason=None)


if __name__ == "__main__":
    # Quick end-to-end check on one file.
    from ingestion.cleaner.normalizer import normalize_file

    repo_root = Path(__file__).resolve().parents[2]
    raw_root = repo_root / "data" / "raw"
    sample = raw_root / "src" / "oss" / "langchain" / "agents.mdx"

    doc = normalize_file(sample)
    result = filter_and_enrich(doc, sample.relative_to(raw_root))

    if result.keep is None:
        print(f"DROPPED: {result.dropped_reason}")
    else:
        e = result.keep
        print(f"KEPT: {e.source_rel_path}")
        print(f"  title:    {e.title}")
        print(f"  product:  {e.product}")
        print(f"  doc_type: {e.doc_type}")
        print(f"  section:  {e.section}")
        print(f"  tokens:   ~{_approx_token_count(e.text)}")
