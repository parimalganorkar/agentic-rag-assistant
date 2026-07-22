"""Response-quality and integrity checks (DataCamp taxonomy, output side).

These are the cheap deterministic checks that run BEFORE any LLM guard, plus one
embedding check. They cover four gaps the security guards never looked at:

  Relevance Validator          — does the answer actually address the question?
  Gibberish Content Filter     — is the answer coherent text at all?
  Duplicate Sentence Eliminator— did the model repeat itself? (REPAIRS, not refuses)
  Source Context Verifier      — do the [S#] citations point at real sources?

Everything here is local: regex, difflib, and the bge-small embedding model that
the retriever already holds in memory. No extra model downloads, no API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

# --- Relevance Validator ----------------------------------------------------
#
# The article's "Relevance Validator" / "Prompt Address Confirmation": compare the
# semantic meaning of the question and the answer. A grounded answer can still be
# an answer to a DIFFERENT question — retrieval drift produces exactly that, and
# no security guard notices because nothing is unsafe or ungrounded.
#
# Threshold is deliberately LOW. bge-small puts a correct technical answer around
# 0.55-0.80 against its question; genuine drift sits below ~0.30. Set at 0.25 so
# this only fires on clear mismatches — it is a backstop, not a quality grader.
RELEVANCE_MIN = 0.25

# Refusals are legitimately dissimilar to the question, so they must be exempt or
# the validator would flag every honest "I don't know" as irrelevant.
_REFUSAL_RE = re.compile(
    r"\b(?:don't|do not|doesn't) have enough information|cannot answer|can't answer", re.I)


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


def relevance_score(question: str, answer: str) -> float | None:
    """Cosine similarity between question and answer. None if not applicable."""
    if not (question or "").strip() or not (answer or "").strip():
        return None
    if _REFUSAL_RE.search(answer):
        return None
    try:
        vecs = _embedder().encode([question, answer], normalize_embeddings=True)
    except Exception:
        return None  # fail open — an embedding hiccup must not refuse a good answer
    return float(vecs[0] @ vecs[1])


# --- Gibberish Content Filter -----------------------------------------------
#
# Catches degenerate generation: repeated single tokens, no sentence structure,
# runaway punctuation. Prose-only — code blocks are stripped first, because valid
# code legitimately looks "unwordlike" and would trip every heuristic here.

_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _prose_only(text: str) -> str:
    return _CODE_FENCE_RE.sub(" ", text or "")


def is_gibberish(answer: str) -> tuple[bool, str]:
    """(is_gibberish, reason). Conservative: only fires on clearly broken output.

    Every threshold here is gated on the text being LONG ENOUGH for the statistic
    to mean anything. A correct terse answer — "The latest version of langgraph is
    1.2.9, released 2026-06-30." — has only ~7 alphabetic words, because digits and
    dates aren't words. An earlier version of this check called that gibberish and
    refused it, which is a far worse failure than missing some real gibberish.
    """
    prose = _prose_only(answer).strip()
    if len(prose) < 200:
        return False, ""  # short answers are legitimate; nothing to measure
    words = _WORD_RE.findall(prose)

    # Lots of characters but almost no words = genuinely degenerate output.
    if len(prose) > 400 and len(words) < 12:
        return True, "long output with almost no words outside code blocks"

    lowered = [w.lower() for w in words]
    # Vocabulary collapse only means something over a decent sample.
    if len(lowered) >= 30:
        unique_ratio = len(set(lowered)) / len(lowered)
        if unique_ratio < 0.18:
            return True, f"vocabulary collapse (unique-word ratio {unique_ratio:.2f})"

    # One token repeated many times in a row = a stuck decoder. Valid prose never
    # does this, so it stays on regardless of length.
    run, longest, prev = 1, 1, None
    for w in lowered:
        run = run + 1 if w == prev else 1
        longest = max(longest, run)
        prev = w
    if longest >= 8:
        return True, f"token repeated {longest}x consecutively"
    return False, ""


# --- Duplicate Sentence Eliminator (REPAIRS) --------------------------------
#
# llama3.1:8b restates the same sentence in different words fairly often. This is
# a repair, not a refusal: dropping a duplicate sentence cannot make an answer
# wrong, so there is no reason to reject the whole response over it.

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def dedupe_sentences(answer: str) -> tuple[str, int]:
    """Remove near-identical repeated sentences from PROSE. Returns (text, n_removed).

    Code fences are held out verbatim — identical lines inside code are normal and
    removing them would corrupt the example.
    """
    if not answer:
        return answer, 0
    blocks = _CODE_FENCE_RE.split(answer)
    fences = _CODE_FENCE_RE.findall(answer)
    removed = 0
    out_blocks = []
    seen: set[str] = set()
    for block in blocks:
        kept = []
        for sent in _SENT_SPLIT_RE.split(block):
            key = _normalise(sent)
            if len(key) > 25 and key in seen:
                removed += 1
                continue
            if len(key) > 25:
                seen.add(key)
            kept.append(sent)
        out_blocks.append(" ".join(k for k in kept if k is not None))
    # reassemble, restoring code fences in their original positions
    rebuilt = out_blocks[0]
    for i, fence in enumerate(fences):
        rebuilt += fence + (out_blocks[i + 1] if i + 1 < len(out_blocks) else "")
    return rebuilt, removed


# --- Source Context Verifier (citation integrity) ---------------------------
#
# The article's "Source Context Verifier". Our prompt asks the model to cite [S1],
# [S2]... A citation pointing at a source that was never retrieved is a fabricated
# attribution: it looks authoritative and is unverifiable by the reader. Purely
# deterministic — we know exactly how many sources we passed in.

_CITE_RE = re.compile(r"\[S(\d+)\]")


def check_citations(answer: str, n_sources: int) -> list[str]:
    """Return citation labels that point outside the retrieved source list."""
    if not answer:
        return []
    bad = []
    for m in _CITE_RE.finditer(answer):
        idx = int(m.group(1))
        if idx < 1 or idx > n_sources:
            label = f"[S{idx}]"
            if label not in bad:
                bad.append(label)
    return bad


# --- PII / secret scrubbing (input side) ------------------------------------
#
# The "Privacy" half of Security and Privacy. A user pasting a real API key into a
# question would otherwise have it copied into prompts, logs and traces. We redact
# before the text travels anywhere. Detection is by KEY SHAPE, not by generic
# high-entropy matching, to keep false positives near zero.

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}")),
    ("langsmith_key", re.compile(r"\blsv2_[A-Za-z0-9_-]{16,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}")),
    ("aws_key_id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{12,}")),
    ("google_key", re.compile(r"\bAIza[A-Za-z0-9_-]{20,}")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Redact credentials/PII. Returns (clean_text, kinds_found)."""
    if not text:
        return text, []
    found: list[str] = []
    out = text
    for kind, pattern in _PII_PATTERNS:
        if pattern.search(out):
            found.append(kind)
            out = pattern.sub(f"[REDACTED_{kind.upper()}]", out)
    return out, found


@dataclass
class QualityResult:
    """Outcome of the deterministic quality tier."""
    answer: str                                   # possibly REPAIRED
    repairs: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)   # unrecoverable → refuse
    relevance: float | None = None
