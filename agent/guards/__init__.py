"""Guardrails, organised as exactly two stages.

    check_input(query, chunks)              -> InputVerdict     BEFORE the LLM
    check_output(query, context, answer, n) -> OutputVerdict    AFTER the LLM

Everything else in this package is an implementation detail. The graph calls
these two functions and nothing else.

WHY TWO STAGES
--------------
Guards were previously numbered 1/2/3/3b/4/5 and scattered across six graph
nodes, which said nothing about when they run or what they defend. The checks
themselves were each added to close a MEASURED attack, so none were removed in
this reorganisation — they were regrouped and re-ordered.

STAGE 1 — INPUT (before the LLM). Everything here is deterministic and runs in
microseconds. The goal is that poisoned or unusable material never enters the
prompt at all.

    1. PII / secret scrubbing     user's text may contain a real API key
    2. Prompt-injection shield    retrieved chunks are attacker-controlled text
    3. Source corroboration       drop chunks outvoted by independent documents

STAGE 2 — OUTPUT (after the LLM), ordered CHEAP -> EXPENSIVE so most answers
never reach the paid tier:

    Tier A  deterministic (~0ms)  gibberish, duplicate sentences, citation
                                  integrity, content safety, symbol allowlist,
                                  URL/host safety
    Tier B  embedding   (~10ms)   relevance validator (bge-small, already loaded)
    Tier C  LLM         (~1-3s)   groundedness fact-check, prompt-address check

REPAIR VS REFUSE
----------------
A guard failure is not automatically a refusal. Cosmetic problems are REPAIRED
(duplicate sentences removed, unsupported sentences stripped) because discarding
a good answer over one bad sentence is itself a failure — that behaviour caused
20+ false refusals on the 100-question eval. Only SAFETY failures refuse:
dangerous content, fabricated APIs, injection compliance, or an answer that is
left with nothing substantive after repair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Sequence

from agent import guardrails as _g
from agent.guards import quality as _q

# Re-exported so callers never need to import the primitives module directly.
scrub_pii = _q.scrub_pii
relevance_score = _q.relevance_score
RELEVANCE_MIN = _q.RELEVANCE_MIN


# ============================================================================
# STAGE 1 — INPUT
# ============================================================================

@dataclass
class InputVerdict:
    query: str                                          # PII-scrubbed
    chunks: list[Any] = field(default_factory=list)     # sanitised + corroborated
    pii_found: list[str] = field(default_factory=list)
    injection_flags: list[str] = field(default_factory=list)
    dropped_chunks: list[dict] = field(default_factory=list)
    source_conflicts: list[dict] = field(default_factory=list)


def check_input(query: str, chunks: Sequence[Any] | None = None) -> InputVerdict:
    """Everything that must happen before text reaches the model."""
    clean_query, pii = scrub_pii(query or "")
    hits = list(chunks or [])

    sanitized = _g.sanitize_chunks(hits)
    flags = sorted({p for d in sanitized.dropped for p in d.get("patterns", [])})

    corroborated = _g.corroborate_chunks(sanitized.kept)

    # Both stages remove chunks, for different reasons: injection text (attacker
    # instructions) and outvoted facts (attacker assertions). The caller wants
    # one list of what left the context and why.
    return InputVerdict(
        query=clean_query,
        chunks=corroborated.kept,
        pii_found=pii,
        injection_flags=flags,
        dropped_chunks=list(sanitized.dropped) + list(corroborated.dropped),
        source_conflicts=corroborated.conflicts,
    )


# ============================================================================
# STAGE 2 — OUTPUT
# ============================================================================

# How much of an answer may be stripped as unsupported before the remainder is
# no longer worth shipping. Above this we refuse rather than hand back a stub.
MAX_STRIP_FRACTION = 0.4
MIN_SENTENCES_AFTER_STRIP = 2

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)


@dataclass
class OutputVerdict:
    answer: str                                        # possibly repaired
    action: str = "pass"                               # pass | repair | refuse
    refuse_reason: str = ""
    repairs: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    relevance: float | None = None
    grounded: bool | None = None
    unsupported: list[str] = field(default_factory=list)
    policy_violation: bool | None = None
    policy_reason: str = ""
    backend: str = ""
    degraded: bool = False        # an LLM guard failed open this turn

    @property
    def refused(self) -> bool:
        return self.action == "refuse"


def _strip_unsupported(answer: str, unsupported: Sequence[str]) -> tuple[str, int]:
    """Remove the sentences that the fact-checker called unsupported.

    The verifier returns claims in its OWN words, not verbatim spans, so matching
    is fuzzy: for each flagged claim, drop the prose sentence that resembles it
    most, provided the resemblance is strong enough to be confident. Code fences
    are never touched — a wrong sentence about code is not a reason to delete the
    code example the user actually needs.
    """
    # Many flagged claims means the answer is broadly unsupported, not that it has
    # a bad sentence. Repairing that would hand back a stub of a wrong answer.
    if not answer or not unsupported or len(unsupported) > 3:
        return answer, 0

    fences = _CODE_FENCE_RE.findall(answer)
    placeholder = "\x00FENCE%d\x00"
    masked = answer
    for i, f in enumerate(fences):
        masked = masked.replace(f, placeholder % i, 1)

    # Split on newlines AS WELL AS sentence boundaries. Answers interleave prose
    # and code, and a prose sentence sitting right after a code block often has no
    # ".!?" between it and the block, so a pure sentence split GLUES it to the
    # fence placeholder — and the fence-skip below then refuses to touch it, so an
    # unsupported prose line adjacent to code could never be stripped and the
    # whole answer got refused instead. Newlines separate them cleanly.
    candidates = [s for part in masked.split("\n") for s in _SENT_SPLIT_RE.split(part)]
    to_remove: list[str] = []
    for claim in unsupported:
        c = re.sub(r"^the claim that\s+", "", (claim or "").strip(), flags=re.I).lower()
        if not c:
            continue
        best, best_score = None, 0.0
        for sent in candidates:
            s = sent.strip().lower()
            if not s or "\x00FENCE" in sent:
                continue
            if s in c or c in s:
                score = 0.95
            else:
                score = SequenceMatcher(None, s, c).ratio()
            if score > best_score:
                best, best_score = sent, score
        # 0.55 was loose enough to match a merely SIMILAR sentence and delete
        # correct content. Repair must be conservative: if we are not confident
        # which sentence the verifier meant, refuse instead of silently editing.
        if best is not None and best_score >= 0.62:
            to_remove.append(best)

    if not to_remove:
        return answer, 0
    # Delete the exact matched spans from the masked text — preserves every
    # newline and code block, unlike a split-then-join rebuild.
    n_dropped = 0
    for sent in to_remove:
        if sent in masked:
            masked = masked.replace(sent, "", 1)
            n_dropped += 1
    rebuilt = re.sub(r"\n{3,}", "\n\n", masked)          # tidy the gap left behind
    for i, f in enumerate(fences):
        rebuilt = rebuilt.replace(placeholder % i, f)
    return rebuilt.strip(), n_dropped


def _tidy_after_removal(text: str) -> str:
    """Clean the punctuation debris left when a citation label is deleted."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)          # " ," -> ","
    text = re.sub(r"\b(in|from|see|per)\s+(and|,)\s+", r"\1 ", text, flags=re.I)
    text = re.sub(r"\(\s*\)", "", text)                  # empty parens
    text = re.sub(r",\s*\.", ".", text)
    return text.strip()


def _substantive(text: str) -> bool:
    """Is there still a real answer here after repairs?"""
    prose = _CODE_FENCE_RE.sub(" ", text or "")
    has_code = bool(_CODE_FENCE_RE.search(text or ""))
    words = len(_q._WORD_RE.findall(prose))
    return has_code or words >= 25


def check_output(
    question: str,
    context: str | None,
    answer: str,
    n_sources: int = 0,
    kind: str = "answer",
    injection_seen: bool = False,
) -> OutputVerdict:
    """Run the output stage.

    `context is None` means nothing external entered the prompt (structured tool
    data, or the clarify path) — the LLM tier is skipped as latency for nothing.
    `kind="clarification"` marks a follow-up QUESTION rather than an answer, so
    answer-shaped checks don't judge it. `injection_seen` reports that the input
    stage found injection text this turn, which changes how a failed-open
    verifier is treated.
    """
    v = OutputVerdict(answer=answer or "")

    if not (answer or "").strip():
        v.action, v.refuse_reason = "refuse", "empty answer"
        return v

    # ---- Tier A: deterministic --------------------------------------------
    bad_cites = _q.check_citations(v.answer, n_sources)
    if bad_cites:
        # A citation to a source that was never retrieved is a fabricated
        # attribution. Strip the label rather than the sentence: the prose may
        # still be right, but the reader must not see a false source. Tidy the
        # leftovers, or removing "[S7]" leaves text like "shown in  and , do X".
        for label in bad_cites:
            v.answer = v.answer.replace(label, "")
        v.answer = _tidy_after_removal(v.answer)
        v.repairs.append(f"removed fabricated citations: {', '.join(bad_cites)}")

    gib, why = _q.is_gibberish(v.answer)
    if gib:
        v.action, v.refuse_reason = "refuse", f"incoherent output: {why}"
        return v

    deduped, n_dupes = _q.dedupe_sentences(v.answer)
    if n_dupes:
        v.answer = deduped
        v.repairs.append(f"removed {n_dupes} duplicate sentence(s)")

    unsafe = _g.scan_unsafe_content(v.answer)
    if unsafe:
        v.safety_flags += unsafe
        v.action = "refuse"
        v.refuse_reason = f"unsafe content: {', '.join(unsafe)}"
        return v

    unknown = _g.scan_unknown_symbols(v.answer)
    if unknown:
        v.safety_flags += unknown
        v.action = "refuse"
        v.refuse_reason = f"undocumented API/package: {', '.join(unknown)}"
        return v

    # ---- Tier B: embedding relevance (TELEMETRY ONLY) ---------------------
    #
    # Measured, not assumed: bge-small scored a blatantly off-topic answer at
    # 0.43 against its question, versus 0.81 for a correct one. The bands overlap
    # with normal variation, so no threshold separates them cleanly. Gating on it
    # would refuse good answers while still missing the bad one it was built for.
    # The score is recorded so a threshold can be calibrated from real data later;
    # until then it must not decide anything.
    if kind == "answer":
        v.relevance = relevance_score(question, v.answer)

    if context is None:
        v.action = "repair" if v.repairs else "pass"
        return v

    # ---- Tier C: LLM ------------------------------------------------------
    ground = _g.verify_grounded(question, context, v.answer)
    v.grounded = ground.grounded
    v.unsupported = list(ground.unsupported or [])
    v.backend = ground.backend
    v.degraded = bool(ground.error)

    # FAIL-OPEN IS AN ATTACK SURFACE. verify_grounded returns grounded=True when
    # the verifier errors, so anyone who can break it (exhaust the API key, kill
    # ollama) silently disables the fact check. That trade is acceptable on a
    # CLEAN turn — a dead network is not evidence an answer is wrong. It is NOT
    # acceptable when the input stage already found injection text this turn:
    # that combination is precisely the one an attacker engineers.
    if ground.error and injection_seen:
        v.action = "refuse"
        v.refuse_reason = (
            f"cannot verify this answer ({ground.error}) and the retrieved "
            "sources contained injection text"
        )
        return v

    if not ground.grounded:
        # REPAIR FIRST. Refusing a mostly-correct answer over one unsupported
        # sentence was the single largest source of false refusals measured.
        n_before = len(_SENT_SPLIT_RE.split(_CODE_FENCE_RE.sub(" ", v.answer)))
        stripped, n_dropped = _strip_unsupported(v.answer, v.unsupported)
        fraction = (n_dropped / n_before) if n_before else 1.0
        remaining = len(_SENT_SPLIT_RE.split(_CODE_FENCE_RE.sub(" ", stripped)))
        if (n_dropped and fraction <= MAX_STRIP_FRACTION
                and remaining >= MIN_SENTENCES_AFTER_STRIP and _substantive(stripped)):
            v.answer = stripped
            v.repairs.append(f"removed {n_dropped} unsupported claim(s)")
        else:
            v.action = "refuse"
            v.refuse_reason = "answer is not supported by the retrieved sources"
            return v

    policy = _g.check_output_policy(question, v.answer, None, True)
    v.policy_violation = policy.violation
    v.policy_reason = policy.reason
    if policy.violation:
        v.safety_flags += list(policy.artifacts or [])
        v.action = "refuse"
        v.refuse_reason = policy.reason or "answer contains content unrelated to the question"
        return v

    v.action = "repair" if v.repairs else "pass"
    return v
