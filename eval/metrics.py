"""Retrieval + routing evaluation metrics.

Every metric here is deterministic and needs no LLM. That keeps eval runs
cheap, fast, and reproducible — you can regenerate every number in this
file in seconds from the testset + a saved retrieval pipeline output.

RETRIEVAL METRICS (compare a ranked list of retrieved files against a set of
expected-relevant files):

  precision_at_k(retrieved, expected, k)
      Of the top-K retrieved files, what fraction are relevant?
      Best when you want the top of the ranking to be dense with hits.

  recall_at_k(retrieved, expected, k)
      Of the expected-relevant files, what fraction appeared in top-K?
      Best when the total set of relevant docs matters (e.g. coverage).

  hit_at_k(retrieved, expected, k)
      Binary: did ANY relevant file appear in top-K? 1 or 0.
      Best for "the system found something useful at all" questions.

  mrr(retrieved, expected)
      Mean Reciprocal Rank — 1 / rank of first relevant hit (0 if none).
      Rewards putting a relevant file at position 1 more than at position 5.

ROUTING METRIC (compare a set of predicted paths to a set of expected ones):

  jaccard(predicted, expected)
      |A ∩ B| / |A ∪ B|. Handles single- and multi-label routing uniformly.

For an "adversarial" question with expected_source_files=[], we treat any
non-empty retrieved list as a partial miss — the retriever WILL return
something even if nothing is relevant. The scorer records this as a special
"adversarial" case and reports the average max-similarity of the top-1 hit
so you can eyeball whether the retriever's own confidence is low (which is
the signal Phase 8's guardrails should use to refuse an answer).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================================
# Retrieval metrics
# ============================================================================

def precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """|top-K ∩ expected| / K."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in expected)
    return hits / min(k, len(top_k))


def recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """|top-K ∩ expected| / |expected|. Adversarial (expected empty) → 1.0."""
    if not expected:
        # If nothing is expected, we can't miss anything → perfect recall
        return 1.0
    top_k_set = set(retrieved[:k])
    return len(top_k_set & expected) / len(expected)


def hit_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    """1.0 if any relevant file is in top-K, else 0.0."""
    if not expected:
        return 0.0
    return 1.0 if any(r in expected for r in retrieved[:k]) else 0.0


def mrr(retrieved: list[str], expected: set[str]) -> float:
    """Reciprocal rank of first hit; 0.0 if none present."""
    if not expected:
        return 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in expected:
            return 1.0 / rank
    return 0.0


# ============================================================================
# Routing metric
# ============================================================================

def jaccard(predicted: set[str], expected: set[str]) -> float:
    """|A ∩ B| / |A ∪ B|. Both empty → 1.0 (both agree there's nothing)."""
    if not predicted and not expected:
        return 1.0
    union = predicted | expected
    if not union:
        return 0.0
    return len(predicted & expected) / len(union)


# ============================================================================
# Aggregate result bundle
# ============================================================================

@dataclass
class RetrievalScore:
    """Per-question retrieval score (file-level metrics)."""
    question_id: str
    question: str
    retrieved_files: list[str]      # unique source_files from top-K chunks, in rank order
    expected_files: set[str]
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    hit_at_5: float = 0.0
    mrr_score: float = 0.0
    is_adversarial: bool = False    # expected_files was empty


@dataclass
class RetrievalReport:
    """Aggregate over all questions in a testset for one retrieval pipeline."""
    pipeline_name: str
    per_question: list[RetrievalScore] = field(default_factory=list)

    def _avg(self, attr: str, exclude_adversarial: bool = True) -> float:
        rows = [
            getattr(r, attr)
            for r in self.per_question
            if not (exclude_adversarial and r.is_adversarial)
        ]
        return sum(rows) / len(rows) if rows else 0.0

    @property
    def n(self) -> int:
        return len(self.per_question)

    @property
    def n_adversarial(self) -> int:
        return sum(1 for r in self.per_question if r.is_adversarial)

    @property
    def avg_precision_at_5(self) -> float:
        return self._avg("precision_at_5")

    @property
    def avg_recall_at_5(self) -> float:
        return self._avg("recall_at_5")

    @property
    def avg_hit_at_5(self) -> float:
        return self._avg("hit_at_5")

    @property
    def avg_mrr(self) -> float:
        return self._avg("mrr_score")


def score_retrieval(
    question_id: str,
    question: str,
    retrieved_files: list[str],
    expected_files: set[str],
    k: int = 5,
) -> RetrievalScore:
    """Compute all retrieval metrics for one question."""
    is_adversarial = not expected_files
    return RetrievalScore(
        question_id=question_id,
        question=question,
        retrieved_files=retrieved_files,
        expected_files=expected_files,
        precision_at_5=precision_at_k(retrieved_files, expected_files, k),
        recall_at_5=recall_at_k(retrieved_files, expected_files, k),
        hit_at_5=hit_at_k(retrieved_files, expected_files, k),
        mrr_score=mrr(retrieved_files, expected_files),
        is_adversarial=is_adversarial,
    )


if __name__ == "__main__":
    # Quick sanity check: hand-compute a few and make sure the numbers agree.
    retrieved = ["a.mdx", "b.mdx", "c.mdx", "d.mdx", "e.mdx"]
    expected = {"b.mdx", "d.mdx", "z.mdx"}

    print(f"retrieved = {retrieved}")
    print(f"expected  = {expected}")
    print(f"  precision@5 = {precision_at_k(retrieved, expected, 5):.3f}  (expected 2/5 = 0.400)")
    print(f"  recall@5    = {recall_at_k(retrieved, expected, 5):.3f}  (expected 2/3 = 0.667)")
    print(f"  hit@5       = {hit_at_k(retrieved, expected, 5):.3f}  (expected 1.000)")
    print(f"  mrr         = {mrr(retrieved, expected):.3f}  (expected 1/2 = 0.500)")

    print()
    print(f"jaccard({{'retrieve'}}, {{'retrieve', 'tool'}}) = {jaccard({'retrieve'}, {'retrieve', 'tool'}):.3f}  (expected 1/2 = 0.500)")
    print(f"jaccard({{'retrieve'}}, {{'retrieve'}}) = {jaccard({'retrieve'}, {'retrieve'}):.3f}  (expected 1.000)")
