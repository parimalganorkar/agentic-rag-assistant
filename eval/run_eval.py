"""Phase 6 — Retrieval evaluation runner.

Runs the same testset through TWO retrieval pipelines:
  A) BASELINE      — dense-only (from Phase 4, `retrieval/dense.py`)
  B) HYBRID+RERANK — dense + BM25 → RRF → cross-encoder (Phase 5, `retrieval/pipeline.py`)

For each pipeline + each question, computes Precision@5, Recall@5, Hit@5,
and MRR. Prints an aggregate comparison so you can point at real numbers.

This is Phase 6's DoD: "you have two real numbers — e.g. faithfulness went
from 0.71 to 0.86 after hybrid + reranking".

Results are also written to eval/results/last_run.json so downstream code
(Phase 8's routing eval, the future write-up) can consume the raw scores.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --k 3         # score top-3 instead of top-5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.metrics import RetrievalReport, RetrievalScore, score_retrieval
from retrieval.dense import dense_search
from retrieval.pipeline import retrieve as hybrid_rerank_retrieve


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = REPO_ROOT / "eval" / "testset.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"


# --- Helpers ----------------------------------------------------------------

def _load_testset(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["questions"]


def _unique_files_in_rank_order(chunks) -> list[str]:
    """A retrieval pipeline returns chunk-level hits, but our expected labels
    are file-level. Collapse the ranked chunks to unique source_files,
    preserving the first-occurrence order."""
    seen: set[str] = set()
    files: list[str] = []
    for c in chunks:
        sf = c.source_file
        if sf and sf not in seen:
            seen.add(sf)
            files.append(sf)
    return files


def _run_one(question: dict, pipeline_fn, pipeline_name: str, k: int) -> tuple[RetrievalScore, list[str]]:
    """Run one question through one pipeline and score it."""
    # Fetch a wider chunk pool so we're likely to hit >=k unique source_files.
    # 15 chunks typically de-duplicates to 8-12 files, safely above k=5.
    chunks = pipeline_fn(question["question"], k=15) if pipeline_name == "hybrid+rerank" else pipeline_fn(question["question"], k=15)
    retrieved_files = _unique_files_in_rank_order(chunks)
    expected_files = set(question.get("expected_source_files", []))
    score = score_retrieval(
        question_id=question["id"],
        question=question["question"],
        retrieved_files=retrieved_files,
        expected_files=expected_files,
        k=k,
    )
    return score, retrieved_files


# --- Reporting --------------------------------------------------------------

def _print_per_question_table(reports: list[RetrievalReport]) -> None:
    """Print a per-question comparison so you can see WHERE each pipeline wins."""
    print()
    print("=" * 120)
    print("PER-QUESTION BREAKDOWN (metric = MRR — higher is better)")
    print("=" * 120)
    header = f"{'id':<5} {'category':<12} {'question':<60} " + \
             " | ".join(f"{r.pipeline_name:>15}" for r in reports)
    print(header)
    print("-" * len(header))

    if not reports or not reports[0].per_question:
        return

    for idx, first_row in enumerate(reports[0].per_question):
        qid = first_row.question_id
        # Trim question to fit column
        q_short = first_row.question
        if len(q_short) > 58:
            q_short = q_short[:55] + "..."
        row = f"{qid:<5} {('adv' if first_row.is_adversarial else 'ok'):<12} {q_short:<60}"
        for r in reports:
            row += f" | {r.per_question[idx].mrr_score:>15.3f}"
        print(row)


def _print_aggregate_table(reports: list[RetrievalReport]) -> None:
    """Print the aggregate metrics side-by-side. This is the DoD headline."""
    print()
    print("=" * 90)
    print("AGGREGATE METRICS — averaged over non-adversarial questions")
    print("=" * 90)

    metric_rows = [
        ("Precision@5", "avg_precision_at_5"),
        ("Recall@5",    "avg_recall_at_5"),
        ("Hit@5",       "avg_hit_at_5"),
        ("MRR",         "avg_mrr"),
    ]

    col_w = 18
    print(f"{'metric':<15}" + "".join(f"{r.pipeline_name:>{col_w}}" for r in reports) + "   delta")
    print("-" * (15 + col_w * len(reports) + 10))
    for label, attr in metric_rows:
        row = f"{label:<15}"
        vals = [getattr(r, attr) for r in reports]
        for v in vals:
            row += f"{v:>{col_w}.3f}"
        if len(vals) == 2:
            delta = vals[1] - vals[0]
            sign = "+" if delta >= 0 else ""
            row += f"   {sign}{delta:.3f}"
        print(row)

    n_ok = reports[0].n - reports[0].n_adversarial
    n_adv = reports[0].n_adversarial
    print()
    print(f"  scored on {n_ok} non-adversarial questions ({n_adv} adversarial excluded from averages)")


def _write_results_json(reports: list[RetrievalReport], k: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "last_run.json"
    payload = {
        "k": k,
        "pipelines": [
            {
                "name": r.pipeline_name,
                "aggregate": {
                    "precision_at_5": r.avg_precision_at_5,
                    "recall_at_5": r.avg_recall_at_5,
                    "hit_at_5": r.avg_hit_at_5,
                    "mrr": r.avg_mrr,
                    "n": r.n,
                    "n_adversarial": r.n_adversarial,
                },
                "per_question": [
                    {
                        "id": s.question_id,
                        "question": s.question,
                        "retrieved_files": s.retrieved_files,
                        "expected_files": sorted(s.expected_files),
                        "precision_at_5": s.precision_at_5,
                        "recall_at_5": s.recall_at_5,
                        "hit_at_5": s.hit_at_5,
                        "mrr": s.mrr_score,
                        "is_adversarial": s.is_adversarial,
                    }
                    for s in r.per_question
                ],
            }
            for r in reports
        ],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# --- Main -------------------------------------------------------------------

def run(k: int = 5) -> int:
    testset = _load_testset(TESTSET_PATH)
    print(f"[eval] loaded {len(testset)} questions from {TESTSET_PATH.name}")

    pipelines = [
        ("baseline (dense)", dense_search),
        ("hybrid+rerank",    hybrid_rerank_retrieve),
    ]
    reports = [RetrievalReport(pipeline_name=name) for name, _ in pipelines]

    for q_idx, question in enumerate(testset, start=1):
        for r_idx, (name, fn) in enumerate(pipelines):
            score, _ = _run_one(question, fn, name, k=k)
            reports[r_idx].per_question.append(score)
        adv_flag = " (adversarial)" if not question.get("expected_source_files") else ""
        print(f"[eval] scored {q_idx}/{len(testset)} — {question['id']}: {question['question'][:60]}{adv_flag}")

    _print_per_question_table(reports)
    _print_aggregate_table(reports)

    out_path = _write_results_json(reports, k=k)
    print(f"\n[eval] wrote raw scores to {out_path.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6 retrieval eval runner")
    ap.add_argument("--k", type=int, default=5, help="top-K for Precision/Recall/Hit metrics (default: 5)")
    args = ap.parse_args()
    return run(k=args.k)


if __name__ == "__main__":
    sys.exit(main())
