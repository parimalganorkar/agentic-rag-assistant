"""Phase 6 (extension) — RAGAS LLM-judged evaluation.

Complements the deterministic file-level metrics in `eval/run_eval.py` with
three reference-free RAGAS metrics scored by a cloud judge LLM (Gemini):

  Faithfulness
      Does the generated ANSWER stay grounded in the RETRIEVED CONTEXTS?
      Every claim is checked against the contexts; unsupported claims drop
      the score. Best signal for "is the RAG hallucinating."

  ResponseRelevancy  (formerly answer_relevancy)
      Does the ANSWER address the QUESTION, or is it off-topic / evasive?
      Uses embeddings to compare the answer to synthetic reverse-generated
      questions.

  LLMContextPrecisionWithoutReference
      Are the RETRIEVED CONTEXTS relevant to the question, weighted by
      rank? The reference-free variant asks the judge LLM directly instead
      of comparing to a ground-truth answer.

Why reference-free
------------------
Our `testset.json` has `expected_source_files` (used by the deterministic
eval) but no ground-truth ANSWERS. That rules out the reference-based
RAGAS metrics (context_recall, answer_correctness). The three above are
the strongest reference-free signals RAGAS ships.

Model roles
-----------
Answerer: llama3.1:8b via Ollama (local) — the actual pipeline under test.
Judge:    gemini-2.5-flash (cloud) — a DIFFERENT model from the answerer, so it
          can't self-favour, and it emits reliable structured JSON (the local
          llama judge produced frequent NaN parse failures). Needs GEMINI_API_KEY
          in .env with BILLING ENABLED — the free tier's ~20 RPM cap throttles a
          100-question run. Eval-only: serving stays fully local.
Embed:    bge-small (local) for the embedding-based ResponseRelevancy metric.

Outputs
-------
  eval/results/last_ragas.json         # per-question + aggregate scores
  eval/results/rag_answers.jsonl       # cache of generated answers per pipeline

Usage
-----
  python -m eval.ragas_eval
  python -m eval.ragas_eval --regenerate   # ignore the answer cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import os

import ollama
from dotenv import load_dotenv

from rag.naive import build_prompt, SYSTEM_PROMPT, LLM_MODEL, OLLAMA_HOST
from retrieval.dense import RetrievedChunk, dense_search
from retrieval.pipeline import retrieve as hybrid_rerank_retrieve


# --- Paths + config ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = REPO_ROOT / "eval" / "testset.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
ANSWERS_CACHE = RESULTS_DIR / "rag_answers.jsonl"
RESULTS_PATH = RESULTS_DIR / "last_ragas.json"

load_dotenv(REPO_ROOT / ".env")

# Answerer = LLM_MODEL (llama3.1:8b, local) — the pipeline under test.
# Judge    = JUDGE_MODEL (gemini, cloud)    — different model → no self-bias,
#            and reliable JSON (the local llama judge produced NaN parse errors).
# gemini-2.5-flash 404s on newer API keys ("no longer available to new users").
# gemini-flash-lite-latest is a current, fast, cheap flash available on this key
# (verified by a live test call) — fast matters because judging is ~500+ calls.
JUDGE_MODEL = "gemini-flash-lite-latest"


# --- Answer generation (cached) --------------------------------------------

@dataclass
class RagRow:
    """One (question, pipeline) sample. Matches what RAGAS expects."""
    question_id: str
    pipeline: str
    question: str
    answer: str
    contexts: list[str]
    retrieved_files: list[str]
    is_adversarial: bool = False
    # Guard telemetry — only populated by the agent pipeline. The retrieval-only
    # pipelines have no guards in the path, so this stays None for them.
    guards: dict | None = None


def _cache_key(question_id: str, pipeline: str) -> str:
    return f"{pipeline}::{question_id}"


def _load_cache() -> dict[str, RagRow]:
    """Load previously-generated (question, pipeline) answers, if any."""
    if not ANSWERS_CACHE.exists():
        return {}
    out: dict[str, RagRow] = {}
    for line in ANSWERS_CACHE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        row = RagRow(**d)
        out[_cache_key(row.question_id, row.pipeline)] = row
    return out


def _append_cache(row: RagRow) -> None:
    ANSWERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with ANSWERS_CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")


def _generate_answer(question: str, hits: list[RetrievedChunk]) -> str:
    """Retrieve → stuff → call Ollama. Same logic as rag/naive.answer(),
    but takes pre-fetched hits so we don't re-run retrieval."""
    if not hits:
        return "I don't have enough information in the retrieved docs to answer that."
    client = ollama.Client(host=OLLAMA_HOST)
    user_prompt = build_prompt(question, hits)
    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        options={"num_predict": 1024, "temperature": 0.0},
    )
    return (response.get("message", {}).get("content") or "").strip()


def build_ragas_rows(
    testset: list[dict],
    pipelines: list[tuple[str, Callable]],
    regenerate: bool = False,
) -> dict[str, list[RagRow]]:
    """For each (pipeline, question) run retrieval + generation, using the
    on-disk cache to skip anything already computed."""
    cache = {} if regenerate else _load_cache()
    if regenerate and ANSWERS_CACHE.exists():
        ANSWERS_CACHE.unlink()

    by_pipeline: dict[str, list[RagRow]] = {name: [] for name, _ in pipelines}
    total = len(pipelines) * len(testset)
    done = 0

    for name, retrieve_fn in pipelines:
        for q in testset:
            done += 1
            key = _cache_key(q["id"], name)
            if key in cache:
                by_pipeline[name].append(cache[key])
                print(f"  [{done}/{total}] cache hit  {name}  {q['id']}")
                continue

            hits = retrieve_fn(q["question"], k=5)
            answer = _generate_answer(q["question"], hits)
            row = RagRow(
                question_id=q["id"],
                pipeline=name,
                question=q["question"],
                answer=answer,
                contexts=[h.text for h in hits],
                retrieved_files=[h.source_file for h in hits],
                is_adversarial=not q.get("expected_source_files"),
            )
            by_pipeline[name].append(row)
            _append_cache(row)
            print(f"  [{done}/{total}] generated  {name}  {q['id']}  ({len(hits)} ctx)")

    return by_pipeline


# --- Agent pipeline (the system we actually serve) --------------------------
#
# The two pipelines above call rag.naive directly, so they measure retrieval +
# generation with NO guards in the path. That is not what the project serves any
# more: every answer now passes GUARDS 1-5, any of which can replace it with a
# refusal. Scoring only the unguarded pipeline would publish quality numbers for
# a system that no longer exists — and would hide the cost of the guards.
#
# This pipeline runs the real LangGraph agent and records, per question, which
# guard fired. That gives the FALSE-REFUSAL rate on 100 legitimate questions,
# which is the number the new Phase 9f content-safety patterns need to justify.

AGENT_PIPELINE_NAME = "agent (guarded)"


def _looks_like_refusal(answer: str) -> bool:
    """Detect the refusal texts the graph actually emits.

    Matches the DISTINCTIVE full phrases, not bare substrings. The old check hit
    on "enough information" alone, which a real answer can contain ("once you have
    enough information about the tools…") — that miscounted valid answers as
    refusals and inflated the refusal rate.
    """
    a = (answer or "").lower()
    return (
        "enough information in the retrieved docs" in a   # NO_CONTEXT_ANSWER
        or "don't have enough information in the retrieved docs" in a
        or "cannot safely" in a
        or "can't safely" in a
    )


async def _agent_rows(testset: list[dict], cache: dict[str, RagRow]) -> list[RagRow]:
    from langchain_core.messages import HumanMessage

    from agent.graph import build_agent
    from agent.mcp_client import warm_tools

    rows: list[RagRow] = []
    async with warm_tools() as tools:
        agent = build_agent(tools)
        for i, q in enumerate(testset, 1):
            key = _cache_key(q["id"], AGENT_PIPELINE_NAME)
            if key in cache:
                rows.append(cache[key])
                print(f"  [agent {i}/{len(testset)}] cache hit  {q['id']}")
                continue

            out = await agent.ainvoke(
                {"query": q["question"], "messages": [HumanMessage(content=q["question"])]},
                {"configurable": {"thread_id": f"ragas-agent-{q['id']}"}},
            )
            answer = out.get("answer") or ""
            hits = out.get("retrieved") or []
            guards = {
                "route": out.get("route"),
                "grounded": out.get("grounded"),
                "policy_violation": out.get("policy_violation"),
                "policy_reason": (out.get("policy_reason") or "")[:160],
                "policy_artifacts": out.get("policy_artifacts"),
                "escalated": out.get("escalated"),
                "dropped_chunks": out.get("dropped_chunks"),
                "source_conflicts": out.get("source_conflicts"),
                "refused": _looks_like_refusal(answer),
            }
            row = RagRow(
                question_id=q["id"],
                pipeline=AGENT_PIPELINE_NAME,
                question=q["question"],
                answer=answer,
                contexts=[getattr(h, "text", "") for h in hits],
                retrieved_files=[getattr(h, "source_file", "?") for h in hits],
                is_adversarial=not q.get("expected_source_files"),
                guards=guards,
            )
            rows.append(row)
            _append_cache(row)
            print(f"  [agent {i}/{len(testset)}] {q['id']:5} route={guards['route']} "
                  f"grounded={guards['grounded']!s:5} policy_viol={guards['policy_violation']!s:5} "
                  f"refused={guards['refused']}")
    return rows


def _print_guard_report(rows: list[RagRow]) -> dict:
    """False-refusal breakdown: on 100 LEGITIMATE questions, nothing should refuse."""
    legit = [r for r in rows if not r.is_adversarial and r.guards]
    n = len(legit)
    refused = [r for r in legit if r.guards.get("refused")]
    by_guard = {
        "guard2_ungrounded": sum(1 for r in legit if r.guards.get("grounded") is False),
        "guard3_policy": sum(1 for r in legit if r.guards.get("policy_violation")),
        "guard5_dropped_chunks": sum(1 for r in legit if r.guards.get("dropped_chunks")),
        "escalated_to_tool": sum(1 for r in legit if r.guards.get("escalated")),
    }
    print()
    print("=" * 90)
    print("GUARD IMPACT ON LEGITIMATE QUESTIONS (false-refusal check)")
    print("=" * 90)
    print(f"  legitimate questions scored : {n}")
    print(f"  ANSWERS REFUSED             : {len(refused)}"
          f"  (rate {len(refused)/n:.3f})" if n else "  n/a")
    for k, v in by_guard.items():
        print(f"  {k:26}: {v}")
    if refused:
        print("\n  refused questions (each one is a potential false refusal):")
        for r in refused:
            art = r.guards.get("policy_artifacts") or r.guards.get("policy_reason") or ""
            print(f"    - {r.question_id:5} {r.question[:60]:60} | {str(art)[:70]}")
    return {"n_legit": n, "refused": len(refused),
            "refusal_rate": round(len(refused) / n, 3) if n else None,
            "by_guard": by_guard,
            "refused_ids": [r.question_id for r in refused]}


# --- RAGAS scoring ----------------------------------------------------------

def _score_with_ragas(rows: list[RagRow]) -> dict[str, list[float]]:
    """Run RAGAS on non-adversarial rows and return per-question scores per metric."""
    # Deferred imports so `--help` etc. don't pay the LangChain load cost.
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        ResponseRelevancy,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — add it to .env (billing enabled).")

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=JUDGE_MODEL, google_api_key=api_key, temperature=0.0)
    )
    # Reuse bge-small locally for embedding-based metrics (ResponseRelevancy) —
    # no extra downloads, no extra API calls, and it's the same model our
    # retriever uses so scores are internally consistent.
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        # strictness=1 → generate ONE reverse-question per answer. RAGAS's default
        # (3) asks Gemini for multiple candidates in a single call, which Gemini
        # rejects ("Multiple candidates is not enabled for this model", 400).
        ResponseRelevancy(llm=judge_llm, embeddings=judge_emb, strictness=1),
        LLMContextPrecisionWithoutReference(llm=judge_llm),
    ]

    non_adv = [r for r in rows if not r.is_adversarial]
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r.question,
            "response": r.answer,
            "retrieved_contexts": r.contexts,
        }
        for r in non_adv
    ])

    # Cloud judge (Gemini) — modest concurrency is fine (no local VRAM bottleneck);
    # keep it friendly to Gemini's rate limits, with retries to absorb 429/503.
    # NOTE: needs billing enabled or the free-tier RPM cap will throttle this.
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=RunConfig(max_workers=4, timeout=180, max_retries=8, max_wait=60),
    )
    # `result` is a Result object; result.to_pandas() gives per-row scores.
    df = result.to_pandas()
    per_metric: dict[str, list[float]] = {}
    for m in metrics:
        col = m.name
        per_metric[col] = df[col].tolist() if col in df.columns else []
    return per_metric


# --- Report -----------------------------------------------------------------

@dataclass
class RagasReport:
    pipeline_name: str
    question_ids: list[str] = field(default_factory=list)
    faithfulness: list[float] = field(default_factory=list)
    response_relevancy: list[float] = field(default_factory=list)
    context_precision: list[float] = field(default_factory=list)

    @staticmethod
    def _avg(xs: list[float]) -> float:
        clean = [x for x in xs if x is not None and not _isnan(x)]
        return sum(clean) / len(clean) if clean else 0.0

    @property
    def avg_faithfulness(self) -> float:
        return self._avg(self.faithfulness)

    @property
    def avg_response_relevancy(self) -> float:
        return self._avg(self.response_relevancy)

    @property
    def avg_context_precision(self) -> float:
        return self._avg(self.context_precision)


def _isnan(x: float) -> bool:
    try:
        return x != x
    except Exception:
        return False


def _print_aggregate(reports: list[RagasReport]) -> None:
    print()
    print("=" * 90)
    print("RAGAS AGGREGATE (LLM-judged, averaged over non-adversarial questions)")
    print("=" * 90)

    metrics = [
        ("Faithfulness",          "avg_faithfulness"),
        ("ResponseRelevancy",     "avg_response_relevancy"),
        ("ContextPrecision",      "avg_context_precision"),
    ]

    col_w = 22
    print(f"{'metric':<22}" + "".join(f"{r.pipeline_name:>{col_w}}" for r in reports) + "   delta")
    print("-" * (22 + col_w * len(reports) + 10))
    for label, attr in metrics:
        row = f"{label:<22}"
        vals = [getattr(r, attr) for r in reports]
        for v in vals:
            row += f"{v:>{col_w}.3f}"
        if len(vals) == 2:
            delta = vals[1] - vals[0]
            sign = "+" if delta >= 0 else ""
            row += f"   {sign}{delta:.3f}"
        print(row)


def _write_results_json(reports: list[RagasReport],
                        guard_report: dict | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_family": "ragas (reference-free)",
        "answer_generator": LLM_MODEL,        # llama3.1:8b (local) — pipeline under test
        "judge_llm": JUDGE_MODEL,             # gemini (cloud) — scores the answers
        "judge_backend": "google-gemini",
        "judge_embeddings": "BAAI/bge-small-en-v1.5",
        "pipelines": [
            {
                "name": r.pipeline_name,
                "aggregate": {
                    "faithfulness":       r.avg_faithfulness,
                    "response_relevancy": r.avg_response_relevancy,
                    "context_precision":  r.avg_context_precision,
                    "n_scored":           len(r.question_ids),
                },
                "per_question": [
                    {
                        "id": qid,
                        "faithfulness":       r.faithfulness[i]       if i < len(r.faithfulness) else None,
                        "response_relevancy": r.response_relevancy[i] if i < len(r.response_relevancy) else None,
                        "context_precision":  r.context_precision[i]  if i < len(r.context_precision) else None,
                    }
                    for i, qid in enumerate(r.question_ids)
                ],
            }
            for r in reports
        ],
        "guard_impact": guard_report,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return RESULTS_PATH


# --- Main -------------------------------------------------------------------

def run(regenerate: bool = False, generate_only: bool = False,
        with_agent: bool = True) -> int:
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))["questions"]
    print(f"[ragas] loaded {len(testset)} questions")

    pipelines: list[tuple[str, Callable]] = [
        ("baseline (dense)", dense_search),
        ("hybrid+rerank",    hybrid_rerank_retrieve),
    ]

    print(f"[ragas] generating answers ({'ignoring cache' if regenerate else 'using cache'})")
    by_pipeline = build_ragas_rows(testset, pipelines, regenerate=regenerate)

    guard_report: dict | None = None
    if with_agent:
        print(f"\n[ragas] generating answers via {AGENT_PIPELINE_NAME} (real graph, guards active)")
        # build_ragas_rows already cleared the cache file when regenerate=True,
        # so re-read it here rather than reusing a stale in-memory copy.
        agent_cache = {} if regenerate else _load_cache()
        agent_rows = asyncio.run(_agent_rows(testset, agent_cache))
        by_pipeline[AGENT_PIPELINE_NAME] = agent_rows
        pipelines = pipelines + [(AGENT_PIPELINE_NAME, None)]
        guard_report = _print_guard_report(agent_rows)

    # Stage 1 (local llama generation) is done and cached to rag_answers.jsonl.
    # With --generate-only we stop here — no Gemini key needed. A later plain
    # run loads these from cache and jumps straight to judging.
    if generate_only:
        n = sum(len(v) for v in by_pipeline.values())
        print(f"\n[ragas] --generate-only: cached {n} answers to "
              f"{ANSWERS_CACHE.relative_to(REPO_ROOT)}. Skipping RAGAS judging.")
        return 0

    reports: list[RagasReport] = []
    for name, _ in pipelines:
        rows = by_pipeline[name]
        non_adv = [r for r in rows if not r.is_adversarial]
        print(f"\n[ragas] scoring {name}  ({len(non_adv)} non-adversarial questions)")
        per_metric = _score_with_ragas(rows)
        reports.append(RagasReport(
            pipeline_name=name,
            question_ids=[r.question_id for r in non_adv],
            faithfulness=per_metric.get("faithfulness", []),
            response_relevancy=per_metric.get("answer_relevancy", per_metric.get("response_relevancy", [])),
            context_precision=per_metric.get("llm_context_precision_without_reference", []),
        ))

    _print_aggregate(reports)
    if guard_report:
        _print_guard_report(by_pipeline[AGENT_PIPELINE_NAME])
    out = _write_results_json(reports, guard_report)
    print(f"\n[ragas] wrote results to {out.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6 RAGAS LLM-judged eval")
    ap.add_argument(
        "--regenerate", action="store_true",
        help="Ignore the on-disk answer cache and regenerate every answer."
    )
    ap.add_argument(
        "--generate-only", action="store_true",
        help="Stage 1 only: generate + cache all answers with llama, then exit "
             "before the Gemini judging stage (no API key needed)."
    )
    ap.add_argument(
        "--no-agent", action="store_true",
        help="Skip the guarded-agent pipeline (retrieval-only baselines)."
    )
    args = ap.parse_args()
    return run(regenerate=args.regenerate, generate_only=args.generate_only,
               with_agent=not args.no_agent)


if __name__ == "__main__":
    sys.exit(main())
