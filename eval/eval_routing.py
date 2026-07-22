"""Phase 8 — score the agent's ROUTER against the handcrafted routing set.

Runs only the router (not the whole graph) on each labeled query and compares
its predicted path set to the expected set with Jaccard — plus exact-match rate
and a per-route breakdown so we can see WHICH paths get confused. This is the
Phase 8 routing-correctness number.

    python -m eval.eval_routing
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from agent.router import _fallback_route, _parse_router_json, _run_router_llm
from eval.metrics import jaccard

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTSET = REPO_ROOT / "eval" / "routing_testset.json"
OUT = REPO_ROOT / "eval" / "results" / "last_routing.json"


def predict(query: str) -> tuple[list[str], bool]:
    """Return (predicted_routes, used_fallback) for one query."""
    raw = _run_router_llm(query)
    parsed = _parse_router_json(raw)
    if parsed is None:
        return _fallback_route(query)["routes"], True
    return parsed["routes"], False


def main() -> None:
    cases = json.loads(TESTSET.read_text(encoding="utf-8"))["queries"]

    rows = []
    jaccards = []
    exact = 0
    fallbacks = 0
    confusion = Counter()  # (expected_primary, predicted_primary) when wrong

    for c in cases:
        expected = set(c["expected_paths"])
        pred_list, used_fallback = predict(c["query"])
        predicted = set(pred_list)
        j = jaccard(predicted, expected)
        is_exact = predicted == expected

        jaccards.append(j)
        exact += int(is_exact)
        fallbacks += int(used_fallback)
        if not is_exact:
            confusion[f"{sorted(expected)} -> {sorted(predicted)}"] += 1

        rows.append({
            "id": c["id"], "query": c["query"],
            "expected": sorted(expected), "predicted": sorted(predicted),
            "jaccard": round(j, 3), "exact": is_exact, "fallback": used_fallback,
        })

    n = len(cases)
    summary = {
        "n": n,
        "mean_jaccard": round(mean(jaccards), 3),
        "exact_match_rate": round(exact / n, 3),
        "exact_matches": exact,
        "fallback_uses": fallbacks,
        "rows": rows,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"ROUTING EVAL  (n={n})")
    print("=" * 64)
    print(f"  mean Jaccard     : {summary['mean_jaccard']}")
    print(f"  exact-match rate : {summary['exact_match_rate']}  ({exact}/{n})")
    print(f"  fallback uses    : {fallbacks}")
    print("\n  misroutes:")
    if confusion:
        for k, v in confusion.most_common():
            print(f"    {v}x  {k}")
    else:
        print("    (none — perfect)")
    print("\n  per-case:")
    for r in rows:
        flag = "  " if r["exact"] else " X"
        print(f"   {flag} [{r['jaccard']:.2f}] {r['id']:4} exp={r['expected']} got={r['predicted']}")
    print(f"\n  wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
