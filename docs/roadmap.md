# Agentic RAG with MCP — Build Roadmap
**Project:** Enterprise Support Knowledge Assistant (LangChain/LangGraph docs corpus)

---

## 1. Component Overview

| # | Component | Role |
|---|---|---|
| 1 | Data cleaning & standardization | Recognize formats, normalize content, filter relevance, track changes |
| 2 | Ingestion & chunking | Turn cleaned docs into retrievable chunks with metadata |
| 3 | Embedding + vector store | Turn chunks into searchable vectors |
| 4 | Hybrid retrieval + reranker | Find the most relevant chunks for a query |
| 5 | LangGraph agent | Decide: retrieve, call a tool, or ask for clarification |
| 6 | MCP server + tools | Give the agent access to live (simulated) data |
| 7 | LLM generation | Produce a grounded, cited answer |
| 8 | RAGAS evaluation | Measure faithfulness / relevance / precision objectively |
| 9 | Guardrails | Catch hallucinated or ungrounded answers before they reach the user |
| 10 | Caching + observability | Keep it fast, keep logs for debugging |
| 11 | Frontend | Chat UI + agent trace + sources panel |

**Build philosophy:** get a dumb, working, end-to-end pipeline first (Phases 1–4), *then* layer in agentic behavior, MCP, and evaluation on top of something that already runs. Don't build all 11 components in parallel — you'll have nothing testable until week 3 and no idea which layer broke when something fails.

---

## 2. Prerequisites

**Environment**
- Python 3.11+
- Git
- A code editor (VS Code is fine)
- ~2GB free disk (docs corpus + embeddings + local vector DB)

**Accounts / keys (all free tier)**
- An LLM API key — Anthropic or OpenAI (you'll spend cents on this project, not dollars)
- Nothing else needs signup: use **Chroma** as the vector store (runs locally, no account, no cost) instead of Pinecone/Weaviate cloud — one less moving part for a portfolio project

**Core libraries** (install as you reach each phase, not all upfront)
```bash
# Phase 1 — data cleaning
pip install python-frontmatter

# Phase 2–3 — ingestion + embedding
pip install langchain langgraph langchain-community
pip install chromadb sentence-transformers rank-bm25

# Phase 7–8 — evaluation
pip install ragas datasets

# Phase 10 — backend
pip install fastapi uvicorn

# Phase 6 — MCP tool server
pip install mcp  # official MCP Python SDK
```

**Knowledge prerequisites:** you already have this covered — Python, basic LangChain/LangGraph familiarity, and your existing FastAPI/Docker experience from the voice pipeline work transfers directly.

---

## 3. Project Layout

```
agentic-rag-mcp/
├── data/
│   ├── raw/                        # cloned langchain-ai/docs (READ-ONLY — never modify)
│   ├── cleaned/                    # normalized .md files output by Phase 1
│   │   └── manifest.json           # tracks format, hash, status per file
│   └── processed/                  # chunked + cleaned JSON ready for embedding
├── ingestion/
│   ├── cleaner/
│   │   ├── __init__.py
│   │   ├── detector.py             # format detection: categorize files by type & relevance
│   │   ├── normalizer.py           # MDX→MD, strip frontmatter & nav cruft
│   │   ├── filter.py               # drop stubs/nav pages, enrich metadata from source path
│   │   └── manifest.py             # SHA256 hashing, manifest read/write, change diffing
│   ├── loader.py                   # reads cleaned .md files via format-registry pattern
│   ├── chunker.py                  # splits into ~300–500 token chunks with metadata
│   └── embed_and_store.py          # embeds chunks + upserts into Chroma (incremental)
├── retrieval/
│   ├── dense.py                    # embedding similarity search
│   ├── sparse.py                   # BM25 keyword search
│   ├── hybrid.py                   # reciprocal rank fusion of dense + sparse
│   └── rerank.py                   # cross-encoder reranking
├── agent/
│   ├── graph.py                    # LangGraph state graph definition
│   ├── nodes.py                    # node functions: retrieve, generate, clarify
│   └── prompts.py
├── mcp_server/
│   ├── server.py                   # FastMCP server (stdio) exposing the tools
│   └── tools.py                    # get_package_version (live PyPI), get_corpus_status (local manifest)
├── eval/
│   ├── generate_testset.py         # builds RAGAS eval set from corpus chunks
│   ├── testset.json
│   └── run_ragas.py
├── guardrails/
│   └── checks.py                   # faithfulness + injection checks
├── api/
│   └── main.py                     # FastAPI backend wrapping the agent
├── frontend/
│   └── ...                         # chat UI (React, or Streamlit for a fast v1)
├── .env
├── requirements.txt
└── README.md
```

---

## 4. Build Order — Phase by Phase

Each phase has a **definition of done** — don't move to the next phase until you can check that box. This is what keeps debugging sane later.

### Phase 0 — Environment setup
- Init repo, virtualenv, folder structure above
- `git clone --depth 1 https://github.com/langchain-ai/docs.git data/raw`
- **Done when:** repo structure exists, docs are cloned, `pip list` shows core deps installed

---

### Phase 1 — Data Cleaning & Standardization
The raw corpus (`data/raw/`) is a full documentation monorepo with ~3,100 files spanning many formats — MDX, Markdown, JSON configs, YAML, Python build scripts, XML sitemaps, CSS, fonts, and images. Most of it is irrelevant noise. This phase produces a clean, uniform set of plain Markdown files in `data/cleaned/` — scoped to `src/oss/langchain/`, `src/oss/langgraph/`, and `src/oss/concepts/` (≈120 files) — that every downstream phase reads from.

Four sub-steps, run in order:

#### 1a — Format Detection (`cleaner/detector.py`)
- Walk `data/raw/` and categorize every file by extension and path:

| Category | Extensions / Paths | Action |
|---|---|---|
| **Relevant content** | `.mdx`, `.md` under `src/oss/langchain/`, `src/oss/langgraph/`, or `src/oss/concepts/` | Process |
| **Irrelevant content** | `.mdx`/`.md` under `src/langsmith/`, `src/snippets/`, `src/oss/python/integrations/`, `src/oss/javascript/`, `src/oss/deepagents/`, `src/oss/reference/`, contributing guides, changelogs | Skip |
| **Non-text assets** | `.json` configs, `.yml`, `.py` build scripts, `.xml`, `.css`, `.woff2`, images | Skip |
| **Name collisions** | `CLAUDE.md`, `AGENTS.md` at repo root (clash with our project memory) | Skip |

- Output: a categorized file list with `(path, format, action)` per entry — used by the next step
- Log counts: how many files of each type found, how many will be processed vs. skipped

#### 1b — Normalization & Standardization (`cleaner/normalizer.py`)
Convert every relevant file to clean, plain Markdown using a format registry:

```python
FORMAT_REGISTRY = {
    ".mdx": clean_mdx,       # strip JSX imports, <Tip>/<Warning>/<CodeGroup> → plain text
    ".md":  clean_markdown,  # strip frontmatter, normalize headings
}
```

The registry is extensible — adding a new format later (e.g. `.rst` or `.pdf`) is a one-line change here plus a matching cleaner function; nothing else in the pipeline moves.

Normalization rules applied to all formats:
- Strip YAML frontmatter (`---` blocks) — extract `title` and `description` as metadata before stripping
- Remove JSX component tags (`<Tip>`, `<Warning>`, `<CodeGroup>`, `<Card>`, etc.) — keep inner text
- Remove import/export statements (MDX-specific)
- Remove navigation artifacts: "Edit this page", "On this page", breadcrumb links, sidebar refs
- Normalize heading levels: ensure every doc starts at `# H1` (some start at `## H2`)
- Collapse multiple blank lines to one; normalize Unicode whitespace

Output: one plain `.md` file per input file, written to `data/cleaned/` mirroring the source path structure

#### 1c — Relevance Filtering (`cleaner/filter.py`)
After normalization, run a secondary filter to drop any pages that slipped through but are still noise:
- Skip files with fewer than 100 tokens (stub pages, pure navigation pages)
- Skip files where >60% of non-empty lines are pure link-only lines (nav/index pages that leaked past the detector)
- Extract and enrich metadata per kept file: `title`, `product` (langchain / langgraph / shared), `doc_type` (concept / how-to / reference / tutorial), `section` (the path segment under `src/oss/`)

Note: code-heavy pages are *not* dropped. Documentation that consists mostly of code examples is exactly what retrieval needs for code-specific queries; retrieval quality is protected downstream by hybrid retrieval + reranking + guardrails, not by aggressive up-front filtering.

Target output: **~100–120 clean `.md` files** from the original ~3,100 raw files — this is your actual RAG corpus

#### 1d — Change Tracking (`cleaner/manifest.py`)
- Compute SHA256 hash of each raw source file *before* cleaning
- Write/update `data/cleaned/manifest.json` with one entry per file:
```json
{
  "src/oss/langchain/concepts/runnable.mdx": {
    "target_file": "data/cleaned/oss/langchain/concepts/runnable.md",
    "original_format": ".mdx",
    "content_hash": "a3f9...",
    "source_commit": "abc1234",
    "ingested_at": "2026-07-05T10:00:00Z",
    "status": "ok"
  }
}
```
- On re-run: load existing manifest, compare hashes → only reprocess files whose hash changed or are new. Log skipped/updated/added counts.
- This is what makes Phase 3's incremental Chroma upsert actually useful — only changed chunks need re-embedding

**Done when:**
1. `data/cleaned/` contains ~100–120 plain `.md` files with no frontmatter, no JSX, no nav cruft
2. `manifest.json` exists and has one entry per cleaned file
3. Re-running the cleaner on an unchanged corpus processes **zero files** (hash check works)
4. You can print 5 random files from `data/cleaned/` and each reads like clean, coherent documentation prose

---

### Phase 2 — Ingestion & Chunking
- `loader.py`: walk `data/cleaned/`, read `.md` files — no format-detection needed here, Phase 1 already normalized everything. Attach metadata from `manifest.json` (`source_file`, `doc_type`, `product`, `section`).
- `chunker.py`: split into ~300–500 token chunks, preserve headers as context, attach `source_file`, `section`, `chunk_id`
- **Done when:** you can print 5 random chunks and each is a coherent, self-contained unit of text — not a sentence cut in half, not a navigation artifact

---

### Phase 3 — Embedding + Vector Store
- Embed all chunks using `sentence-transformers/all-MiniLM-L6-v2` (free, local, good enough for this scale)
- Upsert into Chroma with full metadata; use `chunk_id` as the Chroma document ID so re-runs are incremental upserts, not full reindexes
- **Done when:** raw similarity search for "how do I use a memory checkpoint in LangGraph" returns top 3 results that are actually relevant, by eye

---

### Phase 4 — Naive RAG Baseline (no agent yet)
- Simple linear pipeline: query → dense retrieval (top-5) → stuff into prompt → LLM generates answer
- No LangGraph, no MCP, no hybrid retrieval — the dumbest version that works
- **Done when:** you can ask a question in a terminal script and get back a reasonable, roughly-grounded answer. This is your working baseline — everything after this is improvement, not first-time-it-works risk.

---

### Phase 5 — Hybrid Retrieval + Reranking
- Add `sparse.py` (BM25 over the same chunks)
- Add `hybrid.py` (reciprocal rank fusion of dense + sparse results)
- Add `rerank.py` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, reorders top-20 fused results down to top-5)
- **Done when:** before/after comparison on the same test queries shows hybrid+rerank visibly returns better top results for at least a few queries with exact terms (e.g. specific function names)

---

### Phase 6 — Evaluation Harness (RAGAS + routing correctness)
- Generate a ~50-question eval set from your own chunks (`generate_testset.py`)
- Run RAGAS on Phase 4 baseline AND Phase 5 hybrid pipeline
- Build a small labeled routing set (`eval/routing_testset.json`, ~22 queries tagged: retrieve / get_package_version / get_corpus_status / fetch_live_doc / clarify) + Jaccard-based scoring function — wire into the agent properly in Phase 8 but write the labeled set now while the corpus is fresh
- **Done when:** you have two real numbers — e.g. "faithfulness went from 0.71 to 0.86 after hybrid + reranking." This is your baseline for every phase after this.

**Deterministic retrieval eval** (`eval/run_eval.py`, 91 non-adversarial questions, refreshed
2026-07-21 against the current 2313-chunk corpus):

| metric | baseline (dense) | hybrid + rerank | delta |
|---|---|---|---|
| Precision@5 | 0.212 | 0.212 | +0.000 |
| Recall@5 | 0.868 | 0.861 | −0.007 |
| Hit@5 | 0.890 | 0.890 | +0.000 |
| MRR | 0.760 | 0.760 | −0.001 |

**The refresh changed the story.** The previous baseline (2026-07-10, on the code-poor
2052-chunk corpus) read dense MRR **0.832** vs hybrid **0.638** — hybrid looked clearly worse.
After the snippet code-inlining fix, **hybrid gained +0.12 MRR** (0.638 → 0.760) while dense
lost −0.07 (0.832 → 0.760, expected as the corpus grew ~13% and top slots got more
contested). Inlining restored real code and API identifiers, which is exactly the signal BM25
matches on — so the old numbers were measuring a corpus that structurally handicapped sparse
retrieval, not a weakness of hybrid.

**Honest read:** the two pipelines are now *indistinguishable* on this set (Hit@5 identical,
MRR within 0.001), which almost certainly means hybrid's top-5 is nearly the same as dense's
top-5 here. That is a property of THIS testset and corpus — 116 clean single-domain docs and
auto-generated questions that are semantic paraphrases with ~no exact-term/identifier
queries — not evidence hybrid is useless. Revisiting this with a larger, noisier corpus and
keyword-style questions is a tracked follow-up (see the end-of-build reminder).

---

### Phase 7 — MCP Server + Tools  ✅
- THREE tools that answer what the frozen docs corpus can't (REAL data, no synthetic mock files):
  - `get_package_version(package)` — current version + release date from the **live PyPI** API (free, no key)
  - `get_corpus_status(topic?)` — the docs' pinned commit / freshness / per-product & per-doc_type coverage from our **local manifest**
  - `fetch_live_doc(query?, path?)` — a CURRENT LangChain/LangGraph doc page fetched live from GitHub for topics the frozen corpus lacks (the staleness fix); bge-small ranks in-scope live paths, `min_score` gate returns `matched:false` when out of scope, inlines snippet code over GitHub
- `mcp_server/tools.py` (pure functions, `python -m mcp_server.tools` smoke test) → `mcp_server/server.py` (FastMCP over stdio, prewarms + HF-offline embedder) → `agent/mcp_client.py` (`warm_tools()` holds one session; MCP protocol round-trip). Interactive check: `npx @modelcontextprotocol/inspector python -m mcp_server.server`
- **Done when:** you can query the MCP server directly (the test client or MCP inspector) and get correct tool responses — *before* wiring into the agent. Debug the tool in isolation first.

---

### Phase 8 — LangGraph Agent (this is where it becomes "agentic")  ✅
- Built in `agent/` (state/mcp_client/router/nodes/graph/cli). Pure-LLM router (llama3.1:8b) over the 3 MCP tools via langchain-mcp-adapters; confidence-gated escalation to fetch_live_doc; MemorySaver multi-turn. Routing eval: **Jaccard 0.932, 20/22 exact** (`eval/eval_routing.py` → `eval/results/last_routing.json`). Run: `python -m agent.cli` / `--demo`.
- Replace the linear pipeline from Phase 4 with a LangGraph state graph
- Nodes: `route` (retrieve / get_package_version / get_corpus_status / fetch_live_doc / clarify) → `retrieve` (Phase 5 hybrid) or `call_mcp_tool` (Phase 7) → `generate`; escalate to `fetch_live_doc` when retrieval confidence/coverage is low
- Add LangGraph's `MemorySaver` checkpointer for multi-turn conversation memory
- Run the Phase 6 routing-correctness set (`eval/routing_testset.json`, 5-way) against the live agent and record the score
- **Done when:** docs questions route to retrieval, "latest version?" questions route to `get_package_version`, "how fresh / do you cover X?" route to `get_corpus_status`, new/out-of-scope doc requests route to `fetch_live_doc`, ambiguous questions ask for clarification, and follow-up questions correctly use previous-turn context

---

### Phase 9 — Guardrails  ✅

> **Current architecture (start here).** Guards live in `agent/guards/` and expose exactly
> two functions; the graph calls these and nothing else:
>
> - **`check_input(query, chunks)`** — BEFORE the LLM, all deterministic: PII/secret
>   scrubbing → prompt-injection scan of retrieved chunks → source corroboration.
> - **`check_output(question, context, answer, …)`** — AFTER the LLM, ordered
>   cheap→expensive: deterministic (gibberish, duplicate sentences, citation integrity,
>   content safety, symbol allowlist, URL/host safety) → embedding relevance (telemetry)
>   → LLM tier (groundedness, context-blind policy check).
>
> It **repairs** what it can and refuses only what is unsafe or unsupported.
> `agent/guardrails.py` holds the primitives; the numbered "GUARD 1–5" language below is
> the historical build order, kept because the measurements explain *why* each exists.

- `agent/guardrails.py`: GUARD 1 = regex injection scan on retrieved/live text (pre-LLM); GUARD 2 = LLM groundedness fact-check with pluggable backend (`GUARDRAIL_VERIFIER=ollama|gemini`, local default + auto-fallback) → ungrounded answers replaced by an honest refusal.
- Noise tooling: `eval/noisy_corpus.py --add/--purge` (10 off-topic + 5 poisoned chunks, injected into BOTH Chroma and the BM25 source).
- **Results** (`eval/eval_guardrails.py`, after the materiality-based verifier prompt): both backends 9/9 adversarial refused, 0 attacks landed; false-refusal ollama 0.05 vs gemini 0.15.
- **Injection suite** (`eval/injection_suite.json` + `eval/eval_injections.py`): 8 direct / 12 evasive. Regex catches direct 8/8 but evasive **0/12**. On the evaded set: ollama 2 attacks landed / 0 verifier blocks; gemini 1 landed / 4 blocks → **verifier default is gemini-first with automatic ollama fallback**.
- **GUARD 3 `policy_check`** — answer-vs-question, deliberately context-blind (a poisoned-but-obeyed answer is *grounded*, so GUARD 2 can't see it). **GUARD 3b `scan_unsafe_content`** — deterministic content safety (TLS off, `curl|sh`, `eval(var)`, `shell=True`, hardcoded secrets, `sandbox=False`, data egress to non-official hosts); 7/7 caught, 0 false positives.
- **Red-team suite** (30 attacks: 8 direct / 12 evasive / 10 stealth). GUARD 1 catches direct 8/8, evasive 0/12, stealth 0/10. End-to-end: **attacks landed 3 → 1**, success-when-retrieved **0.231 → 0.077**; evasive class fully closed (0 landed).
- **GUARD 4 `scan_unknown_symbols`** — 13,222-symbol allowlist built from the TRUSTED `data/cleaned/` docs (not the poisonable index); flags framework APIs/packages the docs never mention (fake `create_legacy_agent`, typosquat packages). 4/4 fakes caught, 0 false positives.
- **Final red-team: 35 attacks** (8 direct / 12 evasive / 10 stealth / 5 misinfo). Guard 1 catches direct 8/8 only. End-to-end: **1 attack landed of 18 retrieved — success 0.056** (from 0.333 with Guards 1+2 alone). Evasive and stealth classes fully closed (0 landed).
- **GUARD 5 `corroborate`** — deterministic source corroboration: when retrieved chunks assert incompatible facts, the value backed by more **independent documents** wins and the outlier is dropped pre-generation. Counts distinct source files, not chunks (counting chunks let one poisoned page ballot-stuff and delete the real docs).
- Suite A alone reached 0 attacks landed of 17 retrieved. **That number did not survive contact with new attacks** — see below. Progression on suite A across the build: 0.333 → 0.231 → 0.077 → 0.056 → 0.0.

#### Phase 9f — three independent suites (the real number)

Suite A's 0.0 measured the guards against the 35 strings they were *developed against*. Two disjoint suites (`injection_suite_b.json`, `injection_suite_c.json` — new attacks, new markers, new probe queries, rotated stealth vectors) were written to break that circularity. Run via `--suite a|b|c`, results kept in separate files.

**Held-out suites immediately found four defects suite A could not:**

| Suite | tested | retrieved | landed (before fix) | landed (after fix) |
|---|---|---|---|---|
| A (development set) | 27 | 17 | 0 | 0 |
| B (held out) | 29 | 17 | 2 | **1** |
| C (held out) | 30 | 16 | 2 | **0** |
| **pooled** | **86** | **50** | **4 — 0.080** | **1 — 0.020** |

Defects found and fixed (`scan_unsafe_content`): unsafe deserialization (`pickle.loads` was served verbatim as an answer; also `yaml.load`, `allow_dangerous_deserialization`, `trust_remote_code`); **credential phishing** — the answer told the user to *paste a leaked API key* into an attacker-controlled portal, the worst outcome in any run, missed because both egress checks key off code/config rather than prose; `os.system`; `auth=None` (the disable-check only matched `= False`); wildcard CORS with credentials; and two regex-anchoring bugs (`exec(a, b)` with a second argument, `AWS_SECRET_ACCESS_KEY` having no word boundary before "secret"). Regression-checked 11/11 attacks caught, 0/9 false positives on legitimate text (real provider key-issuance URLs, `yaml.safe_load`, `literal_eval`, security advice about `eval`).

**Also measured: GUARD 1's regex is overfit.** Direct-class catch rate 8/8 on suite A but 6/8 on B and 5/8 on C — equivalent phrasings ("These new instructions override the earlier ones") aren't in the pattern list. All five escapees were stopped downstream, so defense-in-depth held, but the outer layer is narrower than its suite-A score implied.

- **Remaining known failure (1 of 50): suite B `misinfo_03`**, a single-source numeric claim ("subgraphs are capped at a default recursion depth of 50"). Guard 5 correctly declines to act — there is no competing value because *no genuine doc states a nesting cap at all*. Corroboration cannot adjudicate a fact the corpus does not cover. Dropping all single-source numerics was rejected: in a 116-doc corpus most numerics appear exactly once, so that rule trades a rare poisoning for constant false refusals. **This is a threat-model boundary, not a bug** — if an attacker can write to the corpus, an uncorroborated single-source fact is undecidable at generation time. The real fix is provenance control at ingestion, not a generation-time guard.
- Explicitly rejected shortcut: the poisoned chunks carry `product: "noise"` metadata, so Guard 5 could be made to look authoritative by keying on provenance. That scores against a test artifact — real poison arrives through the normal ingestion path with normal metadata.
- Standing caveats: Guard 3's LLM tier is non-deterministic, so single-run numbers move; all three suites are self-authored, so they test what was anticipated; suite C's five behavioural misinfo claims were caught by GUARD 2 (groundedness), not GUARD 5 — the behavioural gap is better covered than expected, the single-source numeric one is not.

#### Phase 9g — the usability measurement that changed everything

The suite scores above say nothing about what the guards cost on REAL questions. Running the 100-question set through the actual agent graph (not `rag.naive`, which nothing serves) produced the number that mattered:

**False-refusal rate: 40 of 91 legitimate questions refused — 0.440.** A system that refuses 44% of valid questions is not usable, whatever its attack-success score says.

Diagnosing those 40 found that most were **bugs, not strictness**:

1. **The policy prompt ended with the bare text `"JSON verdict:"` immediately after the answer**, with no delimiter. Gemini read it as part of the answer under review and reported *"the answer includes a meta-commentary 'JSON verdict:' at the end"* — the guard was flagging our own prompt scaffolding. **~20 of the 40 refusals.** Both verifier prompts now fence the answer in `<<<BEGIN_ANSWER>>>`/`<<<END_ANSWER>>>`.
2. `import X as Y` was split on whitespace, so the **English keyword `as`** was checked as an undocumented API symbol.
3. `localhost` / private IPs were treated as untrusted exfiltration endpoints, refusing "test Studio locally" answers.
4. **GUARD 2 was not broken — the response to it was.** Re-running the verifier on refused answers showed its verdicts were *correct and pedantic*: q3's answer was ~95% right with one genuinely unsupported trailing sentence, and the system discarded the whole thing. Loosening the verifier would have let q4's real hallucination through. The fix was to **repair instead of refuse**: strip the unsupported sentences, keep the rest, and refuse only if >40% must go or nothing substantive remains. Code fences are never stripped.

**Result: 0.440 → 0.100** (9 refusals of 90), with 5 more answers repaired-and-shipped rather than discarded.

#### Phase 9h — white-hat audit (10 findings)

A dedicated adversarial read of the whole agent path, not just the guard functions:

| # | Sev | Finding |
|---|---|---|
| V1 | **critical** | The **clarify path had no output guard at all** — `clarify → finalize` shipped an LLM response built from raw user input, bypassing every guard. Reachable by phrasing a question vaguely. |
| V2 | **high** | `fetch_live_doc(path=…)` interpolated an unvalidated path into the raw.githubusercontent URL — `../../` walked off the pinned repo and made the server fetch **attacker-chosen content**, then clean, trust and feed it to the model. Same hole in the snippet resolver, where the path comes out of already-fetched remote content. |
| V3 | high | PII scrubbing was theatre: it redacted `state["query"]` but the RAW message entered durable history, so a key pasted on turn 1 replayed into every later prompt — and it only ran on the retrieve path. |
| V4 | med | Only `fetch_live_doc` output was scanned. `get_package_version` passes through the PyPI `summary` field, **written by whoever published the package**. |
| V5 | med | Fail-open was a DoS→bypass: exhausting the verifier's API key silently returns `grounded=True`. |
| V6 | med | Sanitised (blanked) tool content still generated a confident answer from the placeholder. |
| V7 | med | The eval harness had no `try/finally` — **this is the bug that left 35 live attack chunks in Chroma and the BM25 index after a killed run**, silently poisoning every later query. |
| V8 | low | The relevance gate was **removed, not tuned**: measured 0.43 for a blatantly off-topic answer vs 0.81 for a correct one — the bands overlap, so no threshold separates them. It would have refused good answers while missing the bad one it was built for. Kept as telemetry rather than leaving fake protection in place. |
| V9 | low | Repair matched at 0.55 similarity, loose enough to delete correct sentences; >3 flagged claims now refuses instead of returning a stub. |
| V10 | low | `re` was never imported in `mcp_server/tools.py` — the new validator would have crashed on first use. |

**Structure was simplified while fixing these:** six numbered guards across four graph nodes collapsed into **two stages with one entry point each** — `guards.check_input()` (PII scrub → injection scan → source corroboration) and `guards.check_output()` (deterministic → embedding → LLM, cheap-first so most answers never reach the paid tier). The graph now has one rule: every LLM-generated path goes through `guard_output`.

#### Phase 9i — final numbers, and an uncomfortable correction

| Metric | Before | After |
|---|---|---|
| False refusals (91 real questions) | 0.440 | **0.099** |
| Attack success (50 retrieved, 3 held-out suites) | 0.020 | **0.040** |

**Final RAGAS (LLM-judged, whole system measured against final code):**

| Metric | dense | hybrid+rerank | agent (guarded) |
|---|---|---|---|
| Faithfulness | 0.835 | 0.809 | **0.719** |
| ResponseRelevancy | 0.929 | 0.956 | **0.851** |
| ContextPrecision | 0.647 | 0.682 | **0.663** |

The guarded agent's faithfulness (0.719) sits ~0.09 below the raw hybrid retriever (0.809), and that gap is the *measurable cost of the safety layer*, not a defect: a refusal is unfaithful-to-context by construction, so the ~10% of rows that safely refuse drag the mean down, and repaired answers have sentences removed. RAGAS cannot distinguish "safely refused" from "hallucinated". Reported honestly, that is ~0.79 faithfulness on answered questions at a 0.099 refusal rate. The earlier 0.467 was the buggy run (40 spurious refusals); this is the first measurement of the system actually served. Routing held at 0.932 post-refactor.

**The security number got worse, and the reason matters more than the number.** Two of the attacks that now land (`misinfo_05`, `stealth_10`) were previously "blocked" by the `"JSON verdict:"` prompt bug making Gemini hallucinate policy violations. They were never genuinely defended — a broken guard was firing semi-randomly and happened to catch them, while simultaneously refusing 20 valid questions. **The old 0.020 was partly propped up by the same bug that made the system unusable.** `stealth_10` was then fixed properly (a real `blocking_stdin_in_server` pattern; suite C went 1 → 0).

- **Remaining 2 of 50, both single-source misinformation** (`misinfo_03`, `misinfo_05`): a poisoned claim with no competing value anywhere in the corpus, so GUARD 5 has nothing to count. Confirmed by measurement, not argued from theory. A phrase-matcher would drive this to zero but would be overfitting to a self-authored suite. **The real fix is provenance control at ingestion.**
- **Done when:** deliberately unanswerable questions return "I don't have enough information" instead of fabricated answers, and noisy chunks don't pollute responses — ✅ met, with the residual risk named above.

---

### Phase 10 — Caching + Observability  ✅

**Observability.** `agent/observability.py` — `configure_tracing()` loads `.env` and, if a
`LANGSMITH_API_KEY` is present, switches LangSmith tracing on (sets both `LANGSMITH_*` and
legacy `LANGCHAIN_*` names); with no key it silently no-ops so the agent still runs fully
offline. Called from `build_agent()`, so every entry point is traced automatically. Each
graph node (`precheck → route → retrieve → guard_input → generate → guard_output →
finalize`) appears as a span with its own latency and token count, and the guard state it
returns (`guard_action`, `guard_repairs`, `grounded`, …) shows in the span output — so the
trace also explains *why* an answer was repaired or refused. Project: `rag-build-agent`.

**Caching** (`agent/cache.py`, one small `Cache` class — LRU + optional TTL + hit/miss
stats; in-process, no Redis). Three layers, outermost first:

| Cache | Key | Scope | TTL |
|---|---|---|---|
| `ANSWER_CACHE` | query + conversation history | whole guarded answer | 900s |
| `VERDICT_CACHE` | backend + question + context + answer | one LLM-guard verdict | 3600s |
| `EMBED_CACHE` | query text | query embedding vector | none (deterministic) |

- **Answer cache** runs in a new `precheck` node *before* routing: a hit skips the router
  LLM call, retrieval, generation and all guards, because **only clean, already-guarded
  answers are stored** (finalize refuses to cache a refusal, a degraded-verifier turn, or a
  clarify turn — caching those would pin a transient failure). Measured: identical query on
  a fresh thread went 58s → ~0s, same answer; a different query correctly missed.
- **Verdict cache** wraps the two LLM guards (`verify_grounded`, `check_output_policy`) —
  the dominant per-answer cost — and stores only successful verdicts, never fail-open errors.
- **Safe for eval by construction:** the caches are in-process and every eval run is a fresh
  process with an empty cache, and eval questions/probes are unique, so caching cannot mask
  an attack or distort a score across runs. History is in the answer key so follow-ups
  ("how do I install it?") never collide with the base question.
- **Done when:** ✅ a LangSmith trace for a real query shows the per-node latency/token
  breakdown; caching verified to take repeat queries to ~0s without changing any answer.

**Honest note:** caching speeds *repeats*; a first-time answer is still ~17s (llama
generation + two Gemini guard calls + rerank), which is inherent to a local LLM. The trace
makes that breakdown visible, which is the point.

---

### Phase 11 — Frontend  ✅

Chosen **Streamlit** (over React/FastAPI-SPA) for a Python-only, single-command UI —
CLAUDE.md had left this "React or Streamlit — TBD".

- `app/runtime.py` — the bridge that makes an async, warm-session agent work under
  Streamlit's re-run-per-interaction model. A dedicated asyncio event loop runs in a
  background thread for the whole process; it enters `warm_tools()` ONCE and builds the
  agent with a `MemorySaver` ONCE, and each turn is submitted with
  `run_coroutine_threadsafe`. So the MCP subprocess, tool session and conversation memory
  survive every rerun. Cached with `@st.cache_resource` → created once per server. This is
  the load-bearing piece; the naive `async with warm_tools()` would respawn the MCP
  subprocess and drop memory on every keystroke.
- `app/streamlit_app.py` — chat thread + per-answer metadata: a **route/tool badge**, a
  **guard badge** (✓ grounded / ✎ repaired / ⚠ refused), latency, a ⚡cached indicator, a
  🔒 PII-redacted notice, and an expandable **Sources** panel (ranked `[S#]` citations from
  the retrieved chunks).
- **Conversation rail** (like a chat app): the sidebar lists every conversation in the
  browser session — switchable, deletable, each auto-named from its first message and
  keeping its own agent memory (its id IS the LangGraph `thread_id`). Stored in
  `st.session_state` (per session, RAM-only; a `SqliteSaver` + disk store would persist it).
- **Honesty in the UI:** source scores are labelled as cross-encoder rerank logits (higher =
  more relevant), NOT shown as a 0-1 percentage, because that is what they are.
- Run: `./env/Scripts/python.exe -m streamlit run app/streamlit_app.py`
- **Verified:** `streamlit.testing.v1.AppTest` runs the script headlessly (builds the real
  runtime, sends a message, gets an answer with route + guard badges — no exception); the
  runtime bridge was unit-tested for multi-turn memory ("how do I install it?" resolved),
  citation sources, and an answer-cache hit at ~0s.
- **Done when:** a stranger could use it without explanation — ✅ (single command, self-
  explanatory sidebar with example prompts).

---

### Phase 12 — Polish
- README with architecture diagram, eval scores, and a GIF/video demo
- Optional: deploy the API on Render/Railway free tier so it's a live link, not just a repo

---

## 5. Where to literally start today

1. Create `ingestion/cleaner/` folder and its four files (`detector.py`, `normalizer.py`, `filter.py`, `manifest.py`)
2. Write `detector.py` first — walk `data/raw/src/oss/`, print the file count by extension and category. This tells you exactly what you're dealing with before you write a single cleaning rule.
3. Write `normalizer.py` for `.mdx` files (the majority) — strip frontmatter, JSX, nav cruft, output plain `.md`
4. Run it on 10 files, inspect the output by eye. Only then wire in the manifest.

That's session one. Everything after Phase 1 depends on having clean, reliable text — it's worth not rushing this part.
