# CLAUDE.md

Project memory for **hybrid-rag-pipeline**. This file is loaded into context at the
start of every Claude Code session. Read it before touching code — it encodes the
non-negotiables that keep this repo's senior signal intact.

---

## 1. What this project is

A production-grade Retrieval-Augmented Generation system whose whole point is the
things most RAG demos skip:

- **Hybrid retrieval** — dense embeddings (Qdrant) **+** sparse BM25 (`rank_bm25`),
  merged with a **hand-written Reciprocal Rank Fusion (RRF, k=60)**, then a
  cross-encoder **rerank** (`BAAI/bge-reranker-base`).
- **Verified citations** — every generated claim is checked against its cited source
  span (lexical overlap + NLI/LLM-judge); the pipeline reports a **measured
  `attribution_rate`**, never a declared one.
- **Rigorous retrieval evaluation** — `recall@k`, `nDCG@k`, `MRR`, RAGAS
  faithfulness/answer-relevance, plus a custom `attribution_rate`, comparing
  **dense-only vs sparse-only vs hybrid vs hybrid+rerank** with **bootstrap CI95**.
- **Optional self-corrective RAG** — a LangGraph `StateGraph`:
  `analyze → retrieve → grade docs → (low relevance) rewrite query & retry → generate → verify → regenerate`.

The differentiator is **not the stack** — it is the evaluation rigor and the
citation verification. A `recall@k` that silently counts wrong, an `attribution_rate`
that is declarative not measured, an RRF tie that is mishandled, or a golden set that
leaks into the index destroys the entire signal. Treat those failure modes as the
things this codebase exists to get right.

## 2. Stack

- **Python 3.11+**, typed throughout.
- **LangChain** (+ **LangGraph** for the agentic layer).
- **Qdrant** (vector store, via Docker Compose), **`rank_bm25`** (sparse),
  **`bge-reranker`** cross-encoder.
- **Anthropic / OpenAI** LLMs; **Pydantic** structured outputs.
- **FastAPI** (async, SSE streaming).
- **RAGAS** + custom metrics for eval; **SQLite** for logs/eval results.
- **ruff + black** (lint/format), **pytest** (tests), **Docker Compose**, **GitHub Actions** CI.

### LLM / Anthropic SDK rules (load-bearing)

Any code that calls Claude (generation, the LLM-judge in verification, doc-grading
and query-rewrite in the agentic layer) MUST use the official **Anthropic Python SDK**
(`anthropic`) — never raw `requests`/`httpx`, never an OpenAI-compatible shim.

- Default model: **`claude-opus-4-8`**. Use **`claude-sonnet-4-6`** for high-volume /
  cost-sensitive paths (e.g. routine LLM-judge calls) when explicitly chosen.
- Use **adaptive thinking**: `thinking={"type": "adaptive"}`. Control depth with
  `output_config={"effort": "high"}`.
- **Do NOT** use the deprecated `budget_tokens` (400 on 4.7/4.8), and **do NOT** pass
  `temperature` / `top_p` / `top_k` (removed on 4.7/4.8 — both 400).
- For structured outputs (Answer/Citation schemas), use `output_config={"format": ...}`
  or `client.messages.parse(...)` with a Pydantic model — **not** assistant-prefill
  (prefill 400s on these models).
- Stream (`client.messages.stream(...)`) for anything with large output or the
  `/query` SSE endpoint; read the final via `.get_final_message()`.
- The LLM-judge in verification is correctness-sensitive — run it on `claude-opus-4-8`
  (or `claude-sonnet-4-6`) with adaptive thinking; never let it silently pass-mark.

## 3. Target architecture

```
Corpus ─▶ src/rag/ingestion/    loaders, chunkers (fixed / recursive / semantic)
       ─▶ src/rag/indexing/     dense embeddings → Qdrant │ sparse → BM25
                                 reproducible build → meta.json
       ─▶ src/rag/retrieval/    dense.py · sparse.py · fusion.py (RRF, k=60) · rerank.py
       ─▶ src/rag/generation/   citation-enforced prompt → Pydantic Answer/Citation
       ─▶ src/rag/verification/ attribution checker (claim ↔ source span)
       ─▶ src/rag/agentic/      corrective_rag.py — LangGraph StateGraph (optional)
       ─▶ src/rag/api/          FastAPI: /ingest /query /eval, SSE, observability
       ─▶ src/rag/eval/         recall@k · nDCG@k · MRR · RAGAS · attribution_rate · bootstrap CI95

src/rag/config.py               Pydantic settings (single source of config truth)
datasets/golden.jsonl           labeled golden set — PROTECTED, never agent-mutated
docs/architecture.md            kept in sync with src/rag
docs/decisions/ADR-NNN-*.md     architecture decision records
```

**Module-boundary rule:** each `src/rag/<module>` owns one stage. The agentic layer
is **composed on top of** retrieval + verification — it must not duplicate fusion,
rerank, or attribution logic. Cross-module contract changes go through `rag-architect`
and get an ADR.

## 4. Build / test / eval commands (Makefile)

Agents and humans drive the repo through `make` targets — prefer these over ad-hoc
commands so behavior is reproducible.

| Target | What it does |
|--------|--------------|
| `make install` | Install deps (pyproject) + pre-commit hooks |
| `make up` / `make down` | Start / stop Qdrant via `docker-compose.yml` |
| `make ingest` | Run ingestion + build dense+sparse index, writing `meta.json` |
| `make serve` | Launch the FastAPI service (async, SSE) |
| `make eval` | Run the full eval harness over `datasets/golden.jsonl` → comparison table + CI95 |
| `make test` | `pytest` (unit + integration) |
| `make lint` | `ruff check` + `black --check` |
| `make fmt` | `ruff check --fix` + `black` |
| `make typecheck` | static type checking |

**Headline-metric rule:** numbers in the README / `docs/` come **only** from a
reproducible `make eval` run. Never invent, round-up, or hand-edit a metric. If a
number isn't reproducible from the harness, it doesn't get published.

## 5. Coding standards

- **Typing:** full type hints on every public function; Pydantic models for all
  structured data (settings, `Answer`, `Citation`, eval records). No bare `dict`
  passing across module boundaries.
- **Lint / format:** `ruff` + `black`, enforced by a PostToolUse hook on save and by
  CI. Keep diffs lint-clean so reviewers read logic, not formatting noise.
- **Tests:** `pytest`. Every retrieval / eval / verification change ships with tests.
  Fusion math must be covered including **tie-stability**; metrics must be covered
  with hand-checked fixtures (a wrong `recall@k` should fail a test, not ship).
- **Reproducibility (`meta.json`):** every index/eval build records env/library
  versions, the **git SHA**, and a **corpus SHA-256**. A rebuild from the same inputs
  must yield identical retrieval metrics. No unsorted sets, no time-seeded shuffles,
  no unpinned models in the build path. This mirrors the author's traffic-repo
  bit-exact reproducibility discipline.
- **No metric leakage / overclaiming:** the golden set is never indexed; bootstrap
  intervals are paired across configurations; `attribution_rate` is measured against
  spans, never asserted.
- **ADRs:** load-bearing decisions (hybrid vs dense, RRF vs weighted fusion, chunking
  strategy, Qdrant schema, citation-verification approach) are captured as numbered
  ADRs in `docs/decisions/` and cross-linked from `docs/architecture.md`.
- **Secrets:** never read or write `.env`; never commit keys. Config flows through
  `config.py` (Pydantic settings) reading the environment.

## 6. Agent team

This repo is built with a multi-agent orchestration setup under `.claude/`. Each agent
owns one boundary of the RAG lifecycle; the strongest treatment (opus) goes to the
places where a subtle bug silently destroys credibility.

| Agent | Model | Owns / when to use |
|-------|-------|--------------------|
| `rag-architect` | opus | System design, ADRs, cross-cutting trade-offs (chunking, RRF vs weighted, Qdrant schema, where LangGraph plugs in). **Use proactively** before any new subsystem or non-trivial change, and to reconcile module contracts. |
| `retrieval-engineer` | sonnet | `ingestion/`, `indexing/`, `retrieval/` — loaders, chunkers, embeddings, vector store, BM25, dense/sparse search, **hand-written RRF (k=60)**, rerank, reproducible `meta.json` builds. Keep fusion math correct and tie-stable. |
| `eval-scientist` | opus | `eval/` + `datasets/golden.jsonl` — recall@k / nDCG@k / MRR, RAGAS, `attribution_rate`, **paired bootstrap CI95**. The repo's headline signal: rigorous, no metric leakage, no overclaiming. |
| `citation-verifier` | sonnet | `generation/` + `verification/` — citation-enforced prompts, Pydantic `Answer`/`Citation` schemas, the anti-hallucination attribution checker. LLM-judge uses `claude-opus-4-8`/`claude-sonnet-4-6` via the SDK with adaptive thinking (never deprecated params). |
| `agentic-graph-engineer` | sonnet | `agentic/corrective_rag.py` — LangGraph `StateGraph`: typed state, conditional edges, loop/retry-budget guards, grading + query-rewrite nodes. Composes on top of retrieval/verification. |
| `api-engineer` | sonnet | `api/` + `config.py` — async `/ingest` `/query` `/eval`, SSE streaming, Pydantic settings, structured JSON logging with request-id, OWASP-aware validation + CORS, observability (p95 latency, cost/request). |
| `adversarial-reviewer` | opus | **Read-only** red-team. Attacks the diff as a skeptical staff engineer: silent retrieval bugs, eval metric leakage/overclaiming, citation gaming, RRF tie edge cases, unbounded LangGraph loops, async/security holes. **Use proactively** after a feature lands, before any commit. Produces severity-tagged findings; **never edits**. |
| `docs-historian` | haiku | `docs/architecture.md`, ADRs, README headline-metric block, Makefile/usage docs. Syncs docs to code and to the latest **reproducible** eval numbers (never invented). Escalates actual decisions to `rag-architect`. |

## 7. Slash commands (how a senior sequences the work)

| Command | Use it to | Orchestration |
|---------|-----------|---------------|
| `/implement-feature` | Build one RAG feature end-to-end | `rag-architect` frames design (+ ADR if load-bearing) → matching domain agent implements → tests added → `adversarial-reviewer` red-teams the diff → `docs-historian` syncs docs. **Stops for human sign-off before any commit.** |
| `/eval-report` | Produce the defensible comparison table | `eval-scientist` runs `make eval` over the golden set, computes metrics + bootstrap CI95 + `attribution_rate`, sanity-checks for leakage/overclaiming, hands numbers to `docs-historian`. **Refuses to publish non-reproducible numbers.** |
| `/adr-new <topic>` | Capture a key decision | `rag-architect` authors context/options/decision/consequences → `docs-historian` assigns the next `ADR-NNN`, formats, and cross-links from `architecture.md`. |
| `/adversarial-review` | Red-team the working diff before commit | `adversarial-reviewer` (read-only, opus) inspects `git diff` → severity-tagged findings → `rag-architect` triages block vs defer → relevant domain agent fixes blockers. Reviewer auto-fixes nothing. |
| `/repro-audit` | Prove bit-exact reproducibility | `retrieval-engineer` rebuilds index + `meta.json` → `eval-scientist` re-runs harness and diffs metrics against the prior committed run → `adversarial-reviewer` confirms nothing nondeterministic leaked. Flags any drift as a regression. |

## 8. Hooks (automation that defends the signal)

Three hooks (Python scripts under `.claude/hooks/`, invoked as
`python ${CLAUDE_PROJECT_DIR}/.claude/hooks/<x>.py`, degrade to exit 0 when optional
deps are missing):

- **PostToolUse / Edit|Write → `format_python.py`** — auto `ruff --fix` + `black` on
  the changed `.py` file so reviews focus on logic, not formatting.
- **PreToolUse / Edit|Write → `protect_golden_and_secrets.py`** — **blocks (exit 2)**
  writes to `datasets/golden.jsonl`, `.env`, or `meta.json` provenance. The golden
  labels and reproducibility provenance must never be silently mutated by an agent —
  doing so invalidates every eval number.
- **Stop → `eval_guard.py`** — if anything under `src/rag/{retrieval,eval,verification}/`
  changed in the working tree, reminds (exit 2) to re-run `make test` + `/eval-report`
  before claiming results. Prevents shipping a metric change without re-measuring.

## 9. House workflow rules

- Branch before committing; commit/push **only when asked**.
- Re-run `make test` + `make lint` before claiming a feature is done.
- After any retrieval/eval/verification change, re-measure via `/eval-report` —
  never report a stale number.
- End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
