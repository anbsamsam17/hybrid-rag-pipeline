# `.claude/` — Multi-Agent Orchestration for hybrid-rag-pipeline

This directory is the project's **agentic engineering setup** for
[Claude Code](https://claude.com/claude-code): a team of specialized subagents,
the orchestration commands that sequence them, and the safety hooks that keep the
repo's headline signal honest. It is committed deliberately — browsing it tells you
exactly how this codebase is built and reviewed.

> This is not cargo-cult configuration. Every agent owns a real boundary of the RAG
> lifecycle, every command encodes how a senior actually sequences the work, and
> every hook defends a specific way this repo could silently lose its credibility.

---

## Philosophy

The repo's whole senior signal lives in two places: **rigorous, defensible retrieval
evaluation** (recall@k / nDCG@k / MRR with bootstrap CI95, comparing dense vs sparse
vs hybrid vs hybrid+rerank) and **measured — not declared — citation attribution**.
Those are exactly the places where a subtle bug is invisible: a `recall@k` that counts
wrong, an `attribution_rate` that never actually checks spans, an RRF tie mishandled,
the golden set leaking into the index. A monolithic "do everything" agent blurs those
boundaries and lets such bugs through.

This setup instead splits the work the way a strong team would:

1. **One owner per `src/rag` boundary.** Each agent has a narrow mandate and a tool
   allowlist, so changes stay inside their module and contracts are explicit.
2. **The differentiators get the strongest model.** Evaluation, architecture, and
   adversarial review run on **opus** — high-volume implementation runs on **sonnet**,
   documentation prose on **haiku**. Reasoning budget goes where a mistake is most
   expensive.
3. **Review and fix are separated.** The adversarial reviewer is **read-only**. It
   finds problems; the relevant domain agent fixes them. This mirrors real staff-level
   practice and prevents a reviewer from rubber-stamping its own changes.
4. **Provenance is structurally protected.** A pre-write hook makes it *impossible* for
   any agent to silently mutate the golden labels or reproducibility provenance that
   every published number depends on.

The payoff: output that is **higher-quality** (the right specialist, with the right
reasoning depth, on each task) and **reviewable** (every feature passes a skeptical
red-team before a human is asked to sign off).

## The agent team

Defined in [`agents/`](agents). Each is a Markdown file with YAML frontmatter
(`name`, `description`, `tools`, `model`) and a system prompt.

| Agent | Model | Responsibility |
|-------|-------|----------------|
| `rag-architect` | opus | System design, ADRs, and cross-cutting trade-offs — chunking strategy, RRF vs weighted fusion, Qdrant schema, where the LangGraph layer plugs in. Arbitrates module contracts. |
| `retrieval-engineer` | sonnet | Ingestion, indexing, and the retrieval core — dense + sparse search, the hand-written **RRF (k=60)**, cross-encoder rerank, and reproducible `meta.json` index builds. |
| `eval-scientist` | opus | The evaluation harness — recall@k / nDCG@k / MRR, RAGAS, custom `attribution_rate`, and **paired bootstrap CI95** across all four retrieval configurations. Guards against metric leakage and overclaiming. |
| `citation-verifier` | sonnet | Generation + verification — citation-enforced prompts, Pydantic `Answer`/`Citation` schemas, and the attribution checker that verifies each claim against its source span. |
| `agentic-graph-engineer` | sonnet | The optional self-corrective RAG layer — a LangGraph `StateGraph` (analyze → retrieve → grade → rewrite & retry → generate → verify → regenerate) with typed state and bounded retry. |
| `api-engineer` | sonnet | The FastAPI service — async `/ingest` `/query` `/eval`, SSE streaming, Pydantic settings, structured JSON logging, OWASP-aware validation, and observability (p95 latency, cost/request). |
| `adversarial-reviewer` | opus | **Read-only** red-team. Attacks the diff for silent retrieval bugs, eval metric leakage/overclaiming, citation gaming, RRF tie edge cases, unbounded LangGraph loops, and async/security holes. Emits severity-tagged findings; edits nothing. |
| `docs-historian` | haiku | Keeps `docs/architecture.md`, ADRs, and the README headline-metric block truthful to the code and to the latest **reproducible** eval numbers. |

**Why this model split?** `eval-scientist`, `rag-architect`, and `adversarial-reviewer`
are opus because the repo's credibility rests on statistics that must be correct and
on interview-defining design trade-offs — exactly where a quiet error is fatal.
Implementation throughput (retrieval/generation/API/LangGraph wiring) runs on sonnet;
high-volume documentation prose runs on haiku, escalating the actual decisions back to
`rag-architect`.

## Orchestration commands

Defined in [`commands/`](commands). They chain the agents into the way a senior
engineer actually works.

| Command | What it orchestrates |
|---------|----------------------|
| `/implement-feature` | **Design → implement → test → red-team → document.** Architect frames the design (+ ADR if load-bearing), the matching domain agent implements with tests, the adversarial reviewer attacks the diff, the historian syncs docs. **Stops for human sign-off before any commit.** |
| `/eval-report` | Runs the full harness and produces the defensible comparison table (dense vs sparse vs hybrid vs hybrid+rerank) with bootstrap CI95 and `attribution_rate`, then refreshes the README headline metric. **Refuses to publish numbers that aren't reproducible from `make eval`.** |
| `/adr-new <topic>` | Scaffolds a numbered Architecture Decision Record (context, options weighed, decision, consequences) and cross-links it from `architecture.md`. |
| `/adversarial-review` | Runs a focused red-team pass over the current `git diff` before committing, targeting the silent-correctness and overclaiming failure modes specific to RAG + rigorous eval. |
| `/repro-audit` | Verifies **bit-exact reproducibility**: rebuilds the index + `meta.json`, re-runs the harness, and diffs metrics against the prior committed run — flagging any drift as a regression. |

## Hooks

Defined in [`hooks/`](hooks) and wired in `settings.json`. All are Python scripts that
read the hook event JSON from stdin and **degrade gracefully** (exit 0 silently) when
optional tooling isn't installed — so they never block a fresh checkout.

| Hook | Event | Why it earns its place |
|------|-------|------------------------|
| `format_python.py` | PostToolUse · `Edit\|Write` | Auto-runs `ruff --fix` + `black` on the changed file, so the adversarial reviewer reviews **real logic, not formatting noise**. |
| `protect_golden_and_secrets.py` | PreToolUse · `Edit\|Write` | **Blocks** (exit 2) any write to `datasets/golden.jsonl`, `.env`, or `meta.json` provenance. Makes it structurally impossible for an agent to silently rewrite the golden labels or reproducibility provenance that **every eval number depends on**. |
| `eval_guard.py` | Stop | If retrieval/eval/verification code changed, reminds (exit 2) to re-run `make test` + `/eval-report` before claiming results — so no metric change ships without re-measuring. |

## Why this yields higher-quality, reviewable output

- **Right specialist, right depth.** Each task is handled by the agent that owns that
  boundary, at a reasoning budget matched to its blast radius — opus where a bug is
  fatal, sonnet/haiku where throughput matters.
- **Adversarial review is built into the loop**, not an afterthought. Every feature is
  attacked by a skeptical opus reviewer before a human is asked to approve, and the
  reviewer is read-only so review and fix never collapse into the same step.
- **The headline numbers can't be faked.** The eval command refuses non-reproducible
  numbers, and the protect hook locks the golden labels and provenance. The comparison
  table a recruiter sees is, by construction, reproducible from `make eval`.
- **Decisions are durable.** Load-bearing trade-offs become ADRs, and documentation is
  kept in lock-step with the code and the latest reproducible metrics.

Full operational detail (commands, standards, the Anthropic SDK rules for the
LLM-judge) lives in [`../CLAUDE.md`](../CLAUDE.md). Personal/machine-specific overrides
belong in `settings.local.json`, which is gitignored.
