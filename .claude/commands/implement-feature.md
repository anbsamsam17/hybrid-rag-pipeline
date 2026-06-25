---
description: Build one RAG feature end-to-end (design -> implement -> test -> red-team -> document) the way a senior engineer would sequence it, stopping for human sign-off before any commit.
argument-hint: <feature, e.g. "add semantic chunking" or "wire cross-encoder rerank">
allowed-tools: [Task, Agent, Read, Grep, Glob, Bash, Edit, Write]
model: claude-opus-4-8
---

You are orchestrating the full lifecycle for a single RAG feature in the
hybrid-rag-pipeline repo. The feature to build is:

    $ARGUMENTS

If `$ARGUMENTS` is empty, ask the user what feature to build and stop until they answer.

Drive the work by delegating to the named subagents in this exact sequence. Do not
skip stages, and do not let one agent do another agent's job. Summarize each stage's
output before moving to the next.

## 1. Design (rag-architect)
Delegate to the **rag-architect** subagent to frame the design: where this feature
plugs into `src/rag/`, the module contracts it touches, and the load-bearing
trade-offs (e.g. chunking strategy, RRF vs weighted fusion, Qdrant schema, where the
LangGraph layer sits). If the decision is load-bearing, have rag-architect write or
update an ADR under `docs/decisions/` (or invoke the /adr-new flow). Capture the
agreed design as the brief for the next stage.

## 2. Implement (the matching domain agent)
Route implementation to exactly one domain owner based on where the code lives:
- ingestion / indexing / retrieval (loaders, chunkers, embeddings, vector_store,
  bm25, dense, sparse, fusion, rerank) -> **retrieval-engineer**
- generation / verification (citation-enforced prompts, Pydantic Answer/Citation
  schemas, attribution checker) -> **citation-verifier**
- the self-corrective LangGraph layer (corrective_rag.py, StateGraph) -> **agentic-graph-engineer**
- FastAPI service / config.py / SSE / observability -> **api-engineer**

Hand the chosen agent the rag-architect brief and have it implement against the
agreed contracts. Keep the differentiator math correct (manual RRF k=60, tie-stable
fusion; measured-not-declared attribution_rate).

## 3. Tests
Have the same domain agent add pytest coverage for the new behavior (edge cases,
fusion tie handling, metric correctness as applicable). Then run the suite:
`make test` (or `python -m pytest`). Iterate until it passes.

## 4. Red-team (adversarial-reviewer)
Delegate to the **adversarial-reviewer** subagent (read-only) to attack the diff:
silent retrieval bugs, eval metric leakage/overclaiming, citation gaming, RRF edge
cases, async/security issues. Collect its severity-tagged findings. Route any
blocking findings back to the relevant domain agent for fixes, then re-run tests.

## 5. Document (docs-historian)
Delegate to the **docs-historian** subagent to sync `docs/architecture.md` and the
README so they stay truthful to the code. It must not invent eval numbers -- if the
change affects metrics, point it at a reproducible `make eval` / /eval-report run.

## 6. Stop for sign-off
Present: the design summary, the diff overview, the test result, the adversarial
findings (and how each was resolved), and the doc updates. Then STOP. Do NOT commit
-- wait for explicit human sign-off before any `git commit`.
