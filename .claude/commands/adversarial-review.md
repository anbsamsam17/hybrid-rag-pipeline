---
description: Run a focused expert red-team pass over the current working diff before committing, targeting the silent-correctness and overclaiming failure modes specific to RAG + rigorous eval.
argument-hint: "[optional: paths/modules to focus on]"
allowed-tools: [Task, Agent, Read, Grep, Glob, Bash]
model: claude-opus-4-8
---

You are running a pre-commit red-team pass on the current working diff. Optional focus:

    $ARGUMENTS

## 1. Establish the diff
Run `git diff` (and `git diff --cached`) and `git status` to identify exactly what
changed. If `$ARGUMENTS` names paths/modules, narrow the review to those.

## 2. Red-team (adversarial-reviewer)
Delegate to the **adversarial-reviewer** subagent. It is READ-ONLY: it inspects the
diff and changed modules and returns a severity-tagged findings list. It must NOT
edit anything. Direct it to hunt the failure modes that sink a senior RAG repo:
- a recall@k / nDCG@k / MRR that silently counts wrong (off-by-one, wrong denominator)
- eval metric leakage / golden-set contamination of the index
- an attribution_rate that is declarative rather than measured against source spans
- RRF tie-handling and k=60 fusion edge cases (non-deterministic ordering)
- unbounded LangGraph loops / missing retry-budget guards
- async correctness and missing input validation on /query (OWASP)
- citation gaming (claims not actually supported by the cited span)

## 3. Triage (rag-architect)
Delegate to the **rag-architect** subagent to triage the findings: decide which BLOCK
the commit versus which can be deferred (with rationale). Produce a clear block/defer
verdict per finding.

## 4. Fix blockers (relevant domain agent)
For each blocking finding, route the fix to the appropriate domain owner
(retrieval-engineer | citation-verifier | agentic-graph-engineer | api-engineer).
The adversarial-reviewer never applies fixes itself. Re-run `make test` after fixes.

## 5. Present
Show the full findings list with severities, the block/defer triage, and the status
of each blocker (fixed / outstanding). Do not commit -- leave that to the human.
