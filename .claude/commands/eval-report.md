---
description: Run the full retrieval/generation evaluation and produce the defensible comparison table (dense vs sparse vs hybrid vs hybrid+rerank) with bootstrap CI95 and the citation attribution_rate -- the repo's headline signal.
argument-hint: "[optional: k values or eval scope, e.g. 'k=5,10']"
allowed-tools: [Task, Agent, Read, Grep, Glob, Bash, Edit, Write]
model: claude-opus-4-8
---

You are producing the repo's headline evaluation report. Optional scope/args:

    $ARGUMENTS

Drive this through the named subagents. Statistical rigor is non-negotiable: every
number published must be reproducible from the harness over the golden set.

## 1. Run + compute (eval-scientist)
Delegate to the **eval-scientist** subagent to:
- Run the evaluation harness over the golden set: `make eval` (or the underlying
  `python -m src.rag.eval ...` entrypoint). Use the scope in `$ARGUMENTS` if given.
- Compute the metrics across all four configurations -- dense-only, sparse-only,
  hybrid, hybrid+rerank: recall@k, nDCG@k, MRR, plus RAGAS faithfulness /
  answer-relevance and the custom attribution_rate.
- Attach paired bootstrap CI95 to each comparison (not point estimates alone).
- Sanity-check for metric leakage and overclaiming: confirm the golden set is not
  contaminating the index, that recall@k actually counts correct hits, that nDCG@k
  is not off-by-one, and that attribution_rate is MEASURED against source spans, not
  declared.

eval-scientist must REFUSE to publish any number it cannot reproduce from the
harness. If `make eval` does not run cleanly, stop and report the failure rather than
fabricating results.

## 2. Refresh docs (docs-historian)
Hand the verified numbers to the **docs-historian** subagent to refresh the README
headline-metric block and the comparison table. docs-historian must use ONLY the
numbers produced in step 1 -- never invented or remembered numbers -- and should
link the table to the reproducible `make eval` invocation.

## 3. Present
Show the final comparison table (the four configs x the metrics, with CI95), the
attribution_rate, any leakage/overclaiming checks that were performed, and the exact
command used to reproduce the run.
