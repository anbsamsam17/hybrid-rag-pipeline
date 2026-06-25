---
description: Verify bit-exact reproducibility of an index/eval run -- that meta.json captures env versions + git SHA + corpus SHA-256, and that a rebuild yields identical retrieval metrics.
argument-hint: "[optional: index/run name or path to audit]"
allowed-tools: [Task, Agent, Read, Grep, Glob, Bash]
model: claude-opus-4-8
---

You are auditing reproducibility, matching the author's traffic-repo standard
(meta.json with env versions + git SHA + data SHA-256, and identical metrics on
rebuild). Optional target:

    $ARGUMENTS

Drive this through the named subagents.

## 1. Rebuild + regenerate provenance (retrieval-engineer)
Delegate to the **retrieval-engineer** subagent to:
- Rebuild the index from the corpus (e.g. `make ingest` / the indexing entrypoint).
- Regenerate `meta.json` and confirm it captures: env/library versions, the git SHA
  of the build, and the SHA-256 of the corpus.
Note: the meta.json provenance file is write-protected by a PreToolUse hook -- the
rebuild process must regenerate it via the build pipeline, not by an agent hand-edit.

## 2. Re-run harness + diff metrics (eval-scientist)
Delegate to the **eval-scientist** subagent to:
- Re-run the evaluation harness (`make eval`) over the golden set.
- Diff the resulting metrics against the prior committed run.
- Report any drift as a reproducibility regression (metrics must match bit-for-bit /
  within stated tolerance for the deterministic configs).

## 3. Confirm determinism (adversarial-reviewer)
Delegate to the **adversarial-reviewer** subagent (read-only) to confirm nothing
nondeterministic leaked in: unsorted sets/dicts feeding output order, time-seeded
shuffles, unpinned model/library versions, RNG without a fixed seed, or filesystem
ordering dependence. It returns findings only -- it applies nothing.

## 4. Present
Report: whether the rebuilt meta.json matches expectations (env + git SHA + corpus
SHA-256), whether metrics reproduced exactly (or where they drifted), and any
determinism risks the reviewer flagged. Call out any reproducibility regression
explicitly.
