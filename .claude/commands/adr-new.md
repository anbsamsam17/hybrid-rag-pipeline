---
description: Scaffold a numbered Architecture Decision Record for a key choice (hybrid retrieval, RRF, chunking strategy, Qdrant, citation verification) in docs/decisions/, following the repo's craftsmanship bar.
argument-hint: <decision title/topic, e.g. "RRF over weighted score fusion">
allowed-tools: [Task, Agent, Read, Grep, Glob, Bash, Edit, Write]
model: claude-opus-4-8
---

You are scaffolding a new Architecture Decision Record. The decision title/topic is:

    $ARGUMENTS

If `$ARGUMENTS` is empty, ask the user for the decision title and stop until they answer.

## 1. Author the decision (rag-architect)
Delegate to the **rag-architect** subagent to write the substance of the ADR:
- Context: the problem and the forces at play in this RAG pipeline.
- Options weighed: the real alternatives, with honest trade-offs (e.g. RRF vs
  weighted fusion; semantic vs fixed-window chunking; Qdrant vs alternatives).
- Decision: what we chose and why.
- Consequences: what this commits us to, what it rules out, follow-ups.
- Evidence: the example or numbers that justify it (cite reproducible eval output
  where the decision is metric-driven -- do not invent numbers).

## 2. Assign id + format + cross-link (docs-historian)
Delegate to the **docs-historian** subagent to:
- Determine the next ADR number by scanning existing `docs/decisions/` files
  (e.g. ADR-0001, ADR-0002, ...) and assign the next NNN.
- Write the file as `docs/decisions/ADR-NNN-<kebab-case-title>.md` using the repo's
  ADR format (Title, Status, Context, Decision, Consequences).
- Cross-link the new ADR from `docs/architecture.md` so it is discoverable.

## 3. Present
Show the assigned ADR number, the file path, and a short summary of the decision.
