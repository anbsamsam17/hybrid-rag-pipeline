"""The citation-enforced grounding prompt.

:func:`build_grounding_prompt` turns the user query plus the retrieved contexts into a
single instruction string that forces *grounded, cited* generation. The contract the
prompt imposes on the model is exactly what :mod:`rag.verification.citations` later
measures, so the two must agree:

* Answer **only** from the provided context; if the context doesn't contain the answer,
  say so plainly instead of guessing (this is what keeps ``attribution_rate`` honest —
  a model that invents facts has nothing to cite).
* For **every** claim, cite the chunk by its ``chunk_id`` and include the **exact**
  supporting quote copied verbatim from that chunk's text.
* Cite **only** the labelled chunk ids shown below — never invent an id.

Each context is labelled with its ``chunk_id`` (and a 1-based ``[n]`` marker for inline
use) and its full text, so the model can both cite by id and drop ``[n]`` markers into
the prose. The function is pure and deterministic: identical inputs yield an identical
prompt string (no timestamps, no set iteration), which keeps generation reproducible and
prompt-cacheable.
"""

from __future__ import annotations

from rag.retrieval.models import RetrievalResult

_SYSTEM_PREAMBLE = (
    "You are a careful retrieval-augmented assistant. Answer the user's question using "
    "ONLY the provided context passages below. Follow these rules exactly:\n"
    "1. If the context does not contain the answer, say so explicitly and do not guess.\n"
    "2. For every claim you make, cite the passage it comes from by its chunk_id, and "
    "include the exact supporting quote copied verbatim from that passage's text.\n"
    "3. Cite ONLY the chunk_ids listed below. Never invent or guess a chunk_id.\n"
    "4. You may add inline markers like [1] in your answer text that correspond to the "
    "numbered passages, but the authoritative attribution is the chunk_id on each citation."
)


def _format_context(index: int, ctx: RetrievalResult) -> str:
    """Render one context as a labelled block the model can cite by chunk_id and [n]."""
    heading = " > ".join(ctx.heading_path) if ctx.heading_path else ""
    header = f"[{index}] chunk_id={ctx.chunk_id} rel_path={ctx.rel_path}"
    if heading:
        header += f" heading={heading}"
    return f"{header}\n{ctx.text}"


def build_grounding_prompt(query: str, contexts: list[RetrievalResult]) -> str:
    """Build the citation-enforced grounding prompt for ``query`` over ``contexts``.

    Args:
        query: The user's question.
        contexts: The retrieved chunks, best first. Each is labelled by ``chunk_id`` (and
            a 1-based ``[n]`` marker) so the model can cite it.

    Returns:
        A single deterministic prompt string. Contains the query and, for every context,
        its ``chunk_id`` and full text. Pure: no I/O, no nondeterministic ordering.
    """
    if contexts:
        blocks = "\n\n".join(
            _format_context(index, ctx) for index, ctx in enumerate(contexts, start=1)
        )
        context_section = f"Context passages:\n\n{blocks}"
    else:
        context_section = "Context passages:\n\n(none provided)"

    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"{context_section}\n\n"
        f"Question: {query}\n\n"
        "Produce a grounded, cited answer now."
    )
