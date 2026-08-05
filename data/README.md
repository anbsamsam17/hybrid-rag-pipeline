# `data/` — corpus & evaluation layout

This project deliberately separates the **open-source engine** from any **proprietary
data**. The same pipeline runs on a public demo corpus (shipped here, so the evaluation
is reproducible by anyone) and, locally, on private documents that are **never committed**.

| Path | Published? | What it holds |
|------|------------|---------------|
| `data/sample/` | ✅ committed | Small **public** traffic-domain corpus (public standards / regulations) used for the demo and the reproducible eval. |
| `data/eval/golden.jsonl` | ✅ committed | Labeled **golden set**: questions → relevant chunk ids. The ground truth for `recall@k`, `nDCG@k`, `MRR`. |
| `data/corpus/` | 🔒 **gitignored** | **Private** documents (e.g. internal business docs) for day-to-day local use. Never published. |
| `../storage/` | 🔒 gitignored | Built indexes, `meta.json` provenance, SQLite logs. |

## Why this split

Putting proprietary documents in a public repository would leak confidential / IP-owned
content. So the corpus is a **runtime input**, not part of the codebase:

- **Public / sample mode** → point the pipeline at `data/sample/`. The eval numbers in
  the README are reproducible from this committed corpus + golden set.
- **Private / daily-use mode** → drop your own documents into `data/corpus/` (gitignored)
  and run the exact same `make ingest` / `make serve`. Nothing leaves your machine.

Configuration (corpus path, models, retrieval params) lives in `src/rag/config.py` and is
driven by environment variables / `.env` — so switching corpora is a config change, not a
code change.
