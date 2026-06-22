# Actual Implementation & Experiments (LT6582 Capstone)

This document describes what is **actually implemented and run** for the capstone report.
It supersedes the aspirational/marketing descriptions in the older `README.md`.

## What runs where
- **Orchestration code**: plain Python (`real_pipeline/`), runs on any machine — no GPU required.
- **Embeddings**: `BAAI/bge-small-en-v1.5` via `fastembed`, **run locally on CPU** (free, private, reproducible). *This replaces the proposal's planned `text-embedding-3-small` (OpenAI) — we use a local open-weights embedding instead.*
- **Vector search**: FAISS (in-memory, `IndexFlatIP`, cosine).
- **LLM**: DeepSeek-V3 (`deepseek-chat`) via the DeepSeek cloud API (needs internet + an API key in `.env`).

## Pipeline components (`real_pipeline/`)

| File | What it does |
|---|---|
| `aes_conditions_v2.py` | Part 1 AES: 4-condition LLM scoring (Prompt A=direct rubric / B=by-dimension × temp 0.1/0.7) on 100 dev essays; reports Pearson r, MSE, and a post-hoc score-calibration analysis. |
| `rag_ablation.py` | Part 2 AWE: **real RAG** (fastembed BGE-small + FAISS over the knowledge base). Generates feedback with RAG **off vs on** for 5 essays × 2 temperatures — a clean ablation (identical base prompt; only the injected retrieved knowledge differs). |
| `agentic_workflow.py` | **Real Agentic Workflow**: 4 dimension experts (Grammar/Vocabulary/Organisation/Content), each with its own RAG retrieval → a **self-correction verifier** that drops unsupported/hallucinated findings → a composer that writes the final feedback. Implements decompose → tool-use → self-correct. |
| `extract_items.py` | Decomposes free-form feedback into discrete flagged-error items (for human audit). |
| `build_audit_tool.py` | Generates a single-file, **blind** HTML audit tool for the Feedback Auditing group. |

## Honest notes (read before writing the report)
- **Agentic Workflow** is implemented as an **explicit multi-step orchestration of LLM calls** (decomposition + per-dimension RAG + self-correction + compose), *not* via the LangChain framework named in the older README. The behaviour matches the proposal's "plan / execute / monitor / self-correct" description.
- The **self-correction drop-rate is a tunable parameter** (it varied 0–30% across verifier strictness settings; the committed balanced setting drops ~6%). Whether agentic genuinely reduces false positives vs single-pass is decided by the **human audit**, not this automatic number.
- The Part-2 audit uses **human judgement** as the gold standard (the 5 essays are ICLE learner essays without ELLIPSE analytic scores).
- The knowledge base in `knowledge_base/` is a **bootstrap placeholder**; the official, curated knowledge base is produced separately by the Knowledge Base group.

## Results (`results/`)
- `aes_proposal_predictions.csv` — per-essay AES predictions (4 conditions).
- `rag_ablation/` — RAG off-vs-on feedback (paired, 5 essays × 2 temps).
- `agentic/` — agentic feedback + self-correction logs.
- `audit/` — extracted items, the blind audit tool, and the metrics script.

## How to run
1. `pip install -r requirements.txt` (plus `fastembed faiss-cpu`).
2. Put `DEEPSEEK_API_KEY=...` in a `.env` file at the repo root (never commit it — it is gitignored).
3. Run any script in `real_pipeline/` from the repo root.
