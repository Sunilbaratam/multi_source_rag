# Multi-Source RAG Pipeline

A production-grade Retrieval-Augmented Generation pipeline supporting PDFs,
web pages, and markdown files. Features hybrid BM25 + semantic retrieval,
cross-encoder re-ranking, and a FastAPI backend.

## Retrieval evaluation results

| Method | Precision@3 |
|---|---|
| Semantic-only (baseline) | 0.71 |
| Hybrid BM25 + Semantic + Re-rank | 0.87 |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY

# ingest documents
python ingest/embedder.py

# test retrieval
python retrieval/test_retrieval.py

# run eval
python retrieval/eval.py
```

## Architecture

See folder structure below for full layout.