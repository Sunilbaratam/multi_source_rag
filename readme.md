# Multi-Source RAG Pipeline

A production-grade Retrieval-Augmented Generation pipeline that ingests PDFs,
web pages, and markdown files and answers questions with cited, grounded responses.
**100% local and free** — no API keys required.

## Live demo

[Streamlit app link here after deploying]

## Architecture

```
Sources (PDF / Web / Markdown)
        ↓
   Ingestion pipeline
   └── loaders.py       — parse and clean
   └── chunker.py       — split into 512-char chunks
   └── embedder.py      — embed with all-MiniLM-L6-v2 → ChromaDB
        ↓
   Hybrid retrieval (per query)
   ├── Semantic search  → top 20 (ChromaDB cosine similarity)
   ├── BM25 search      → top 20 (keyword frequency)
   └── RRF fusion       → top 10 → Cross-encoder rerank → top 3
        ↓
   Generation (Ollama llama3.2)
   └── Citation-aware prompt → streamed answer with [1][2][3] citations
        ↓
   FastAPI backend  +  Streamlit frontend
```

## Evaluation results

### Retrieval (Week 2)

| Method | Precision@3 |
|---|---|
| Semantic-only (baseline) | 0.71 |
| Hybrid BM25 + Semantic + Re-rank | 0.87 |

### Generation quality (Week 4 — RAGAS)

| Metric | Score |
|---|---|
| Faithfulness | — |
| Answer relevancy | — |
| Context precision | — |

*(Run `python3 eval/ragas_eval.py` to populate these)*

## Quick start

```bash
# 1. Clone and install
git clone <your-repo>
cd rag-pipeline
pip install -r requirements.txt

# 2. Install and start Ollama
brew install ollama
ollama pull llama3.2
ollama serve          # keep running in a separate terminal

# 3. Ingest documents
python3 ingest/embedder.py

# 4a. Run the Streamlit UI
streamlit run frontend/app.py

# 4b. Or run the API
uvicorn api.main:app --reload --port 8000
```

## Docker

```bash
docker compose up
# API  → http://localhost:8000
# UI   → http://localhost:8501
```

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) | Free, local, 384-dim |
| Vector DB | ChromaDB | Zero setup, persists to disk |
| Keyword search | BM25 (rank-bm25) | Complements semantic search |
| Re-ranker | ms-marco-MiniLM cross-encoder | Improves precision@3 by ~22% |
| LLM | Ollama llama3.2 | Free, local, no API key |
| API | FastAPI | Async, streaming, typed |
| Frontend | Streamlit | Fast to build, easy to deploy |
| Eval | RAGAS | Industry-standard RAG metrics |

## Project structure

```
rag-pipeline/
├── db.py                    ← single source of truth for ChromaDB
├── ingest/
│   ├── loaders.py           ← PDF, web, markdown loaders
│   ├── chunker.py           ← text splitting
│   └── embedder.py          ← embed + store
├── retrieval/
│   ├── retriever.py         ← BM25 + semantic + RRF + cross-encoder
│   ├── filters.py           ← metadata filter builder
│   ├── eval.py              ← precision@3 evaluation
│   └── test_retrieval.py    ← side-by-side comparison
├── generation/
│   └── chain.py             ← RAG chain with citations + memory
├── api/
│   └── main.py              ← FastAPI endpoints
├── frontend/
│   └── app.py               ← Streamlit chat UI
├── eval/
│   └── ragas_eval.py        ← RAGAS faithfulness/relevancy/precision
├── data/raw/                ← your source documents
├── chroma_db/               ← vector store (auto-created)
├── Dockerfile
└── docker-compose.yml
```