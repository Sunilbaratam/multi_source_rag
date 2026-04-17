"""
main.py — FastAPI backend for the RAG pipeline.
Week 3, Day 3-4.

Endpoints:
    POST /ingest          — ingest a file or URL into ChromaDB
    POST /query           — ask a question, get a streamed answer
    GET  /sources         — list all ingested sources
    DELETE /sources/{id}  — remove a source by chunk_id prefix

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile
import shutil

from ingest.loaders import load_source
from ingest.chunker import chunk_documents
from ingest.embedder import embed_and_store
from generation.chain import RAGChain
from db import get_vectorstore


app = FastAPI(
    title="Multi-Source RAG Pipeline",
    description="Hybrid BM25 + semantic retrieval with cross-encoder re-ranking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# one chain per session_id — simple in-memory store
# (use Redis for production)
_chains: dict[str, RAGChain] = {}


def get_chain(session_id: str = "default") -> RAGChain:
    if session_id not in _chains:
        _chains[session_id] = RAGChain()
    return _chains[session_id]


# ── request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    source_type: Optional[str] = None   # filter: "pdf", "web", "markdown"
    stream: bool = True


class IngestURLRequest(BaseModel):
    url: str


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "endpoints": ["/ingest", "/ingest/url", "/query", "/sources"],
    }


@app.post("/ingest/url")
def ingest_url(req: IngestURLRequest):
    """
    Ingest a web URL into ChromaDB.

    Example:
        curl -X POST http://localhost:8000/ingest/url \\
             -H "Content-Type: application/json" \\
             -d '{"url": "https://en.wikipedia.org/wiki/RAG"}'
    """
    try:
        docs   = load_source(req.url)
        chunks = chunk_documents(docs)
        count  = embed_and_store(chunks)
        return {
            "status":       "success",
            "source":       req.url,
            "chunks_stored": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest an uploaded file (.pdf or .md) into ChromaDB.

    Example:
        curl -X POST http://localhost:8000/ingest/file \\
             -F "file=@report.pdf"
    """
    allowed = {".pdf", ".md", ".markdown", ".txt"}
    suffix  = os.path.splitext(file.filename)[-1].lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    # save upload to temp file (loaders need a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        docs   = load_source(tmp_path)
        chunks = chunk_documents(docs)
        count  = embed_and_store(chunks)
        return {
            "status":        "success",
            "filename":      file.filename,
            "chunks_stored": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/query")
async def query(req: QueryRequest):
    """
    Ask a question against ingested documents.
    Returns a streamed answer with inline citations.

    Example:
        curl -X POST http://localhost:8000/query \\
             -H "Content-Type: application/json" \\
             -d '{"question": "What is RAG?", "stream": true}'
    """
    chain = get_chain(req.session_id)

    # build optional metadata filter
    filter_dict = None
    if req.source_type:
        filter_dict = {"source_type": {"$eq": req.source_type}}

    if req.stream:
        # Server-Sent Events streaming
        def event_stream():
            for token in chain.stream(req.question, filter=filter_dict):
                # escape newlines for SSE format
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        # non-streaming — return full result at once
        result = chain.query(req.question, filter=filter_dict)
        return result


@app.get("/sources")
def list_sources():
    """
    List all ingested sources with chunk counts.

    Example:
        curl http://localhost:8000/sources
    """
    db = get_vectorstore()
    result = db._collection.get(include=["metadatas"])
    metadatas = result["metadatas"]

    # group by source
    sources: dict[str, dict] = {}
    for meta in metadatas:
        src = meta.get("source", "unknown")
        if src not in sources:
            sources[src] = {
                "source":      src,
                "source_type": meta.get("source_type", ""),
                "chunk_count": 0,
                "ingested_at": meta.get("ingested_at", ""),
            }
        sources[src]["chunk_count"] += 1

    return {
        "total_chunks":  len(metadatas),
        "total_sources": len(sources),
        "sources":       list(sources.values()),
    }


@app.delete("/sources/{chunk_id_prefix}")
def delete_source(chunk_id_prefix: str):
    """
    Delete all chunks whose chunk_id starts with the given prefix.
    Get the chunk_id from the /sources response.

    Example:
        curl -X DELETE http://localhost:8000/sources/3a7f
    """
    db = get_vectorstore()
    result = db._collection.get(include=["metadatas"])

    ids_to_delete = [
        id_ for id_, meta in zip(result["ids"], result["metadatas"])
        if meta.get("chunk_id", "").startswith(chunk_id_prefix)
        or meta.get("source", "").endswith(chunk_id_prefix)
    ]

    if not ids_to_delete:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found matching '{chunk_id_prefix}'",
        )

    db._collection.delete(ids=ids_to_delete)
    return {
        "status":  "deleted",
        "removed": len(ids_to_delete),
    }


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in _chains:
        _chains[session_id].reset_history()
    return {"status": "reset", "session_id": session_id}