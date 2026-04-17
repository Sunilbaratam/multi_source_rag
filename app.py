import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List

# Import your existing logic
from ingest.loaders import load_sources
from ingest.chunker import chunk_documents
from ingest.embedder import embed_and_store, similarity_search

app = FastAPI(title="Multi-Source RAG API", version="1.1.0")

# --- Schemas ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class IngestURLRequest(BaseModel):
    url: str

# --- Background Task Logic ---
def process_ingestion(sources: List[str]):
    """The heavy lifting: Load -> Chunk -> Embed"""
    docs = load_sources(sources)
    chunks = chunk_documents(docs)
    embed_and_store(chunks)

# --- Endpoints ---

@app.post("/ingest/file")
async def ingest_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF or text file to the vector store."""
    if not file.filename.endswith(('.pdf', '.txt', '.md')):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Save file temporarily to disk
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run embedding in background so user doesn't wait
    background_tasks.add_task(process_ingestion, [temp_path])
    
    return {"message": f"Ingestion started for {file.filename}. It will be searchable in a few moments."}

@app.post("/ingest/url")
async def ingest_url(request: IngestURLRequest, background_tasks: BackgroundTasks):
    """Ingest content from a website URL."""
    background_tasks.add_task(process_ingestion, [request.url])
    return {"message": f"Crawling and embedding {request.url} in the background."}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Retrieve relevant chunks for a question."""
    hits = similarity_search(request.question, k=request.top_k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "score": round(float(score), 4)
        } for doc, score in hits
    ]

@app.delete("/reset")
async def reset_database():
    from embedder import get_vectorstore
    db = get_vectorstore()
    db.delete_collection()
    return {"message": "Vector database cleared."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)