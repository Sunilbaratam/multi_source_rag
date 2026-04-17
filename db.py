"""
db.py — Single source of truth for the ChromaDB connection.
Both embedder.py and retriever.py import from here.
Uses HuggingFace all-MiniLM-L6-v2 — runs locally, completely free.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "./chroma_db"
COLLECTION = "rag_pipeline"

_embeddings = None  # module-level cache — model loads once per process


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the shared embedding model.
    Cached so the 80MB model is only loaded once per Python process,
    not on every function call.
    """
    global _embeddings
    if _embeddings is None:
        print("[EMBED] Loading all-MiniLM-L6-v2 (first run downloads ~80MB)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},   # change to "mps" on Apple Silicon for speed
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent ChromaDB collection."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )