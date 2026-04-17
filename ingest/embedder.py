"""
embedder.py — Embeds chunks and upserts them into ChromaDB.
Run this after chunker.py to populate your vector store.
Embedding model lives in db.py — do not redefine it here.
"""

from typing import List
from langchain_core.documents import Document
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db import get_vectorstore


def embed_and_store(chunks: List[Document]) -> int:
    """
    Embed chunks and upsert into ChromaDB.
    Uses chunk_id as the document ID so re-running the same source
    is idempotent (updates rather than duplicates).

    Returns: number of chunks stored.
    """
    db = get_vectorstore()

    ids   = [c.metadata["chunk_id"] for c in chunks]
    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]

    db.add_texts(texts=texts, metadatas=metas, ids=ids)
    print(f"[EMBED] Stored {len(chunks)} chunks in ChromaDB at './chroma_db'")
    return len(chunks)


def similarity_search(query: str, k: int = 5):
    """Quick test helper — returns top-k chunks for a query string."""
    db = get_vectorstore()
    return db.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from loaders import load_sources
    from chunker import chunk_documents

    sources = ["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    docs   = load_sources(sources)
    chunks = chunk_documents(docs)
    embed_and_store(chunks)

    print("\nTest query: 'what is retrieval augmented generation?'")
    hits = similarity_search("what is retrieval augmented generation?", k=3)
    for doc, score in hits:
        print(f"  Score {score:.3f} | {doc.metadata.get('source_type')} | "
              f"{doc.page_content[:120]}...")