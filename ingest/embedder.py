"""
embedder.py — Embeds chunks and upserts them into ChromaDB.
Run this after chunker.py to populate your vector store.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


CHROMA_DIR = "./chroma_db"
COLLECTION  = "multi_source_rag_pipeline"


# 1. Install the local embedding package
# pip install langchain-huggingface

from langchain_huggingface import HuggingFaceEmbeddings

def get_vectorstore() -> Chroma:
    # This downloads a model to your Mac and runs it locally for FREE
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

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
    print(f"[EMBED] Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DIR}'")
    return len(chunks)


def similarity_search(query: str, k: int = 5) -> List[Document]:
    """Quick test helper — returns top-k chunks for a query string."""
    db = get_vectorstore()
    results = db.similarity_search_with_score(query, k=k)
    return results


# if __name__ == "__main__":
#     from loaders import load_sources
#     from chunker import chunk_documents

#     sources = ["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
#     docs   = load_sources(sources)
#     chunks = chunk_documents(docs)
#     embed_and_store(chunks)

#     print("\nTest query: 'what is retrieval augmented generation?'")
#     hits = similarity_search("what is retrieval augmented generation?", k=3)
#     print(hits)
#     for doc, score in hits:
#         print(f"  Score {score:.3f} | {doc.metadata.get('source_type')} | "
#               f"{doc.page_content[:120]}...")