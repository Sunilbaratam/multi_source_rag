"""
test_retrieval.py — Quick manual test to run on Day 1 and Day 3.
Prints side-by-side comparison of semantic vs BM25 vs hybrid results.

Run after you have documents in ChromaDB:
    python test_retrieval.py
"""

import os
from retriever import HybridRetriever, _get_vectorstore, BM25Retriever

QUERY = "what is retrieval augmented generation?"


def print_docs(label: str, docs, max_chars: int = 120):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    for i, doc in enumerate(docs, 1):
        src  = doc.metadata.get("source_type", "?")
        ce   = doc.metadata.get("ce_score", "")
        rrf  = doc.metadata.get("rrf_score", "")
        bm25 = doc.metadata.get("bm25_score", "")
        score_str = ""
        if ce   != "": score_str += f"ce={ce:.3f} "
        if rrf  != "": score_str += f"rrf={rrf:.4f} "
        if bm25 != "": score_str += f"bm25={bm25:.2f}"
        text = doc.page_content[:max_chars].replace("\n", " ")
        print(f"  [{i}] [{src}] {score_str}")
        print(f"       {text}...")


def main():
    print(f"\nQuery: \"{QUERY}\"\n")

    # --- semantic only (Week 1 baseline) ---
    vs = _get_vectorstore()
    sem_docs = vs.similarity_search(QUERY, k=3)
    print_docs("Semantic search (Week 1 baseline)", sem_docs)

    # --- BM25 only ---
    bm25 = BM25Retriever(vs)
    bm25_docs = bm25.search(QUERY, k=3)
    print_docs("BM25 keyword search", bm25_docs)

    # --- hybrid (full pipeline) ---
    retriever = HybridRetriever()
    hybrid_docs = retriever.retrieve(QUERY, top_k=3)
    print_docs("Hybrid: BM25 + Semantic + RRF + Cross-encoder", hybrid_docs)

    print(f"\n{'='*55}")
    print("  Inspect the differences above.")
    print("  Hybrid should return the most relevant chunks.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()