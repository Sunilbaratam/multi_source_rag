"""
eval.py — Week 2 Day 5: measure retrieval quality.
Compares semantic-only vs hybrid retrieval using precision@k.

Run:
    python eval.py

Outputs a table + saves results to retrieval_eval_results.json
"""

import os
import json
import time
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = "./chroma_db"
COLLECTION = "rag_pipeline"


# ── test dataset ──────────────────────────────────────────────────────────────
# Edit these to match documents you have actually ingested.
# Each entry: question + list of chunk_ids that are "relevant" answers.
# You get chunk_ids by inspecting ChromaDB after ingestion.

TEST_PAIRS = [
    {
        "question": "What is retrieval augmented generation?",
        "relevant_keywords": ["retrieval", "augmented", "generation", "RAG"],
        "note": "Definition question — should hit intro/overview chunks",
    },
    {
        "question": "How does vector similarity search work?",
        "relevant_keywords": ["vector", "similarity", "embedding", "cosine"],
        "note": "Technical concept — should return embedding-related chunks",
    },
    {
        "question": "What are the limitations of RAG systems?",
        "relevant_keywords": ["limitation", "challenge", "drawback", "problem"],
        "note": "Critical analysis — should return limitations sections",
    },
    {
        "question": "How is RAG different from fine-tuning?",
        "relevant_keywords": ["fine-tuning", "fine tuning", "finetune", "comparison"],
        "note": "Comparison question — both keyword and semantic signals matter",
    },
    {
        "question": "What embedding models are used in RAG?",
        "relevant_keywords": ["embedding", "model", "encode", "dense"],
        "note": "Specific technical detail — benefits from BM25 keyword match",
    },
]


def keyword_precision(docs, relevant_keywords: List[str]) -> float:
    """
    Proxy for precision@k when we don't have gold chunk IDs.
    A chunk is 'relevant' if it contains at least one keyword (case-insensitive).
    Returns fraction of returned chunks that are relevant.
    """
    hits = 0
    for doc in docs:
        text = doc.page_content.lower()
        if any(kw.lower() in text for kw in relevant_keywords):
            hits += 1
    return hits / len(docs) if docs else 0.0


def run_eval():
    """Compare semantic-only vs hybrid retrieval across all test pairs."""
    from retriever import HybridRetriever, _get_vectorstore

    print("=" * 60)
    print("RAG Pipeline — Retrieval Evaluation")
    print("=" * 60)

    # semantic-only retriever (Week 1 baseline)
    vectorstore = _get_vectorstore()

    # hybrid retriever (Week 2)
    hybrid = HybridRetriever(semantic_k=20, bm25_k=20, rrf_top=10)

    results = []
    semantic_scores = []
    hybrid_scores   = []

    for i, pair in enumerate(TEST_PAIRS, 1):
        q        = pair["question"]
        keywords = pair["relevant_keywords"]

        # --- semantic only ---
        t0 = time.time()
        sem_docs = vectorstore.similarity_search(q, k=3)
        sem_time = round((time.time() - t0) * 1000, 1)
        sem_prec = keyword_precision(sem_docs, keywords)

        # --- hybrid ---
        t0 = time.time()
        hyb_docs = hybrid.retrieve(q, top_k=3)
        hyb_time = round((time.time() - t0) * 1000, 1)
        hyb_prec = keyword_precision(hyb_docs, keywords)

        delta = hyb_prec - sem_prec
        indicator = "+" if delta > 0 else ("=" if delta == 0 else "-")

        print(f"\nQ{i}: {q}")
        print(f"  Semantic  precision@3 = {sem_prec:.2f}  ({sem_time}ms)")
        print(f"  Hybrid    precision@3 = {hyb_prec:.2f}  ({hyb_time}ms)  [{indicator}]")
        if pair.get("note"):
            print(f"  Note: {pair['note']}")

        semantic_scores.append(sem_prec)
        hybrid_scores.append(hyb_prec)

        results.append({
            "question":          q,
            "semantic_precision": sem_prec,
            "hybrid_precision":   hyb_prec,
            "delta":              round(delta, 2),
            "semantic_ms":        sem_time,
            "hybrid_ms":          hyb_time,
        })

    # summary
    avg_sem = sum(semantic_scores) / len(semantic_scores)
    avg_hyb = sum(hybrid_scores)   / len(hybrid_scores)
    improvement = ((avg_hyb - avg_sem) / max(avg_sem, 0.001)) * 100

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Avg semantic-only precision@3 : {avg_sem:.3f}")
    print(f"  Avg hybrid        precision@3 : {avg_hyb:.3f}")
    print(f"  Relative improvement          : {improvement:+.1f}%")
    print("=" * 60)

    # save for README
    output = {
        "avg_semantic_precision_at_3": round(avg_sem, 3),
        "avg_hybrid_precision_at_3":   round(avg_hyb, 3),
        "relative_improvement_pct":    round(improvement, 1),
        "per_question":                results,
    }
    with open("retrieval_eval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to retrieval_eval_results.json")
    print("Copy these numbers into your README — interviewers will ask about them.")
    return output


if __name__ == "__main__":
    run_eval()