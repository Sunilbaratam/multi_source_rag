"""
retriever.py — Hybrid retrieval: BM25 + semantic search fused with RRF,
re-ranked by a cross-encoder. Drop-in replacement for pure vector search.

Pipeline:
  query
    ├── ChromaDB semantic search  → top 20
    ├── BM25 keyword search       → top 20
    └── RRF merge                 → top 10
          └── Cross-encoder       → final top k (default 3)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from db import get_vectorstore as _db_get_vectorstore


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── vector store ──────────────────────────────────────────────────────────────

def _get_vectorstore() -> Chroma:
    # delegates to db.py — same model, same DB, no duplication
    return _db_get_vectorstore()


# ── BM25 retriever ────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Keyword retriever built over all chunks currently in ChromaDB.
    Rebuilt each time from the live collection — no separate index to maintain.
    """

    def __init__(self, vectorstore: Chroma):
        # pull every document out of ChromaDB to build the BM25 corpus
        collection = vectorstore._collection
        result = collection.get(include=["documents", "metadatas"])

        self.docs: List[Document] = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(result["documents"], result["metadatas"])
        ]

        # tokenise: lowercase, split on whitespace
        tokenised = [doc.page_content.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenised)
        print(f"[BM25] Index built over {len(self.docs)} chunks")

    def search(self, query: str, k: int = 20) -> List[Document]:
        """Return top-k chunks by BM25 score."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # pair each doc with its score and sort descending
        scored = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top = scored[:k]

        results = []
        for doc, score in top:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": round(float(score), 4)},
            )
            results.append(doc_copy)

        return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    results_lists: List[List[Document]],
    k: int = 60,
    top_n: int = 10,
) -> List[Document]:
    """
    Merge multiple ranked lists into one using RRF.
    Score for doc d = Σ  1 / (k + rank_i(d))  across all lists.

    Args:
        results_lists : list of ranked Document lists (semantic, BM25, …)
        k             : RRF constant (60 is standard; higher = smoother)
        top_n         : how many merged results to return

    Returns:
        Re-ranked Documents with rrf_score added to metadata.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for result_list in results_lists:
        for rank, doc in enumerate(result_list):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:40])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc   # keep latest version

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    fused = []
    for doc_id in sorted_ids[:top_n]:
        doc = doc_map[doc_id]
        fused_doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rrf_score": round(scores[doc_id], 6)},
        )
        fused.append(fused_doc)

    return fused


# ── cross-encoder re-ranker ───────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Re-scores (query, chunk) pairs together — much more accurate than
    cosine similarity alone because it sees both texts simultaneously.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"[RERANK] Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        """
        Score each (query, doc) pair and return the top_k highest-scoring docs.
        Adds ce_score to each document's metadata.
        """
        if not docs:
            return []

        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for doc, score in scored[:top_k]:
            reranked_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "ce_score": round(float(score), 4)},
            )
            results.append(reranked_doc)

        return results


# ── metadata filter builder ───────────────────────────────────────────────────

def build_filter(
    source_type: Optional[str] = None,
    file_name: Optional[str] = None,
    domain: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build a ChromaDB `where` filter dict from optional parameters.
    Pass to semantic_search() to narrow results before retrieval.

    Examples:
        build_filter(source_type="pdf")
        build_filter(file_name="report.pdf")
        build_filter(source_type="web", domain="arxiv.org")
    """
    conditions = []

    if source_type:
        conditions.append({"source_type": {"$eq": source_type}})
    if file_name:
        conditions.append({"file_name": {"$eq": file_name}})
    if domain:
        conditions.append({"domain": {"$eq": domain}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── main hybrid retriever ─────────────────────────────────────────────────────

class HybridRetriever:
    """
    Unified retriever that combines semantic search, BM25, RRF, and
    cross-encoder re-ranking into a single `.retrieve(query)` call.

    Usage:
        retriever = HybridRetriever()
        docs = retriever.retrieve("what is RAG?", top_k=3)
    """

    def __init__(self, semantic_k: int = 20, bm25_k: int = 20, rrf_top: int = 10):
        self.semantic_k = semantic_k
        self.bm25_k     = bm25_k
        self.rrf_top    = rrf_top

        self.vectorstore = _get_vectorstore()
        self.bm25        = BM25Retriever(self.vectorstore)
        self.reranker    = CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Full hybrid retrieval pipeline.

        Args:
            query  : natural language question
            top_k  : number of final chunks to return (after re-ranking)
            filter : optional ChromaDB metadata filter (from build_filter())

        Returns:
            top_k Documents, sorted by cross-encoder relevance score.
        """

        # 1. semantic search
        semantic_results = self.vectorstore.similarity_search(
            query,
            k=self.semantic_k,
            filter=filter,
        )

        # 2. BM25 keyword search (filters applied post-hoc)
        bm25_results = self.bm25.search(query, k=self.bm25_k)
        if filter:
            bm25_results = _apply_filter(bm25_results, filter)

        # 3. RRF merge
        fused = reciprocal_rank_fusion(
            [semantic_results, bm25_results],
            top_n=self.rrf_top,
        )

        # 4. cross-encoder re-rank
        final = self.reranker.rerank(query, fused, top_k=top_k)

        return final

    def retrieve_with_scores(self, query: str, top_k: int = 3) -> List[dict]:
        """
        Same as retrieve() but returns dicts with all intermediate scores —
        useful for debugging and for your eval baseline.
        """
        docs = self.retrieve(query, top_k=top_k)
        return [
            {
                "content":    doc.page_content,
                "source":     doc.metadata.get("source", ""),
                "source_type": doc.metadata.get("source_type", ""),
                "chunk_id":   doc.metadata.get("chunk_id", ""),
                "ce_score":   doc.metadata.get("ce_score", 0),
                "rrf_score":  doc.metadata.get("rrf_score", 0),
            }
            for doc in docs
        ]


def _apply_filter(docs: List[Document], filter: Dict) -> List[Document]:
    """Apply a simple equality filter to BM25 results post-hoc."""
    def matches(meta: dict) -> bool:
        for key, condition in filter.items():
            if key == "$and":
                return all(matches_condition(meta, c) for c in condition)
            return matches_condition(meta, {key: condition})
        return True

    def matches_condition(meta: dict, condition: dict) -> bool:
        for field, op in condition.items():
            if isinstance(op, dict):
                op_key, op_val = next(iter(op.items()))
                if op_key == "$eq" and meta.get(field) != op_val:
                    return False
            elif meta.get(field) != op:
                return False
        return True

    return [d for d in docs if matches(d.metadata)]