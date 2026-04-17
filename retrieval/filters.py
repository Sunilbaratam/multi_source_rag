"""
filters.py — Metadata filtering helpers for Day 4.
Build ChromaDB-compatible `where` dicts to narrow retrieval
by source type, file name, domain, or date range.

Usage:
    from filters import FilterBuilder
    f = FilterBuilder().source_type("pdf").domain("arxiv.org").build()
    docs = retriever.retrieve(query, filter=f)
"""

from typing import Optional, Dict, Any
from datetime import datetime


class FilterBuilder:
    """
    Fluent builder for ChromaDB `where` filter dicts.
    Chain methods and call .build() at the end.

    Examples:
        FilterBuilder().source_type("pdf").build()
        FilterBuilder().source_type("web").domain("arxiv.org").build()
        FilterBuilder().ingested_after("2024-01-01").build()
    """

    def __init__(self):
        self._conditions = []

    def source_type(self, type_: str) -> "FilterBuilder":
        """Filter to a specific source type: 'pdf', 'web', or 'markdown'."""
        self._conditions.append({"source_type": {"$eq": type_}})
        return self

    def file_name(self, name: str) -> "FilterBuilder":
        """Filter to a specific file name, e.g. 'report.pdf'."""
        self._conditions.append({"file_name": {"$eq": name}})
        return self

    def domain(self, domain: str) -> "FilterBuilder":
        """Filter web sources to a specific domain, e.g. 'arxiv.org'."""
        self._conditions.append({"domain": {"$eq": domain}})
        return self

    def ingested_after(self, date_str: str) -> "FilterBuilder":
        """
        Filter to chunks ingested after a given date.
        date_str format: 'YYYY-MM-DD'
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d").isoformat()
        self._conditions.append({"ingested_at": {"$gte": dt}})
        return self

    def ingested_before(self, date_str: str) -> "FilterBuilder":
        """Filter to chunks ingested before a given date."""
        dt = datetime.strptime(date_str, "%Y-%m-%d").isoformat()
        self._conditions.append({"ingested_at": {"$lte": dt}})
        return self

    def build(self) -> Optional[Dict[str, Any]]:
        """
        Build the final ChromaDB `where` dict.
        Returns None if no conditions (meaning: no filter applied).
        """
        if not self._conditions:
            return None
        if len(self._conditions) == 1:
            return self._conditions[0]
        return {"$and": self._conditions}


# ── convenience functions ─────────────────────────────────────────────────────

def pdfs_only() -> Dict:
    return FilterBuilder().source_type("pdf").build()

def web_only() -> Dict:
    return FilterBuilder().source_type("web").build()

def markdown_only() -> Dict:
    return FilterBuilder().source_type("markdown").build()

def from_domain(domain: str) -> Dict:
    return FilterBuilder().source_type("web").domain(domain).build()

def recent(days: int = 7) -> Dict:
    """Chunks ingested in the last N days."""
    from datetime import timedelta
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    return FilterBuilder().ingested_after(cutoff).build()


# ── usage demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from retriever import HybridRetriever

    retriever = HybridRetriever()

    print("--- PDF sources only ---")
    docs = retriever.retrieve(
        "what is retrieval augmented generation?",
        top_k=3,
        filter=pdfs_only(),
    )
    for d in docs:
        print(f"  [{d.metadata['source_type']}] ce={d.metadata.get('ce_score',0):.3f} | {d.page_content[:100]}...")

    print("\n--- Web sources only ---")
    docs = retriever.retrieve(
        "what is retrieval augmented generation?",
        top_k=3,
        filter=web_only(),
    )
    for d in docs:
        print(f"  [{d.metadata['source_type']}] ce={d.metadata.get('ce_score',0):.3f} | {d.page_content[:100]}...")

    print("\n--- Custom filter: arxiv.org only ---")
    docs = retriever.retrieve(
        "transformer architecture",
        top_k=3,
        filter=from_domain("arxiv.org"),
    )
    for d in docs:
        print(f"  [{d.metadata.get('domain','')}] {d.page_content[:100]}...")