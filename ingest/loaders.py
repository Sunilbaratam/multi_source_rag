"""
loaders.py — Multi-source document loaders for RAG pipeline.
Supports: PDF files, web URLs, and markdown files.
Each loader returns a list of LangChain Document objects with rich metadata.
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List

import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_chunk_id(source: str, page: int, index: int) -> str:
    """Stable, unique ID for each chunk — used for deduplication and citations."""
    raw = f"{source}::{page}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _base_metadata(source: str, source_type: str) -> dict:
    return {
        "source": source,
        "source_type": source_type,
        "ingested_at": datetime.utcnow().isoformat(),
    }


# ── PDF loader ────────────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF and return one Document per page.
    Metadata includes: file name, page number, total pages, file size.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    loader = PyPDFLoader(str(path))
    raw_pages = loader.load()

    docs = []
    total_pages = len(raw_pages)

    for page_num, page in enumerate(raw_pages, start=1):
        text = page.page_content.strip()
        if not text:           # skip blank pages
            continue

        metadata = {
            **_base_metadata(str(path), "pdf"),
            "file_name":    path.name,
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "page":         page_num,
            "total_pages":  total_pages,
            "chunk_id":     _make_chunk_id(str(path), page_num, 0),
        }
        docs.append(Document(page_content=text, metadata=metadata))

    print(f"[PDF]  {path.name} — {len(docs)} pages loaded")
    return docs


# ── Web loader ────────────────────────────────────────────────────────────────

def load_web(url: str) -> List[Document]:
    """
    Scrape a web page, strip boilerplate, return a single clean Document.
    Falls back to LangChain WebBaseLoader if custom scraping fails.
    Metadata includes: URL, page title, domain, word count.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (RAG-Pipeline/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # remove noise elements
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "form", "iframe"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else url

        # prefer <article> or <main>, fall back to <body>
        content_el = (
            soup.find("article")
            or soup.find("main")
            or soup.find("body")
        )
        text = content_el.get_text(separator="\n", strip=True) if content_el else ""

        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        metadata = {
            **_base_metadata(url, "web"),
            "title":      title,
            "domain":     domain,
            "word_count": len(text.split()),
            "chunk_id":   _make_chunk_id(url, 0, 0),
        }

    except Exception as e:
        print(f"[WEB]  Custom scrape failed ({e}), falling back to WebBaseLoader")
        loader = WebBaseLoader(url)
        raw = loader.load()
        if not raw:
            raise ValueError(f"Could not load content from: {url}")
        text = raw[0].page_content
        metadata = {
            **_base_metadata(url, "web"),
            "chunk_id": _make_chunk_id(url, 0, 0),
        }

    doc = Document(page_content=text.strip(), metadata=metadata)
    print(f"[WEB]  {url[:60]}{'...' if len(url)>60 else ''} — {len(text.split())} words loaded")
    return [doc]


# ── Markdown loader ───────────────────────────────────────────────────────────

def load_markdown(file_path: str) -> List[Document]:
    """
    Load a markdown file, split on ## headings into logical sections,
    and return one Document per section.
    Metadata includes: file name, section heading, section index.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    if path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError(f"Expected a .md file, got: {path.suffix}")

    raw_text = path.read_text(encoding="utf-8")

    # split into sections on any heading (# or ##)
    import re
    sections = re.split(r"\n(?=#{1,2} )", raw_text)

    docs = []
    for idx, section in enumerate(sections):
        text = section.strip()
        if not text:
            continue

        # extract heading from first line
        first_line = text.splitlines()[0]
        heading = first_line.lstrip("#").strip() if first_line.startswith("#") else "Introduction"

        metadata = {
            **_base_metadata(str(path), "markdown"),
            "file_name":     path.name,
            "section":       heading,
            "section_index": idx,
            "chunk_id":      _make_chunk_id(str(path), 0, idx),
        }
        docs.append(Document(page_content=text, metadata=metadata))

    print(f"[MD]   {path.name} — {len(docs)} sections loaded")
    return docs


# ── unified loader ────────────────────────────────────────────────────────────

def load_source(source: str) -> List[Document]:
    """
    Auto-detect source type and route to the right loader.

    Args:
        source: a file path (.pdf / .md / .markdown) or a http(s):// URL

    Returns:
        List of LangChain Document objects ready for chunking.
    """
    s = source.strip()

    if s.startswith("http://") or s.startswith("https://"):
        return load_web(s)

    path = Path(s)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return load_markdown(s) if False else load_pdf(s)  # keeps type checker happy
    if ext in {".md", ".markdown"}:
        return load_markdown(s)

    # try as plain text fallback
    print(f"[AUTO] Unknown extension '{ext}', attempting plain text load")
    loader = TextLoader(s, encoding="utf-8")
    raw = loader.load()
    for doc in raw:
        doc.metadata.update(_base_metadata(s, "text"))
    return raw


# ── batch loader ──────────────────────────────────────────────────────────────

def load_sources(sources: List[str]) -> List[Document]:
    """
    Load multiple sources (mix of PDFs, URLs, markdown files) in one call.
    Failures are logged and skipped — does not abort the whole batch.

    Returns:
        Flat list of all Documents from all sources.
    """
    all_docs: List[Document] = []
    failed: List[str] = []

    for src in sources:
        try:
            docs = load_source(src)
            all_docs.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load '{src}': {e}")
            failed.append(src)

    print(f"\n--- Load summary ---")
    print(f"  Sources attempted : {len(sources)}")
    print(f"  Sources loaded    : {len(sources) - len(failed)}")
    print(f"  Documents created : {len(all_docs)}")
    if failed:
        print(f"  Failed            : {failed}")

    return all_docs


# # ── quick test ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     # Test with one of each source type.
#     # Replace these with real files / URLs you want to ingest.
#     test_sources = [
#         "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
#         # "./data/raw/sample.pdf",
#         # "./data/raw/notes.md",
#     ]

#     docs = load_sources(test_sources)

#     print(f"\nSample document:")
#     print(f"  Content preview : {docs[0].page_content[:200]}...")
#     print(f"  Metadata        : {docs[0].metadata}")