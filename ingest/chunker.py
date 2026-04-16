"""
chunker.py — Splits raw Documents into smaller chunks ready for embedding.
Preserves all source metadata and adds chunk-level fields.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """
    Split documents into overlapping chunks.

    - PDFs/web pages use character splitting (512 chars, 64 overlap).
    - Markdown sections are already split by heading in the loader,
      so they get a larger chunk size to stay coherent.
    - Chunk index and total chunk count are added to metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []

    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")

        # Markdown sections are already semantically split — use larger window
        if source_type == "markdown":
            md_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=128,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            raw_chunks = md_splitter.split_documents([doc])
        else:
            raw_chunks = splitter.split_documents([doc])

        # enrich each chunk with position metadata
        for i, chunk in enumerate(raw_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(raw_chunks)
            chunk.metadata["chunk_size"]  = len(chunk.page_content)
            # make chunk_id unique per-chunk (not per-source-page)
            import hashlib
            raw_id = f"{doc.metadata.get('source')}::{doc.metadata.get('page', 0)}::{i}"
            chunk.metadata["chunk_id"] = hashlib.md5(raw_id.encode()).hexdigest()[:12]

        all_chunks.extend(raw_chunks)

    print(f"[CHUNK] {len(docs)} docs → {len(all_chunks)} chunks "
          f"(avg {len(all_chunks)//max(len(docs),1)} per doc)")
    return all_chunks


# if __name__ == "__main__":
#     from loaders import load_sources
#     docs = load_sources(["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"])
#     chunks = chunk_documents(docs)
#     print(f"\nSample chunk:")
#     print(f"  Content : {chunks[0].page_content[:200]}...")
#     print(f"  Metadata: {chunks[0].metadata}")