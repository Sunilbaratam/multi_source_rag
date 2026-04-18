"""
app.py — Streamlit chat UI for the RAG pipeline.
Week 4, Day 1-2.

Run:
    streamlit run frontend/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from generation.chain import RAGChain
from ingest.loaders import load_source
from ingest.chunker import chunk_documents
from ingest.embedder import embed_and_store
from db import get_vectorstore


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide",
)

st.title("Multi-Source RAG Pipeline")
st.caption("Hybrid BM25 + Semantic retrieval · Cross-encoder re-ranking · Ollama LLM · 100% local & free")


# ── session state ─────────────────────────────────────────────────────────────

if "chain" not in st.session_state:
    with st.spinner("Loading models (first run may take ~30s)..."):
        st.session_state.chain = RAGChain()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []


# ── sidebar — document ingestion ──────────────────────────────────────────────

with st.sidebar:
    st.header("Ingest documents")

    # URL ingestion
    st.subheader("From URL")
    url_input = st.text_input("Paste a web URL", placeholder="https://en.wikipedia.org/wiki/...")
    if st.button("Ingest URL", disabled=not url_input):
        with st.spinner(f"Ingesting {url_input[:50]}..."):
            try:
                docs   = load_source(url_input)
                chunks = chunk_documents(docs)
                count  = embed_and_store(chunks)
                st.session_state.ingested_sources.append(
                    {"type": "web", "name": url_input[:60], "chunks": count}
                )
                st.success(f"Stored {count} chunks")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    # File ingestion
    st.subheader("From file")
    uploaded = st.file_uploader(
        "Upload PDF or Markdown",
        type=["pdf", "md", "markdown", "txt"],
    )
    if uploaded and st.button("Ingest file"):
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                # save to temp file
                import tempfile, shutil
                suffix = os.path.splitext(uploaded.name)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(uploaded, tmp)
                    tmp_path = tmp.name

                docs   = load_source(tmp_path)
                chunks = chunk_documents(docs)
                count  = embed_and_store(chunks)
                os.unlink(tmp_path)

                st.session_state.ingested_sources.append(
                    {"type": "file", "name": uploaded.name, "chunks": count}
                )
                st.success(f"Stored {count} chunks from {uploaded.name}")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    # Source filter
    st.subheader("Filter by source type")
    source_filter = st.selectbox(
        "Search across",
        ["All sources", "PDFs only", "Web only", "Markdown only"],
    )

    filter_map = {
        "All sources":    None,
        "PDFs only":      {"source_type": {"$eq": "pdf"}},
        "Web only":       {"source_type": {"$eq": "web"}},
        "Markdown only":  {"source_type": {"$eq": "markdown"}},
    }
    active_filter = filter_map[source_filter]

    st.divider()

    # Ingested sources list
    st.subheader("Ingested sources")
    try:
        db = get_vectorstore()
        total_chunks = db._collection.count()
        st.metric("Total chunks in DB", total_chunks)
    except Exception:
        st.metric("Total chunks in DB", 0)

    if st.session_state.ingested_sources:
        for src in st.session_state.ingested_sources[-5:]:  # show last 5
            icon = "📄" if src["type"] == "file" else "🌐"
            st.caption(f"{icon} {src['name']} — {src['chunks']} chunks")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chain.reset_history()
        st.rerun()


# ── main chat area ────────────────────────────────────────────────────────────

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # show sources expander for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**[{src['index']}]** `{src['source_type']}` — "
                        f"{src['source'][:80]}  \n"
                        f"*Relevance score: {src['ce_score']:.3f}*  \n"
                        f"> {src['preview'][:200]}..."
                    )


# chat input
if question := st.chat_input("Ask a question about your documents..."):
    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # stream assistant response
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        full_answer = ""
        try:
            # get sources first (non-streaming) for display
            result = st.session_state.chain.query(
                question,
                filter=active_filter,
            )
            full_answer = result["answer"]
            sources     = result["sources"]

            # simulate streaming display
            displayed = ""
            for char in full_answer:
                displayed += char
                answer_placeholder.markdown(displayed + "▌")

            answer_placeholder.markdown(full_answer)

            # show sources expander
            with st.expander("Sources used", expanded=False):
                for src in sources:
                    st.markdown(
                        f"**[{src['index']}]** `{src['source_type']}` — "
                        f"{src['source'][:80]}  \n"
                        f"*Relevance score: {src['ce_score']:.3f}*  \n"
                        f"> {src['preview'][:200]}..."
                    )

        except Exception as e:
            full_answer = f"Error: {e}"
            answer_placeholder.error(full_answer)
            sources = []

    # save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": sources if "sources" in dir() else [],
    })