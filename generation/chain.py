"""
chain.py — RAG generation chain with citation-aware streaming responses.
Wires the hybrid retriever → citation prompt → LLM → streamed answer.

Uses Ollama (free, local) for generation — no API key needed.
Install: https://ollama.com  then run:  ollama pull llama3.2

Week 3, Day 1-2.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Generator
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from retrieval.retriever import HybridRetriever


# ── model config ──────────────────────────────────────────────────────────────

# Change this to any model you have pulled in Ollama.
# Good options ranked by speed vs quality on Mac:
#   "llama3.2"      — best quality, ~4GB, recommended
#   "llama3.2:1b"   — fastest, ~1GB, lower quality
#   "mistral"       — good balance, ~4GB
#   "phi3"          — very fast, good for testing
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"   # default Ollama port


# ── prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the context provided below.

Rules:
- For every claim you make, add a citation like [1], [2] referring to the source number.
- If the answer is not in the context, say exactly: "I don't have enough information to answer this."
- Do not make up information. Do not use your training knowledge — only the context.
- Be concise and clear.

Context:
{context}

Sources:
{sources}"""


# ── context formatter ─────────────────────────────────────────────────────────

def format_context(docs: List[Document]) -> tuple[str, str]:
    """
    Format retrieved docs into context block + numbered sources list.
    Returns: (context_text, sources_text)
    """
    context_parts = []
    sources_parts = []

    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[{i}] {doc.page_content.strip()}")

        src     = doc.metadata.get("source", "unknown")
        stype   = doc.metadata.get("source_type", "")
        page    = doc.metadata.get("page", "")
        section = doc.metadata.get("section", "")

        detail = ""
        if page:     detail = f"page {page}"
        elif section: detail = f"section: {section}"

        sources_parts.append(
            f"[{i}] ({stype}) {src}" + (f" — {detail}" if detail else "")
        )

    return "\n\n".join(context_parts), "\n".join(sources_parts)


# ── faithfulness checker ──────────────────────────────────────────────────────

def check_faithfulness(answer: str, context: str, llm: ChatOllama) -> dict:
    """
    Verify each claim in the answer is grounded in the retrieved context.
    Uses LLM-as-judge pattern — runs the same local Ollama model.
    Returns: {faithful: bool, issues: list[str]}
    """
    check_prompt = f"""You are a fact-checker. Given an answer and its source context,
identify any claims in the answer that are NOT supported by the context.

Context:
{context}

Answer:
{answer}

List any unsupported claims. If all claims are supported, reply with just "FAITHFUL".
Be brief — one line per issue."""

    result = llm.invoke([HumanMessage(content=check_prompt)])
    text = result.content.strip()

    if "FAITHFUL" in text.upper() and len(text) < 20:
        return {"faithful": True, "issues": []}

    issues = [line.strip() for line in text.splitlines() if line.strip()]
    return {"faithful": False, "issues": issues}


# ── main RAG chain ────────────────────────────────────────────────────────────

class RAGChain:
    """
    Full RAG chain: retrieve → format → prompt → stream.
    Runs 100% locally — Ollama for LLM, HuggingFace for embeddings.

    Usage:
        chain = RAGChain()
        for token in chain.stream("What is RAG?"):
            print(token, end="", flush=True)

        result = chain.query("What is RAG?")
        print(result["answer"])
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        top_k: int = 3,
        temperature: float = 0,
    ):
        self.retriever = HybridRetriever()
        self.llm = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,   # 0 = deterministic, factual
        )
        self.top_k = top_k
        self.conversation_history: List = []

        print(f"[CHAIN] Using Ollama model: {model} at {OLLAMA_BASE_URL}")

    def _build_messages(self, question: str, context: str, sources: str) -> List:
        system = SystemMessage(
            content=SYSTEM_PROMPT.format(context=context, sources=sources)
        )
        messages = [system] + self.conversation_history
        messages.append(HumanMessage(content=question))
        return messages

    def stream(
        self,
        question: str,
        filter: dict = None,
    ) -> Generator[str, None, None]:
        """
        Stream the answer token by token from Ollama.
        Yields string chunks as they arrive.
        """
        docs = self.retriever.retrieve(question, top_k=self.top_k, filter=filter)
        context, sources = format_context(docs)
        messages = self._build_messages(question, context, sources)

        full_answer = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content
            full_answer += token
            yield token

        # save to conversation memory
        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=full_answer))
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]

    def query(
        self,
        question: str,
        filter: dict = None,
        check_faith: bool = False,
    ) -> dict:
        """
        Non-streaming query. Returns full result dict with answer + sources.
        """
        docs = self.retriever.retrieve(question, top_k=self.top_k, filter=filter)
        context, sources_text = format_context(docs)
        messages = self._build_messages(question, context, sources_text)

        response = self.llm.invoke(messages)
        answer = response.content.strip()

        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=answer))
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]

        sources_list = [
            {
                "index":       i,
                "source":      doc.metadata.get("source", ""),
                "source_type": doc.metadata.get("source_type", ""),
                "chunk_id":    doc.metadata.get("chunk_id", ""),
                "ce_score":    doc.metadata.get("ce_score", 0),
                "preview":     doc.page_content[:200],
            }
            for i, doc in enumerate(docs, 1)
        ]

        result = {"answer": answer, "sources": sources_list}

        if check_faith:
            result.update(check_faithfulness(answer, context, self.llm))

        return result

    def reset_history(self):
        """Clear conversation memory — start a fresh session."""
        self.conversation_history = []


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chain = RAGChain()

    print("=== Non-streaming query ===")
    result = chain.query("What is retrieval augmented generation?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['source_type']} | ce={s['ce_score']:.3f} | {s['preview'][:80]}...")

    print("\n\n=== Streaming query ===")
    print("Answer: ", end="")
    for token in chain.stream("What are the main limitations of RAG?"):
        print(token, end="", flush=True)
    print()

    print("\n\n=== Multi-turn follow-up ===")
    result2 = chain.query("How does it compare to fine-tuning?")
    print(f"\nAnswer:\n{result2['answer']}")