"""
ragas_eval.py — RAGAS evaluation for the RAG pipeline.
Week 4, Day 1.

Fixed:
  - TimeoutError: increased timeout, sequential evaluation, smaller model
  - KeyError 'question': adapted to newer RAGAS DataFrame schema

Run:
    python3 eval/ragas_eval.py
"""

import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from retrieval.retriever import HybridRetriever
from generation.chain import RAGChain


# ── test dataset ──────────────────────────────────────────────────────────────
# Keep this small (3-5 questions) — each one makes multiple Ollama calls
# during RAGAS evaluation, and local models are slow.

TEST_QUESTIONS = [
    {
        "question": "What is retrieval augmented generation?",
        "ground_truth": "RAG is a technique that enhances LLMs by retrieving relevant documents from an external knowledge base before generating an answer.",
    },
    {
        "question": "What are the main components of a RAG system?",
        "ground_truth": "A RAG system consists of a retriever that fetches relevant documents and a generator that produces answers based on those documents.",
    },
    {
        "question": "What are the limitations of RAG systems?",
        "ground_truth": "RAG systems can struggle with retrieval quality, latency, handling conflicting information, and scalability of the knowledge base.",
    },
]


# ── build eval dataset ────────────────────────────────────────────────────────

def build_eval_dataset() -> Dataset:
    retriever = HybridRetriever()
    chain     = RAGChain()

    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    print("Running test questions through RAG pipeline...\n")

    for i, item in enumerate(TEST_QUESTIONS, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"[{i}/{len(TEST_QUESTIONS)}] {q}")

        docs          = retriever.retrieve(q, top_k=3)
        context_texts = [doc.page_content for doc in docs]
        result        = chain.query(q)
        answer        = result["answer"]
        chain.reset_history()

        questions.append(q)
        answers.append(answer)
        contexts_list.append(context_texts)
        ground_truths.append(gt)

        print(f"  Answer: {answer[:120]}...\n")

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })


# ── run evaluation ────────────────────────────────────────────────────────────

def run_eval():
    print("=" * 60)
    print("RAGAS Evaluation — Multi-Source RAG Pipeline")
    print("=" * 60)

    # phi3 is much faster than llama3.2 for RAGAS judge calls
    # if you don't have phi3: ollama pull phi3
    # fallback: keep llama3.2 but expect ~10 min runtime
    judge_model = "phi3"
    print(f"\nUsing '{judge_model}' as RAGAS judge (faster than llama3.2)")
    print("If you don't have it: ollama pull phi3\n")

    ollama_llm = LangchainLLMWrapper(
        ChatOllama(
            model=judge_model,
            base_url="http://localhost:11434",
            timeout=300,          # 5 min per call — prevents TimeoutError
        )
    )

    hf_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    dataset = build_eval_dataset()

    # RunConfig: sequential (max_workers=1) prevents parallel timeout pile-up
    run_config = RunConfig(
        max_workers=1,       # sequential — slower but no timeouts
        timeout=300,         # 5 min per metric call
        max_retries=2,
    )

    print("Running RAGAS metrics sequentially (takes ~5-10 min locally)...")

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ollama_llm,
        embeddings=hf_embeddings,
        run_config=run_config,
    )

    df = results.to_pandas()
    print("\nDataFrame columns available:", df.columns.tolist())

    # ── safely extract scores ─────────────────────────────────────────────────
    def safe_mean(col):
        if col in df.columns:
            return round(float(df[col].dropna().mean()), 3)
        return "n/a"

    scores = {
        "faithfulness":      safe_mean("faithfulness"),
        "answer_relevancy":  safe_mean("answer_relevancy"),
        "context_precision": safe_mean("context_precision"),
    }

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Faithfulness      : {scores['faithfulness']}  (target > 0.80)")
    print(f"  Answer relevancy  : {scores['answer_relevancy']}  (target > 0.75)")
    print(f"  Context precision : {scores['context_precision']}  (target > 0.70)")
    print("=" * 60)

    # ── per-question breakdown (column-safe) ──────────────────────────────────
    print("\nPer-question breakdown:")
    for i, row in df.iterrows():
        # RAGAS newer versions may store question differently
        q_text = ""
        for col in ["question", "user_input", "input"]:
            if col in df.columns:
                q_text = str(row[col])[:60]
                break
        if not q_text:
            q_text = TEST_QUESTIONS[i]["question"][:60] if i < len(TEST_QUESTIONS) else f"Q{i+1}"

        print(f"\n  Q{i+1}: {q_text}...")
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            val = f"{row[metric]:.3f}" if metric in df.columns and not __import__('math').isnan(row[metric]) else "n/a"
            print(f"    {metric:<22}: {val}")

    # ── save results ──────────────────────────────────────────────────────────
    per_q = []
    for i, row in df.iterrows():
        q_text = TEST_QUESTIONS[i]["question"] if i < len(TEST_QUESTIONS) else f"Q{i+1}"
        entry  = {"question": q_text}
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            if metric in df.columns:
                val = row[metric]
                entry[metric] = round(float(val), 3) if val == val else None  # nan check
        per_q.append(entry)

    output = {"summary": scores, "per_question": per_q}
    out_path = os.path.join(os.path.dirname(__file__), "ragas_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to eval/ragas_results.json")
    print("\nCopy into your README:\n")
    print(f"| Metric             | Score |")
    print(f"|--------------------|-------|")
    print(f"| Faithfulness       | {scores['faithfulness']} |")
    print(f"| Answer relevancy   | {scores['answer_relevancy']} |")
    print(f"| Context precision  | {scores['context_precision']} |")

    return scores


if __name__ == "__main__":
    run_eval()