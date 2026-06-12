import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import ollama


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.retrieval.retriever import retrieve
from src.generation.generator import build_prompt, generate_answer
from src.agents.agent import classify_intent


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASETS_DIR = BASE_DIR / "data" / "datasets"
RESULTS_DIR = BASE_DIR / "data" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Test Questions

# Since TeleQnA requires setup, we use these built-in test questions
# that cover all 4 intent types
TEST_QUESTIONS = [
    {
        "question": "What are the physical channels defined in 5G NR?",
        "expected_keywords": ["PDSCH", "PDCCH", "PUSCH", "PUCCH", "physical channel"],
        "intent": "spec_lookup"
    },
    {
        "question": "What is the architecture of NG-RAN?",
        "expected_keywords": ["gNB", "AMF", "UPF", "Xn", "NG interface"],
        "intent": "spec_lookup"
    },
    {
        "question": "What interfaces does a gNB support?",
        "expected_keywords": ["NG", "Xn", "F1", "E1", "interface"],
        "intent": "spec_lookup"
    },
    {
        "question": "Why would a gNB experience high packet loss?",
        "expected_keywords": ["NG-U", "PDCP", "loss", "packet", "synchronization"],
        "intent": "rca"
    },
    {
        "question": "What is beamforming in 5G NR?",
        "expected_keywords": ["beam", "antenna", "spatial", "MIMO", "beamforming"],
        "intent": "general_qna"
    },
    {
        "question": "What is the role of AMF in 5G core network?",
        "expected_keywords": ["AMF", "mobility", "authentication", "access", "management"],
        "intent": "spec_lookup"
    },
    {
        "question": "What are the frequency bands used in 5G NR?",
        "expected_keywords": ["FR1", "FR2", "GHz", "band", "frequency"],
        "intent": "spec_lookup"
    },
    {
        "question": "What is handover procedure in NR?",
        "expected_keywords": ["handover", "Xn", "measurement", "target", "source"],
        "intent": "spec_lookup"
    }
]


# Faithfulness scorer
def score_faithfulness(answer: str, chunks: list[dict]) -> float:
    """
    Measures how grounded the answer is in the retrieved context.
    Uses Mistral to check if each claim is supported by the context.
    """
    if not chunks or not answer:
        return 0.0

    context = "\n\n".join([chunk["text"] for chunk in chunks[:3]])

    faithfulness_prompt = f"""You are an evaluator checking if an answer is faithful to the source context.

CONTEXT:
{context[:2000]}

ANSWER:
{answer[:1000]}

Score the faithfulness of the answer from 0.0 to 1.0:
- 1.0 = Every claim in the answer is directly supported by the context
- 0.5 = Some claims are supported, some are not
- 0.0 = Answer contains claims not found in the context

Respond with ONLY a number between 0.0 and 1.0, nothing else.
Example: 0.85"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": faithfulness_prompt}]
        )
        score = float(response["message"]["content"].strip())
        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
    except:
        return 0.5  # Default if parsing fails
    

# Answer relevancy scorer
def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Measures how relevant the answer is to the question.
    """
    relevancy_prompt = f"""You are an evaluator checking if an answer is relevant to the question.

QUESTION: {question}

ANSWER: {answer[:1000]}

Score the relevancy from 0.0 to 1.0:
- 1.0 = Answer directly and completely addresses the question
- 0.5 = Answer partially addresses the question
- 0.0 = Answer does not address the question at all

Respond with ONLY a number between 0.0 and 1.0, nothing else.
Example: 0.90"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": relevancy_prompt}]
        )
        score = float(response["message"]["content"].strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5
    

# Keyword accuracy scorer
def score_keyword_accuracy(answer: str, expected_keywords: list[str]) -> float:
    """
    Measures what percentage of expected keywords appear in the answer.
    """
    if not expected_keywords:
        return 0.0

    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


# Context recall scorer
def score_context_recall(question: str, chunks: list[dict]) -> float:
    """
    Measures if the retrieved chunks contain the information
    needed to answer the question.
    """
    if not chunks:
        return 0.0

    context = "\n\n".join([chunk["text"] for chunk in chunks[:3]])

    recall_prompt = f"""You are an evaluator checking if retrieved context contains 
enough information to answer a question.

QUESTION: {question}

RETRIEVED CONTEXT:
{context[:2000]}

Score the context recall from 0.0 to 1.0:
- 1.0 = Context contains all information needed to answer the question
- 0.5 = Context contains some relevant information
- 0.0 = Context does not contain information to answer the question

Respond with ONLY a number between 0.0 and 1.0, nothing else.
Example: 0.80"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": recall_prompt}]
        )
        score = float(response["message"]["content"].strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5
    

# MRR scorer
def score_mrr(question: str, chunks: list[dict]) -> float:
    """
    Computes Mean Reciprocal Rank — is the most relevant
    chunk ranked at the top?
    """
    if not chunks:
        return 0.0

    question_lower = question.lower()
    question_words = set(question_lower.split())

    for rank, chunk in enumerate(chunks, start=1):
        chunk_words = set(chunk["text"].lower().split())
        # Check overlap between question words and chunk words
        overlap = len(question_words & chunk_words) / len(question_words)
        if overlap > 0.3:  # 30% word overlap = relevant chunk
            return 1.0 / rank

    return 0.0


# Running full evaluation
def run_evaluation():
    """
    Runs full evaluation over all test questions and saves report.
    """
    print("\n" + "="*60)
    print("RANsight — Evaluation Pipeline")
    print("="*60)
    print(f"Running {len(TEST_QUESTIONS)} test questions...\n")

    results = []

    for i, test in enumerate(TEST_QUESTIONS):
        question = test["question"]
        expected_keywords = test["expected_keywords"]

        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] {question}")
        print("-" * 50)

        try:
            # Retrieve chunks
            chunks = retrieve(question, top_k=5)

            # Generate answer
            prompt = build_prompt(question, chunks)
            answer = generate_answer(prompt)

            # Score all metrics
            print("  Scoring metrics...")
            faithfulness = score_faithfulness(answer, chunks)
            relevancy = score_answer_relevancy(question, answer)
            keyword_acc = score_keyword_accuracy(answer, expected_keywords)
            context_recall = score_context_recall(question, chunks)
            mrr = score_mrr(question, chunks)

            result = {
                "question": question,
                "intent": test["intent"],
                "answer_preview": answer[:300],
                "metrics": {
                    "faithfulness": round(faithfulness, 4),
                    "answer_relevancy": round(relevancy, 4),
                    "keyword_accuracy": round(keyword_acc, 4),
                    "context_recall": round(context_recall, 4),
                    "mrr": round(mrr, 4)
                },
                "sources": [
                    {"source": c["source"], "page": c["page_number"]}
                    for c in chunks
                ]
            }

            results.append(result)

            print(f"  ✅ Faithfulness:     {faithfulness:.2%}")
            print(f"  ✅ Answer Relevancy: {relevancy:.2%}")
            print(f"  ✅ Keyword Accuracy: {keyword_acc:.2%}")
            print(f"  ✅ Context Recall:   {context_recall:.2%}")
            print(f"  ✅ MRR:              {mrr:.4f}")

            # Small delay to avoid overwhelming Ollama
            time.sleep(1)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    # Calculate averages
    if results:
        avg_metrics = {
            "avg_faithfulness": np.mean([r["metrics"]["faithfulness"] for r in results]),
            "avg_answer_relevancy": np.mean([r["metrics"]["answer_relevancy"] for r in results]),
            "avg_keyword_accuracy": np.mean([r["metrics"]["keyword_accuracy"] for r in results]),
            "avg_context_recall": np.mean([r["metrics"]["context_recall"] for r in results]),
            "avg_mrr": np.mean([r["metrics"]["mrr"] for r in results])
        }

        # Save report
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(results),
            "average_metrics": {k: round(v, 4) for k, v in avg_metrics.items()},
            "detailed_results": results
        }

        report_path = RESULTS_DIR / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"  Questions evaluated: {len(results)}")
        print(f"  Faithfulness:        {avg_metrics['avg_faithfulness']:.2%} (target: >90%)")
        print(f"  Answer Relevancy:    {avg_metrics['avg_answer_relevancy']:.2%} (target: >80%)")
        print(f"  Keyword Accuracy:    {avg_metrics['avg_keyword_accuracy']:.2%} (target: >80%)")
        print(f"  Context Recall:      {avg_metrics['avg_context_recall']:.2%} (target: >85%)")
        print(f"  MRR:                 {avg_metrics['avg_mrr']:.4f} (target: >0.75)")
        print(f"\n  Report saved to: {report_path.name}")
        print("="*60)


if __name__ == "__main__":
    run_evaluation()


