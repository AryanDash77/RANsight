import ollama
from pathlib import Path
import sys

# Add project root to path so we can import from src/retrieval
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.retrieval.retriever import retrieve


# Building the Prompt
def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Builds a citation-grounded prompt from query and retrieved chunks.
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['source']}, Page {chunk['page_number']}]\n{chunk['text']}"
        )
    
    context = "\n\n".join(context_parts)

    # Build the full prompt
    prompt = f"""You are RANsight, an expert Telecom RAN Assistant specializing in 3GPP specifications.

Answer the question using ONLY the context provided below.
For every claim you make, cite the source using [Source X] notation.
If the answer is not in the context, say "I cannot find this information in the provided specifications."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    return prompt


# Generating answer
def generate_answer(prompt: str, model: str = "mistral") -> str:
    """
    Sends prompt to Mistral via Ollama and returns the response.
    """
    print("\nGenerating answer with Mistral...\n")

    response = ollama.chat(
        model = model,
        messages = [
            {
                "role": "system",
                "content": "You are RANsight, an expert Telecom RAN Assistant. Always cite your sources."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# Displaying the results
def display_results(query: str, answer: str, chunks: list[dict]):
    """
    Prints the query, sources used, and final answer cleanly.
    """
    print("=" * 60)
    print("RANsight — Telecom RAN Assistant")
    print("=" * 60)
    
    print(f"\n📌 QUERY:\n{query}\n")
    
    print("📚 SOURCES RETRIEVED:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk['source']} — Page {chunk['page_number']} "
              f"(Rerank Score: {chunk.get('rerank_score', 0):.4f})")
    
    print(f"\n🤖 ANSWER:\n{answer}")
    print("\n" + "=" * 60)


# Main RAG pipeline
def ask(query: str) -> str:
    """
    Full RAG pipeline — retrieve + generate.
    """
    # Step 1: Retrieve relevant chunks
    print(f"🔍 Retrieving relevant chunks...")
    chunks = retrieve(query, top_k=5)

    if not chunks:
        return "No relevant information found in the knowledge base."

    # Step 2: Build prompt
    prompt = build_prompt(query, chunks)

    # Step 3: Generate answer
    answer = generate_answer(prompt)

    # Step 4: Display results
    display_results(query, answer, chunks)

    return answer


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What are the physical channels in 5G NR?",
        "What is the architecture of NG-RAN?",
    ]

    for query in test_queries:
        ask(query)
        print("\n")

        