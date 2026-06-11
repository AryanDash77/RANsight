import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer , CrossEncoder


BASE_DIR = Path(__file__).resolve().parent.parent.parent

INDEX_DIR = BASE_DIR / "data" / "index"


# Load FAISS index and metadata
def load_index_and_metadata():
    """
    Loads the FAISS index and chunk metadata from disk.
    """

    print("Loading FAISS index...")
    index_path = INDEX_DIR / "faiss_index.bin"
    index = faiss.read_index(str(index_path))
    print(f"FAISS index loaded = {index.ntotal:,} vectors")

    print("Loading chunk metadata...")
    metadata_path = INDEX_DIR / "chunks_metadata.pkl"
    with open(metadata_path,"rb") as f:
        chunks = pickle.load(f)
    print(f"Metadata loaded - {len(chunks):,} chunks")

    return index, chunks


# Build BM25 index
def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    Builds a BM25 index from chunk texts.
    """

    print("\nBuilding BM25 index...")

    # Tokenize each chunk by splitting on whitespace
    tokenized_chunks = [chunk["text"].lower().split() for chunk in chunks]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)

    print(f"BM25 index built with {len(tokenized_chunks):,} documents")
    return bm25


# Dense Retrieval (FAISS)
def dense_search(query: str, index, chunks: list[dict], model: SentenceTransformer, top_k: int = 10) -> list[dict]:
    """
    Searches FAISS index using dense embeddings.
    """
    # Convert query to embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Search FAISS index
    scores, indices = index.search(query_embedding, top_k)

    # Build results list
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        chunk = chunks[idx].copy()
        chunk["dense_score"] = float(score)
        results.append(chunk)

    return results


# Sparse Retrieval (BM25)
def sparse_search(query: str, bm25: BM25Okapi, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """
    Searches using BM25 keyword matching.
    """
    # Tokenize query
    tokenized_query = query.lower().split()

    # Get BM25 scores for all chunks
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Build results list
    results = []
    for idx in top_indices:
        if scores[idx] == 0:  # Skip zero-score results
            continue
        chunk = chunks[idx].copy()
        chunk["sparse_score"] = float(scores[idx])
        results.append(chunk)

    return results


# Combine and deduplicate results
def combine_results(dense_results: list[dict], sparse_results: list[dict]) -> list[dict]:
    """
    Combines dense and sparse results, removing duplicates.
    """
    combined = {}

    # Add dense results
    for chunk in dense_results:
        chunk_id = chunk["chunk_id"]
        combined[chunk_id] = chunk

    # Add sparse results (only if not already in combined)
    for chunk in sparse_results:
        chunk_id = chunk["chunk_id"]
        if chunk_id not in combined:
            combined[chunk_id] = chunk

    return list(combined.values())


# Re-rank with CrossEncoder
def rerank_results(query: str, candidates: list[dict], cross_encoder: CrossEncoder, top_k: int = 5) -> list[dict]:
    """
    Re-ranks candidates using a cross-encoder model.
    """
    if not candidates:
        return []

    print(f"  Re-ranking {len(candidates)} candidates...")

    # Build query-chunk pairs for cross-encoder
    pairs = [[query, chunk["text"]] for chunk in candidates]

    # Get cross-encoder scores
    scores = cross_encoder.predict(pairs)

    # Attach scores to chunks
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    # Sort by rerank score descending
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # Return top-k
    return reranked[:top_k]


# Main Retrieval Pipeline
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Full hybrid retrieval pipeline for a given query.
    """
    print(f"\nQuery: {query}")
    print("-" * 50)

    # Load index and metadata
    index, chunks = load_index_and_metadata()

    # Load models
    print("\nLoading embedding model...")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    print("Loading cross-encoder model...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Build BM25 index
    bm25 = build_bm25_index(chunks)

    # Dense search
    print("\nRunning dense search...")
    dense_results = dense_search(query, index, chunks, embed_model, top_k=10)
    print(f"  Dense results: {len(dense_results)}")

    # Sparse search
    print("Running sparse search...")
    sparse_results = sparse_search(query, bm25, chunks, top_k=10)
    print(f"  Sparse results: {len(sparse_results)}")

    # Combine
    combined = combine_results(dense_results, sparse_results)
    print(f"  Combined unique results: {len(combined)}")

    # Re-rank
    final_results = rerank_results(query, combined, cross_encoder, top_k=top_k)

    # Display results
    print(f"\n✅ Top {len(final_results)} results:\n")
    for i, chunk in enumerate(final_results):
        print(f"Result {i+1}:")
        print(f"  Source: {chunk['source']}")
        print(f"  Page: {chunk['page_number']}")
        print(f"  Rerank Score: {chunk['rerank_score']:.4f}")
        print(f"  Text preview: {chunk['text'][:150]}...")
        print()

    return final_results


if __name__ == "__main__":
    # Test with a sample telecom query
    results = retrieve("What are the physical channels in 5G NR?")

