import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss


BASE_DIR = Path(__file__).resolve().parent.parent.parent

CHUNKS_DIR = BASE_DIR / "data" / "chunks"
INDEX_DIR = BASE_DIR / "data" / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_all_chunks() -> list[dict]:
    """
    Loads all chunk JSON files into a single list.
    """
    all_chunks = []

    chunk_files = list(CHUNKS_DIR.glob("*_chunks.json"))

    if not chunk_files:
        print("No chunk files found in data/chunks/")
        return []

    print(f"Found {len(chunk_files)} chunk files...\n")

    for chunk_file in chunk_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {chunk_file.name}")

    print(f"\nTotal chunks loaded: {len(all_chunks):,}")
    return all_chunks


def generate_embeddings(chunks: list[dict], model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Converts chunk text into vector embeddings using BGE model.
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts):,} chunks...")
    print("This may take a few minutes...\n")

    # Generate embeddings in batches for efficiency
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for cosine similarity search
    )

    print(f"\nEmbedding shape: {embeddings.shape}")
    return embeddings, model


def build_faiss_index(embeddings: np.ndarray, chunks: list[dict]):
    """
    Builds a FAISS index from embeddings and saves everything to disk.
    """
    print("\nBuilding FAISS index...")

    # Get embedding dimension
    dimension = embeddings.shape[1]

    # Create FAISS index using cosine similarity (Inner Product on normalized vectors)
    index = faiss.IndexFlatIP(dimension)

    # Add all embeddings to the index
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal:,} vectors")

    # Save FAISS index to disk
    index_path = INDEX_DIR / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved to {index_path}")

    # Save chunk metadata separately using pickle
    # We need this to retrieve the original text and source after a search
    metadata_path = INDEX_DIR / "chunks_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunk metadata saved to {metadata_path}")

    return index


def main():
    # Step 1: Load all chunks
    chunks = load_all_chunks()
    if not chunks:
        return

    # Step 2: Generate embeddings
    embeddings, model = generate_embeddings(chunks)

    # Step 3: Build and save FAISS index
    index = build_faiss_index(embeddings, chunks)

    print("\n✅ Embedding pipeline complete!")
    print(f"   Total vectors in index: {index.ntotal:,}")
    print(f"   Index saved to: data/index/faiss_index.bin")
    print(f"   Metadata saved to: data/index/chunks_metadata.pkl")


if __name__ == "__main__":
    main()


