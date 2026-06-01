import json
import os
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent

PARSED_DIR = BASE_DIR / "data" / "parsed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def chunk_pages(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Splits pages into overlapping chunks of roughly chunk_size characters.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        source = page["source"]
        page_number = page["page_number"]

        # Split text into chunks with overlap
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Get the chunk text
            chunk_text = text[start:end]

            # Skip very short chunks (less than 50 characters)
            if len(chunk_text.strip()) < 50:
                break

            chunk = {
                "chunk_id": chunk_id,
                "source": source,
                "page_number": page_number,
                "text": chunk_text.strip(),
                "char_count": len(chunk_text.strip())
            }

            chunks.append(chunk)
            chunk_id += 1

            # Move forward by chunk_size minus overlap
            start += chunk_size - overlap

    return chunks


def save_chunks(chunks: list[dict], output_path: Path):
    """
    Saves chunks to a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {output_path.name}")


def process_all_documents():
    """
    Reads all parsed JSONs and chunks them.
    """
    json_files = list(PARSED_DIR.glob("*.json"))

    if not json_files:
        print("No parsed JSON files found in data/parsed/")
        return

    print(f"Found {len(json_files)} documents to chunk...\n")

    total_chunks = 0

    for json_path in tqdm(json_files, desc="Chunking documents"):
        print(f"\nProcessing: {json_path.name}")

        # Load parsed pages
        with open(json_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        # Chunk the pages
        chunks = chunk_pages(pages)

        # Save chunks
        output_filename = json_path.stem + "_chunks.json"
        output_path = CHUNKS_DIR / output_filename
        save_chunks(chunks, output_path)

        print(f"  Pages: {len(pages)} → Chunks: {len(chunks)}")
        total_chunks += len(chunks)

    print(f"\n✅ Done! Total chunks created: {total_chunks:,}")


if __name__ == "__main__":
    process_all_documents()


