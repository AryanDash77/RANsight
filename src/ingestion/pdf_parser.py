import fitz
import pdfplumber
import json
import os
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent.parent                  # base project path

RAW_DIR = BASE_DIR / "data" / "raw"
PARSED_DIR = BASE_DIR / "data" / "parsed"


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extracts text from each page of a PDF.
    Returns a list of dictionaries, one per page.
    """
    pages = []
    
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract raw text from the page
        text = page.get_text("text")
        
        # Skip empty pages
        if not text.strip():
            continue
        
        # Build a structured dictionary for each page
        page_data = {
            "source": pdf_path.name,        # e.g. "3GPP_TS_38_300_R16.pdf"
            "page_number": page_num + 1,     # human-readable page number
            "text": text.strip(),            # clean extracted text
            "char_count": len(text.strip())  # useful for filtering later
        }
        
        pages.append(page_data)
    
    doc.close()
    return pages


def save_parsed_output(pages: list[dict], output_path: Path):
    """
    Saves extracted pages to a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(pages)} pages to {output_path.name}")



def process_all_pdfs():
    """
    Processes all PDFs in the raw folder.
    """
    # Find all PDFs in raw folder
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDFs found in data/raw/ folder!")
        return
    
    print(f"Found {len(pdf_files)} PDFs to process...\n")
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\nProcessing: {pdf_path.name}")
        
        # Extract text
        pages = extract_text_from_pdf(pdf_path)
        
        # Create output filename
        # e.g. "3GPP_TS_38_300_R16.pdf" → "3GPP_TS_38_300_R16.json"
        output_filename = pdf_path.stem + ".json"
        output_path = PARSED_DIR / output_filename
        
        # Save to JSON
        save_parsed_output(pages, output_path)
        
        print(f"  Pages extracted: {len(pages)}")
        print(f"  Total characters: {sum(p['char_count'] for p in pages):,}")


if __name__ == "__main__":
    process_all_pdfs()