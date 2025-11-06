# src/ingest.py
import os
import json
from pypdf import PdfReader
from transformers import AutoTokenizer

# Settings
MAX_TOKENS = 512
OVERLAP = 128
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

def pdf_to_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=False)['input_ids']
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_ids = tokens[i:i+MAX_TOKENS]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        i += MAX_TOKENS - OVERLAP
    return chunks

def process_pdfs(input_dir, output_path):
    all_chunks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(input_dir, filename)
            chunks = pdf_to_chunks(path)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "pdf": filename,
                    "chunk_id": idx,
                    "text": chunk
                })
            print(f"Processed {filename} → {len(chunks)} chunks")
    # Save all chunks
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"All chunks saved to {output_path}")

if __name__ == "__main__":
    INPUT_DIR = "./data"      # folder with PDFs
    OUTPUT_PATH = "./data/chunks.json"
    process_pdfs(INPUT_DIR, OUTPUT_PATH)
