# src/embed_index.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Settings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "./data/faiss.index"
META_PATH = "./data/meta.json"
CHUNKS_PATH = "./data/chunks.json"

def build_index(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    texts = [chunk["text"] for chunk in chunks]
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)  # CPU-friendly HNSW
    index.hnsw.efSearch = 64
    index.add(embeddings.astype("float32"))

    # Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"FAISS index saved → {INDEX_PATH}")
    print(f"Metadata saved → {META_PATH}")

if __name__ == "__main__":
    build_index(CHUNKS_PATH)
