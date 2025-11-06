# src/utils.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_index(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def embed_query(query, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    emb = model.encode([query], convert_to_numpy=True)
    return emb.astype("float32")

def retrieve(query, index, meta, top_k=5):
    q_emb = embed_query(query)
    D, I = index.search(q_emb, top_k)
    results = [meta[i] for i in I[0]]
    return results

def build_prompt(query, chunks):
    context = "\n\n".join([f"{c['text']}" for c in chunks])
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    return prompt
