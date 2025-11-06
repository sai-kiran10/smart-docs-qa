# src/qa_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils import load_index, retrieve, build_prompt # pyright: ignore[reportMissingImports]
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

app = FastAPI(title="Smart Document Q&A")

# Load index & metadata
INDEX_PATH = "./data/faiss.index"
META_PATH = "./data/meta.json"
index, meta = load_index(INDEX_PATH, META_PATH)

# Load small CPU-friendly LLM
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

class Query(BaseModel):
    q: str

def generate_answer(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/query")
def answer_question(query: Query):
    top_chunks = retrieve(query.q, index, meta, top_k=5)
    prompt = build_prompt(query.q, top_chunks)
    answer = generate_answer(prompt)
    return {"answer": answer, "sources": [c["pdf"] for c in top_chunks]}

@app.get("/")
def root():
    return {"message": "Smart Document Q&A API is running"}
