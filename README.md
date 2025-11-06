# 📄 Smart Document Q&A System

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📝 Project Overview
The **Smart Document Q&A System** allows users to query a large collection of PDF documents and get **context-aware answers** in under 2 seconds. It leverages **retrieval-augmented generation (RAG)** to provide accurate responses from domain-specific data.  
Powered by **Hugging Face Transformers** and **FAISS**, it efficiently searches and retrieves relevant content.

---

## 🎯 Uses of this Project
- Quickly answer questions from large PDF collections.  
- Useful for **knowledge management**, **customer support**, and **internal documentation** search.  
- Can be extended to any **domain-specific PDF corpus**.  
- Serves as a foundation for building AI-powered **chatbots** or **document assistants**.

---

## 🛠 Technical Stack
- **Python 3.13** 🐍  
- **PyTorch (CPU/GPU compatible)** 🔥  
- **Hugging Face Transformers** 🤗  
- **Sentence Transformers** 🧠  
- **LangChain** (for retrieval workflows) 🔗  
- **FAISS** (vector search) 🎯  
- **FastAPI** (backend API server) 🚀  
- **Uvicorn** (ASGI server) ⚡  
- **Docker** (optional, for deployment) 🐳  

---

## ⚡ Project Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/smart-docs-qa.git
cd smart-docs-qa
```

### 2️⃣ Create & activate virtual environment
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
# OR
source .venv/bin/activate       # macOS/Linux
```

### 3️⃣ Install dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4️⃣ Add PDFs
Place your PDF files in the `data/` folder.  
Example structure:
```
data/
 ├─ document1.pdf
 ├─ document2.pdf
 └─ ...
```

### 5️⃣ Ingest PDFs
```bash
python src/ingest.py
```
- Converts PDFs to text chunks for embedding.

### 6️⃣ Build FAISS index
```bash
python src/embed_index.py
```
- Generates vector embeddings and saves `faiss.index`.

### 7️⃣ Start the API server
```bash
python -m uvicorn src.qa_server:app --reload --host 127.0.0.1 --port 8000
```

---

## 🧪 How to Test

### Using Swagger UI
1. Open in browser: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
2. Use the `POST /query` endpoint.  
3. Example query JSON:
```json
{
  "q": "What is the refund policy?"
}
```

### Using Python script
Create `test_query.py`:
```python
import requests

url = "http://127.0.0.1:8000/query"
data = {"q": "What is the refund policy?"}

response = requests.post(url, json=data)
print(response.json())
```
Run:
```bash
python test_query.py
```

---

## 🚀 Future Scope
- Add **dynamic PDF upload** endpoint for live ingestion.  
- Integrate a **frontend** using React, Gradio, or Streamlit for a user-friendly interface.  
- Deploy on **cloud platforms** (AWS, Heroku, Azure).  
- Optimize for **GPU inference** to reduce latency further.  
- Extend to **multi-language PDF support** or domain-specific fine-tuning.

---

## ✍️ Author
- **Sai Kiran Vasa**
