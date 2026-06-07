# 🔍 RAG Pipeline with LangGraph + Groq + FAISS

A production-style **Retrieval-Augmented Generation (RAG)** pipeline built with:
- **LangGraph** — multi-agent orchestration
- **Groq** — blazing-fast LLM inference (llama3-70b-8192)
- **FAISS** — local semantic vector search
- **Sentence Transformers** — free local embeddings
- **Token Mapper Agent** — real-time per-node token usage tracking

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- A free Groq API key → https://console.groq.com

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Edit `.env` and add your Groq key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### 4. Add Your PDF
Place your PDF file in this folder and note its filename.

### 5. Launch the Notebook
```bash
jupyter notebook pipeline.ipynb
```
Then run all cells top to bottom (Cell → Run All).

---

## 🏗️ Pipeline Architecture

```
PDF Input
    │
    ▼
[Node 1: ingest_pdf]        ← Parse PDF pages → text chunks
    │
    ▼
[Node 2: embed_and_index]   ← Embed chunks → FAISS index
    │
    ▼
[Node 3: retrieve]          ← Semantic search → top-k chunks
    │
    ▼
[Node 4: generate_answer]   ← Groq LLM → grounded answer
    │
    ▼
[Node 5: token_summary]     ← Token Mapper → per-node token table
```

---

## 📊 Token Mapper Agent

After every run, the **Token Mapper Agent** prints a table like:

```
╒══════════════════════╤═══════════════╤════════════════╤═══════════╕
│ Node                 │ Input Tokens  │ Output Tokens  │ Est. Cost │
╞══════════════════════╪═══════════════╪════════════════╪═══════════╡
│ ingest_pdf           │ —             │ —              │ $0.0000   │
│ embed_and_index      │ —             │ —              │ $0.0000   │
│ retrieve             │ —             │ —              │ $0.0000   │
│ generate_answer      │ 1,248         │ 312            │ $0.0002   │
│ TOTAL                │ 1,248         │ 312            │ $0.0002   │
╘══════════════════════╧═══════════════╧════════════════╧═══════════╛
```

---

## ⚙️ Configuration (Cell 2 of the notebook)

| Parameter | Default | Description |
|---|---|---|
| `PDF_PATH` | `"sample.pdf"` | Path to your PDF |
| `GROQ_MODEL` | `"llama3-70b-8192"` | Groq model to use |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `4` | Number of chunks to retrieve |

---

## 📁 Project Structure

```
D:\rag-langgraph-pipeline\
├── pipeline.ipynb      ← Main notebook (start here)
├── requirements.txt    ← Python dependencies
├── .env                ← Your API key (keep private!)
├── README.md           ← This file
└── your_file.pdf       ← Drop your PDF here
```
