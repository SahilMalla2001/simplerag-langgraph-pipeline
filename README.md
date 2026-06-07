# RAG Pipeline with LangGraph + Google Gemini (FREE) + FAISS

A two-agent **Retrieval-Augmented Generation (RAG)** pipeline for question-answering over PDF documents.

- **LangGraph** — multi-agent orchestration
- **Google Gemini 2.5 Flash** — LLM inference on the free tier (no credit card)
- **FAISS** — local semantic vector search
- **Sentence Transformers** — local embeddings, runs fully offline
- **Token Mapper Agent** — real-time per-node token usage tracking

---

## Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- A free Gemini API key → https://aistudio.google.com/apikey (no credit card, takes 30 seconds)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Edit `.env` and add your Gemini key:
```
GEMINI_API_KEY=AIzaSy...
```

### 4. Add Your PDF
Drop your PDF file into this folder.

### 5. Launch the Notebook
```bash
python -m jupyter notebook pipeline.ipynb
```
- Run **Cell 0** first to install packages into the active kernel
- Edit `PDF_PATH` in Cell 2 to match your filename
- Edit `QUESTION` in Cell 8
- Run all cells top to bottom

---

## Pipeline Architecture

```
PDF Input
    |
[ingest_pdf]        Parse PDF pages into overlapping text chunks
    |
[embed_and_index]   Embed chunks locally via sentence-transformers -> FAISS index
    |
[retrieve]          Semantic search: embed question -> find top-K similar chunks
    |
[generate_answer]   Build RAG prompt -> call Gemini 2.5 Flash -> grounded answer
    |
[token_summary]     Token Mapper Agent finalises per-node usage record
    |
   END
```

---

## Two Agents

### Agent 1 — RAG Agent
Handles the full retrieval and generation pipeline across 4 LangGraph nodes:
`ingest_pdf` → `embed_and_index` → `retrieve` → `generate_answer`

### Agent 2 — Token Mapper Agent
Wraps every node with `start_node()` / `end_node()` hooks and records:
- Input tokens and output tokens per node (sourced directly from the Gemini API response)
- Elapsed time per node in milliseconds
- A styled HTML summary table rendered at the end of each run

Since the pipeline runs on the **Gemini free tier**, cost is always **$0.00**.

---

## Token Mapper Output (example)

| Node | Input Tokens | Output Tokens | Total Tokens | Time (ms) | Cost |
|---|---|---|---|---|---|
| ingest_pdf | — | — | — | 108 | $0.00 [FREE] |
| embed_and_index | — | — | — | 347 | $0.00 [FREE] |
| retrieve | — | — | — | 21 | $0.00 [FREE] |
| generate_answer | 673 | 41 | 714 | 3,460 | $0.00 [FREE] |
| token_summary | — | — | — | 0 | $0.00 [FREE] |
| **TOTAL** | **673** | **41** | **714** | **3,936** | **$0.00 [FREE TIER]** |

---

## Configuration (Cell 2)

| Parameter | Default | Description |
|---|---|---|
| `PDF_PATH` | your filename | PDF file in the project folder |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model to use (all options are free tier) |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K` | `4` | Number of chunks to retrieve per question |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model (offline, no API) |

---

## Notebook Structure

| Cell | Purpose |
|---|---|
| 0 | Auto-install all packages into the active Jupyter kernel |
| 1 | Imports and API key validation |
| 2 | Configuration |
| 3 | Token Mapper Agent (Agent 2) |
| 4 | PipelineState schema definition |
| 5 | Load local embedding model |
| 6 | 5 RAG pipeline node functions (Agent 1) |
| 7 | LangGraph graph build and compile |
| 8 | Run the pipeline — shows answer only |
| 9 | Token Mapper Agent report |
| 10 | Follow-up questions (reuses FAISS index, no re-embedding) |

---

## Project Structure

```
rag-langgraph-pipeline/
├── pipeline.ipynb      <- Main notebook (start here)
├── run_pipeline.py     <- Same pipeline as a terminal script
├── requirements.txt    <- Python dependencies
├── .env                <- Your API key (keep private, gitignored)
├── .gitignore          <- Ignores .env and your PDF
└── README.md           <- This file
```

---

## Free Tier Limits (Gemini 2.5 Flash)

- 500 requests / day
- 1,000,000 tokens / day
- Cost: $0.00
