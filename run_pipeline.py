"""
run_pipeline.py — RAG Pipeline (terminal version)

Equivalent to running all cells of pipeline.ipynb from the command line.
Use this for quick testing or automation without opening Jupyter.

Usage:
    python run_pipeline.py
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import TypedDict, List, Any
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# ── Environment ────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    raise EnvironmentError(
        "GEMINI_API_KEY not set!\n"
        "  Edit .env and add: GEMINI_API_KEY=AIzaSy...\n"
        "  Get a free key at: https://aistudio.google.com/apikey"
    )
print(f"[OK] Gemini API key loaded (...{GEMINI_API_KEY[-6:]})")

# ── Imports ────────────────────────────────────────────────────
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
from google.genai import types as genai_types
from langgraph.graph import StateGraph, END
import pandas as pd

print(f"[OK] Python {sys.version.split()[0]}")
print("[OK] All imports successful!")

# ── Configuration ──────────────────────────────────────────────
PDF_PATH        = "GRE_score.pdf"       # <- change to your PDF filename
GEMINI_MODEL    = "gemini-2.5-flash"    # free tier
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K           = 4
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # runs locally, no API needed
QUESTION        = "What is this document about?"  # <- change your question

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print(f"[OK] Config: PDF={PDF_PATH} | Model={GEMINI_MODEL} | TOP_K={TOP_K}")

# ── Token Mapper Agent (Agent 2) ───────────────────────────────
@dataclass
class NodeTokenRecord:
    """Stores token usage and timing for one LangGraph node."""
    node_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: float = 0.0

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class TokenMapperAgent:
    """
    Agent 2 — Real-Time Token Mapper.

    Wraps every LangGraph node with start_node() / end_node() hooks.
    Token counts come directly from the Gemini API response.
    Call report() to print the per-node summary table.
    """

    def __init__(self):
        self.records: List[NodeTokenRecord] = []
        self._timers = {}

    def start_node(self, name: str):
        self._timers[name] = time.perf_counter()
        print(f"  --> [{name}] starting...")

    def end_node(self, name: str, input_tokens: int = 0, output_tokens: int = 0):
        elapsed = 0.0
        if name in self._timers:
            elapsed = (time.perf_counter() - self._timers.pop(name)) * 1000
        self.records.append(NodeTokenRecord(
            node_name=name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed,
        ))
        has_tokens = (input_tokens + output_tokens) > 0
        tok_str = f"{input_tokens} in / {output_tokens} out" if has_tokens else "(no LLM call)"
        print(f"  <-- [{name}] done  |  tokens: {tok_str}  |  {elapsed:.0f} ms")

    def report(self):
        """Print a per-node token usage table to the terminal."""
        print()
        print("=" * 64)
        print("  TOKEN MAPPER AGENT — Per-Node Usage")
        print("=" * 64)
        header = f"{'Node':<20} {'In':>6} {'Out':>6} {'Total':>7} {'ms':>8}   Cost"
        print(header)
        print("-" * 64)
        total_in = total_out = 0
        total_ms = 0.0
        for r in self.records:
            total_in  += r.input_tokens
            total_out += r.output_tokens
            total_ms  += r.elapsed_ms
            in_s  = str(r.input_tokens)  if r.input_tokens  else "—"
            out_s = str(r.output_tokens) if r.output_tokens else "—"
            tot_s = str(r.total_tokens)  if r.total_tokens  else "—"
            print(f"{r.node_name:<20} {in_s:>6} {out_s:>6} {tot_s:>7} {r.elapsed_ms:>7.0f}ms  $0.00 [FREE]")
        print("-" * 64)
        print(f"{'TOTAL':<20} {total_in:>6} {total_out:>6} {total_in+total_out:>7} {total_ms:>7.0f}ms  $0.00 [FREE TIER]")
        print("=" * 64)

    def reset(self):
        self.records.clear()
        self._timers.clear()


token_agent = TokenMapperAgent()
print("[OK] Token Mapper Agent ready.")

# ── Shared State Schema ────────────────────────────────────────
class PipelineState(TypedDict, total=False):
    """State that flows through the entire LangGraph pipeline."""
    pdf_path: str
    question: str
    chunks: List[str]
    chunk_meta: List[dict]
    embeddings: Any
    faiss_index: Any
    retrieved: List[str]
    retrieved_meta: List[dict]
    answer: str
    token_usage: dict


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into overlapping fixed-size character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ── Load Embedding Model ───────────────────────────────────────
print(f"\nLoading embedding model '{EMBEDDING_MODEL}' (local, free)...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("[OK] Embedding model ready.")

# ── NODE 1: ingest_pdf ─────────────────────────────────────────
def ingest_pdf(state: PipelineState) -> PipelineState:
    """Parse the PDF page by page and split text into overlapping chunks."""
    token_agent.start_node("ingest_pdf")

    pdf_path = state["pdf_path"]
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    chunks = []
    meta = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(page_chunks):
            chunks.append(ch)
            meta.append({"page": page_num, "chunk_index": i})

    token_agent.end_node("ingest_pdf")
    print(f"     {len(reader.pages)} pages -> {len(chunks)} chunks")
    return {**state, "chunks": chunks, "chunk_meta": meta}


# ── NODE 2: embed_and_index ────────────────────────────────────
def embed_and_index(state: PipelineState) -> PipelineState:
    """Embed all chunks locally using sentence-transformers and build a FAISS index."""
    token_agent.start_node("embed_and_index")

    chunks = state["chunks"]
    print(f"     Embedding {len(chunks)} chunks locally (free, no API)...")

    emb = embedder.encode(
        chunks,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine similarity
    index.add(emb)

    token_agent.end_node("embed_and_index")
    print(f"     FAISS index built: {index.ntotal} vectors, dim={emb.shape[1]}")
    return {**state, "embeddings": emb, "faiss_index": index}


# ── NODE 3: retrieve ───────────────────────────────────────────
def retrieve(state: PipelineState) -> PipelineState:
    """Embed the question and retrieve top-K most similar chunks via FAISS."""
    token_agent.start_node("retrieve")

    q_emb = embedder.encode(
        [state["question"]],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = state["faiss_index"].search(q_emb, TOP_K)
    retrieved      = [state["chunks"][i]     for i in indices[0] if i >= 0]
    retrieved_meta = [state["chunk_meta"][i] for i in indices[0] if i >= 0]

    score_str = ", ".join(f"{s:.3f}" for s in scores[0])
    print(f"     Top-{TOP_K} similarity scores: [{score_str}]")
    for m in retrieved_meta:
        print(f"       -> Page {m['page']}, chunk #{m['chunk_index']}")

    token_agent.end_node("retrieve")
    return {**state, "retrieved": retrieved, "retrieved_meta": retrieved_meta}


# ── NODE 4: generate_answer ────────────────────────────────────
def generate_answer(state: PipelineState) -> PipelineState:
    """Build the RAG prompt from retrieved chunks and call Gemini (free tier)."""
    token_agent.start_node("generate_answer")

    context_parts = []
    for chunk, m in zip(state["retrieved"], state["retrieved_meta"]):
        context_parts.append(f"[Page {m['page']}]\n{chunk}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a precise document assistant. "
        "Answer ONLY using the context below. "
        "If the answer is not present, say: This information is not in the document.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {state['question']}\n\n"
        "ANSWER:"
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    answer = response.text.strip()
    usage  = response.usage_metadata
    token_usage = {
        "input_tokens":  usage.prompt_token_count     or 0,
        "output_tokens": usage.candidates_token_count or 0,
        "total_tokens":  usage.total_token_count      or 0,
    }

    token_agent.end_node(
        "generate_answer",
        input_tokens=token_usage["input_tokens"],
        output_tokens=token_usage["output_tokens"],
    )
    return {**state, "answer": answer, "token_usage": token_usage}


# ── NODE 5: token_summary (Token Mapper Agent node) ────────────
def token_summary(state: PipelineState) -> PipelineState:
    """Signal node — triggers the Token Mapper to finalize its record."""
    token_agent.start_node("token_summary")
    token_agent.end_node("token_summary")
    return state


# ── Build LangGraph ────────────────────────────────────────────
builder = StateGraph(PipelineState)
builder.add_node("ingest_pdf",      ingest_pdf)
builder.add_node("embed_and_index", embed_and_index)
builder.add_node("retrieve",        retrieve)
builder.add_node("generate_answer", generate_answer)
builder.add_node("token_summary",   token_summary)
builder.set_entry_point("ingest_pdf")
builder.add_edge("ingest_pdf",      "embed_and_index")
builder.add_edge("embed_and_index", "retrieve")
builder.add_edge("retrieve",        "generate_answer")
builder.add_edge("generate_answer", "token_summary")
builder.add_edge("token_summary",   END)
pipeline = builder.compile()
print("\n[OK] LangGraph pipeline compiled!")

# ── Run ────────────────────────────────────────────────────────
token_agent.reset()

print()
print("=" * 60)
print(f"  PDF:      {PDF_PATH}")
print(f"  Question: {QUESTION}")
print(f"  Model:    {GEMINI_MODEL}  [FREE TIER]")
print("=" * 60)
print()

final_state = pipeline.invoke({
    "pdf_path": PDF_PATH,
    "question": QUESTION,
})

# ── Answer ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("  ANSWER")
print("=" * 60)
print(final_state["answer"])

# ── Token Report ───────────────────────────────────────────────
token_agent.report()

if "token_usage" in final_state:
    u = final_state["token_usage"]
    print()
    print("Raw Gemini token counts:")
    print(f"  Prompt tokens:    {u['input_tokens']:,}")
    print(f"  Candidate tokens: {u['output_tokens']:,}")
    print(f"  Total tokens:     {u['total_tokens']:,}")
    print()
    print("Gemini 2.5 Flash: 500 req/day | 1M tokens/day | $0.00")
