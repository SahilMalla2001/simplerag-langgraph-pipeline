"""
Run the full RAG pipeline end-to-end as a plain Python script.
This is equivalent to running all cells in pipeline.ipynb.
"""

import os, time, warnings
from pathlib import Path
from typing import TypedDict, List, Any
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# ── Environment ───────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    raise EnvironmentError("GEMINI_API_KEY not set in .env!")
print(f"[OK] Gemini API key loaded (...{GEMINI_API_KEY[-6:]})")

# ── Imports ───────────────────────────────────────────────────
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
from google.genai import types as genai_types
from langgraph.graph import StateGraph, END
import pandas as pd

print("[OK] All imports successful!")

# ── Configuration ─────────────────────────────────────────────
PDF_PATH        = "GRE_score.pdf"
GEMINI_MODEL    = "gemini-2.5-flash"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K           = 4
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QUESTION        = "What were the Verbal Reasoning, Quantitative Reasoning, and Analytical Writing scores?"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
print(f"[OK] Config: PDF={PDF_PATH}, Model={GEMINI_MODEL}, TOP_K={TOP_K}")

# ── Token Mapper Agent ────────────────────────────────────────
@dataclass
class NodeTokenRecord:
    node_name:     str
    input_tokens:  int   = 0
    output_tokens: int   = 0
    elapsed_ms:    float = 0.0

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class TokenMapperAgent:
    def __init__(self):
        self.records: List[NodeTokenRecord] = []
        self._timers = {}

    def start_node(self, name: str):
        self._timers[name] = time.perf_counter()
        print(f"  --> [{name}] starting...")

    def end_node(self, name: str, input_tokens=0, output_tokens=0):
        elapsed = 0.0
        if name in self._timers:
            elapsed = (time.perf_counter() - self._timers.pop(name)) * 1000
        self.records.append(NodeTokenRecord(
            node_name=name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed,
        ))
        tok = f"{input_tokens} in / {output_tokens} out" if (input_tokens + output_tokens) else "(no LLM call)"
        print(f"  <-- [{name}] done  |  tokens: {tok}  |  {elapsed:.0f} ms")

    def report(self):
        print()
        print("=" * 62)
        print("  TOKEN MAPPER AGENT — Per-Node Usage")
        print("=" * 62)
        header = f"{'Node':<20} {'In Tok':>8} {'Out Tok':>8} {'Total':>8} {'Time(ms)':>10}  Cost"
        print(header)
        print("-" * 62)
        ti = to = 0
        tm = 0.0
        for r in self.records:
            ti += r.input_tokens
            to += r.output_tokens
            tm += r.elapsed_ms
            in_s  = str(r.input_tokens)  if r.input_tokens  else "—"
            out_s = str(r.output_tokens) if r.output_tokens else "—"
            tot_s = str(r.total_tokens)  if r.total_tokens  else "—"
            print(f"{r.node_name:<20} {in_s:>8} {out_s:>8} {tot_s:>8} {r.elapsed_ms:>9.0f}ms  $0.00 [FREE]")
        print("-" * 62)
        print(f"{'TOTAL':<20} {ti:>8} {to:>8} {ti+to:>8} {tm:>9.0f}ms  $0.00 [FREE TIER]")
        print("=" * 62)

    def reset(self):
        self.records.clear()
        self._timers.clear()


token_agent = TokenMapperAgent()
print("[OK] Token Mapper Agent ready.")

# ── Helpers ───────────────────────────────────────────────────
def _chunk_text(text, size, overlap):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]

# ── Load embedding model ──────────────────────────────────────
print(f"\nLoading embedding model '{EMBEDDING_MODEL}' (local, free)...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("[OK] Embedding model ready.")

# ── Pipeline State Schema ──────────────────────────────────────
class PipelineState(TypedDict, total=False):
    pdf_path:       str
    question:       str
    chunks:         List[str]
    chunk_meta:     List[dict]
    embeddings:     Any
    faiss_index:    Any
    retrieved:      List[str]
    retrieved_meta: List[dict]
    answer:         str
    token_usage:    dict

# ── NODE 1: ingest_pdf ────────────────────────────────────────
def ingest_pdf(state: PipelineState) -> PipelineState:
    token_agent.start_node("ingest_pdf")
    if not Path(state["pdf_path"]).exists():
        raise FileNotFoundError(f"PDF not found: {state['pdf_path']}")
    reader = PdfReader(state["pdf_path"])
    chunks, meta = [], []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for i, chunk in enumerate(_chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append(chunk)
            meta.append({"page": page_num, "chunk_index": i})
    token_agent.end_node("ingest_pdf")
    print(f"     {len(reader.pages)} pages -> {len(chunks)} chunks")
    return {**state, "chunks": chunks, "chunk_meta": meta}

# ── NODE 2: embed_and_index ───────────────────────────────────
def embed_and_index(state: PipelineState) -> PipelineState:
    token_agent.start_node("embed_and_index")
    chunks = state["chunks"]
    print(f"     Embedding {len(chunks)} chunks locally (free, no API)...")
    emb = embedder.encode(
        chunks, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    token_agent.end_node("embed_and_index")
    print(f"     FAISS index: {index.ntotal} vectors, dim={emb.shape[1]}")
    return {**state, "embeddings": emb, "faiss_index": index}

# ── NODE 3: retrieve ──────────────────────────────────────────
def retrieve(state: PipelineState) -> PipelineState:
    token_agent.start_node("retrieve")
    q_emb = embedder.encode(
        [state["question"]], convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")
    scores, indices = state["faiss_index"].search(q_emb, TOP_K)
    retrieved      = [state["chunks"][i]     for i in indices[0] if i >= 0]
    retrieved_meta = [state["chunk_meta"][i] for i in indices[0] if i >= 0]
    print(f"     Top-{TOP_K} scores: [{', '.join(f'{s:.3f}' for s in scores[0])}]")
    for m in retrieved_meta:
        print(f"       -> Page {m['page']}, chunk #{m['chunk_index']}")
    token_agent.end_node("retrieve")
    return {**state, "retrieved": retrieved, "retrieved_meta": retrieved_meta}

# ── NODE 4: generate_answer ───────────────────────────────────
def generate_answer(state: PipelineState) -> PipelineState:
    token_agent.start_node("generate_answer")
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {m['page']}]\n{chunk}"
        for i, (chunk, m) in enumerate(zip(state["retrieved"], state["retrieved_meta"]))
    )
    prompt = (
        "You are a precise document assistant. "
        "Answer ONLY using the context below. "
        "If not found say: 'This information is not in the document.' "
        "Cite the chunk number(s) used.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {state['question']}\n\nANSWER:"
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

# ── NODE 5: token_summary ─────────────────────────────────────
def token_summary(state: PipelineState) -> PipelineState:
    token_agent.start_node("token_summary")
    token_agent.end_node("token_summary")
    return state

# ── Build LangGraph ───────────────────────────────────────────
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
print("[OK] LangGraph pipeline compiled!")

# ── RUN ───────────────────────────────────────────────────────
print()
print("=" * 62)
print(f"  PDF:      {PDF_PATH}")
print(f"  Question: {QUESTION}")
print(f"  Model:    {GEMINI_MODEL}  [FREE TIER]")
print("=" * 62)
print()

token_agent.reset()
final_state = pipeline.invoke({
    "pdf_path": PDF_PATH,
    "question": QUESTION,
})

# ── Results ───────────────────────────────────────────────────
print()
print("=" * 62)
print("  ANSWER")
print("=" * 62)
print(final_state["answer"])

print()
print("=" * 62)
print("  RETRIEVED SOURCE CHUNKS")
print("=" * 62)
for i, (chunk, meta) in enumerate(
    zip(final_state["retrieved"], final_state["retrieved_meta"]), 1
):
    print(f"\n[Chunk {i}] Page {meta['page']}, chunk #{meta['chunk_index']}")
    print("-" * 40)
    print(chunk[:300] + ("..." if len(chunk) > 300 else ""))

# ── Token Report ──────────────────────────────────────────────
token_agent.report()

if "token_usage" in final_state:
    u = final_state["token_usage"]
    print()
    print("Raw Gemini token counts:")
    print(f"  Prompt tokens:    {u['input_tokens']:,}")
    print(f"  Candidate tokens: {u['output_tokens']:,}")
    print(f"  Total tokens:     {u['total_tokens']:,}")
    print()
    print("Free tier limits (Gemini 2.5 Flash):")
    print("  15 requests/min | 1,000,000 tokens/day | $0.00")
