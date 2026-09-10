# 📄 RAG Production App

An **evaluation-driven Retrieval-Augmented Generation (RAG)** system for PDF document Q&A — built to measure and improve retrieval quality, not just demo it.

Upload PDFs through a Streamlit UI, ingest them into a vector store, and ask natural-language questions. Every retrieval returns structured chunks with similarity scores, and an automated eval harness scores pipeline changes against ground-truth Q&A pairs.

---

## ✨ Features

### ✅ Implemented today

- **📥 PDF ingestion** — Upload PDFs; they are chunked, embedded, and stored in Qdrant
- **🔍 Hybrid retrieval** — BM25 keyword search + dense cosine search over OpenAI embeddings (`text-embedding-3-large`, 3072-dim), fused with Reciprocal Rank Fusion
- **🎯 Cross-encoder re-ranking** — The fused candidate pool is re-scored with a local ONNX cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2` via `fastembed`) before the final top-k is picked; toggle back to dense-only retrieval per-query via `retrieval_mode`
- **🤖 LLM answers** — `gpt-4o-mini` answers from retrieved context only
- **⚡ Durable workflows** — Inngest orchestrates ingest and query as step-based functions (retries, observability, execution traces)
- **🖥️ Streamlit UI** — Upload documents and ask questions in the browser
- **📊 Evaluation harness** — Ground-truth Q&A dataset with retrieval metrics (Recall@k, MRR) and LLM-as-judge answer scoring
- **🛡️ Ingest guardrails** — Throttle (2/min global) and per-source rate limit (2 per 4 hours per PDF)
- **📦 Structured retrieval output** — Each query returns `retrieved_chunks` with `text`, `source`, and `score`

### 🚧 Planned / roadmap

- SQLite query telemetry and logging
- RAGAS integration
- Benchmarked before/after metrics published in README

---

## 🏗️ Architecture

```
Streamlit UI (streamlit_app.py)
    │
    ├─ Upload PDF ──► save to uploads/ ──► Inngest event: rag/ingest_pdf
    │                                              │
    │                                              ▼
    │                                    FastAPI + Inngest (app/main.py)
    │                                    rag_ingest_pdf:
    │                                      1. load-and-chunk   (app/data_loader)
    │                                      2. embed-and-upsert (app/data_loader + app/vector_db)
    │
    └─ Ask question ──► Inngest event: rag/query_pdf_ai
                               │
                               ▼
                         rag_query_pdf_ai:
                           1. embed-and-search (app/data_loader + app/vector_db)
                           2. llm-answer       (OpenAI via Inngest AI)
                               │
                               ▼
                         Streamlit polls Inngest API for run output
```

### Ingest flow

1. PDF saved locally under `uploads/`
2. `PDFReader` (LlamaIndex) extracts text; `SentenceSplitter` chunks it (~1000 chars, 200 overlap)
3. OpenAI `text-embedding-3-large` embeds each chunk
4. Deterministic UUID per chunk (`source_id` + index); payload `{source, text}`
5. Vectors upserted into Qdrant collection `docs` on `localhost:6333`

### Query flow

1. Question embedded with the same model
2. Retrieval (`retrieval_mode`, default `hybrid`):
   - `hybrid` — dense Qdrant search + BM25 keyword search over the collection, fused by Reciprocal Rank Fusion, then re-ranked by a local cross-encoder; final `retrieved_chunks[i].score` is the cross-encoder relevance score
   - `dense` — the original cosine-similarity-only path, for reproducing pre-hybrid baseline numbers
3. Chunks formatted into a prompt
4. `gpt-4o-mini` generates an answer from context only
5. Response: `answer`, `sources`, `num_contexts`, `retrieved_chunks`

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.13 |
| **Package manager** | [uv](https://docs.astral.sh/uv/) |
| **API / workflows** | FastAPI, Inngest |
| **Vector DB** | Qdrant |
| **Embeddings & LLM** | OpenAI (`text-embedding-3-large`, `gpt-4o-mini`) |
| **Keyword search** | `rank-bm25` (BM25Okapi) |
| **Re-ranking** | `fastembed` `TextCrossEncoder` (`Xenova/ms-marco-MiniLM-L-6-v2`, ONNX, local/CPU) |
| **PDF parsing** | LlamaIndex (`PDFReader`, `SentenceSplitter`) |
| **UI** | Streamlit |
| **Eval** | Custom harness (`eval/eval_harness.py`), fpdf2 for sample PDFs |

---

## 📋 Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)
- [Inngest CLI](https://www.inngest.com/docs/local-development) (`inngest-cli`)
- OpenAI API key

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/divyshah03/agentic-rag-system-with-telemetry-and-evaluation-harness.git
cd agentic-rag-system-with-telemetry-and-evaluation-harness
uv sync
```

### 2. Configure environment

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Optional:

```bash
INNGEST_API_BASE=http://127.0.0.1:8288/v1   # default; used by Streamlit polling
```

### 3. Start Qdrant (Docker)

```bash
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

> `qdrant_storage/` is gitignored — it's local runtime data, recreated by re-ingesting.

### 4. Start Inngest dev server

```bash
inngest dev
```

Dashboard: http://127.0.0.1:8288

### 5. Start the FastAPI backend

```bash
uv run uvicorn app.main:app
```

API: http://127.0.0.1:8000

> If you see `address already in use`, a server is already running on port 8000. Use it or stop the old process first.

### 6. Start the Streamlit UI

```bash
uv run streamlit run streamlit_app.py
```

UI: http://localhost:8501

### 7. Use the app

1. Upload a PDF in Streamlit → triggers ingest via Inngest → chunks land in Qdrant
2. Ask a question → query runs via Inngest → answer and sources appear in Streamlit

---

## 📊 Evaluation Harness

The `eval/` directory contains a standalone harness that drives the pipeline through the same Inngest event contract as Streamlit — no duplicated retrieval logic.

```bash
# Ensure Qdrant, Inngest dev server, and uvicorn are running first

# Smoke test (5 questions)
uv run python eval/eval_harness.py --ingest --limit 5

# Full eval run
uv run python eval/eval_harness.py

# Label a run after a pipeline change
uv run python eval/eval_harness.py --run-label hybrid-retrieval

# Reproduce the original dense-only baseline for comparison
uv run python eval/eval_harness.py --retrieval-mode dense --run-label dense-baseline-repro
```

The harness runs against its own Qdrant collection (`eval_docs` by default, override with `--collection`), separate from whatever collection (`docs` by default) you use for manual/Streamlit testing — so eval runs stay reproducible regardless of what you've uploaded for ad-hoc testing.

By default the harness benchmarks the current hybrid retrieval path (`--retrieval-mode hybrid`, matching `rag_query_pdf_ai`'s own default) with no flags needed; pass `--retrieval-mode dense` to fall back to the original cosine-only path for an apples-to-apples before/after comparison.

**Metrics:**
- **Retrieval** — Recall@k, MRR (did the right source appear in top-k?)
- **Answer quality** — LLM-as-judge (1–5 scale, pass threshold ≥ 4)

**Dataset:** `eval/dataset/qa_dataset.json`  
**Sample PDFs:** `eval/pdfs/` (generated via `eval/generate_sample_pdfs.py`)  
**Results:** `eval/results/*.json` (gitignored; regenerate by re-running the harness)

### Evaluation-driven development workflow

1. Instrument the pipeline — every query returns retrieved chunks with scores
2. Build a ground-truth Q&A dataset targeting real failure modes
3. Score every pipeline version against that dataset
4. Change retrieval or generation, re-run the harness, compare metrics
5. Only call it an improvement when the numbers prove it

---

## 📁 Project Structure

```
RAGProductionApp/
├── app/                        # Importable application package
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + Inngest ingest/query functions
│   ├── data_loader.py          # PDF loading, chunking, OpenAI embeddings
│   ├── vector_db.py            # Qdrant wrapper (upsert, search)
│   └── custom_types.py         # Pydantic models for Inngest step I/O
├── streamlit_app.py            # Upload + query UI
├── eval/
│   ├── eval_harness.py         # Automated eval runner
│   ├── generate_sample_pdfs.py
│   ├── dataset/qa_dataset.json
│   ├── pdfs/                   # Sample documents for eval
│   └── results/                # Eval output (gitignored)
├── pyproject.toml              # Dependencies (uv)
├── uv.lock                     # Locked dependency versions
├── uploads/                    # User-uploaded PDFs (local, gitignored)
└── qdrant_storage/             # Qdrant on-disk data (local, gitignored)
```

### File reference

| File | Purpose |
|------|---------|
| `app/main.py` | Backend entry point. Registers `rag_ingest_pdf` and `rag_query_pdf_ai` Inngest functions |
| `streamlit_app.py` | Frontend: PDF upload, question form, Inngest event polling |
| `app/data_loader.py` | `load_and_chunk_pdf()`, `embed_texts()` |
| `app/vector_db.py` | `QdrantStorage` — auto-creates `docs` collection, upsert + `query_points` search |
| `app/custom_types.py` | `RAGChunkAndSrc`, `RAGUpsertresult`, `RAGSearchResult`, `RetrievedChunk`, `RAGQueryResult` |
| `eval/eval_harness.py` | End-to-end eval: ingest sample PDFs, run Q&A, score retrieval + answers |

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | Embeddings and LLM generation |
| `INNGEST_API_BASE` | ❌ | `http://127.0.0.1:8288/v1` | Inngest REST API for Streamlit/eval polling |

---

## 🛡️ Ingest Guardrails

The ingest function (`rag_ingest_pdf`) has two limits to prevent runaway embedding costs:

| Limit | Setting | Scope |
|-------|---------|-------|
| **Throttle** | 2 runs / minute | Global |
| **Rate limit** | 2 runs / 4 hours | Per `source_id` (PDF filename) |

---

## 🧩 Services at a Glance

| Service | Port | Command |
|---------|------|---------|
| Qdrant | 6333 | `docker run -p 6333:6333 qdrant/qdrant` |
| Inngest dev | 8288 | `inngest dev` |
| FastAPI | 8000 | `uv run uvicorn app.main:app` |
| Streamlit | 8501 | `uv run streamlit run streamlit_app.py` |

You can shut down Qdrant and FastAPI while keeping Inngest running — it will just sit idle until the backend is back.

---

## 🔒 What stays local (not on GitHub)

| Path | Why |
|------|-----|
| `.env` | API keys and secrets |
| `.venv/` | Virtual environment |
| `qdrant_storage/` | Qdrant runtime database |
| `uploads/` | User-uploaded PDFs |
| `eval/results/*.json` | Generated eval output |
| `__pycache__/` | Python bytecode cache |

---

## 👤 Author

**Divy Shah** — [divyrshah3@gmail.com](mailto:divyrshah3@gmail.com)
