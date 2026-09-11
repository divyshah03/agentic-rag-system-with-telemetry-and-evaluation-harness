# 📄 RAG Production App

An **evaluation-driven Retrieval-Augmented Generation (RAG)** system for PDF document Q&A — built to measure and improve retrieval quality, not just demo it.

Upload PDFs through a Streamlit UI, ingest them into a vector store, and ask natural-language questions. Every retrieval returns structured chunks with similarity scores, and an automated eval harness scores pipeline changes against ground-truth Q&A pairs.

---

## ✨ Features

### ✅ Implemented today

- **📥 PDF ingestion** — Upload PDFs; they are chunked, embedded, and stored in Qdrant
- **🔍 Hybrid retrieval** — BM25 keyword search + dense cosine search over OpenAI embeddings (`text-embedding-3-large`, 3072-dim), fused with Reciprocal Rank Fusion
- **🎯 Cross-encoder re-ranking** — The fused candidate pool is re-scored with a local ONNX cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2` via `fastembed`) before the final top-k is picked; toggle back to dense-only retrieval per-query via `retrieval_mode`
- **🧭 Agentic query routing** — When the top retrieved chunk's cross-encoder score falls below its own relevant/irrelevant boundary (< 0), the query is automatically re-retrieved with a much wider candidate pool before answering; toggle with `enable_routing`
- **🤖 LLM answers** — `gpt-4o-mini` answers from retrieved context only
- **⚡ Durable workflows** — Inngest orchestrates ingest and query as step-based functions (retries, observability, execution traces)
- **🖥️ Streamlit UI** — Upload documents and ask questions in the browser
- **📊 Evaluation harness** — Ground-truth Q&A dataset with retrieval metrics (Recall@k, MRR) and LLM-as-judge answer scoring
- **🛡️ Ingest guardrails** — Throttle (2/min global) and per-source rate limit (2 per 4 hours per PDF)
- **📦 Structured retrieval output** — Each query returns `retrieved_chunks` with `text`, `source`, and `score`

### 🚧 Planned / roadmap

- SQLite query telemetry and logging
- RAGAS integration

---

## 🏗️ Architecture

```
Streamlit UI (streamlit_app.py)
    │
    ├─ Upload PDF ──► POST {BACKEND_URL}/uploads ──► saved to backend's uploads/
    │                                                        │
    │                                                        ▼
    │                                    FastAPI + Inngest (app/main.py)
    │                                    sends Inngest event: rag/ingest_pdf
    │                                              │
    │                                              ▼
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

The upload goes through the backend (rather than Streamlit sending the Inngest
event directly with a local file path) so ingestion works when Streamlit and
the backend are deployed on separate hosts — see [Deployment](#-deployment).

### Ingest flow

1. PDF uploaded via `POST /uploads` and saved under `uploads/` on the backend
2. `PDFReader` (LlamaIndex) extracts text — one `Document` per PDF page. All pages are joined into a single document-level string *before* splitting, so `SentenceSplitter` (1000 tokens, 200 token overlap) can chunk across what used to be a hard page boundary. (Joining first matters: chunking per page independently means overlap can never bridge a page break, and short pages may never even reach the chunk-size threshold at all — see AGENTS.md.)
3. OpenAI `text-embedding-3-large` embeds each chunk
4. Deterministic UUID per chunk (`source_id` + index); payload `{source, text}`
5. Vectors upserted into the Qdrant collection `docs` at `QDRANT_URL` (`localhost:6333` by default, a Qdrant Cloud cluster in production)

### Query flow

1. Question embedded with the same model
2. Retrieval (`retrieval_mode`, default `hybrid`):
   - `hybrid` — dense Qdrant search + BM25 keyword search over the collection, fused by Reciprocal Rank Fusion, then re-ranked by a local cross-encoder; final `retrieved_chunks[i].score` is the cross-encoder relevance score. When `enable_routing` is true (default), a low-confidence top result triggers one re-retrieval pass with a much wider candidate pool before continuing — see [Agentic Query Routing](#-agentic-query-routing) below
   - `dense` — the original cosine-similarity-only path, for reproducing pre-hybrid baseline numbers; routing never applies here
3. Chunks formatted into a prompt
4. `gpt-4o-mini` generates an answer from context only
5. Response: `answer`, `sources`, `num_contexts`, `retrieved_chunks`, `routing`

---

## 🧭 Agentic Query Routing

When hybrid retrieval's top result doesn't look trustworthy, the query is automatically
re-retrieved with a wider candidate pool before answering, instead of confidently answering from
whatever came back.

**Confidence signal:** the cross-encoder's top-1 relevance score, thresholded at `0` — the
reranker's own trained relevant/irrelevant decision boundary (MS MARCO cross-encoders are trained
as a binary classifier around that boundary). This is deliberately *not* fit to this project's
eval labels: with only one failure in the 80-question eval set, no threshold could be validated
against pass/fail outcomes without overfitting to a sample size of one. Real trigger rate on the
eval set: 11/80 (13.75%).

**Re-retrieval strategy:** on low confidence, re-run hybrid search once with a much larger
candidate pool (20 → 60) and re-rank. Chosen over a dense-only fallback because it directly
answers what the trigger condition says ("of the candidates we gave the reranker, none looked
good") — no retry loop; if the widened pass is still low-confidence, that result is returned as-is.

**Known limitation, found and kept rather than hidden:** this signal does *not* catch every retrieval
failure. Investigating `pr_018` (an open-ended eval question) after the hybrid-retrieval rollout
showed the failure mode was a confident-but-incomplete retrieval — the top chunk scored +1.61
(clearly "relevant" by the same threshold) while a second necessary chunk, orphaned from its
section header by a PDF-page-boundary chunking bug, scored −11.3 (indistinguishable from actual
noise) and never resurfaced. No signal computed from final top-k scores can detect evidence that
was already discarded before scoring — that's a chunking/coverage problem, not something a
confidence gate on retrieval output can fix. Routing targets the more common, structurally
different "nothing retrieved looked relevant at all" failure instead.

*Update:* the chunking bug behind `pr_018` (per-page-independent chunking, see AGENTS.md) has since
been fixed — `pr_018` now passes, and passes confidently enough that routing doesn't even need to
fire for it. The general point stands as an architectural limitation of any post-hoc,
final-top-k confidence signal, even though this specific instance is resolved: routing is a
safety net for "nothing looked relevant," not a substitute for correct chunking.

`enable_routing` (default `true`) toggles this per-query; `eval_harness.py` exposes it as
`--enable-routing` / `--no-enable-routing` for routing-on vs. routing-off comparison runs, and
records `routing.triggered`/`routing.initial_confidence`/`routing.final_confidence` per question
in the results JSON plus a `routing_trigger_rate` in the run summary.

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

Copy `.env.example` to `.env` and fill in your OpenAI key:

```bash
cp .env.example .env
```

```bash
OPENAI_API_KEY=sk-your-key-here
INNGEST_DEV=1   # targets `inngest dev` instead of Inngest Cloud
```

The other vars in `.env.example` (`QDRANT_URL`, `QDRANT_API_KEY`, `INNGEST_EVENT_KEY`,
`INNGEST_SIGNING_KEY`, `BACKEND_URL`) are only needed for the hosted setup — see
[Deployment](#-deployment) and [Environment Variables](#️-environment-variables).

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

# Compare routing on vs. off (routing is on by default)
uv run python eval/eval_harness.py --no-enable-routing --run-label routing-off
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

### 📈 Benchmarks: baseline → final

Same 80-question dataset (55 factual, 25 open-ended), 0 errors in both runs.

| Metric | Baseline (dense-only) | Final (hybrid + routing + chunking fix) |
|---|---|---|
| Exact-match accuracy | 98.2% | **100.0%** |
| Judge pass rate | 100.0% | 100.0% |
| Avg judge score | 4.92 / 5 | 4.92 / 5 |
| Overall pass rate | 98.75% | **100.0%** |
| Retrieval hit-rate | 100.0% | 100.0% |
| Routing trigger rate | n/a (feature didn't exist) | 13.75% |

- **Baseline** — `eval/results/eval_baseline_20260829_232315.json`: cosine-only dense retrieval, before hybrid search, re-ranking, agentic routing, or the chunking fix existed.
- **Final** — `eval/results/eval_chunk-fix-rerun_20260911_141534.json`: BM25 + dense hybrid retrieval fused by RRF, cross-encoder re-ranking, confidence-gated re-retrieval, and page-joined chunking (see [Agentic Query Routing](#-agentic-query-routing) and AGENTS.md).
- The single point of factual-accuracy headroom in the baseline (98.2% → 100%) was exactly one failure mode: a source document's list item got orphaned from its section header by a chunking bug (`pr_018`, detailed above) — closed by joining PDF pages before splitting, not by retrieval tuning.
- Routing's 13.75% trigger rate has no baseline column to compare against since confidence-gated re-retrieval didn't exist yet — it's there to show the safety net actually engages on real queries, not that it moved this particular metric (this small eval corpus mostly didn't have anything left to find once widening the pool).

---

## 📁 Project Structure

```
RAGProductionApp/
├── app/                        # Importable application package
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + Inngest ingest/query functions
│   ├── data_loader.py          # PDF loading, chunking, OpenAI embeddings
│   ├── vector_db.py            # Qdrant wrapper (upsert, search, scroll_all)
│   ├── hybrid_retrieval.py     # BM25 index, RRF fusion, cross-encoder re-ranking
│   ├── query_router.py         # Confidence gate + widened-pool re-retrieval
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
| `app/vector_db.py` | `QdrantStorage` — auto-creates collection, upsert + `query_points` search + `scroll_all` |
| `app/hybrid_retrieval.py` | `hybrid_search()` — BM25 + dense fusion (RRF) + cross-encoder re-ranking; `invalidate_bm25_cache()` |
| `app/query_router.py` | `route_query()` — confidence gate on `hybrid_search()`'s top-1 score, re-retrieves with a wider pool on low confidence |
| `app/custom_types.py` | `RAGChunkAndSrc`, `RAGUpsertresult`, `RAGSearchResult`, `RetrievedChunk` |
| `eval/eval_harness.py` | End-to-end eval: ingest sample PDFs, run Q&A, score retrieval + answers |

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill in real values for local dev. In hosted
environments these are set on each platform's dashboard instead — see
[Deployment](#-deployment) for exactly which var goes where.

| Variable | Where it's used | Required | Default | Description |
|----------|------------------|----------|---------|-------------|
| `OPENAI_API_KEY` | Backend | ✅ | — | Embeddings and LLM generation |
| `INNGEST_DEV` | Backend, Streamlit, eval | ❌ | unset | Set to `1` locally to target `inngest dev` instead of Inngest Cloud |
| `QDRANT_URL` | Backend | ❌ | `http://localhost:6333` | Qdrant instance to connect to (Qdrant Cloud cluster URL when deployed) |
| `QDRANT_API_KEY` | Backend | ❌ (✅ for Qdrant Cloud) | — | API key for Qdrant Cloud |
| `INNGEST_EVENT_KEY` | Backend, Streamlit, eval | ❌ (✅ for Inngest Cloud) | — | Authenticates sending events to Inngest Cloud |
| `INNGEST_SIGNING_KEY` | Backend, Streamlit | ❌ (✅ for Inngest Cloud) | — | Verifies Inngest Cloud's webhook calls to the backend; also used as the bearer token when polling the hosted run-status API |
| `INNGEST_API_BASE` | Streamlit, eval | ❌ | `http://127.0.0.1:8288/v1` | Inngest REST API base for run-status polling (`https://api.inngest.com/v1` in production) |
| `BACKEND_URL` | Streamlit | ❌ | `http://127.0.0.1:8000` | Public URL of the deployed FastAPI backend |

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

## 🚀 Deployment

Each local piece maps to a hosted equivalent:

| Local | Hosted |
|-------|--------|
| Qdrant (Docker) | [Qdrant Cloud](https://cloud.qdrant.io) free cluster |
| `inngest dev` | [Inngest Cloud](https://www.inngest.com) free tier |
| `uvicorn app.main:app` | [Render](https://render.com) free Web Service |
| `streamlit run streamlit_app.py` | [Streamlit Community Cloud](https://streamlit.io/cloud) |

**Render (backend) setup:**
- Build command: `pip install uv && uv sync --frozen --no-dev`
- Start command: `uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Set `PYTHON_VERSION=3.13` (or the equivalent runtime setting) — this project requires Python ≥3.13
- Env vars: `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `INNGEST_SIGNING_KEY`, `INNGEST_EVENT_KEY`

**Inngest Cloud setup:** create an app, copy its Event Key and Signing Key. Once
the backend is deployed, sync the app against `https://<your-render-app>.onrender.com/api/inngest`
(the default path `inngest.fast_api.serve()` mounts — no path configuration needed).

**Streamlit Community Cloud setup:** deploy `streamlit_app.py` from this repo,
then set `BACKEND_URL`, `INNGEST_EVENT_KEY`, `INNGEST_SIGNING_KEY`, and
`INNGEST_API_BASE=https://api.inngest.com/v1` in the app's **Secrets** panel.
`OPENAI_API_KEY` is not needed here — only the backend calls OpenAI.

**Free-tier cold starts:** Render's free Web Service sleeps after 15 minutes
idle; the first request afterward can take 30-60s to wake up. The UI shows a
caption warning about this rather than paying for an always-on instance or
running a keep-alive job — proportionate for a demo link.

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
