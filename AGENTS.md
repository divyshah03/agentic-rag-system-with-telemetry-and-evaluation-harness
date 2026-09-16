# AGENTS.md

## Project layout

- `app/` — importable application package: `main.py` (FastAPI + Inngest functions), `data_loader.py`, `vector_db.py`, `hybrid_retrieval.py`, `query_router.py`, `custom_types.py`. Import these as `from app.vector_db import QdrantStorage`.
- `streamlit_app.py` — Streamlit UI at the repo root. Posts uploads and questions to the backend (`POST /uploads`, `POST /query`) and polls the Inngest REST API for run output; it does not import `app/` and does not use the Inngest SDK. Keep it that way: it runs on a separate host from the backend, and an `inngest.Inngest` client cached across Streamlit reruns breaks on the second query with `RuntimeError: Event loop is closed`, because each `asyncio.run()` closes the loop its pooled connections were bound to.
- `eval/` — evaluation harness, ground-truth dataset, and sample PDFs. Drives the pipeline only through Inngest events, so it needs no changes when retrieval internals change.

## Commands

- **Install:** `uv sync`
- **Backend:** `uv run uvicorn app.main:app` (port 8000)
- **UI:** `uv run streamlit run streamlit_app.py` (port 8501)
- **Eval:** `uv run python eval/eval_harness.py --ingest --limit 5`

There is no test suite or linter configured.

## Services

The pipeline needs three processes besides the UI. Do not start them or run a live ingest/query without confirming with the user first, since ingestion spends OpenAI credits.

| Service | Port | How to start |
|---------|------|--------------|
| Qdrant | 6333 | `docker run -p 6333:6333 qdrant/qdrant` |
| Inngest dev server | 8288 | `inngest dev` |
| FastAPI backend | 8000 | `uv run uvicorn app.main:app` |

`OPENAI_API_KEY` must be set in `.env` before the backend or eval harness will work.

## Conventions worth preserving

- **Inngest identity:** function IDs are `f"{app_id}-{fn_id}"`, derived only from the explicit `app_id="rag_app"` and `fn_id` strings in `app/main.py`. They do not depend on module paths. Changing either string makes Inngest treat it as a brand-new function.
- **`source_id` is not a path.** The ingest event's `source_id` is stored as `payload["source"]` in Qdrant and compared against the `source_pdf` field in `eval/dataset/qa_dataset.json`. Keep these as bare filenames (`test.pdf`), not paths, or retrieval metrics will silently score 0.
- **Qdrant collections are isolated by an event field, not hardcoded.** `rag/ingest_pdf` and `rag/query_pdf_ai` both read an optional `collection` field off `event.data` (default `"docs"`). Manual/Streamlit usage omits it and stays on `docs`; `eval_harness.py` defaults `--collection` to `eval_docs` so eval runs never see PDFs uploaded for manual testing, and vice versa. Qdrant auto-creates a collection on first use, so a new collection name just works with no migration step.
- **The BM25 index is an in-memory cache over Qdrant, not a second source of truth.** `app/hybrid_retrieval.py` builds it by scrolling a collection's payloads and caches it per collection name for the process's lifetime. `rag_ingest_pdf` calls `invalidate_bm25_cache(collection)` after every upsert so the next query rebuilds it — if you ever write to Qdrant by a path other than `rag_ingest_pdf` (a script, a notebook), call `invalidate_bm25_cache` yourself or hybrid queries will keyword-search a stale corpus.
- `rag_query_pdf_ai` reads `retrieval_mode` off `event.data` (default `"hybrid"`; `"dense"` reproduces the original cosine-only path). `eval_harness.py` exposes this as `--retrieval-mode`.
- **Those two defaults are env-driven, and event data always wins.** `app/main.py` reads `DEFAULT_RETRIEVAL_MODE` from `RETRIEVAL_MODE` and `DEFAULT_ENABLE_ROUTING` from `ENABLE_ROUTING`, but only as fallbacks for events that omit the field. `eval_harness.py` always sends both explicitly, which is what keeps the published benchmarks reproducible regardless of server config — keep it that way. The hosted Render deploy sets `RETRIEVAL_MODE=dense`, because the cross-encoder cannot run in 512 MB: `import app.main` alone is ~300 MB, loading the model reaches ~457 MB, and one rerank of the default 20-candidate pool peaks near 990 MB (~2.2 GB for routing's widened 60). `fastembed`'s `rerank()` defaults to `batch_size=64`, so the whole pool is one batch; even `batch_size=1` peaks at ~483 MB, so this is not a batching bug to fix but a plan ceiling. `dense` is the only mode that never constructs the reranker. Do not "fix" the live demo by re-enabling hybrid there — it OOMs, and Inngest reports it as the misleading "No step output was produced before the request failed".
- **`PDFReader` returns one `Document` per PDF page, not one per file.** `app/data_loader.py`'s `load_and_chunk_pdf` joins every page's text into a single string *before* calling `SentenceSplitter.split_text()`. Splitting per page instead (the original, buggy behavior) silently breaks `chunk_overlap` — overlap only bridges gaps within one `split_text()` call, never across separate calls — and on short PDFs (any page under the 1000-token `chunk_size`) it means chunking never engages at all, so each "chunk" is just a raw page. This caused a real eval failure (`pr_018`, see README's Agentic Query Routing section) where a bullet list got split across a page break and the second half lost its section header. Keep pages joined before splitting.
- **Routing (`app/query_router.py`) only applies to the `hybrid` path.** `route_query()` re-retrieves with a wider candidate pool when the top cross-encoder score is below `0` (the reranker's own relevant/irrelevant boundary, not a value fit to this project's eval labels — see README's Agentic Query Routing section for why, including a documented case it doesn't catch). `rag_query_pdf_ai` reads `enable_routing` off `event.data` (default `True`); `retrieval_mode="dense"` bypasses routing entirely regardless of `enable_routing`, since dense mode exists specifically to reproduce the untouched pre-hybrid baseline. `eval_harness.py` exposes the toggle as `--enable-routing`/`--no-enable-routing` and records per-question `routing` telemetry plus a `routing_trigger_rate` summary field.
- Keyword arguments in `app/` are written with spaces around `=` (`limit = 2`). Match the surrounding file.
