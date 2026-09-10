# AGENTS.md

## Project layout

- `app/` — importable application package: `main.py` (FastAPI + Inngest functions), `data_loader.py`, `vector_db.py`, `custom_types.py`. Import these as `from app.vector_db import QdrantStorage`.
- `streamlit_app.py` — Streamlit UI at the repo root. Sends Inngest events and polls for run output; it does not import `app/`.
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
- Keyword arguments in `app/` are written with spaces around `=` (`limit = 2`). Match the surrounding file.
