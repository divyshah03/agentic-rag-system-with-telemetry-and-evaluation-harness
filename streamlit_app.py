from pathlib import Path
import time

import streamlit as st
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title = "RAG Ingest PDF", page_icon = "📄", layout = "centered")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
DEMO_PDF_PATH = Path(__file__).parent / "eval" / "pdfs" / "employee_handbook.pdf"


def post_backend(path: str, **kwargs) -> dict:
    # Render's free tier can take well over a minute to cold-start a sleeping
    # backend, so the first request after inactivity gets a generous timeout;
    # a retry (now against an already-awake backend) gets a short one.
    url = f"{BACKEND_URL}{path}"
    try:
        resp = requests.post(url, timeout = 120, **kwargs)
    except requests.exceptions.ReadTimeout:
        resp = requests.post(url, timeout = 30, **kwargs)

    if not resp.ok:
        # Reported rather than raised: Streamlit Cloud redacts the text of
        # uncaught exceptions, which hides the status code needed to tell a
        # missing endpoint (stale deploy) from a backend-side failure.
        st.error(f"Backend returned HTTP {resp.status_code} for POST {path}.")
        st.code(resp.text[:1000] or "(empty response body)")
        st.stop()

    return resp.json()


def trigger_ingest(filename: str, file_bytes: bytes) -> None:
    # Uploaded through the backend (not saved locally) since Streamlit and the
    # backend run on separate hosts once deployed — only the backend's own
    # filesystem is guaranteed to be readable by the ingest step that follows.
    post_backend("/uploads", files = {"file": (filename, file_bytes, "application/pdf")})


st.title("Upload a PDF to Ingest")
st.caption(
    "First request after a period of inactivity can take up to two minutes "
    "while the backend wakes up from its free-tier sleep."
)

if st.button("Load demo document (employee handbook)"):
    with st.spinner("Uploading and triggering ingestion — this can take up to 2 minutes if the backend is waking up..."):
        trigger_ingest(DEMO_PDF_PATH.name, DEMO_PDF_PATH.read_bytes())
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {DEMO_PDF_PATH.name}")

uploaded = st.file_uploader("Or choose your own PDF", type = ["pdf"], accept_multiple_files = False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion — this can take up to 2 minutes if the backend is waking up..."):
        trigger_ingest(uploaded.name, uploaded.getvalue())
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {uploaded.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


def request_query(question: str, top_k: int) -> str:
    # The backend sends the Inngest event, so this frontend needs no Inngest
    # credentials and no event loop of its own.
    return post_backend("/query", json = {"question": question, "top_k": top_k})["event_id"]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    headers = {}
    signing_key = os.getenv("INNGEST_SIGNING_KEY")
    # The local dev server's REST API is unauthenticated; Inngest Cloud's isn't.
    # NOTE: verify this is the auth scheme Inngest Cloud's dashboard docs
    # describe for this endpoint before relying on it in production.
    if signing_key and "127.0.0.1" not in url and "localhost" not in url:
        headers["Authorization"] = f"Bearer {signing_key}"
    resp = requests.get(url, headers = headers)

    if not resp.ok:
        # Same reasoning as post_backend: a redacted traceback here would hide
        # whether polling failed on auth (401) or on the run itself.
        st.error(f"Inngest run-status API returned HTTP {resp.status_code}.")
        st.code(resp.text[:1000] or "(empty response body)")
        st.stop()

    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 300.0, poll_interval_s: float = 1.0) -> dict:
    # A first query on a cold free-tier backend legitimately runs for minutes:
    # the cross-encoder model is downloaded and loaded, and the BM25 index is
    # rebuilt from Qdrant, before any reranking happens.
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                st.error(f"The Inngest run {status.lower()}. See the run's timeline in the Inngest dashboard.")
                st.stop()
        if time.time() - start > timeout_s:
            # Reported rather than raised so the last status survives Streamlit
            # Cloud's redaction: no status at all means no function run was ever
            # created for the event (typically an unsynced Inngest app), which
            # is a different problem from a run that is merely slow.
            if last_status is None:
                st.error(
                    f"No Inngest run appeared for event {event_id} within {timeout_s:.0f}s. "
                    "The event was accepted but nothing picked it up — check that the Inngest "
                    "Cloud app is synced against the deployed backend's /api/inngest."
                )
            else:
                st.error(
                    f"Timed out after {timeout_s:.0f}s waiting for the answer "
                    f"(last run status: {last_status})."
                )
            st.stop()
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value = 1, max_value = 20, value = 5, step = 1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Backend fires the Inngest event and hands back its id
            event_id = request_query(question.strip(), int(top_k))
            # Poll the Inngest API for the run's output
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            retrieved_chunks = output.get("retrieved_chunks", [])
            routing = output.get("routing") or {}

        st.subheader("Answer")
        st.write(answer or "(No answer)")

        if routing.get("triggered"):
            st.caption(
                f"🧭 Routing triggered — low-confidence top result, re-retrieved "
                f"with a wider pool (strategy: {routing.get('strategy')})."
            )

        if retrieved_chunks:
            with st.expander(f"Sources ({len(retrieved_chunks)} chunks retrieved)"):
                for chunk in retrieved_chunks:
                    st.markdown(f"**{chunk.get('source', '?')}** — score: `{chunk.get('score'):.3f}`")
                    st.caption(chunk.get("text", ""))