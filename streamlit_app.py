import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title = "RAG Ingest PDF", page_icon = "📄", layout = "centered")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
DEMO_PDF_PATH = Path(__file__).parent / "eval" / "pdfs" / "employee_handbook.pdf"


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    # Falls back to the INNGEST_DEV env var: set INNGEST_DEV=1 locally, leave
    # unset when deployed so the SDK sends events to Inngest Cloud instead of
    # a local dev server.
    return inngest.Inngest(app_id = "rag_app")


def trigger_ingest(filename: str, file_bytes: bytes) -> None:
    # Uploaded through the backend (not saved locally) since Streamlit and the
    # backend run on separate hosts once deployed — only the backend's own
    # filesystem is guaranteed to be readable by the ingest step that follows.
    resp = requests.post(
        f"{BACKEND_URL}/uploads",
        files = {"file": (filename, file_bytes, "application/pdf")},
        timeout = 60,
    )
    resp.raise_for_status()


st.title("Upload a PDF to Ingest")
st.caption(
    "First request after a period of inactivity can take up to a minute "
    "while the backend wakes up from its free-tier sleep."
)

if st.button("Load demo document (employee handbook)"):
    with st.spinner("Uploading and triggering ingestion..."):
        trigger_ingest(DEMO_PDF_PATH.name, DEMO_PDF_PATH.read_bytes())
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {DEMO_PDF_PATH.name}")

uploaded = st.file_uploader("Or choose your own PDF", type = ["pdf"], accept_multiple_files = False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        trigger_ingest(uploaded.name, uploaded.getvalue())
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {uploaded.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name = "rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
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
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value = 1, max_value = 20, value = 5, step = 1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's output
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")