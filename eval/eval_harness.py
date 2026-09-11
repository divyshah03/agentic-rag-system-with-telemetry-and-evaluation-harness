"""Standalone evaluation harness for the RAG pipeline.

Drives the pipeline entirely through its public Inngest event contract (the same
send-event-then-poll-for-output pattern already used by streamlit_app.py) so this script never
duplicates retrieval/generation logic living in app/main.py / app/vector_db.py. That also means it keeps
working unmodified after future changes to retrieval (e.g. hybrid search) or routing (e.g. an
agentic router), as long as the `rag/query_pdf_ai` event's input/output shape stays the same.

Usage:
    # one-time: make sure Qdrant, `inngest-cli dev`, and `uv run uvicorn app.main:app` are running
    uv run python eval/eval_harness.py --ingest --limit 5   # smoke test
    uv run python eval/eval_harness.py                      # full run
    uv run python eval/eval_harness.py --run-label hybrid-retrieval   # after a pipeline change
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import inngest
import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

EVAL_DIR = Path(__file__).parent
DEFAULT_DATASET = EVAL_DIR / "dataset" / "qa_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

# source_id -> pdf path, used by --ingest. source_id must match the "source_pdf" values used in
# the dataset, since that's what retrieval metrics compare against.
SAMPLE_PDFS = {
    "test.pdf": EVAL_DIR / "pdfs" / "test.pdf",
    "product_requirements.pdf": EVAL_DIR / "pdfs" / "product_requirements.pdf",
    "employee_handbook.pdf": EVAL_DIR / "pdfs" / "employee_handbook.pdf",
}

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_PASS_THRESHOLD = 4

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator judging whether a generated answer \
correctly and adequately answers a question, compared to a reference expected answer.

Score the generated answer from 1 to 5:
5 = fully correct and complete
4 = mostly correct, only minor omissions
3 = partially correct, missing significant information or containing minor inaccuracies
2 = largely incorrect or missing most key information
1 = completely incorrect, irrelevant, or a refusal to answer

Respond with ONLY a JSON object of the form:
{"score": <int 1-5>, "reasoning": "<one sentence explaining the score>"}"""

_inngest_client: inngest.Inngest | None = None
_openai_client: AsyncOpenAI | None = None


def get_inngest_client() -> inngest.Inngest:
    global _inngest_client
    if _inngest_client is None:
        # Falls back to the INNGEST_DEV env var (set INNGEST_DEV=1 to target a
        # local dev server; leave unset to target Inngest Cloud).
        _inngest_client = inngest.Inngest(app_id="rag_app")
    return _inngest_client


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


async def send_event(name: str, data: dict) -> str:
    client = get_inngest_client()
    result = await client.send(inngest.Event(name=name, data=data))
    return result[0]


def _fetch_runs(event_id: str) -> list[dict]:
    url = f"{inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])


async def wait_for_run_output(
    event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5
) -> dict:
    start = time.monotonic()
    last_status = None
    while True:
        runs = await asyncio.to_thread(_fetch_runs, event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status} for event {event_id}")
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (event {event_id}, last status: {last_status})"
            )
        await asyncio.sleep(poll_interval_s)


async def run_query(
    question: str,
    top_k: int,
    collection: str,
    retrieval_mode: str,
    enable_routing: bool,
    timeout_s: float = 120.0,
) -> dict:
    event_id = await send_event(
        "rag/query_pdf_ai",
        {
            "question": question,
            "top_k": top_k,
            "collection": collection,
            "retrieval_mode": retrieval_mode,
            "enable_routing": enable_routing,
        },
    )
    return await wait_for_run_output(event_id, timeout_s=timeout_s)


async def run_ingest(pdf_path: str, source_id: str, collection: str, timeout_s: float = 180.0) -> dict:
    event_id = await send_event(
        "rag/ingest_pdf", {"pdf_path": pdf_path, "source_id": source_id, "collection": collection}
    )
    return await wait_for_run_output(event_id, timeout_s=timeout_s)


async def ingest_sample_pdfs(collection: str) -> None:
    print(f"Ingesting {len(SAMPLE_PDFS)} sample PDFs into Qdrant collection '{collection}' "
          f"(respecting the pipeline's 2/min ingest throttle, this may take a minute or two)...")
    for source_id, path in SAMPLE_PDFS.items():
        if not path.exists():
            print(f"  ! skipping {source_id}: file not found at {path}")
            continue
        print(f"  ingesting {source_id} ...", end=" ", flush=True)
        try:
            result = await run_ingest(str(path), source_id, collection)
            print(f"ingested {result.get('ingested', '?')} chunks")
        except Exception as e:
            print(f"FAILED: {e}")


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")

# Structural/connective words excluded from token-presence matching so that paraphrases like
# "OptiSense Corp, 16 weeks" vs "...supplied by OptiSense Corp and its lead time is 16 weeks"
# still match on their content words.
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was",
    "were", "be", "been", "being", "with", "that", "which", "this", "these", "those", "it",
    "its", "as", "by", "from",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _stem(token: str) -> str:
    # Light plural handling ("reads" -> "read", "bytes" -> "byte") without a full stemmer.
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def exact_match_score(expected_answer: str, variants: list[str] | None, generated: str) -> bool:
    """Order-independent, stopword-filtered, plural-tolerant containment check: a candidate
    matches if every one of its content words appears somewhere in the generated answer
    (directly, or via simple plural stemming). Deterministic and auditable, but tolerant of
    natural paraphrasing (word order, connective phrases) that a strict substring check isn't."""
    normalized_generated_text = normalize(generated)
    generated_tokens = set(normalized_generated_text.split())
    generated_stems = {_stem(t) for t in generated_tokens}

    for candidate in [expected_answer] + list(variants or []):
        normalized_candidate = normalize(candidate)
        if len(normalized_candidate) < 2 and not normalized_candidate.isdigit():
            continue
        cand_tokens = [t for t in normalized_candidate.split() if t not in _STOPWORDS]
        if not cand_tokens:
            # Candidate was entirely stopwords/trivial; fall back to a raw substring check.
            if normalized_candidate and normalized_candidate in normalized_generated_text:
                return True
            continue
        if all(t in generated_tokens or _stem(t) in generated_stems for t in cand_tokens):
            return True
    return False


async def judge_answer(question: str, expected_answer: str, generated_answer: str) -> dict:
    client = get_openai_client()
    user_prompt = (
        f"Question: {question}\n\n"
        f"Reference (expected) answer:\n{expected_answer}\n\n"
        f"Generated answer to evaluate:\n{generated_answer}\n\n"
        "Score the generated answer's accuracy and completeness relative to the reference answer."
    )
    try:
        response = await client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        score = int(parsed["score"])
        score = max(1, min(5, score))
        return {"score": score, "reasoning": parsed.get("reasoning", "")}
    except Exception as e:
        return {"score": 1, "reasoning": f"judge call/parse failed: {e}"}


def retrieval_metrics(source_pdf: str, retrieved_chunks: list[dict]) -> dict:
    sources_ranked = [c.get("source") for c in retrieved_chunks]
    scores_ranked = [c.get("score") for c in retrieved_chunks]
    source_rank = next(
        (i + 1 for i, s in enumerate(sources_ranked) if s == source_pdf), None
    )
    return {
        "retrieved_sources": sources_ranked,
        "retrieved_scores": scores_ranked,
        "source_in_topk": source_rank is not None,
        "source_rank": source_rank,
    }


async def evaluate_item(
    item: dict,
    top_k: int,
    collection: str,
    retrieval_mode: str,
    enable_routing: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        base = {
            "id": item["id"],
            "question": item["question"],
            "question_type": item["question_type"],
            "source_pdf": item["source_pdf"],
            "expected_answer": item["expected_answer"],
        }
        try:
            output = await run_query(item["question"], top_k, collection, retrieval_mode, enable_routing)
        except Exception as e:
            return {**base, "error": str(e), "scoring": {"passed": False}}

        generated = output.get("answer", "")
        retrieved_chunks = output.get("retrieved_chunks", [])
        retrieval = retrieval_metrics(item["source_pdf"], retrieved_chunks)

        if item["question_type"] == "factual":
            passed = exact_match_score(
                item["expected_answer"], item.get("acceptable_variants"), generated
            )
            scoring = {"method": "exact_match", "passed": passed}
        else:
            judge = await judge_answer(item["question"], item["expected_answer"], generated)
            passed = judge["score"] >= JUDGE_PASS_THRESHOLD
            scoring = {
                "method": "llm_judge",
                "passed": passed,
                "judge_score": judge["score"],
                "judge_reasoning": judge["reasoning"],
            }

        return {
            **base,
            "generated_answer": generated,
            "num_contexts": output.get("num_contexts"),
            "sources": output.get("sources"),
            "scoring": scoring,
            "retrieval": retrieval,
            "routing": output.get("routing"),
        }


def compute_summary(
    results: list[dict],
    run_label: str,
    top_k: int,
    collection: str,
    retrieval_mode: str,
    enable_routing: bool,
    dataset_path: Path,
) -> dict:
    total = len(results)
    errored = [r for r in results if "error" in r]
    factual = [r for r in results if r["question_type"] == "factual"]
    open_ended = [r for r in results if r["question_type"] == "open_ended"]
    scored = [r for r in results if "error" not in r]

    exact_match_accuracy = (
        mean(1.0 if r["scoring"]["passed"] else 0.0 for r in factual) if factual else None
    )
    judge_pass_rate = (
        mean(1.0 if r["scoring"]["passed"] else 0.0 for r in open_ended) if open_ended else None
    )
    avg_judge_score = (
        mean(r["scoring"]["judge_score"] for r in open_ended if "judge_score" in r["scoring"])
        if any("judge_score" in r["scoring"] for r in open_ended)
        else None
    )
    overall_pass_rate = mean(1.0 if r["scoring"]["passed"] else 0.0 for r in results) if results else None
    retrieval_hit_rate = (
        mean(1.0 if r["retrieval"]["source_in_topk"] else 0.0 for r in scored) if scored else None
    )
    routed = [r for r in scored if r.get("routing")]
    routing_trigger_rate = (
        mean(1.0 if r["routing"]["triggered"] else 0.0 for r in routed) if routed else None
    )

    return {
        "run_label": run_label,
        "timestamp": datetime.now().isoformat(),
        "top_k": top_k,
        "collection": collection,
        "retrieval_mode": retrieval_mode,
        "enable_routing": enable_routing,
        "dataset_path": str(dataset_path),
        "total_questions": total,
        "factual_questions": len(factual),
        "open_ended_questions": len(open_ended),
        "errors": len(errored),
        "exact_match_accuracy": exact_match_accuracy,
        "judge_pass_rate": judge_pass_rate,
        "avg_judge_score": avg_judge_score,
        "overall_pass_rate": overall_pass_rate,
        "retrieval_hit_rate": retrieval_hit_rate,
        "routing_trigger_rate": routing_trigger_rate,
    }


def print_summary(summary: dict) -> None:
    def fmt_pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    def fmt_num(v):
        return "n/a" if v is None else f"{v:.2f}"

    print("\n" + "=" * 60)
    print(f"  Eval run: {summary['run_label']}  ({summary['timestamp']})")
    print("=" * 60)
    print(f"  Collection:           {summary['collection']}")
    print(f"  Retrieval mode:       {summary['retrieval_mode']}")
    print(f"  Routing enabled:      {summary['enable_routing']}")
    print(f"  Questions:            {summary['total_questions']} "
          f"({summary['factual_questions']} factual, {summary['open_ended_questions']} open-ended)")
    print(f"  Errors:               {summary['errors']}")
    print(f"  Exact-match accuracy: {fmt_pct(summary['exact_match_accuracy'])}")
    print(f"  Judge pass rate:      {fmt_pct(summary['judge_pass_rate'])}")
    print(f"  Avg judge score:      {fmt_num(summary['avg_judge_score'])} / 5")
    print(f"  Overall pass rate:    {fmt_pct(summary['overall_pass_rate'])}")
    print(f"  Retrieval hit-rate:   {fmt_pct(summary['retrieval_hit_rate'])}  (correct source in top-k)")
    print(f"  Routing trigger rate: {fmt_pct(summary['routing_trigger_rate'])}  (low-confidence re-retrieval fired)")
    print("=" * 60 + "\n")


async def main_async(args: argparse.Namespace) -> None:
    if args.ingest:
        await ingest_sample_pdfs(args.collection)

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text())
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"Running {len(dataset)} questions against collection '{args.collection}' "
          f"(top_k={args.top_k}, retrieval_mode={args.retrieval_mode}, "
          f"enable_routing={args.enable_routing}, concurrency={args.concurrency})...")
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        evaluate_item(item, args.top_k, args.collection, args.retrieval_mode, args.enable_routing, semaphore)
        for item in dataset
    ]
    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        result = await coro
        results.append(result)
        status = "ERROR" if "error" in result else ("PASS" if result["scoring"]["passed"] else "FAIL")
        print(f"  [{i}/{len(dataset)}] {result['id']} ... {status}")

    results.sort(key=lambda r: r["id"])
    summary = compute_summary(
        results, args.run_label, args.top_k, args.collection, args.retrieval_mode, args.enable_routing, dataset_path
    )
    print_summary(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{args.run_label}_{timestamp}.json"
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"Results written to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="path to the Q&A dataset JSON")
    parser.add_argument("--top-k", type=int, default=5, help="top_k passed to rag/query_pdf_ai")
    parser.add_argument("--collection", default="eval_docs",
                         help="Qdrant collection to ingest into / query against, isolated from "
                              "manual/Streamlit testing data (default: eval_docs)")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "dense"], default="hybrid",
                         help="retrieval strategy passed to rag/query_pdf_ai (default: hybrid, "
                              "matching the server default; use 'dense' to reproduce the "
                              "original dense-only baseline)")
    parser.add_argument("--enable-routing", action=argparse.BooleanOptionalAction, default=True,
                         help="enable confidence-gated re-retrieval on the hybrid path (default: "
                              "true, matching the server default; pass --no-enable-routing for a "
                              "routing-off comparison run)")
    parser.add_argument("--run-label", default="baseline", help="tag for this run, e.g. 'baseline' or 'hybrid-retrieval'")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
    parser.add_argument("--concurrency", type=int, default=3, help="max concurrent in-flight questions")
    parser.add_argument("--ingest", action="store_true", help="ingest the 3 sample PDFs before running")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        sys.exit(1)
